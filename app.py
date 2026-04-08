"""
server/app.py — OpenEnv-compliant FastAPI server for PriceStabilizerEnv
========================================================================
Exposes the three required OpenEnv HTTP endpoints:
  POST /reset          → initialise episode, return observation
  POST /step           → execute action, return observation + reward
  GET  /state          → return current episode state
  GET  /health         → liveness probe (must return 200)
  GET  /actions        → list available actions

The automated hackathon ping hits /health and then calls reset().
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our environment
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from price_stabilizer_env import (
    PriceStabilizerEnv,
    ACTIONS,
    ACTION_LIST_TEXT,
    grade_episode,
    INFLATION_TARGET,
    INFLATION_BAND_LOW,
    INFLATION_BAND_HIGH,
    MAX_STEPS,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PriceStabilizerEnv",
    description="India monetary policy RL environment — OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global environment instance (one per container — stateful) ────────────────
_env: Optional[PriceStabilizerEnv] = None
_total_reward: float = 0.0
_episode_history: list = []


# ── Pydantic models ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    scenario: Optional[str] = None   # baseline | stagflation | external_shock | ...
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action_id: int                    # 0–14


class Observation(BaseModel):
    observation: str                  # human-readable macro dashboard text
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = {}


class StateResponse(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    scenario: Optional[str] = None
    total_reward: float = 0.0
    done: bool = False
    macro: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — hackathon validator pings this first."""
    return {"status": "ok", "environment": "PriceStabilizerEnv-v1"}


@app.get("/")
def root():
    """Root endpoint — returns environment info."""
    return {
        "environment": "PriceStabilizerEnv-v1",
        "description": "India monetary policy RL environment",
        "endpoints": ["/health", "/reset", "/step", "/state", "/actions"],
        "inflation_target": INFLATION_TARGET,
        "inflation_band": [INFLATION_BAND_LOW, INFLATION_BAND_HIGH],
        "max_steps": MAX_STEPS,
        "n_actions": len(ACTIONS),
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = None):
    """
    OpenEnv required: initialise a new episode.
    Returns initial observation.
    """
    global _env, _total_reward, _episode_history

    if req is None:
        req = ResetRequest()

    scenario = req.scenario
    # Normalise "baseline" → None (env treats None as baseline)
    if scenario == "baseline":
        scenario = None

    _env = PriceStabilizerEnv(scenario=scenario, seed=req.seed)
    obs_text = _env.reset()
    _total_reward = 0.0
    _episode_history = []

    return Observation(
        observation=obs_text,
        reward=0.0,
        done=False,
        metadata={
            "scenario": req.scenario or "baseline",
            "seed": req.seed,
            "max_steps": MAX_STEPS,
            "inflation_target": INFLATION_TARGET,
            "action_space_size": len(ACTIONS),
        },
    )


@app.post("/step", response_model=Observation)
def step(req: StepRequest):
    """
    OpenEnv required: execute one policy action.
    Returns next observation, reward, done flag.
    """
    global _env, _total_reward, _episode_history

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")

    if req.action_id not in ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_id {req.action_id}. Valid range: 0–{len(ACTIONS)-1}",
        )

    next_obs, reward, done, info = _env.step(req.action_id)
    _total_reward += reward

    state = info["state"]
    _episode_history.append({
        "step": state["step"],
        "action_id": req.action_id,
        "action_name": ACTIONS[req.action_id]["name"],
        "reward": reward,
        "cpi_inflation": state["cpi_inflation"],
        "gdp_growth": state["gdp_growth"],
        "unemployment_rate": state["unemployment_rate"],
        "repo_rate": state["repo_rate"],
    })

    metadata = {
        "action_taken": ACTIONS[req.action_id]["name"],
        "action_type": ACTIONS[req.action_id]["type"],
        "step": state["step"],
        "episode_total_reward": round(_total_reward, 4),
        "reward_breakdown": info["reward_breakdown"],
        "state": {
            "cpi_inflation": round(state["cpi_inflation"], 2),
            "gdp_growth": round(state["gdp_growth"], 2),
            "unemployment_rate": round(state["unemployment_rate"], 2),
            "repo_rate": round(state["repo_rate"], 2),
            "currency_pressure": round(state["currency_pressure"], 2),
        },
    }

    # If episode is done, attach grading
    if done and _env.state is not None:
        grading = grade_episode(_env.history, _env.state, _total_reward)
        metadata["grading"] = grading

    return Observation(
        observation=next_obs,
        reward=round(reward, 4),
        done=done,
        metadata=metadata,
    )


@app.get("/state", response_model=StateResponse)
def state():
    """
    OpenEnv required: return current episode metadata.
    """
    global _env, _total_reward

    if _env is None or _env.state is None:
        return StateResponse(step_count=0, done=False)

    s = _env.state
    return StateResponse(
        step_count=s.step,
        scenario=_env.scenario or "baseline",
        total_reward=round(_total_reward, 4),
        done=s.done,
        macro={
            "cpi_inflation": round(s.cpi_inflation, 2),
            "gdp_growth": round(s.gdp_growth, 2),
            "unemployment_rate": round(s.unemployment_rate, 2),
            "repo_rate": round(s.repo_rate, 2),
            "currency_pressure": round(s.currency_pressure, 2),
            "oil_price_shock": round(s.oil_price_shock, 2),
            "global_food_shock": round(s.global_food_shock, 2),
        },
    )


@app.get("/actions")
def actions():
    """Return full action catalogue."""
    return {
        "n_actions": len(ACTIONS),
        "actions": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "type": v["type"],
            }
            for k, v in ACTIONS.items()
        ],
    }