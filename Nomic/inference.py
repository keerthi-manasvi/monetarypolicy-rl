"""
inference.py  — Price Stabilizer: Monetary Policy RL Simulator
=================================================================
OpenEnv Hackathon Round 1 submission entry point.

Required by the hackathon spec:
  • defines API_BASE_URL, MODEL_NAME, HF_TOKEN
  • uses OpenAI Client for all LLM calls
  • emits structured stdout logs: [START], [STEP], [END]

Author: OpenEnv Hackathon
Environment: PriceStabilizerEnv-v1 (India monetary policy)
"""

import os
import sys
import json
import re
import time
from typing import Optional

from openai import OpenAI
from price_stabilizer_env import (
    PriceStabilizerEnv,
    ACTIONS,
    ACTION_LIST_TEXT,
    grade_episode,
    MAX_STEPS,
    INFLATION_TARGET,
    INFLATION_BAND_LOW,
    INFLATION_BAND_HIGH,
)

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN:     str = os.getenv("HF_TOKEN",     "")   # required at runtime

# Fallback to OpenAI if explicitly requested
_use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
if _use_openai:
    API_BASE_URL = "https://api.openai.com/v1"
    MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ── OpenAI-compatible client (works with HF Inference & OpenAI) ───────────────
client = OpenAI(
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "no-key"),
    base_url=API_BASE_URL,
)

# ── System prompt for the LLM policy agent ────────────────────────────────────
SYSTEM_PROMPT = """You are an expert monetary policy economist for the Reserve Bank of India (RBI).
Your objective is to act as the monetary policy committee, choosing the right policy instruments each month to:
1. Keep headline CPI inflation close to the 4% target (tolerance band: 2%–6%)
2. Preserve real GDP growth (ideally above 6.5%)
3. Reduce unemployment
4. Maintain INR exchange rate stability
5. Avoid excessive policy oscillation (don't flip-flop)

You will receive the current macroeconomic dashboard. Analyse it carefully and choose exactly ONE action.

Respond with ONLY a JSON object in this exact format (no other text):
{
  "action_id": <integer 0-14>,
  "reasoning": "<brief 2-3 sentence explanation of your decision>"
}

Available actions:
""" + ACTION_LIST_TEXT


def call_llm_agent(observation: str, step: int, max_retries: int = 3) -> tuple[int, str]:
    """
    Call the LLM agent to decide the next policy action.
    Returns (action_id, reasoning).
    """
    user_message = (
        f"Month {step}/{MAX_STEPS} — Analyse the following India macro dashboard and choose your policy action:\n\n"
        f"{observation}\n\n"
        "Respond with ONLY the JSON object as specified in your instructions."
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=256,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)
            action_id = int(parsed.get("action_id", 0))
            reasoning = str(parsed.get("reasoning", "No reasoning provided."))

            if action_id not in ACTIONS:
                action_id = 0
                reasoning = f"(Clamped to hold — original id out of range.) {reasoning}"

            return action_id, reasoning

        except json.JSONDecodeError:
            # Try to extract action_id with regex fallback
            match = re.search(r'"action_id"\s*:\s*(\d+)', raw if 'raw' in dir() else "")
            if match:
                return int(match.group(1)) % len(ACTIONS), "Parsed via regex fallback."
            time.sleep(1)

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [WARN] LLM call failed after {max_retries} attempts: {e}", file=sys.stderr)
                return 0, f"Error: {str(e)} — defaulting to hold_policy."
            time.sleep(2 ** attempt)

    return 0, "Max retries exceeded — holding policy."


def run_episode(
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run one complete episode of the Price Stabilizer environment.
    Emits [START], [STEP], [END] structured logs to stdout.
    Returns the grading report.
    """
    env = PriceStabilizerEnv(scenario=scenario, seed=seed)

    # ── [START] log ───────────────────────────────────────────────────────────
    start_payload = {
        "event": "START",
        "environment": env.metadata["name"],
        "model": MODEL_NAME,
        "scenario": scenario or "baseline",
        "max_steps": MAX_STEPS,
        "inflation_target": INFLATION_TARGET,
        "inflation_band": [INFLATION_BAND_LOW, INFLATION_BAND_HIGH],
    }
    print(f"[START] {json.dumps(start_payload)}", flush=True)

    obs = env.reset()
    total_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        # Get agent decision
        action_id, reasoning = call_llm_agent(obs, step)
        action_name = ACTIONS[action_id]["name"]

        # Execute step
        next_obs, reward, done, info = env.step(action_id)
        total_reward += reward

        # ── [STEP] log ────────────────────────────────────────────────────────
        step_payload = {
            "event": "STEP",
            "step": step,
            "action_id": action_id,
            "action": action_name,
            "reasoning": reasoning,
            "reward": reward,
            "episode_total_reward": round(total_reward, 4),
            "cpi_inflation": round(info["state"]["cpi_inflation"], 2),
            "gdp_growth": round(info["state"]["gdp_growth"], 2),
            "unemployment_rate": round(info["state"]["unemployment_rate"], 2),
            "repo_rate": round(info["state"]["repo_rate"], 2),
            "reward_breakdown": info["reward_breakdown"],
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

        if verbose:
            print(
                f"  Step {step:2d} | Action: {action_name:<30s} | "
                f"Reward: {reward:+.2f} | CPI: {step_payload['cpi_inflation']:.2f}% | "
                f"GDP: {step_payload['gdp_growth']:.2f}%",
                file=sys.stderr,
            )

        obs = next_obs

        if done:
            break

    # ── Grade the episode ──────────────────────────────────────────────────────
    grading_report = grade_episode(env.history, env.state, total_reward)

    # ── [END] log ──────────────────────────────────────────────────────────────
    end_payload = {
        "event": "END",
        "scenario": scenario or "baseline",
        "steps_completed": len(env.history),
        "total_reward": round(total_reward, 4),
        "grading": grading_report,
    }
    print(f"[END] {json.dumps(end_payload)}", flush=True)

    if verbose:
        _pretty_grade(grading_report)

    return grading_report


def _pretty_grade(report: dict) -> None:
    """Print a human-friendly grade summary to stderr."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("  EPISODE GRADING REPORT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Overall Score : {report['overall_score']} / 10.0", file=sys.stderr)
    print(f"  Grade         : {report['grade']}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Component Scores:", file=sys.stderr)
    for k, v in report["component_scores"].items():
        print(f"    {k:<30s}: {v:.2f}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Statistics:", file=sys.stderr)
    for k, v in report["statistics"].items():
        print(f"    {k:<30s}: {v}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Final State:", file=sys.stderr)
    for k, v in report["final_state"].items():
        print(f"    {k:<30s}: {v}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Price Stabilizer — OpenEnv Hackathon Demo")
    parser.add_argument(
        "--scenario",
        choices=["baseline", "stagflation", "high_growth_overheating", "recession_risk", "external_shock", "random"],
        default="baseline",
        help="Starting scenario for the episode",
    )
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed")
    parser.add_argument("--verbose", action="store_true",   help="Print human-readable progress to stderr")
    args = parser.parse_args()

    scenario = None if args.scenario == "baseline" else args.scenario

    report = run_episode(scenario=scenario, seed=args.seed, verbose=args.verbose)
    sys.exit(0 if report.get("grade") not in ("F",) else 1)