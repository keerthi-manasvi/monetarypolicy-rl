"""
demo.py — Price Stabilizer: Quick Demo (no LLM API required)
=============================================================
Runs a single episode with a rule-based heuristic agent to validate
the environment works correctly before deploying.

Usage:
    python demo.py [--scenario stagflation|recession_risk|external_shock|random]
"""

import json
import sys
import argparse
from price_stabilizer_env import (
    PriceStabilizerEnv,
    ACTIONS,
    grade_episode,
    INFLATION_TARGET,
    INFLATION_BAND_LOW,
    INFLATION_BAND_HIGH,
    GDP_GROWTH_STAGNATION,
)


def heuristic_agent(state) -> int:
    """
    Simple rule-based heuristic:
      - If inflation very high (>7%) → hike repo 50bp
      - If inflation high (>6%) → hike repo 25bp
      - If inflation in band AND growth low (<4.5%) → cut repo 25bp
      - If growth collapsing (<3%) → cut repo 50bp + sell bonds
      - If oil shock AND inflation high → sell bonds
      - Otherwise hold
    """
    inf = state.cpi_inflation
    gdp = state.gdp_growth
    unemp = state.unemployment_rate

    if inf > 7.5:
        return 2   # increase_repo_rate_50bp
    elif inf > 6.0:
        return 1   # increase_repo_rate_25bp
    elif inf < INFLATION_BAND_LOW and gdp < 4.0:
        return 4   # decrease_repo_rate_50bp (stimulate)
    elif inf < INFLATION_BAND_HIGH and gdp < GDP_GROWTH_STAGNATION:
        return 3   # decrease_repo_rate_25bp (mild stimulus)
    elif gdp < 3.0:
        return 11  # buy_bonds_small (inject liquidity)
    elif unemp > 12.0:
        return 8   # decrease_crr_50bp (ease bank lending)
    else:
        return 0   # hold_policy


def run_demo(scenario=None, seed=42, verbose=True):
    env = PriceStabilizerEnv(scenario=scenario, seed=seed)
    obs = env.reset()

    print("\n" + "=" * 65)
    print("  PRICE STABILIZER — DEMO RUN  (Heuristic Agent)")
    print(f"  Scenario: {scenario or 'baseline'}  |  Seed: {seed}")
    print("=" * 65)
    print("\nInitial State:")
    print(obs)
    print()

    start_payload = {
        "event": "START",
        "environment": "PriceStabilizerEnv-v1",
        "model": "heuristic_agent",
        "scenario": scenario or "baseline",
    }
    print(f"[START] {json.dumps(start_payload)}", flush=True)

    total_reward = 0.0

    for step in range(1, 13):
        action_id = heuristic_agent(env.state)
        action_name = ACTIONS[action_id]["name"]

        obs, reward, done, info = env.step(action_id)
        total_reward += reward

        step_payload = {
            "event": "STEP",
            "step": step,
            "action_id": action_id,
            "action": action_name,
            "reward": reward,
            "episode_total_reward": round(total_reward, 4),
            "cpi_inflation":    round(info["state"]["cpi_inflation"], 2),
            "gdp_growth":       round(info["state"]["gdp_growth"], 2),
            "unemployment_rate":round(info["state"]["unemployment_rate"], 2),
            "repo_rate":        round(info["state"]["repo_rate"], 2),
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

        if verbose:
            print(
                f"  Step {step:2d} | {action_name:<32} | "
                f"R={reward:+5.2f} | "
                f"CPI={step_payload['cpi_inflation']:.2f}% | "
                f"GDP={step_payload['gdp_growth']:.2f}% | "
                f"Unemp={step_payload['unemployment_rate']:.2f}%"
            )

        if done:
            break

    grading = grade_episode(env.history, env.state, total_reward)

    end_payload = {
        "event": "END",
        "steps_completed": len(env.history),
        "total_reward": round(total_reward, 4),
        "grading": grading,
    }
    print(f"[END] {json.dumps(end_payload)}", flush=True)

    # ── Pretty print grading ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GRADING REPORT")
    print("=" * 65)
    print(f"  Overall Score : {grading['overall_score']} / 10.0  (Grade: {grading['grade']})")
    print()
    print("  Component Scores:")
    for k, v in grading["component_scores"].items():
        bar = "█" * int(v) + "░" * (10 - int(v))
        print(f"    {k:<32}: {bar} {v:.2f}")
    print()
    print("  Statistics:")
    for k, v in grading["statistics"].items():
        print(f"    {k:<35}: {v}")
    print("=" * 65 + "\n")

    return grading


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=["baseline", "stagflation", "high_growth_overheating",
                 "recession_risk", "external_shock", "random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenario = None if args.scenario == "baseline" else args.scenario
    run_demo(scenario=scenario, seed=args.seed)
