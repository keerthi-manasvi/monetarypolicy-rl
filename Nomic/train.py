"""
train.py — Multi-episode training loop with lesson accumulation
Runs N episodes of inference.py, tracks scores, shows improvement trend.
"""

import subprocess
import json
import sys
import os
import argparse
import re
from datetime import datetime

def run_episode(scenario: str, seed: int, episode_num: int) -> dict | None:
    """Run one episode via inference.py and parse the END log."""
    cmd = [
        sys.executable, "inference.py",
        "--scenario", scenario,
        "--seed", str(seed),
        "--verbose"
    ]
    
    print(f"\n{'='*60}", flush=True)
    print(f"  EPISODE {episode_num} | Scenario: {scenario} | Seed: {seed}", flush=True)
    print(f"{'='*60}", flush=True)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print verbose output (goes to stderr in inference.py)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    
    # Parse [END] line from stdout
    for line in result.stdout.splitlines():
        if line.startswith("[END]"):
            try:
                payload = json.loads(line[6:])
                return payload.get("grading")
            except json.JSONDecodeError:
                pass
    
    return None


def print_progress_table(history: list):
    """Print a table of all episodes so far."""
    print("\n" + "="*75, flush=True)
    print(f"  TRAINING PROGRESS ({len(history)} episodes)", flush=True)
    print("="*75, flush=True)
    print(f"  {'Ep':>3}  {'Score':>6}  {'Grade':>5}  {'Avg CPI':>8}  {'Avg GDP':>8}  {'Inf Mgmt':>9}  {'Growth':>7}", flush=True)
    print(f"  {'-'*3}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}", flush=True)
    
    for h in history:
        ep   = h["episode"]
        sc   = h["score"]
        gr   = h["grade"]
        cpi  = h["avg_cpi"]
        gdp  = h["avg_gdp"]
        inf  = h["inflation_management"]
        grow = h["growth_preservation"]
        
        # Colour-code grade
        grade_str = f"[{gr}]"
        print(f"  {ep:>3}  {sc:>6.2f}  {grade_str:>5}  {cpi:>7.2f}%  {gdp:>7.2f}%  {inf:>9.2f}  {grow:>7.2f}", flush=True)
    
    if len(history) > 1:
        scores = [h["score"] for h in history]
        best   = max(scores)
        latest = scores[-1]
        trend  = latest - scores[0]
        print(f"\n  Best score : {best:.2f}", flush=True)
        print(f"  Latest     : {latest:.2f}", flush=True)
        print(f"  Trend      : {trend:+.2f} (first → latest)", flush=True)
    
    print("="*75 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-episode training loop")
    parser.add_argument("--scenario", default="external_shock",
                        choices=["baseline", "stagflation", "high_growth_overheating",
                                 "recession_risk", "external_shock", "random"])
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--seed-start", type=int, default=42,
                        help="Starting seed (increments each episode)")
    parser.add_argument("--same-seed", action="store_true",
                        help="Use the same seed every episode (for controlled comparison)")
    args = parser.parse_args()

    print(f"\n🏦 Price Stabilizer — Training Loop", flush=True)
    print(f"   Scenario : {args.scenario}", flush=True)
    print(f"   Episodes : {args.episodes}", flush=True)
    print(f"   Seed mode: {'fixed' if args.same_seed else 'incrementing'}", flush=True)
    print(f"   Started  : {datetime.now().strftime('%H:%M:%S')}", flush=True)

    history = []
    
    for ep in range(1, args.episodes + 1):
        seed = args.seed_start if args.same_seed else args.seed_start + ep - 1
        
        grading = run_episode(args.scenario, seed, ep)
        
        if grading is None:
            print(f"  [WARN] Episode {ep} failed to parse grading.", flush=True)
            continue
        
        history.append({
            "episode": ep,
            "score": grading["overall_score"],
            "grade": grading["grade"],
            "avg_cpi": grading["statistics"]["avg_cpi_inflation"],
            "avg_gdp": grading["statistics"]["avg_gdp_growth"],
            "inflation_management": grading["component_scores"]["inflation_management"],
            "growth_preservation":  grading["component_scores"]["growth_preservation"],
            "employment":           grading["component_scores"]["employment"],
            "policy_stability":     grading["component_scores"]["policy_stability"],
        })
        
        # Print running progress after each episode
        print_progress_table(history)
    
    # Save full history
    out_path = f"training_history_{args.scenario}.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()