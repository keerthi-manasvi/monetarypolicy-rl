"""
inference.py  — Price Stabilizer: Monetary Policy RL Simulator
=================================================================
OpenEnv Hackathon Round 1 submission entry point.

Required by the hackathon spec:
  • defines API_BASE_URL, MODEL_NAME, HF_TOKEN
  • uses OpenAI Client for all LLM calls
  • emits structured stdout logs per spec:
      [START] task=<task_name> env=<benchmark> model=<model_name>
      [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
      [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Author: OpenEnv Hackathon
Environment: PriceStabilizerEnv-v1 (India monetary policy)
"""

import os
import sys
import json
import re
import time
from typing import List, Optional

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

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN:     str = os.getenv("HF_TOKEN",     "")

_use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
if _use_openai:
    API_BASE_URL = "https://api.openai.com/v1"
    MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")

TASK_NAME           = os.getenv("PRICE_STABILIZER_TASK",      "monetary-policy")
BENCHMARK           = os.getenv("PRICE_STABILIZER_BENCHMARK", "PriceStabilizerEnv-v1")
SUCCESS_SCORE_THRESHOLD = 0.6   # overall_score / 10 >= 0.6  → success

# ── OpenAI-compatible client ───────────────────────────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "no-key"),
    base_url=API_BASE_URL,
)

# ── System prompt ──────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are an expert monetary policy economist for the Reserve Bank of India (RBI).
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


# ── Stdout logging helpers (spec-compliant) ────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise action string: no spaces (use underscores) to keep line parseable
    action_safe = action.replace(" ", "_")
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Lesson persistence (cross-episode learning) ───────────────────────────────
def lessons_path(scenario: str) -> str:
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, f"lessons_{scenario}.json")


def load_lessons(scenario: str) -> str:
    path = lessons_path(scenario)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        lessons = json.load(f)
    recent = lessons[-5:]
    return "\n".join(
        f"Past episode score {l['score']}: {l['lesson']}" for l in recent
    )


def call_llm_for_summary(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=256,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def save_lesson(scenario: str, score: float, history: list) -> None:
    summary_prompt = (
        f"You just completed a monetary policy episode.\n"
        f"Score: {score}/10\n"
        f"Trajectory: {json.dumps(history, indent=2)}\n\n"
        f"In 2 sentences, what should a future agent do differently in this scenario to score higher?"
    )
    lesson = call_llm_for_summary(summary_prompt)

    path = lessons_path(scenario)
    lessons = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lessons = json.load(f)
    lessons.append({"score": score, "lesson": lesson})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lessons, f, indent=2)


def get_system_prompt(scenario: str) -> str:
    lessons = load_lessons(scenario)
    lessons_section = (
        "LESSONS FROM PAST EPISODES:\n" + lessons + "\n\nUse these lessons to make better decisions."
        if lessons
        else "LESSONS FROM PAST EPISODES:\nNone yet."
    )
    return f"{BASE_SYSTEM_PROMPT}\n\n{lessons_section}\n"


# ── LLM policy agent ───────────────────────────────────────────────────────────
def call_llm_agent(
    observation: str,
    step: int,
    scenario: str,
    episode_history: list,
    max_retries: int = 3,
) -> tuple[int, str]:
    """Call the LLM to decide the next policy action. Returns (action_id, reasoning)."""

    history_text = "\n".join(
        f"Month {h['step']}: Action={h['action']} → CPI={h['cpi']:.1f}% "
        f"GDP={h['gdp']:.1f}% Reward={h['reward']:+.1f}"
        for h in episode_history
    ) or "None yet — this is the first step."

    user_message = (
        f"Month {step}/{MAX_STEPS}\n\n"
        f"PAST DECISIONS THIS EPISODE:\n{history_text}\n\n"
        f"CURRENT DASHBOARD:\n{observation}\n\n"
        f"Review what has and hasn't worked. Then choose your action.\n"
        f"Respond with ONLY the JSON object as specified in your instructions."
    )

    raw = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": get_system_prompt(scenario)},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=256,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",           "", raw)

            parsed    = json.loads(raw)
            action_id = int(parsed.get("action_id", 0))
            reasoning = str(parsed.get("reasoning", "No reasoning provided."))

            if action_id not in ACTIONS:
                action_id = 0
                reasoning = f"(Clamped to hold — original id out of range.) {reasoning}"

            return action_id, reasoning

        except json.JSONDecodeError:
            match = re.search(r'"action_id"\s*:\s*(\d+)', raw)
            if match:
                return int(match.group(1)) % len(ACTIONS), "Parsed via regex fallback."
            time.sleep(1)

        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"[DEBUG] LLM call failed after {max_retries} attempts: {exc}", flush=True)
                return 0, f"Error: {exc} — defaulting to hold_policy."
            time.sleep(2 ** attempt)

    return 0, "Max retries exceeded — holding policy."


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run one complete episode of the Price Stabilizer environment.
    Emits spec-compliant [START], [STEP], [END] lines to stdout.
    Returns the grading report.
    """
    current_scenario = scenario or "baseline"
    env = PriceStabilizerEnv(scenario=scenario, seed=seed)

    rewards:         List[float] = []
    episode_history: list        = []
    steps_taken:     int         = 0
    score:           float       = 0.0
    success:         bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action_id, reasoning = call_llm_agent(obs, step, current_scenario, episode_history)
            action_name = ACTIONS[action_id]["name"]

            next_obs, reward, done, info = env.step(action_id)

            rewards.append(reward)
            steps_taken = step

            error = info.get("error") or None
            log_step(step=step, action=action_name, reward=reward, done=done, error=error)

            if verbose:
                cpi = info["state"]["cpi_inflation"]
                gdp = info["state"]["gdp_growth"]
                print(
                    f"  Step {step:2d} | {action_name:<30s} | "
                    f"Reward: {reward:+.2f} | CPI: {cpi:.2f}% | GDP: {gdp:.2f}% | "
                    f"{reasoning}",
                    file=sys.stderr,
                )

            episode_history.append({
                "step":   step,
                "action": action_name,
                "cpi":    info["state"]["cpi_inflation"],
                "gdp":    info["state"]["gdp_growth"],
                "reward": reward,
            })

            obs = next_obs

            if done:
                break

        total_reward   = sum(rewards)
        grading_report = grade_episode(env.history, env.state, total_reward)

        # Normalise score to [0, 1] using the 0-10 overall_score
        score   = min(max(grading_report["overall_score"] / 10.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close() if hasattr(env, "close") else None
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    save_lesson(current_scenario, grading_report["overall_score"], episode_history)

    if verbose:
        _pretty_grade(grading_report)

    return grading_report


# ── Human-readable grade summary ──────────────────────────────────────────────
def _pretty_grade(report: dict) -> None:
    print("\n" + "=" * 60, file=sys.stderr)
    print("  EPISODE GRADING REPORT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Overall Score : {report['overall_score']} / 10.0", file=sys.stderr)
    print(f"  Grade         : {report['grade']}",                file=sys.stderr)
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
        choices=["baseline", "stagflation", "high_growth_overheating",
                 "recession_risk", "external_shock", "random"],
        default="baseline",
        help="Starting scenario for the episode",
    )
    parser.add_argument("--seed",    type=int,          default=42,  help="Random seed")
    parser.add_argument("--verbose", action="store_true",            help="Print human-readable progress to stderr")
    args = parser.parse_args()

    scenario = None if args.scenario == "baseline" else args.scenario
    report   = run_episode(scenario=scenario, seed=args.seed, verbose=args.verbose)
    sys.exit(0 if report.get("grade") not in ("F",) else 1)