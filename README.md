---
title: Monetary Policy RL
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
short_description: India monetary policy RL environment — OpenEnv compliant
---

# 🏦 Price Stabilizer: Monetary Policy RL Environment

**OpenEnv-compliant** RL environment where an LLM agent acts as the Reserve Bank of India (RBI) Monetary Policy Committee.

## Environment

- **Name**: `PriceStabilizerEnv-v1`
- **Action space**: 15 discrete actions (repo rate, CRR, SLR, OMO bond operations)
- **Observation**: India macroeconomic dashboard (CPI, GDP, unemployment, FX pressure, commodity shocks)
- **Episode length**: 12 steps (one fiscal year of monthly MPC decisions)
- **Reward**: Multi-objective — inflation control (30%), growth preservation (25%), employment (20%), FX stability (15%), policy consistency (10%)

## HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute policy action |
| `/state` | GET | Current episode state |
| `/actions` | GET | Full action catalogue |

## Quick Start

```python
import requests

BASE = "https://karunarapoluuu-monetary-policy-rl.hf.space"

# Reset
obs = requests.post(f"{BASE}/reset", json={"scenario": "baseline", "seed": 42}).json()
print(obs["observation"])

# Step
result = requests.post(f"{BASE}/step", json={"action_id": 1}).json()
print(f"Reward: {result['reward']}, Done: {result['done']}")

# State
state = requests.get(f"{BASE}/state").json()
print(state)
```

## Scenarios

- `baseline` — Normal 2023–24 India conditions (CPI ~5.5%, GDP ~6.5%)
- `stagflation` — High inflation + low growth (CPI ~8.5%, GDP ~3.5%)
- `high_growth_overheating` — Overheated economy (CPI ~7.2%, GDP ~9%)
- `recession_risk` — Near-recession (GDP ~2.5%, CPI ~2.8%)
- `external_shock` — Oil + food price shock (oil +25%, food +20%)
- `random` — Randomised stress test

## Grading

Episodes are graded A–F based on:
- % of steps with CPI in the 2–6% target band
- Average GDP growth vs 7% potential
- Average unemployment vs 7% benchmark
- Policy stability (no flip-flopping)
- Cumulative reward

## Environment Variables Required

```
API_BASE_URL   # HuggingFace router or OpenAI endpoint
MODEL_NAME     # meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN       # Your HuggingFace API token
```

## OpenEnv Hackathon

Built for the **Meta × Hugging Face × PyTorch OpenEnv Hackathon** (India Round 1).