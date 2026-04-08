# 🏦 Price Stabilizer: Monetary Policy RL Simulator

**OpenEnv Hackathon Round 1 — Meta × Hugging Face**

> An RL environment where an AI agent acts as the Reserve Bank of India's Monetary Policy Committee — navigating inflation, growth, jobs, and geopolitical shocks through monetary policy instruments.

[![HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/price-stabilizer)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](https://huggingface.co/openenv)

---

## 🎯 Overview

**Price Stabilizer** is an India-specific macroeconomic simulation environment built for the OpenEnv framework. An LLM agent observes a rich macroeconomic dashboard and decides monthly monetary policy actions — exactly as the RBI MPC does in real life.

The agent must balance:
- 📉 Bringing CPI inflation to the 4% target (tolerance: 2–6%)
- 📈 Preserving real GDP growth (ideally ≥6.5%)
- 👷 Reducing unemployment
- 💱 Maintaining INR exchange rate stability
- ⚖️ Avoiding excessive policy oscillation

---

## 🏗️ Environment Design

### State Space (19 variables)

| Category | Variables |
|----------|-----------|
| **Inflation** | Headline CPI, Food Inflation, Core Inflation |
| **Growth** | Real GDP growth, IIP (Industrial Output), Private Consumption |
| **Labour** | Unemployment Rate (CMIE benchmark) |
| **External** | USD/INR pressure, Current Account Deficit, Forex Reserves Index |
| **Shocks** | Oil Price Shock (Brent deviation), Global Food Price Shock |
| **Policy Instruments** | Repo Rate, Reverse Repo, CRR, SLR, Net OMO Bond Balance |
| **Sentiment** | Market Sentiment (-1 to +1), Geopolitical Risk (0 to 1) |

### Action Space (15 actions)

| ID | Action | Type |
|----|--------|------|
| 0  | Hold policy steady | Neutral |
| 1  | Increase repo rate +25bp | Hawkish |
| 2  | Increase repo rate +50bp | Strongly Hawkish |
| 3  | Decrease repo rate -25bp | Dovish |
| 4  | Decrease repo rate -50bp | Strongly Dovish |
| 5  | Increase reverse repo +25bp | Hawkish |
| 6  | Decrease reverse repo -25bp | Dovish |
| 7  | Increase CRR +50bp | Hawkish |
| 8  | Decrease CRR -50bp | Dovish |
| 9  | Increase SLR +100bp | Hawkish |
| 10 | Decrease SLR -100bp | Dovish |
| 11 | OMO: Buy bonds (1 unit = ₹50K Cr) | Dovish |
| 12 | OMO: Buy bonds large (3 units) | Strongly Dovish |
| 13 | OMO: Sell bonds (1 unit) | Hawkish |
| 14 | OMO: Sell bonds large (3 units) | Strongly Hawkish |

### Episode Structure
- **12 steps** = 1 fiscal year of monthly MPC decisions
- Each step, the agent observes the macro dashboard and chooses one action
- Macro state evolves via calibrated India-specific dynamics

---

## 🏆 Reward Function

The multi-objective reward function balances:

```
reward = inflation_component    (weight: ~40%)
       + growth_component       (weight: ~25%)
       + unemployment_component (weight: ~15%)
       + currency_stability     (weight: ~10%)
       + policy_consistency     (weight: ~10%)
```

**Key incentives:**
- ✅ +4.0 when CPI is inside 2–6% comfort band
- ✅ +3.0 when GDP growth ≥ 6.5%
- ✅ +2.0 when unemployment ≤ 7%
- ❌ −5 × deviation when CPI > 8% (runaway inflation)
- ❌ −4 × deviation when GDP < 2% (growth collapse)
- ✅ +1.5 for effective shock response
- ❌ −2.0 if unemployment spikes 0.5pp+ in a single step

Reward is clipped to **[-10, +10]** per step.

---

## 📊 Grading System

Each episode is graded on 5 components (0–10 each):

| Component | Weight | Description |
|-----------|--------|-------------|
| Inflation Management | 30% | % of steps with CPI in target band |
| Growth Preservation | 25% | Average GDP growth vs potential |
| Employment | 20% | Average unemployment vs 7% benchmark |
| Policy Stability | 15% | Absence of flip-flop behaviour |
| Cumulative Reward | 10% | Normalised total episode reward |

**Grades:** A (≥8.0) · B (≥6.5) · C (≥5.0) · D (≥3.5) · F (<3.5)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/price-stabilizer
cd price-stabilizer
pip install -r requirements.txt
```

### Demo (no API key required)

```bash
# Baseline scenario with heuristic agent
python demo.py

# Try different stress scenarios
python demo.py --scenario stagflation
python demo.py --scenario recession_risk
python demo.py --scenario external_shock
python demo.py --scenario random --seed 123
```

### LLM Agent (requires HF Token)

```bash
export HF_TOKEN=hf_your_token_here
python inference.py --scenario stagflation --verbose --seed 42
```

### Environment only (Python API)

```python
from environments.price_stabilizer_env import PriceStabilizerEnv, grade_episode

env = PriceStabilizerEnv(scenario="stagflation", seed=42)
obs = env.reset()

for step in range(12):
    # Your agent decides action
    action_id = your_agent(obs)  # 0–14
    obs, reward, done, info = env.step(action_id)
    print(f"Step {step+1}: reward={reward:.2f}, CPI={info['state']['cpi_inflation']:.2f}%")
    if done:
        break

report = grade_episode(env.history, env.state, cumulative_reward)
print(f"Grade: {report['grade']}  Score: {report['overall_score']}/10")
```

---

## 📋 OpenEnv Interface Compliance

| Requirement | Status |
|-------------|--------|
| `reset() -> str` | ✅ |
| `step(action_id) -> (obs, reward, done, info)` | ✅ |
| `get_action_space() -> list[str]` | ✅ |
| `inference.py` in project root | ✅ |
| `API_BASE_URL` defined | ✅ |
| `MODEL_NAME` defined | ✅ |
| `HF_TOKEN` defined | ✅ |
| OpenAI Client for LLM calls | ✅ |
| `[START]` log format | ✅ |
| `[STEP]` log format | ✅ |
| `[END]` log format | ✅ |

---

## 🧪 Scenarios

| Scenario | Description | Initial CPI | Initial GDP |
|----------|-------------|-------------|-------------|
| `baseline` | Normal India conditions (2023–24) | 5.5% | 6.5% |
| `stagflation` | High inflation + weak growth + oil shock | 8.5% | 3.5% |
| `high_growth_overheating` | Boom with rising prices | 7.2% | 9.0% |
| `recession_risk` | Growth collapse, low inflation | 2.8% | 2.5% |
| `external_shock` | Oil/food/FX triple shock | 7.0% | 6.0% |
| `random` | Randomised stress test | Random | Random |

---

## 📁 Project Structure

```
price-stabilizer/
├── inference.py              ← OpenEnv entry point (LLM agent runner)
├── demo.py                   ← Quick demo with heuristic agent
├── app.py                    ← Gradio Spaces UI
├── requirements.txt
├── README.md
├── environments/
│   ├── __init__.py
│   └── price_stabilizer_env.py  ← Core environment (state, actions, reward, grading)
└── tests/
    └── test_env.py           ← Unit tests
```

---

## 🔬 Technical Details

### Macro Dynamics
- **Inflation** modelled via oil pass-through, food shocks, demand-pull, and monetary transmission lag
- **GDP** responds to tightness score (repo + CRR + SLR + OMO net effect)
- **Unemployment** follows India-calibrated Okun's Law with a GDP gap proxy
- **Currency** responds to rate differentials, inflation, oil, and geopolitical risk
- **Shocks** follow mean-reverting AR(1) processes with calibrated volatility

### Instrument Constraints
| Instrument | Min | Max |
|-----------|-----|-----|
| Repo Rate | 3.0% | 10.0% |
| Reverse Repo | Repo-0.5 floor | Repo ceiling |
| CRR | 3.0% | 10.0% |
| SLR | 18.0% | 30.0% |

---

## 🤗 Hugging Face Spaces

Live demo: **[https://huggingface.co/spaces/YOUR_USERNAME/price-stabilizer](https://huggingface.co/spaces/YOUR_USERNAME/price-stabilizer)**

Choose a scenario, pick your agent (heuristic or LLM), and watch the RBI navigate the economy.

---

## 📜 License

MIT License — see [LICENSE](LICENSE)
