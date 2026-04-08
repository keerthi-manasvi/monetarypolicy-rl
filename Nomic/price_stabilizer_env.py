"""
Price Stabilizer: Monetary Policy RL Environment
India-specific macroeconomic simulation for OpenEnv hackathon.
"""

import json
import random
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# ── India-specific macro constants ──────────────────────────────────────────
INFLATION_TARGET = 4.0          # RBI's headline CPI target (%)
INFLATION_BAND_LOW = 2.0        # Lower tolerance band
INFLATION_BAND_HIGH = 6.0       # Upper tolerance band
REPO_RATE_MIN = 3.0             # Historical floor (%)
REPO_RATE_MAX = 10.0            # Historical ceiling (%)
CRR_MIN = 3.0                   # Min CRR (%)
CRR_MAX = 10.0                  # Max CRR (%)
SLR_MIN = 18.0                  # Min SLR (%)
SLR_MAX = 30.0                  # Max SLR (%)
GDP_GROWTH_STAGNATION = 4.0     # Below this = growth concern (%)
UNEMPLOYMENT_TARGET = 7.0       # Approximate CMIE benchmark (%)
BOND_UNIT = 50_000              # Crore INR per bond operation unit
MAX_STEPS = 12                  # 12 months / 1 fiscal year


@dataclass
class MacroState:
    """Complete macroeconomic state of the Indian economy."""
    # Core inflation metrics
    cpi_inflation: float        # Headline CPI (%)
    food_inflation: float       # Food price index (%)
    core_inflation: float       # Excluding food & fuel (%)

    # Growth metrics
    gdp_growth: float           # Real GDP growth rate (%)
    industrial_output: float    # IIP growth (%)
    private_consumption: float  # Private demand growth (%)

    # Labour
    unemployment_rate: float    # CMIE unemployment (%)

    # External sector
    currency_pressure: float    # USD/INR change (positive = depreciation %)
    current_account_deficit: float  # CAD as % of GDP
    forex_reserves: float       # USD billion equivalent index (60–100)

    # Commodity shocks
    oil_price_shock: float      # Brent crude deviation from baseline (%)
    global_food_shock: float    # Global food price index deviation (%)

    # Policy instruments (current stance)
    repo_rate: float            # RBI repo rate (%)
    reverse_repo_rate: float    # Reverse repo (%)
    crr: float                  # Cash Reserve Ratio (%)
    slr: float                  # Statutory Liquidity Ratio (%)
    bonds_outstanding: float    # OMO net bond balance (units)

    # Sentiment / risk
    market_sentiment: float     # -1 (panic) to +1 (euphoria)
    geopolitical_risk: float    # 0 (calm) to 1 (high risk)

    # Episode metadata
    step: int = 0
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_observation_text(self) -> str:
        """Human-readable state for LLM agent."""
        lines = [
            "=== INDIA MACRO DASHBOARD ===",
            f"[Inflation]",
            f"  Headline CPI: {self.cpi_inflation:.2f}%  (target: {INFLATION_TARGET}%  band: {INFLATION_BAND_LOW}–{INFLATION_BAND_HIGH}%)",
            f"  Food Inflation: {self.food_inflation:.2f}%",
            f"  Core Inflation: {self.core_inflation:.2f}%",
            "",
            f"[Growth]",
            f"  Real GDP Growth: {self.gdp_growth:.2f}%",
            f"  Industrial Output (IIP): {self.industrial_output:.2f}%",
            f"  Private Consumption: {self.private_consumption:.2f}%",
            "",
            f"[Labour]",
            f"  Unemployment Rate: {self.unemployment_rate:.2f}%  (benchmark: {UNEMPLOYMENT_TARGET}%)",
            "",
            f"[External Sector]",
            f"  USD/INR Pressure: {self.currency_pressure:+.2f}%  (positive = INR depreciation)",
            f"  Current Account Deficit: {self.current_account_deficit:.2f}% of GDP",
            f"  Forex Reserves Index: {self.forex_reserves:.1f} / 100",
            "",
            f"[Commodity Shocks]",
            f"  Oil Price Shock: {self.oil_price_shock:+.2f}%",
            f"  Global Food Shock: {self.global_food_shock:+.2f}%",
            "",
            f"[Current Policy Stance]",
            f"  Repo Rate: {self.repo_rate:.2f}%",
            f"  Reverse Repo Rate: {self.reverse_repo_rate:.2f}%",
            f"  CRR: {self.crr:.2f}%",
            f"  SLR: {self.slr:.2f}%",
            f"  Net OMO Bond Balance: {self.bonds_outstanding:.0f} units",
            "",
            f"[Market]",
            f"  Market Sentiment: {self.market_sentiment:+.2f}  (-1=panic, +1=euphoria)",
            f"  Geopolitical Risk: {self.geopolitical_risk:.2f}  (0=calm, 1=high)",
            "",
            f"[Episode] Step {self.step}/{MAX_STEPS}",
        ]
        return "\n".join(lines)


# ── Action catalogue ────────────────────────────────────────────────────────
ACTIONS: Dict[int, Dict[str, Any]] = {
    0:  {"name": "hold_policy",             "description": "Hold all policy instruments unchanged",                  "type": "neutral"},
    1:  {"name": "increase_repo_rate_25bp",  "description": "Raise repo rate by 25 basis points (hawkish)",          "type": "hawkish"},
    2:  {"name": "increase_repo_rate_50bp",  "description": "Raise repo rate by 50 basis points (strongly hawkish)", "type": "hawkish"},
    3:  {"name": "decrease_repo_rate_25bp",  "description": "Cut repo rate by 25 basis points (dovish)",             "type": "dovish"},
    4:  {"name": "decrease_repo_rate_50bp",  "description": "Cut repo rate by 50 basis points (strongly dovish)",    "type": "dovish"},
    5:  {"name": "increase_reverse_repo_25bp","description": "Raise reverse repo 25bp (absorb liquidity)",           "type": "hawkish"},
    6:  {"name": "decrease_reverse_repo_25bp","description": "Cut reverse repo 25bp (inject liquidity)",             "type": "dovish"},
    7:  {"name": "increase_crr_50bp",        "description": "Raise CRR by 50bp (tighten bank liquidity)",            "type": "hawkish"},
    8:  {"name": "decrease_crr_50bp",        "description": "Cut CRR by 50bp (ease bank liquidity)",                 "type": "dovish"},
    9:  {"name": "increase_slr_100bp",       "description": "Raise SLR by 100bp (tighten)",                         "type": "hawkish"},
    10: {"name": "decrease_slr_100bp",       "description": "Cut SLR by 100bp (ease)",                              "type": "dovish"},
    11: {"name": "buy_bonds_small",          "description": "OMO: Buy bonds worth 1 unit (inject ₹50K Cr liquidity)","type": "dovish"},
    12: {"name": "buy_bonds_large",          "description": "OMO: Buy bonds worth 3 units (large liquidity injection)","type": "dovish"},
    13: {"name": "sell_bonds_small",         "description": "OMO: Sell bonds worth 1 unit (absorb ₹50K Cr liquidity)","type": "hawkish"},
    14: {"name": "sell_bonds_large",         "description": "OMO: Sell bonds worth 3 units (large absorption)",      "type": "hawkish"},
}

ACTION_LIST_TEXT = "\n".join(
    f"  {k}: {v['name']} — {v['description']}"
    for k, v in ACTIONS.items()
)


def get_action_names() -> List[str]:
    return [v["name"] for v in ACTIONS.values()]


# ── Transition dynamics ──────────────────────────────────────────────────────
def apply_action(state: MacroState, action_id: int) -> MacroState:
    """
    Apply policy action and evolve macro state with realistic India-like dynamics.
    Returns the *next* state.
    """
    s = state.to_dict()

    # ── 1. Apply direct instrument changes ──────────────────────────────────
    if action_id == 1:   s["repo_rate"] = min(s["repo_rate"] + 0.25, REPO_RATE_MAX)
    elif action_id == 2: s["repo_rate"] = min(s["repo_rate"] + 0.50, REPO_RATE_MAX)
    elif action_id == 3: s["repo_rate"] = max(s["repo_rate"] - 0.25, REPO_RATE_MIN)
    elif action_id == 4: s["repo_rate"] = max(s["repo_rate"] - 0.50, REPO_RATE_MIN)
    elif action_id == 5: s["reverse_repo_rate"] = min(s["reverse_repo_rate"] + 0.25, s["repo_rate"])
    elif action_id == 6: s["reverse_repo_rate"] = max(s["reverse_repo_rate"] - 0.25, REPO_RATE_MIN - 0.5)
    elif action_id == 7: s["crr"] = min(s["crr"] + 0.50, CRR_MAX)
    elif action_id == 8: s["crr"] = max(s["crr"] - 0.50, CRR_MIN)
    elif action_id == 9:  s["slr"] = min(s["slr"] + 1.00, SLR_MAX)
    elif action_id == 10: s["slr"] = max(s["slr"] - 1.00, SLR_MIN)
    elif action_id == 11: s["bonds_outstanding"] += 1
    elif action_id == 12: s["bonds_outstanding"] += 3
    elif action_id == 13: s["bonds_outstanding"] -= 1
    elif action_id == 14: s["bonds_outstanding"] -= 3

    # ── 2. Compute effective monetary tightness score ────────────────────────
    # Higher repo/crr/slr + bond selling = tighter
    repo_neutral = 6.5   # RBI's approximate neutral rate reference
    tightness = (
        (s["repo_rate"] - repo_neutral) * 0.5
        + (s["crr"] - 4.5) * 0.3
        + (s["slr"] - 22.0) * 0.1
        - s["bonds_outstanding"] * 0.05   # more bonds bought = more loose
    )

    # ── 3. Inflation dynamics ────────────────────────────────────────────────
    oil_pass_through   = s["oil_price_shock"] * 0.10
    food_pass_through  = s["global_food_shock"] * 0.12
    demand_pull        = max(0, (s["private_consumption"] - 6.0)) * 0.08
    monetary_effect    = -tightness * 0.6         # tightening reduces inflation
    sentiment_effect   = -s["market_sentiment"] * 0.05
    noise              = random.gauss(0, 0.15)

    delta_cpi = oil_pass_through + food_pass_through + demand_pull + monetary_effect + sentiment_effect + noise
    s["cpi_inflation"] = max(0.5, s["cpi_inflation"] + delta_cpi)

    # Food inflation: driven more by global shocks + monsoon noise
    s["food_inflation"] = max(0.0, s["food_inflation"]
                              + food_pass_through * 1.5
                              + random.gauss(0, 0.3)
                              - tightness * 0.10)

    # Core inflation: more persistent, monetary policy has lagged effect
    s["core_inflation"] = max(0.5, s["core_inflation"]
                              + demand_pull * 0.8
                              - tightness * 0.20
                              + random.gauss(0, 0.10))

    # ── 4. Growth dynamics ───────────────────────────────────────────────────
    rate_effect_gdp = -tightness * 0.35             # tightening hurts growth
    oil_effect_gdp  = -s["oil_price_shock"] * 0.05  # oil shocks hurt growth
    consume_lag     = (s["private_consumption"] - 6.0) * 0.2
    gdp_noise       = random.gauss(0, 0.2)

    s["gdp_growth"] = max(-3.0, s["gdp_growth"] + rate_effect_gdp + oil_effect_gdp + consume_lag * 0.1 + gdp_noise)

    s["industrial_output"] = max(-5.0, s["industrial_output"]
                                 - tightness * 0.4
                                 + random.gauss(0, 0.4))

    s["private_consumption"] = max(1.0, s["private_consumption"]
                                   - tightness * 0.25
                                   + random.gauss(0, 0.3))

    # ── 5. Labour market ─────────────────────────────────────────────────────
    # Higher growth → lower unemployment (Okun's Law, lagged)
    gdp_gap = s["gdp_growth"] - 6.5   # potential GDP ≈ 6.5% for India
    s["unemployment_rate"] = max(3.0, s["unemployment_rate"]
                                 - gdp_gap * 0.15
                                 + tightness * 0.10
                                 + random.gauss(0, 0.15))

    # ── 6. External sector ───────────────────────────────────────────────────
    rate_diff = s["repo_rate"] - 5.0   # vs global benchmark proxy
    s["currency_pressure"] = (
        - rate_diff * 0.3              # higher rates attract capital → INR appreciates
        + s["oil_price_shock"] * 0.04  # oil shock hurts INR
        + s["cpi_inflation"] * 0.05    # inflation erodes INR
        + random.gauss(0, 0.3)
        + s["geopolitical_risk"] * 0.5
    )
    s["currency_pressure"] = max(-4.0, min(6.0, s["currency_pressure"]))

    s["current_account_deficit"] = max(-1.0, min(5.0,
        s["current_account_deficit"]
        + s["oil_price_shock"] * 0.03
        - rate_diff * 0.1
        + random.gauss(0, 0.1)
    ))

    s["forex_reserves"] = max(30.0, min(100.0,
        s["forex_reserves"]
        - s["currency_pressure"] * 1.5
        + random.gauss(0, 1.0)
    ))

    # ── 7. Evolve shocks (mean-reverting AR(1)) ──────────────────────────────
    s["oil_price_shock"]   = s["oil_price_shock"] * 0.75 + random.gauss(0, 2.0)
    s["global_food_shock"] = s["global_food_shock"] * 0.70 + random.gauss(0, 1.5)
    s["geopolitical_risk"] = max(0.0, min(1.0,
        s["geopolitical_risk"] * 0.85 + random.uniform(0, 0.15)
    ))
    s["market_sentiment"] = max(-1.0, min(1.0,
        s["market_sentiment"] * 0.80
        - (s["cpi_inflation"] - INFLATION_TARGET) * 0.05
        + random.gauss(0, 0.15)
    ))

    # ── 8. Clamp all rates to valid ranges ───────────────────────────────────
    s["repo_rate"]         = round(max(REPO_RATE_MIN, min(REPO_RATE_MAX, s["repo_rate"])), 2)
    s["reverse_repo_rate"] = round(max(REPO_RATE_MIN - 0.5, min(s["repo_rate"], s["reverse_repo_rate"])), 2)
    s["crr"]               = round(max(CRR_MIN, min(CRR_MAX, s["crr"])), 2)
    s["slr"]               = round(max(SLR_MIN, min(SLR_MAX, s["slr"])), 2)

    s["step"] = state.step + 1
    s["done"] = s["step"] >= MAX_STEPS

    return MacroState(**s)


# ── Reward function ──────────────────────────────────────────────────────────
def compute_reward(prev: MacroState, curr: MacroState, action_id: int) -> Tuple[float, Dict[str, float]]:
    """
    Multi-objective reward for the RBI monetary policy controller.
    Returns (total_reward, component_breakdown).
    """
    components: Dict[str, float] = {}

    # ── Inflation component (±4 range, most important) ──────────────────────
    inf_dev = curr.cpi_inflation - INFLATION_TARGET
    if INFLATION_BAND_LOW <= curr.cpi_inflation <= INFLATION_BAND_HIGH:
        components["inflation_target"] = 4.0   # in the comfort zone
    else:
        components["inflation_target"] = -2.0 * (abs(inf_dev) ** 1.2)

    # Penalise runaway inflation heavily
    if curr.cpi_inflation > 8.0:
        components["runaway_inflation_penalty"] = -5.0 * (curr.cpi_inflation - 8.0)
    else:
        components["runaway_inflation_penalty"] = 0.0

    # Bonus for moving inflation closer to target
    prev_dev = abs(prev.cpi_inflation - INFLATION_TARGET)
    curr_dev = abs(curr.cpi_inflation - INFLATION_TARGET)
    components["inflation_improvement"] = (prev_dev - curr_dev) * 2.0

    # ── Growth component ─────────────────────────────────────────────────────
    if curr.gdp_growth >= 6.5:
        components["gdp_growth"] = 3.0
    elif curr.gdp_growth >= GDP_GROWTH_STAGNATION:
        components["gdp_growth"] = (curr.gdp_growth - GDP_GROWTH_STAGNATION) / 2.5 * 2.0
    else:
        components["gdp_growth"] = -2.5 * (GDP_GROWTH_STAGNATION - curr.gdp_growth)

    # Prevent collapse
    if curr.gdp_growth < 2.0:
        components["growth_collapse_penalty"] = -4.0 * (2.0 - curr.gdp_growth)
    else:
        components["growth_collapse_penalty"] = 0.0

    # ── Unemployment component ────────────────────────────────────────────────
    if curr.unemployment_rate <= UNEMPLOYMENT_TARGET:
        components["unemployment"] = 2.0
    else:
        components["unemployment"] = -1.5 * (curr.unemployment_rate - UNEMPLOYMENT_TARGET)

    # Penalise if unemployment worsened significantly
    if curr.unemployment_rate > prev.unemployment_rate + 0.5:
        components["unemployment_deterioration"] = -2.0
    else:
        components["unemployment_deterioration"] = 0.0

    # ── Currency stability component ─────────────────────────────────────────
    abs_cp = abs(curr.currency_pressure)
    if abs_cp < 1.0:
        components["currency_stability"] = 1.5
    elif abs_cp < 2.5:
        components["currency_stability"] = 0.5 - (abs_cp - 1.0) * 0.5
    else:
        components["currency_stability"] = -2.0 * (abs_cp - 2.5)

    # ── Policy stability / consistency ───────────────────────────────────────
    # Penalise large sudden swings (e.g., 50bp cut immediately after 50bp hike)
    if action_id in (2, 14):   # large aggressive moves
        # Check if previous action was in the opposite direction
        components["policy_stability"] = -0.5
    elif action_id == 0:        # hold = slightly rewarded for calmness
        components["policy_stability"] = 0.5
    else:
        components["policy_stability"] = 0.2

    # Penalise hitting constraint limits (instrument clipped)
    if curr.repo_rate in (REPO_RATE_MIN, REPO_RATE_MAX):
        components["constraint_violation"] = -1.0
    else:
        components["constraint_violation"] = 0.0

    # ── Shock response bonus ─────────────────────────────────────────────────
    big_shock = abs(curr.oil_price_shock) > 5 or abs(curr.global_food_shock) > 5
    action_type = ACTIONS[action_id]["type"]
    if big_shock and curr.cpi_inflation > INFLATION_BAND_HIGH and action_type == "hawkish":
        components["shock_response"] = 1.5
    elif big_shock and curr.gdp_growth < GDP_GROWTH_STAGNATION and action_type == "dovish":
        components["shock_response"] = 1.5
    else:
        components["shock_response"] = 0.0

    # ── Depression / deflation scenario ─────────────────────────────────────
    if curr.cpi_inflation < INFLATION_BAND_LOW and curr.gdp_growth < 4.0:
        # Agent should be stimulating
        if action_type == "dovish":
            components["deflation_response"] = 1.0
        else:
            components["deflation_response"] = -1.0
    else:
        components["deflation_response"] = 0.0

    total = sum(components.values())
    # Normalise to roughly [-10, +10]
    total = max(-10.0, min(10.0, total))

    return round(total, 4), {k: round(v, 4) for k, v in components.items()}


# ── Environment class ────────────────────────────────────────────────────────
class PriceStabilizerEnv:
    """
    OpenEnv-compatible monetary policy RL environment for the Reserve Bank of India.

    Interface:
        reset() -> observation (str)
        step(action_id: int) -> (observation, reward, done, info)
        get_action_space() -> list[str]
        render() -> str
    """

    metadata = {
        "name": "PriceStabilizerEnv-v1",
        "description": "India monetary policy RL environment — balance inflation, growth, jobs.",
        "version": "1.0.0",
        "author": "OpenEnv Hackathon Team",
    }

    def __init__(self, scenario: Optional[str] = None, seed: Optional[int] = None):
        self.scenario = scenario
        if seed is not None:
            random.seed(seed)
        self.state: Optional[MacroState] = None
        self.history: List[Dict] = []
        self._episode_reward: float = 0.0

    # ── OpenEnv required methods ─────────────────────────────────────────────

    def reset(self) -> str:
        """Reset environment and return initial observation string."""
        self.state = self._build_initial_state()
        self.history = []
        self._episode_reward = 0.0
        return self.state.to_observation_text()

    def step(self, action_id: int) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one policy decision step.
        Returns: (observation, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        if action_id not in ACTIONS:
            raise ValueError(f"Invalid action_id {action_id}. Valid range: 0–{len(ACTIONS)-1}")

        prev_state = self.state
        next_state = apply_action(self.state, action_id)
        reward, reward_breakdown = compute_reward(prev_state, next_state, action_id)

        self.state = next_state
        self._episode_reward += reward

        info = {
            "action_taken": ACTIONS[action_id]["name"],
            "action_type": ACTIONS[action_id]["type"],
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "episode_total_reward": round(self._episode_reward, 4),
            "step": next_state.step,
            "state": next_state.to_dict(),
        }

        self.history.append({
            "step": next_state.step,
            "action_id": action_id,
            "action_name": ACTIONS[action_id]["name"],
            "reward": reward,
            "cpi_inflation": next_state.cpi_inflation,
            "gdp_growth": next_state.gdp_growth,
            "unemployment_rate": next_state.unemployment_rate,
            "repo_rate": next_state.repo_rate,
        })

        obs = next_state.to_observation_text()
        return obs, reward, next_state.done, info

    def get_action_space(self) -> List[str]:
        """Return list of action names (OpenEnv interface)."""
        return get_action_names()

    def render(self) -> str:
        """Return current state as a human-readable string."""
        if self.state is None:
            return "Environment not initialised. Call reset() first."
        return self.state.to_observation_text()

    def get_action_prompt(self) -> str:
        """Return action listing suitable for LLM prompt injection."""
        return (
            "Available policy actions (choose one action_id 0–14):\n"
            + ACTION_LIST_TEXT
        )

    # ── Scenario builder ─────────────────────────────────────────────────────

    def _build_initial_state(self) -> MacroState:
        """Build initial macro state, optionally from named scenario."""

        # Baseline India macro (roughly 2023–24 conditions)
        base = dict(
            cpi_inflation    = 5.5,
            food_inflation   = 7.0,
            core_inflation   = 4.5,
            gdp_growth       = 6.5,
            industrial_output= 4.5,
            private_consumption=6.5,
            unemployment_rate= 8.0,
            currency_pressure= 1.0,
            current_account_deficit=1.8,
            forex_reserves   = 75.0,
            oil_price_shock  = 0.0,
            global_food_shock= 0.0,
            repo_rate        = 6.50,
            reverse_repo_rate= 6.25,
            crr              = 4.50,
            slr              = 18.00,
            bonds_outstanding= 0.0,
            market_sentiment = 0.1,
            geopolitical_risk= 0.3,
            step             = 0,
            done             = False,
        )

        if self.scenario == "stagflation":
            base.update(cpi_inflation=8.5, food_inflation=11.0, gdp_growth=3.5,
                        oil_price_shock=15.0, unemployment_rate=10.0,
                        currency_pressure=3.5, market_sentiment=-0.5)

        elif self.scenario == "high_growth_overheating":
            base.update(cpi_inflation=7.2, gdp_growth=9.0, private_consumption=11.0,
                        core_inflation=6.0, unemployment_rate=5.5, market_sentiment=0.8)

        elif self.scenario == "recession_risk":
            base.update(gdp_growth=2.5, cpi_inflation=2.8, industrial_output=-2.0,
                        unemployment_rate=12.0, market_sentiment=-0.7,
                        private_consumption=2.0)

        elif self.scenario == "external_shock":
             base.update(oil_price_shock=10.0,   
                global_food_shock=8.0,  
                currency_pressure=3.0, 
                geopolitical_risk=0.9,
                forex_reserves=55.0,
                cpi_inflation=7.0)

        elif self.scenario == "random":
            # Randomised stress test
            base["cpi_inflation"]     = random.uniform(2.0, 10.0)
            base["gdp_growth"]        = random.uniform(1.5, 9.0)
            base["unemployment_rate"] = random.uniform(5.0, 14.0)
            base["oil_price_shock"]   = random.gauss(0, 10)
            base["global_food_shock"] = random.gauss(0, 8)
            base["geopolitical_risk"] = random.uniform(0, 1)

        return MacroState(**base)


# ── Grading logic ────────────────────────────────────────────────────────────
def grade_episode(history: List[Dict], final_state: MacroState, total_reward: float) -> Dict[str, Any]:
    """
    Evaluate a completed episode against key policy objectives.
    Returns structured grading report.
    """
    if not history:
        return {"error": "No history to grade"}

    steps = len(history)
    avg_cpi   = sum(h["cpi_inflation"] for h in history) / steps
    avg_gdp   = sum(h["gdp_growth"] for h in history) / steps
    avg_unemp = sum(h["unemployment_rate"] for h in history) / steps
    cpi_in_band = sum(1 for h in history if INFLATION_BAND_LOW <= h["cpi_inflation"] <= INFLATION_BAND_HIGH) / steps

    # Policy oscillation: count direction reversals
    actions = [h["action_id"] for h in history]
    action_types = [ACTIONS[a]["type"] for a in actions]
    reversals = sum(
        1 for i in range(1, len(action_types))
        if action_types[i] != action_types[i-1]
        and action_types[i] != "neutral"
        and action_types[i-1] != "neutral"
    )
    oscillation_score = max(0.0, 1.0 - reversals / max(1, steps))

    # Scores out of 10
    inflation_score = min(10.0, cpi_in_band * 10.0)
    growth_score    = min(10.0, max(0.0, avg_gdp / 7.0 * 10.0))
    employment_score= min(10.0, max(0.0, (1 - max(0, avg_unemp - 7.0) / 7.0) * 10.0))
    stability_score = oscillation_score * 10.0
    reward_score    = min(10.0, max(0.0, (total_reward / (MAX_STEPS * 4.0)) * 10.0))

    overall = (
        inflation_score * 0.30
        + growth_score  * 0.25
        + employment_score * 0.20
        + stability_score * 0.15
        + reward_score  * 0.10
    )

    grade = (
        "A" if overall >= 8.0 else
        "B" if overall >= 6.5 else
        "C" if overall >= 5.0 else
        "D" if overall >= 3.5 else "F"
    )

    return {
        "overall_score": round(overall, 2),
        "grade": grade,
        "component_scores": {
            "inflation_management": round(inflation_score, 2),
            "growth_preservation":  round(growth_score, 2),
            "employment":           round(employment_score, 2),
            "policy_stability":     round(stability_score, 2),
            "cumulative_reward":    round(reward_score, 2),
        },
        "statistics": {
            "avg_cpi_inflation":   round(avg_cpi, 2),
            "avg_gdp_growth":      round(avg_gdp, 2),
            "avg_unemployment":    round(avg_unemp, 2),
            "steps_in_target_band_pct": round(cpi_in_band * 100, 1),
            "policy_reversals":    reversals,
            "total_reward":        round(total_reward, 4),
            "steps_completed":     steps,
        },
        "final_state": {
            "cpi_inflation":   round(final_state.cpi_inflation, 2),
            "gdp_growth":      round(final_state.gdp_growth, 2),
            "unemployment":    round(final_state.unemployment_rate, 2),
            "repo_rate":       round(final_state.repo_rate, 2),
            "currency_pressure": round(final_state.currency_pressure, 2),
        },
    }
