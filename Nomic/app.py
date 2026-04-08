"""
app.py — Hugging Face Spaces deployment for Price Stabilizer
============================================================
Interactive Gradio UI: run the monetary policy RL environment
with an LLM agent OR the built-in heuristic, and visualise results.
"""

import os
import json
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from price_stabilizer_env import (
    PriceStabilizerEnv,
    ACTIONS,
    grade_episode,
    INFLATION_TARGET,
    INFLATION_BAND_LOW,
    INFLATION_BAND_HIGH,
    GDP_GROWTH_STAGNATION,
    MAX_STEPS,
)

# ── Try to load LLM agent; fall back to heuristic gracefully ────────────────
def _heuristic_agent(state) -> tuple[int, str]:
    inf, gdp, unemp = state.cpi_inflation, state.gdp_growth, state.unemployment_rate
    if inf > 7.5:   return 2,  "Inflation critically high → hike 50bp"
    if inf > 6.0:   return 1,  "Inflation above band → hike 25bp"
    if inf < INFLATION_BAND_LOW and gdp < 4.0:
                    return 4,  "Deflationary risk + weak growth → cut 50bp"
    if inf < INFLATION_BAND_HIGH and gdp < GDP_GROWTH_STAGNATION:
                    return 3,  "Growth concerns with manageable inflation → cut 25bp"
    if gdp < 3.0:   return 11, "Sharp growth slowdown → inject liquidity via OMO"
    if unemp > 12:  return 8,  "High unemployment → ease CRR to boost lending"
    return 0, "Macroeconomic indicators balanced → hold policy steady"


def run_simulation(scenario: str, agent_type: str, hf_token: str, seed: int):
    """Run a full episode and return results for Gradio."""
    env_scenario = None if scenario == "Baseline" else scenario.lower().replace(" ", "_")

    # Determine agent
    use_llm = agent_type == "LLM Agent (Meta Llama)" and hf_token.strip()
    if use_llm:
        try:
            from openai import OpenAI
            from inference import SYSTEM_PROMPT, call_llm_agent
            os.environ["HF_TOKEN"] = hf_token.strip()
            os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1/"
            os.environ["MODEL_NAME"] = "meta-llama/Llama-3.1-8B-Instruct"
        except Exception:
            use_llm = False

    env = PriceStabilizerEnv(scenario=env_scenario, seed=int(seed))
    obs = env.reset()

    logs, history_rows = [], []
    total_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if use_llm:
            from inference import call_llm_agent as llm_call
            action_id, reasoning = llm_call(obs, step)
        else:
            action_id, reasoning = _heuristic_agent(env.state)

        obs, reward, done, info = env.step(action_id)
        total_reward += reward
        s = info["state"]

        row = {
            "Step": step,
            "Action": ACTIONS[action_id]["name"],
            "Reward": round(reward, 2),
            "Total Reward": round(total_reward, 2),
            "CPI %": round(s["cpi_inflation"], 2),
            "GDP %": round(s["gdp_growth"], 2),
            "Unemp %": round(s["unemployment_rate"], 2),
            "Repo %": round(s["repo_rate"], 2),
            "FX Pressure": round(s["currency_pressure"], 2),
        }
        history_rows.append(row)
        logs.append(f"Step {step:2d} | {ACTIONS[action_id]['name']:<32} | R={reward:+.2f} | CPI={s['cpi_inflation']:.2f}% | Reasoning: {reasoning}")

        if done:
            break

    grading = grade_episode(env.history, env.state, total_reward)
    df = pd.DataFrame(history_rows)

    # ── Build Plotly chart ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "CPI Inflation (%)", "Real GDP Growth (%)",
            "Repo Rate (%)", "Unemployment (%)",
            "Cumulative Reward", "FX Pressure (%)"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    steps = df["Step"].tolist()

    def add_line(row, col, y, name, color):
        fig.add_trace(go.Scatter(x=steps, y=y, mode="lines+markers",
                                 name=name, line=dict(color=color, width=2)), row=row, col=col)

    add_line(1, 1, df["CPI %"].tolist(),         "CPI",     "#e74c3c")
    add_line(1, 2, df["GDP %"].tolist(),          "GDP",     "#27ae60")
    add_line(2, 1, df["Repo %"].tolist(),         "Repo",    "#2980b9")
    add_line(2, 2, df["Unemp %"].tolist(),        "Unemp",   "#8e44ad")
    add_line(3, 1, df["Total Reward"].tolist(),   "Reward",  "#f39c12")
    add_line(3, 2, df["FX Pressure"].tolist(),    "FX",      "#1abc9c")

    # Target bands
    fig.add_hline(y=INFLATION_TARGET, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hrect(y0=INFLATION_BAND_LOW, y1=INFLATION_BAND_HIGH,
                  fillcolor="rgba(39,174,96,0.08)", line_width=0, row=1, col=1)
    fig.add_hline(y=GDP_GROWTH_STAGNATION, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(
        height=700,
        title_text=f"Price Stabilizer — {scenario} Scenario  |  Agent: {agent_type}",
        showlegend=False,
        template="plotly_white",
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )

    # ── Format grading as markdown ────────────────────────────────────────────
    grade_md = f"""
### 📊 Grading Report

**Overall Score:** {grading['overall_score']} / 10.0 &nbsp; | &nbsp; **Grade:** `{grading['grade']}`

| Component | Score |
|-----------|-------|
| Inflation Management | {grading['component_scores']['inflation_management']} |
| Growth Preservation | {grading['component_scores']['growth_preservation']} |
| Employment | {grading['component_scores']['employment']} |
| Policy Stability | {grading['component_scores']['policy_stability']} |
| Cumulative Reward | {grading['component_scores']['cumulative_reward']} |

**Statistics:**
- Avg CPI: {grading['statistics']['avg_cpi_inflation']}%  (target: {INFLATION_TARGET}%)
- Avg GDP: {grading['statistics']['avg_gdp_growth']}%
- Steps in target band: {grading['statistics']['steps_in_target_band_pct']}%
- Policy reversals: {grading['statistics']['policy_reversals']}
- Total reward: {grading['statistics']['total_reward']}
"""

    log_text = "\n".join(logs)
    return fig, df[["Step","Action","Reward","CPI %","GDP %","Unemp %","Repo %"]].to_html(index=False), grade_md, log_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Price Stabilizer — Monetary Policy RL",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="green",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css="""
    .container { max-width: 1200px; margin: auto; }
    .gradio-container { font-family: 'IBM Plex Mono', monospace !important; }
    h1 { color: #1a3c5e; letter-spacing: -0.5px; }
    .grade-box { border: 2px solid #2980b9; border-radius: 8px; padding: 16px; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🏦 Price Stabilizer: Monetary Policy RL Simulator
        ### Reserve Bank of India — Macroeconomic Stabilisation Agent

        An **RL environment** where an AI agent acts as the RBI Monetary Policy Committee.
        The agent observes inflation, GDP, unemployment, and external shocks, then chooses
        policy actions (repo rate, CRR, SLR, OMO bond operations) to stabilise the Indian economy.

        **OpenEnv Hackathon Round 1** | Meta × Hugging Face
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            scenario_dd = gr.Dropdown(
                label="📍 Starting Scenario",
                choices=["Baseline", "Stagflation", "High Growth Overheating",
                         "Recession Risk", "External Shock", "Random"],
                value="Baseline",
            )
            agent_dd = gr.Dropdown(
                label="🤖 Policy Agent",
                choices=["Heuristic (Rule-Based)", "LLM Agent (Meta Llama)"],
                value="Heuristic (Rule-Based)",
            )
            hf_token_box = gr.Textbox(
                label="🔑 HF Token (for LLM Agent only)",
                placeholder="hf_...",
                type="password",
                value="",
            )
            seed_slider = gr.Slider(label="🎲 Random Seed", minimum=1, maximum=999, step=1, value=42)
            run_btn = gr.Button("▶ Run Episode", variant="primary")

        with gr.Column(scale=3):
            chart = gr.Plot(label="Episode Metrics")

    with gr.Row():
        grade_out  = gr.Markdown(label="Grading Report", elem_classes=["grade-box"])

    with gr.Row():
        table_out  = gr.HTML(label="Step-by-Step History")

    with gr.Row():
        log_out    = gr.Textbox(label="📋 Agent Log", lines=14, max_lines=20)

    run_btn.click(
        fn=run_simulation,
        inputs=[scenario_dd, agent_dd, hf_token_box, seed_slider],
        outputs=[chart, table_out, grade_out, log_out],
    )

    gr.Markdown(
        """
        ---
        **Actions:** Repo Rate ±25/50bp · Reverse Repo ±25bp · CRR ±50bp · SLR ±100bp · OMO Buy/Sell Bonds  
        **Reward:** Inflation near target +4 · Growth preservation +3 · Employment +2 · FX stability +1.5 · Policy consistency +0.5  
        **Episode:** 12 steps = 1 fiscal year of monthly MPC decisions
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)