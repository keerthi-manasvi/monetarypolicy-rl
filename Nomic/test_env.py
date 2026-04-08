"""
tests/test_env.py — Unit tests for PriceStabilizerEnv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from price_stabilizer_env import (
    PriceStabilizerEnv,
    ACTIONS,
    apply_action,
    compute_reward,
    grade_episode,
    REPO_RATE_MIN, REPO_RATE_MAX,
    CRR_MIN, CRR_MAX,
    SLR_MIN, SLR_MAX,
    MAX_STEPS,
)


@pytest.fixture
def env():
    e = PriceStabilizerEnv(seed=42)
    e.reset()
    return e


def test_reset_returns_string(env):
    obs = env.reset()
    assert isinstance(obs, str)
    assert "CPI" in obs


def test_action_space_length():
    env = PriceStabilizerEnv()
    env.reset()
    actions = env.get_action_space()
    assert len(actions) == len(ACTIONS)


def test_step_returns_tuple(env):
    obs, reward, done, info = env.step(0)
    assert isinstance(obs, str)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_increments_step(env):
    env.reset()
    _, _, _, info = env.step(0)
    assert info["step"] == 1


def test_episode_ends_at_max_steps(env):
    env.reset()
    done = False
    for _ in range(MAX_STEPS):
        _, _, done, _ = env.step(0)
    assert done is True


def test_all_actions_execute(env):
    for action_id in ACTIONS:
        e = PriceStabilizerEnv(seed=42)
        e.reset()
        obs, reward, done, info = e.step(action_id)
        assert isinstance(reward, float), f"Action {action_id} returned non-float reward"


def test_repo_rate_clamped(env):
    env.reset()
    # Apply many aggressive hikes
    for _ in range(20):
        env.step(2)  # increase_repo_rate_50bp
    assert env.state.repo_rate <= REPO_RATE_MAX


def test_repo_rate_floor(env):
    env.reset()
    for _ in range(20):
        env.step(4)  # decrease_repo_rate_50bp
    assert env.state.repo_rate >= REPO_RATE_MIN


def test_crr_bounds(env):
    env.reset()
    for _ in range(20):
        env.step(7)   # increase_crr_50bp
    assert env.state.crr <= CRR_MAX
    env.reset()
    for _ in range(20):
        env.step(8)   # decrease_crr_50bp
    assert env.state.crr >= CRR_MIN


def test_slr_bounds(env):
    env.reset()
    for _ in range(20):
        env.step(9)   # increase_slr_100bp
    assert env.state.slr <= SLR_MAX
    env.reset()
    for _ in range(20):
        env.step(10)  # decrease_slr_100bp
    assert env.state.slr >= SLR_MIN


def test_reward_is_bounded(env):
    env.reset()
    for action_id in ACTIONS:
        e = PriceStabilizerEnv(seed=1)
        e.reset()
        _, reward, _, _ = e.step(action_id)
        assert -10.0 <= reward <= 10.0, f"Reward {reward} out of bounds for action {action_id}"


def test_grading_report_structure(env):
    env.reset()
    for _ in range(MAX_STEPS):
        env.step(0)
    report = grade_episode(env.history, env.state, 0.0)
    assert "overall_score" in report
    assert "grade" in report
    assert report["grade"] in ("A", "B", "C", "D", "F")
    assert 0.0 <= report["overall_score"] <= 10.0


def test_all_scenarios():
    for scenario in ["stagflation", "high_growth_overheating", "recession_risk", "external_shock", "random"]:
        env = PriceStabilizerEnv(scenario=scenario, seed=7)
        obs = env.reset()
        assert isinstance(obs, str)


def test_reward_components_sum(env):
    env.reset()
    prev = env.state
    env.step(1)
    curr = env.state
    total, breakdown = compute_reward(prev, curr, 1)
    assert abs(total - max(-10.0, min(10.0, sum(breakdown.values())))) < 0.01


def test_render_before_reset():
    env = PriceStabilizerEnv()
    result = env.render()
    assert "not initialised" in result.lower() or "reset" in result.lower()


def test_invalid_action_raises(env):
    with pytest.raises(ValueError):
        env.step(999)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
