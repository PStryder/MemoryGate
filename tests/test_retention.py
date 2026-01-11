import pytest

from core.services.retention_service import apply_decay_tick, apply_fetch_bump, apply_floor, clamp_score


def test_clamp_score_bounds():
    assert clamp_score(-5.0, -3.0, 1.0) == -3.0
    assert clamp_score(2.0, -3.0, 1.0) == 1.0
    assert clamp_score(0.5, -3.0, 1.0) == 0.5


def test_apply_floor():
    assert apply_floor(-5.0, -2.0) == -2.0
    assert apply_floor(-1.0, -2.0) == -1.0


def test_apply_fetch_bump_formula():
    assert apply_fetch_bump(0.0, 0.4) == pytest.approx(0.4)
    assert apply_fetch_bump(1.0, 0.4) == pytest.approx(1.0)
    assert apply_fetch_bump(-3.0, 0.4) == pytest.approx(-1.8)


def test_apply_decay_tick():
    assert apply_decay_tick(0.5, 0.1, 2.0) == pytest.approx(0.3)
