import random

import pytest

from src.data.anchor_sampling import (
    AnchorSpec,
    choose_anchor_count,
    sample_positions_with_gap,
)


def _spec(empty_prob: float = 0.0, min_gap: int = 1) -> AnchorSpec:
    return AnchorSpec(empty_prob=empty_prob, min_gap=min_gap)


def test_choose_anchor_count_empty():
    spec = _spec(empty_prob=1.0)
    for _ in range(20):
        assert choose_anchor_count(D_axis=8, spec=spec) == 0


def test_choose_anchor_count_range_gap_1():
    # D_axis=8, min_gap=1 → max_K = 7, K ∈ [1, 7].
    random.seed(0)
    spec = _spec(min_gap=1)
    seen = set()
    for _ in range(200):
        K = choose_anchor_count(D_axis=8, spec=spec)
        assert 1 <= K <= 7
        seen.add(K)
    assert seen == set(range(1, 8))


def test_choose_anchor_count_range_gap_larger():
    # D_axis=8, min_gap=3 → max_K = min(7, 7//3+1) = 3, K ∈ [1, 3].
    random.seed(0)
    spec = _spec(min_gap=3)
    seen = set()
    for _ in range(200):
        K = choose_anchor_count(D_axis=8, spec=spec)
        assert 1 <= K <= 3
        seen.add(K)
    assert seen == {1, 2, 3}


def test_sample_positions_K_zero():
    assert sample_positions_with_gap(D_axis=8, K=0, min_gap=1) == []
    assert sample_positions_with_gap(D_axis=8, K=0, min_gap=3) == []


def test_sample_positions_gap_1_distinct():
    random.seed(1)
    for _ in range(20):
        pos = sample_positions_with_gap(D_axis=16, K=5, min_gap=1)
        assert len(pos) == 5
        assert len(set(pos)) == 5
        assert all(0 <= p < 16 for p in pos)


def test_sample_positions_gap_respected():
    random.seed(2)
    for _ in range(50):
        pos = sample_positions_with_gap(D_axis=32, K=6, min_gap=4)
        assert len(pos) == 6
        assert all(0 <= p < 32 for p in pos)
        s = sorted(pos)
        for a, b in zip(s, s[1:]):
            assert b - a >= 4


def test_sample_positions_infeasible_raises():
    # K=5, min_gap=3, D=10: need last position >= 4*3=12 > 9. Infeasible.
    with pytest.raises(ValueError, match="infeasible"):
        sample_positions_with_gap(D_axis=10, K=5, min_gap=3)


def test_sample_positions_boundary_feasible():
    # K=4, min_gap=2, D=8: positions 0,2,4,6 is tight but valid.
    random.seed(3)
    pos = sample_positions_with_gap(D_axis=8, K=4, min_gap=2)
    assert len(pos) == 4
    s = sorted(pos)
    for a, b in zip(s, s[1:]):
        assert b - a >= 2
