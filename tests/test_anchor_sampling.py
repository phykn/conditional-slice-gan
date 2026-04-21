import random

import pytest

from src.data.anchor_sampling import (
    axis_index,
    choose_anchor_count,
    sample_positions_with_gap,
)


def test_axis_index_axis_0():
    idx = axis_index(0, 3)
    assert idx == (slice(None), 3, slice(None), slice(None))


def test_axis_index_axis_1():
    idx = axis_index(1, 5)
    assert idx == (slice(None), slice(None), 5, slice(None))


def test_axis_index_axis_2():
    idx = axis_index(2, 7)
    assert idx == (slice(None), slice(None), slice(None), 7)


def test_choose_anchor_count_empty():
    for _ in range(20):
        K = choose_anchor_count(
            D_axis=8, empty_prob=1.0, full_prob=0.0,
            sparse_min=1, sparse_max=7,
        )
        assert K == 0


def test_choose_anchor_count_full():
    for _ in range(20):
        K = choose_anchor_count(
            D_axis=8, empty_prob=0.0, full_prob=1.0,
            sparse_min=1, sparse_max=7,
        )
        assert K == 8


def test_choose_anchor_count_sparse_range():
    random.seed(0)
    seen = set()
    for _ in range(200):
        K = choose_anchor_count(
            D_axis=8, empty_prob=0.0, full_prob=0.0,
            sparse_min=3, sparse_max=5,
        )
        assert 3 <= K <= 5
        seen.add(K)
    assert seen == {3, 4, 5}


def test_sample_positions_K_zero():
    assert sample_positions_with_gap(D_axis=8, K=0, min_gap=1) == []
    assert sample_positions_with_gap(D_axis=8, K=0, min_gap=3) == []


def test_sample_positions_full_requires_gap_1():
    pos = sample_positions_with_gap(D_axis=8, K=8, min_gap=1)
    assert pos == list(range(8))
    with pytest.raises(AssertionError):
        sample_positions_with_gap(D_axis=8, K=8, min_gap=2)


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
