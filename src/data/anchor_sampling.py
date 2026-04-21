import random


def axis_index(anchor_axis: int, k: int) -> tuple:
    """Index tuple selecting slice k along spatial `anchor_axis` of a (C, D, H, W) array.
    Works on both torch tensors and numpy arrays."""
    return (slice(None),) + tuple(k if i == anchor_axis else slice(None) for i in range(3))


def choose_anchor_count(
    D_axis: int,
    empty_prob: float,
    full_prob: float,
    sparse_min: int,
    sparse_max: int | None,
) -> int:
    assert empty_prob >= 0.0 and full_prob >= 0.0
    assert empty_prob + full_prob <= 1.0
    smax = D_axis - 1 if sparse_max is None else sparse_max
    assert 1 <= sparse_min <= smax <= D_axis - 1

    r = random.random()
    if r < empty_prob:
        return 0
    if r < empty_prob + full_prob:
        return D_axis
    return random.randint(sparse_min, smax)


def sample_positions_with_gap(D_axis: int, K: int, min_gap: int) -> list[int]:
    """Sample K distinct positions in [0, D_axis) such that any two differ by at least `min_gap`.

    Uses the bijection p_i = q_i + i * (min_gap - 1) with 0 <= q_0 < q_1 < ... < q_{K-1}
    < D_axis - (K-1)*(min_gap-1), so no rejection sampling is needed.
    """
    assert min_gap >= 1, f"min_gap must be >= 1; got {min_gap}"
    assert 0 <= K <= D_axis, f"K={K} out of range [0, {D_axis}]"
    if K == 0:
        return []
    if K == D_axis:
        assert min_gap == 1, (
            f"K={K} (full) with min_gap={min_gap} is infeasible; full regime requires min_gap=1"
        )
        return list(range(D_axis))
    reduced = D_axis - (K - 1) * (min_gap - 1)
    if reduced < K:
        raise ValueError(
            f"infeasible: K={K} positions with min_gap={min_gap} in D={D_axis}"
        )
    q = sorted(random.sample(range(reduced), K))
    return [q_i + i * (min_gap - 1) for i, q_i in enumerate(q)]
