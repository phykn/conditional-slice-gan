import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorSpec:
    """Regime probabilities and structural constraints for anchor synthesis.

    - `axis`: spatial axis (0/1/2) along which anchors are planted.
    - `empty_prob`: probability of drawing K=0 (unconditional regime).
      Otherwise K is drawn uniformly in `[1, max_K]` where
      `max_K = min(D_axis - 1, (D_axis - 1) // min_gap + 1)` — i.e. the largest
      count that still fits under `min_gap` spacing along the anchor axis.
    - `min_gap`: minimum distance between any two planted positions (>= 1).
    """

    axis: int
    empty_prob: float
    min_gap: int


def axis_index(anchor_axis: int, k: int) -> tuple:
    """Index tuple selecting slice k along spatial `anchor_axis` of a (C, D, H, W) array.
    Works on both torch tensors and numpy arrays."""
    return (slice(None),) + tuple(
        k if i == anchor_axis else slice(None) for i in range(3)
    )


def max_anchors_under_gap(D_axis: int, min_gap: int) -> int:
    """Largest K such that K positions with spacing >= min_gap fit in [0, D_axis).
    Capped at D_axis - 1 so the full regime (K = D_axis) is never reached."""
    return min(D_axis - 1, (D_axis - 1) // min_gap + 1)


def choose_anchor_count(D_axis: int, spec: AnchorSpec) -> int:
    if random.random() < spec.empty_prob:
        return 0
    return random.randint(1, max_anchors_under_gap(D_axis, spec.min_gap))


def sample_positions_with_gap(D_axis: int, K: int, min_gap: int) -> list[int]:
    """Sample K distinct positions in [0, D_axis) such that any two differ by at least `min_gap`.

    Uses the bijection p_i = q_i + i * (min_gap - 1) with 0 <= q_0 < q_1 < ... < q_{K-1}
    < D_axis - (K-1)*(min_gap-1), so no rejection sampling is needed.
    """
    assert min_gap >= 1, f"min_gap must be >= 1; got {min_gap}"
    assert 0 <= K <= D_axis, f"K={K} out of range [0, {D_axis}]"
    if K == 0:
        return []
    reduced = D_axis - (K - 1) * (min_gap - 1)
    if reduced < K:
        raise ValueError(
            f"infeasible: K={K} positions with min_gap={min_gap} in D={D_axis}"
        )
    q = sorted(random.sample(range(reduced), K))
    return [q_i + i * (min_gap - 1) for i, q_i in enumerate(q)]
