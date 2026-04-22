import random
from dataclasses import dataclass

_VALID_K_DISTS = ("uniform", "log_uniform")


@dataclass(frozen=True)
class AnchorSpec:
    """Regime probabilities and structural constraints for anchor synthesis.

    - `empty_prob`: probability of drawing K=0 (unconditional regime).
      Otherwise K is drawn over `[1, max_K]` where
      `max_K = min(D_axis - 1, (D_axis - 1) // min_gap + 1)` — i.e. the largest
      count that still fits under `min_gap` spacing along axis 0.
    - `min_gap`: minimum distance between any two planted positions (>= 1).
    - `k_dist`: distribution over `[1, max_K]` when the sparse regime is chosen.
      `"uniform"` draws each K with equal probability; `"log_uniform"` weights
      `P(K=k) ∝ 1/k`, which oversamples small K to match typical inference usage
      (users supply 1–4 anchors) while still covering large K.
    """

    empty_prob: float
    min_gap: int
    k_dist: str = "uniform"

    def __post_init__(self) -> None:
        if not 0.0 <= self.empty_prob <= 1.0:
            raise ValueError(f"empty_prob must be in [0, 1]; got {self.empty_prob}")
        if self.min_gap < 1:
            raise ValueError(f"min_gap must be >= 1; got {self.min_gap}")
        if self.k_dist not in _VALID_K_DISTS:
            raise ValueError(
                f"k_dist must be one of {_VALID_K_DISTS}; got {self.k_dist!r}"
            )


def max_anchors_under_gap(D_axis: int, min_gap: int) -> int:
    """Largest K such that K positions with spacing >= min_gap fit in [0, D_axis).
    Capped at D_axis - 1 so the full regime (K = D_axis) is never reached."""
    return min(D_axis - 1, (D_axis - 1) // min_gap + 1)


def choose_anchor_count(D_axis: int, spec: AnchorSpec) -> int:
    if random.random() < spec.empty_prob:
        return 0
    max_K = max_anchors_under_gap(D_axis, spec.min_gap)
    if spec.k_dist == "uniform":
        return random.randint(1, max_K)
    ks = range(1, max_K + 1)
    return random.choices(ks, weights=[1.0 / k for k in ks], k=1)[0]


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
