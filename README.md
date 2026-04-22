# conditional-slice-gan

A conditional Slice-GAN: one 3D U-Net generator supervised by **three independent 2D critics** (one per spatial axis) under WGAN-GP, plus an L1 reconstruction loss at anchor positions. A single checkpoint handles both unconditional generation and sparse-conditional generation — switched implicitly by the number of anchor images supplied at inference.

## Why this exists

Classic SliceGAN trains one generator against one 2D cross-section of a material. To synthesize a different microstructure, you retrain from scratch — one structure, one model.

This project collapses that loop: **train once, condition at inference**. Supply one or more cross-section images (with their positions along a chosen axis), and the same model generates a 3D volume that honors them. Change the conditions, get a different volume. No per-structure retraining.

## Data: training pool vs. inference conditions

Real-world slicing (FIB, microtome, serial-section SEM) produces a **handful of sparse cross-sections along a single physical axis**, not a dense 3D stack. The design leans into that reality:

- **Training data**: a **2D image pool** — cross-section images gathered from various sources. They do not need to come from the same 3D sample, nor to be spatially aligned. The pool teaches the model what a cross-section of this material *looks like* (a distribution), not where each one sits.
- **Inference input**: user-supplied `(image, position)` pairs — as many as you captured, however you want them placed.
- **No 3D ground-truth volume is required**. Conditions are synthesized from the 2D pool at training time (see below), so the expensive 3D data you would otherwise need simply isn't in the loop.

## How training works

At each step we **synthesize** anchor conditions on the fly:

1. Draw `K` from two regimes — `K = 0` (empty, probability `empty_prob`) or `K ∈ [1, max_K]` (sparse, remaining probability). `max_K = min(D_axis - 1, (D_axis - 1) // min_gap + 1)` — the largest count that still fits under `min_gap` spacing, so `min_gap` alone caps how many anchors can appear.
2. Sample `K` 2D images from the anchor-axis pool.
3. Plant them at `K` distinct positions along `anchor.axis`, separated by at least `min_gap`, to build `(sparse, mask)`.
4. Feed `(sparse, mask)` to the 3D U-Net generator; it emits a full `(B, C, D, H, W)` volume.
5. Update losses (below).
6. Repeat — new regime, new `K`, new positions, new images every step.

Two losses split the work:

| Loss | Who | Enforces |
|------|-----|----------|
| **recon (L1)** | generator | At `mask == 1` positions, the output slice equals the planted condition image. |
| **critic (WGAN-GP)** | three 2D critics (one per axis) | Along each axis, generated slice statistics match the real pool. |
| *— (emergent)* | 3-axis critic | 3D coherence: what looks real along every axis simultaneously is, in practice, a coherent volume. |

**Why this generalizes at inference.** The recon loss trains an *identity map* on many random `(image, position)` pairs: `mask == 1` positions must reproduce their input. That is a property of the learned function, not an inference-time patch. At test time, supply new pairs and the generator applies the same identity skill — placed conditions reappear in the output, while the critics' learned coherence fills the rest.

## Constraints

**Single anchor axis.** Physical slicing cuts one direction only, so every anchor — both the ones synthesized during training and the ones a user supplies at inference — sits perpendicular to `anchor.axis`. The other two axes are governed solely by their critics' distribution pressure.

**Minimum gap between anchors (`min_gap`).** Two unrelated anchors placed on adjacent slots are a contradiction for the model: the critic demands smooth transitions, while recon demands "*this here, that there.*" The solution is to spread anchors. `min_gap` enforces this during training by construction (exact sampling via `sample_positions_with_gap`) and is recommended at inference too. For realistic FIB spacing, 4 or 8 is typical.

## Inference: three regimes, one model

Input size alone selects the mode — no flag, no separate checkpoint:

| Anchors | Behavior | Use case |
|---------|----------|----------|
| **0**   | vanilla Slice-GAN; random plausible volume | unconditional generation |
| **1**   | volume containing that slice | minimum-information structural guess |
| **N ≥ 2** | volume honoring every supplied slice | FIB / serial-section reconstruction |

Rules at inference: anchors must lie on the training `anchor.axis`; spacing between anchors should respect `min_gap`; more anchors ⇒ closer match to the intended structure; no retraining.

## Quick start

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Put 2D training images under `data.images.shared` (or per-axis directories). Supported extensions: `.png .jpg .jpeg .bmp .tif .tiff`. Each image must be at least as large as the axis-specific crop: axis 0 → `(H, W)`, axis 1 → `(D, W)`, axis 2 → `(D, H)`.

3. Review `configs/default.yaml`. The main knobs:
    - `data` — `train_shape`, `in_channels` (1 or 3), `images.{shared,axis0,axis1,axis2}`
    - `anchor` — `axis`, `empty_prob`, `min_gap`
    - `generator` — `enc_channels`, `dec_channels`, `noise_channels`, `output` (`tanh` or `softmax`)
    - `critic` — `channels`, `kernels`, `strides`, `paddings`
    - `optimizer` — `lr`, `betas`
    - `trainer` — `gp_lambda`, `recon_lambda`, `gen_freq`, `steps`, `save_freq`
    - `device` — `"cuda"` or `"cpu"`

4. Train:
    ```bash
    python run_train.py                                         # uses configs/default.yaml
    python run_train.py --config configs/default.yaml --images-dir data/my_pool
    ```

5. Predict: programmatically via `src/inference/predictor.py::Predictor` — see `notebooks/05_predict.ipynb` for a worked example.

## Monitoring

```bash
tensorboard --logdir run
```

Scalars under `train/`: `critic_real_score`, `critic_fake_score`, `wass_dist`, `gp`, `loss`, and (every `gen_freq` steps) `generator_loss`, `recon_loss`, `adv_loss`.

## Config conventions

- `data.in_channels` must equal `critic.channels[0]` (1 for grayscale, 3 for RGB).
- `generator` uses `enc_channels` / `dec_channels` (same length); the first encoder block takes `in_channels + 1 mask` and the final layer projects back to `in_channels` via a 1×1 Conv3d. You set `in_channels` once on `data`, not on generator.
- `data.train_shape` must be divisible by `2 ** len(enc_channels)`.
- `anchor.min_gap ≥ 1`. Larger values reduce the number of anchors per batch — the upper bound on K is derived automatically.
- `data.images` must resolve to a per-axis pool for each of 0/1/2, either through `shared` or through `axisK` overrides.

`validate_config(cfg)` runs at startup and refuses to proceed on mismatch.

### Switching to RGB

Set `data.in_channels: 3` and `critic.channels[0]: 3`. The generator handles the encoder/decoder input adjustment internally.

## Project layout

```
conditional-slice-gan/
├── run_train.py
├── requirements.txt
├── configs/default.yaml
├── src/
│   ├── builder.py                  # composition root (validate_config + build_* fns)
│   ├── data/
│   │   ├── image_dataset.py        # per-axis 2D image pools, load_image
│   │   └── anchor_sampling.py      # choose_anchor_count, sample_positions_with_gap
│   ├── model/
│   │   ├── generator.py            # UNet3DGenerator
│   │   └── critic.py               # Critic2D
│   ├── training/
│   │   ├── trainer.py              # ConditionalSliceGANTrainer
│   │   └── penalty.py              # gradient_penalty
│   └── inference/
│       ├── predictor.py            # Predictor
│       └── io.py                   # load_anchor_spec, save_volume (.npy)
├── notebooks/                      # interactive walkthroughs (dataset / model / loss / trainer / predict)
└── tests/                          # pytest suite (CPU-only)
```

## Tests

```bash
pytest tests/ -v
```
