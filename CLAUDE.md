# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Training entry point:
```bash
python run_train.py                                                 # uses configs/default.yaml
python run_train.py --config configs/default.yaml --images-dir data/my_pool
```

Prediction is invoked programmatically via `src/inference/predictor.py::Predictor` (see `notebooks/05_predict.ipynb`).

Dependencies: `pip install -r requirements.txt` (PyTorch from the CUDA 12.6 wheel index).

Tests: `pytest tests/ -v` (CPU-only, <30s).

Outputs land in `run/<YYYYMMDD_HHMMSS>/`: TensorBoard events in `logs/`, model weights in `weights/` (`generator.pth`, `critic_{0,1,2}.pth`), and the resolved config copied to `config.yaml`.

## Architecture

This is a **conditional Slice-GAN**: a 3D U-Net generator is supervised by **three independent 2D critics** (one per spatial axis). The generator consumes `(sparse, mask)` and produces a 3D volume; critics judge 2D slices along axes 0/1/2. Anchor images are positioned by being physically placed in the sparse volume at their target index along **axis 0** — no separate index encoding. Training mixes two regimes: `K = 0` (unconditional, probability `empty_prob`) and `K ∈ [1, max_K]` (sparse fill) where `max_K` is the largest count that fits under `min_gap` spacing along axis 0. At inference the same checkpoint accepts any `K ≤ D`. Training requires **only 2D image pools** — no 3D ground-truth volume. Anchors are synthesized at each step by sampling K images from the axis-0 pool and planting them at distinct positions along axis 0, separated by at least `min_gap`. Users are expected to orient their data so the physical slicing direction is axis 0.

Data flow per training step (`src/training/trainer.py`):
1. **Anchor synthesis**: `Trainer._make_anchor_batch` picks a regime (empty / sparse) and `K` **once per batch** via `choose_anchor_count`. For each sample it draws `K` positions along axis 0 via `sample_positions_with_gap` (honoring `min_gap`), samples `B*K` 2D images from the axis-0 pool, and places them into `(sparse, mask)`. The K distribution in the sparse regime is controlled by `anchor.k_dist` (`uniform` or `log_uniform`).
2. **Generator forward (no grad)**: `netG(sparse, mask)` emits `(B, C, D, H, W)` once per step; `_slice_along_axis` flattens to 2D slices `(B*S, C, H, W)` via einops per axis.
3. **Critic updates**: all three critics are updated each step against the shared `fake_3d`, with real 2D slices drawn per-axis from `ImageDataset.sample(axis, count=B*S_axis)`. Each critic trains with WGAN-GP loss (`fake_score - real_score + gp`); `src/training/penalty.py::gradient_penalty` computes GP after subsampling fake slices to match the real batch size. For axis 0, slices at anchor positions are dropped before scoring (they are planted ground truth, not generated distribution).
4. **Generator update**: every `gen_freq` steps the generator is updated against the mean of all three critics' fake scores plus `recon_lambda × L1(fake, sparse)` at mask positions. At mask==1 positions `sparse` is the planted condition image, so the recon term drives identity mapping. When `K = 0` (empty regime), recon is zero.

The sole data source is `ImageDataset.sample(axis, count)`: it feeds both critic reals (along the current critic axis) and anchor plants (along axis 0).

### Config wiring

Everything is composed by explicit `build_*` functions in `src/builder.py` driven by `configs/default.yaml`. There is no Hydra `instantiate`: swapping the generator or critic class means editing the builder, not the YAML.

`validate_config(cfg)` asserts at startup:
- `cfg.data.in_channels == cfg.critic.channels[0]` (generator output channels are projected via an internal 1×1 Conv3d, so `dec_channels[-1]` is a feature width, not image channels).
- `len(cfg.generator.enc_channels) == len(cfg.generator.dec_channels)`.
- `cfg.anchor.empty_prob ∈ [0, 1]`.
- `cfg.anchor.min_gap ≥ 1`. The sparse-regime K upper bound is derived from `min_gap` and `D_axis`.
- `cfg.anchor.k_dist ∈ {"uniform", "log_uniform"}`. `log_uniform` weights `P(K=k) ∝ 1/k`, oversampling small K to match typical inference usage (users tend to supply 1–4 anchors).
- `cfg.data.train_shape` divisible by total stride (`2 ** len(enc_channels)`).
- `cfg.data.images` must resolve to a per-axis pool for each of 0/1/2 (via `data.images.shared` fallback or per-axis `axis0`/`axis1`/`axis2` overrides).

RGB is enabled by setting `data.in_channels` and `critic.channels[0]` to `3` (encoder input = `3 + 1 mask = 4`, projected in by the first encoder block).

### Generator output mode

`UNet3DGenerator.output` selects the final activation: `"tanh"` (matches the `[-1, 1]` dataset normalization) or `"softmax"` (categorical voxels — requires `in_channels` to equal the number of classes; v1 focuses on `tanh`).

### Generator noise

Two noise sources feed the generator: (a) a `noise_channels`-wide Gaussian latent concatenated to the bottleneck, and (b) per-decoder additive Gaussian noise with a learned per-channel scale (StyleGAN-B style), applied inside every `UpBlock3D` **except the final one**. The final decoder block also drops instance norm so the output distribution is not squashed. This dual injection counters the failure mode where the recon term in K-heavy regimes pushes the generator to ignore its noise input and produce near-deterministic volumes.

### Inference

`src/inference/predictor.py::Predictor.predict(anchor_images, anchor_indices, shape=None, axis=None, seed=None)` loads a trained run and generates a volume. `anchor_images` are **uint8** arrays (`(H, W)` or `(H, W, C)`, 0–255); the predictor normalizes to `[-1, 1]` and applies `sdimg.image` gray↔RGB conversion to match `in_channels`. Shape defaults to train shape; user-supplied shape must be ≤ 2× per-dim and divisible by total stride. There is **no inference-time anchor overwrite** — anchor fidelity is driven only by the L1 training loss, so neighbors remain consistent with predicted anchor values. When `seed` is provided the predictor sets the **global** `torch.manual_seed` — this is required because the per-decoder noise injections inside `UNet3DGenerator` sample via `torch.randn_like`, which ignores per-call `Generator` objects.

`src/inference/io.py` provides `load_anchor_spec` (YAML parsing) and `save_volume` (always `.npy`).
