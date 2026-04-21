# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Training entry point:
```bash
python run_train.py                                                 # uses src/config/default.yaml
python run_train.py --config src/config/default.yaml --voxel-path data/foo.npy
```

Prediction entry point:
```bash
python run_predict.py --run-dir run/<timestamp> --anchors anchors.yaml --output out.npy
```

Dependencies: `pip install -r requirements.txt` (PyTorch from the CUDA 12.6 wheel index; `tifffile` for TIFF volumes).

Tests: `pytest tests/ -v` (CPU-only, <30s).

Outputs land in `run/<YYYYMMDD_HHMMSS>/`: TensorBoard events in `logs/`, model weights in `weights/` (`generator.pth`, `critic_{0,1,2}.pth`), and the resolved config copied to `config.yaml`.

## Architecture

This is a **conditional Slice-GAN**: a 3D U-Net generator is supervised by **three independent 2D critics** (one per spatial axis). The generator consumes `(sparse, mask)` and produces a 3D volume; critics judge 2D slices along axes 0/1/2. Anchor images are positioned by being physically placed in the sparse volume at their target index — no separate index encoding. A single checkpoint handles three regimes: `K = 0` (unconditional), `K ∈ [1, D−1]` (sparse fill), `K = D` (identity). Critic real 2D slices now come from per-axis image pools (or a shared fallback), not from the 3D volume. The 3D volume is optional — when absent, training is unconditional (K=0 only).

Data flow per training step (`src/training/trainer.py`):
1. **Critic real data**: drawn per-axis from `ImageDataset.sample(axis, count=B*S_axis)` — each axis has its own 2D image pool, with an optional shared fallback.
2. **Anchor source**: `VoxelDataset` (optional) provides sub-volumes; `_batch_anchor_sample` picks a regime (empty / full / sparse) **once per batch** and builds `(sparse, mask)` for every sample via `place_anchor_slices`. When `VoxelDataset` is absent, `sparse`/`mask` are zero tensors and `K=0` is forced.
3. `axis = global_step % 3` selects the critic to update.
4. `netG(sparse, mask)` emits `(B, C, D, H, W)`; `_slice_along_axis` flattens to 2D slices `(B*S, C, H, W)` via einops.
5. The selected critic is trained with WGAN-GP loss (`fake_score - real_score + gp`); `src/training/penalty.py::gradient_penalty` computes GP after subsampling fake slices to match the real batch size.
6. Every `gen_freq` steps the generator is updated against the mean of all three critics' fake scores plus `recon_lambda × L1(fake, sub)` at mask positions. When `K = 0` (empty regime) or voxel is absent, the recon term is zero.

Two independent sources feed the trainer each step: `ImageDataset.sample(axis, count)` for real 2D slices (per-axis pools, optional shared fallback), and an optional `itertools.cycle(DataLoader(VoxelDataset))` for 3D sub-volumes (anchor source only). `VoxelDataset.__len__` is still driven by `cfg.data.steps_per_epoch`.

### Config wiring

Everything is composed by explicit `build_*` functions in `src/builder.py` driven by `src/config/default.yaml`. There is no Hydra `instantiate`: swapping the generator or critic class means editing the builder, not the YAML.

`validate_config(cfg)` asserts at startup:
- `cfg.data.in_channels == cfg.critic.channels[0]` (generator output channels are projected via an internal 1×1 Conv3d, so `dec_channels[-1]` is a feature width, not image channels).
- `len(cfg.generator.enc_channels) == len(cfg.generator.dec_channels)`.
- `cfg.anchor.axis ∈ {0, 1, 2}`; `empty_prob + full_prob ≤ 1`; sparse range valid.
- `cfg.data.train_shape` divisible by total stride (`2 ** len(enc_channels)`).
- `cfg.data.images` must resolve to a per-axis pool for each of 0/1/2 (via `data.images.shared` fallback or per-axis `axis0`/`axis1`/`axis2` overrides).
- When `cfg.data.voxel_path` is null, `cfg.anchor.empty_prob` must be 1.0 (forces unconditional regime).

RGB is enabled by setting `data.in_channels` and `critic.channels[0]` to `3` (encoder input = `3 + 1 mask = 4`, projected in by the first encoder block).

### Generator output mode

`UNet3DGenerator.output` selects the final activation: `"tanh"` (matches the `[-1, 1]` dataset normalization) or `"softmax"` (categorical voxels — requires `in_channels` to equal the number of classes; v1 focuses on `tanh`).

### Inference

`src/inference/predictor.py::Predictor.predict(anchor_images, anchor_indices, shape=None, axis=None, seed=None)` loads a trained run and generates a volume. Shape defaults to train shape; user-supplied shape must be ≤ 2× per-dim and divisible by total stride. There is **no inference-time anchor overwrite** — anchor fidelity is driven only by the L1 training loss, so neighbors remain consistent with predicted anchor values.

`run_predict.py` wraps this with a YAML anchor-spec CLI. `src/inference/io.py` handles image loading (with `in_channels`-aware grayscale conversion via `cv2`) and volume output dispatch (`.npy` vs `.tif/.tiff`).
