# conditional-slice-gan

Slice-GAN trainer for synthesizing 3D volumes from 2D image data. A single 3D generator is supervised by three independent 2D critics — one per spatial axis — trained with WGAN-GP.

## What You Get
- **Single-image → 3D volume**: one 2D image trains a generator that produces `(B, C, X, Y, Z)` volumes; each axis is critiqued independently by a 2D critic that sees slices along that axis.
- **Grayscale and RGB** supported via `data.in_channels` (1 or 3), enforced end-to-end by `check_channel_consistency(cfg)`.
- **WGAN-GP** with random fake-slice subsampling in `src/training/penalty.py::gradient_penalty`.
- Per-run artifacts in `run/<YYYYMMDD_HHMMSS>/`: `config.yaml` snapshot, TensorBoard events in `logs/`, and weights in `weights/` (`generator.pth`, `critic_{0,1,2}.pth`, overwritten every `trainer.save_freq` steps; no resume).
- Explicit `build_*` composition in `src/builder.py` — no Hydra `instantiate`, no reflection. Swapping a generator/critic class is a code edit, not a YAML trick.
- pytest suite (27 cases, CPU-only, < 5s).

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Place a training image (a single 2D image is enough for isotropic synthesis):
- path: `images/sample.png` (or anywhere — override via `--image-path`)
3. Review `src/config/default.yaml`:
- `data` — `image_path`, `image_size`, `in_channels`, `steps_per_epoch`
- `generator` / `critic` — `channels` (node list, first/last must satisfy the consistency check), `kernels`, `strides`, `paddings`, `output` (`"tanh"` or `"softmax"`)
- `optimizer` — `lr`, `betas`
- `trainer` — `gp_lambda`, `gen_batch_size`, `gen_freq`, `steps`, `save_freq`
- `device` — `"cuda"` or `"cpu"`
4. Start training:
```bash
python run_train.py
# or with overrides:
python run_train.py --config src/config/default.yaml --image-path images/foo.png
```

## Monitoring
```bash
tensorboard --logdir run
```
Scalars logged under `train/`: `critic_real_score`, `critic_fake_score`, `wass_dist`, `gp`, `loss`, and `generator_loss` (every `gen_freq` steps).

## Config Conventions

- `generator.channels` is a **node list**: `channels[0]` must equal `latent_shape[0]` (latent input channels); `channels[-1]` must equal `data.in_channels` (output image channels). The list has `len(kernels) + 1` entries.
- `critic.channels` is also a node list: `channels[0]` must equal `data.in_channels`; `channels[-1]` is 1 (scalar score).
- `check_channel_consistency(cfg)` runs at startup and refuses to proceed on mismatch — the three relevant values are `data.in_channels`, `generator.channels[-1]`, `critic.channels[0]`, plus the latent alignment `generator.channels[0] == generator.latent_shape[0]`.

### Switching to RGB

Set all three to `3`:
```yaml
data:
  in_channels: 3
generator:
  channels: [32, 1024, 512, 128, 32, 3]
critic:
  channels: [3, 64, 128, 256, 512, 1]
```

## Architecture

Data flow per training step (`src/training/trainer.py`):
1. `axis = global_step % 3` selects which critic to update.
2. `netG.sample(gen_batch_size)` emits `(B, C, X, Y, Z)`. `SliceGANTrainer.slice_along_axis` rearranges to 2D slices `(B*S, C, H, W)` via einops.
3. The selected critic is trained with `fake_score - real_score + gp` where `gp` subsamples the fake slices to match the real batch size.
4. Every `gen_freq` steps the generator is updated against the mean of all three critics' fake scores.

Three `DataLoader`s (one per axis) are each wrapped in `itertools.cycle` so `next(loader)` is infinite. `SliceDataset.__len__` is driven by `cfg.data.steps_per_epoch`.

## Project Layout
```
conditional-slice-gan/
├── run_train.py
├── requirements.txt
├── src/
│   ├── builder.py                # composition root (build_* functions + consistency check)
│   ├── config/default.yaml
│   ├── data/dataset.py           # SliceDataset
│   ├── model/
│   │   ├── generator.py          # Generator3D, UpBlock3D
│   │   └── critic.py             # Critic2D, DownBlock2D
│   └── training/
│       ├── trainer.py            # SliceGANTrainer
│       └── penalty.py            # gradient_penalty
└── tests/                        # pytest suite (CPU-only)
```

## Tests
```bash
pytest tests/ -v
```
