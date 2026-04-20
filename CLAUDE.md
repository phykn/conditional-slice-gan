# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Training entry point:
```bash
python run_train.py                                  # uses src/config/default.yaml
python run_train.py --config src/config/default.yaml --image-path images/foo.png
```

Dependencies: `pip install -r requirements.txt` (PyTorch from the CUDA 12.6 wheel index).

Tests: `pytest tests/ -v` (CPU-only, <30s).

Outputs land in `run/<YYYYMMDD_HHMMSS>/`: TensorBoard events in `logs/`, model weights in `weights/` (`generator.pth`, `critic_{0,1,2}.pth`), and the resolved config copied to `config.yaml`.

## Architecture

This is a **Slice-GAN-style** trainer: a single 3D generator is supervised by **three independent 2D critics**, one per spatial axis. The generator outputs a 3D volume; each critic sees 2D slices along axis 0/1/2. This lets a 3D volume be learned from 2D image data.

Data flow per training step (`src/training/trainer.py`):
1. `axis = global_step % 3` selects the critic to update.
2. `netG.sample(gen_batch_size)` emits `(B, C, X, Y, Z)`. `SliceGANTrainer.slice_along_axis` rearranges to 2D slices `(B*S, C, H, W)` via einops.
3. The selected critic is trained with WGAN-GP loss (`fake_score - real_score + gp`); `src/training/penalty.py::gradient_penalty` computes GP after subsampling fake slices to match the real batch size.
4. Every `gen_freq` steps the generator is updated against the mean of all three critics' fake scores (`loss = -stack(scores).mean()`).

Three `DataLoader`s are constructed — one per axis — and each is wrapped in `itertools.cycle` so `next(loader)` is infinite. `SliceDataset.__len__` is driven by `cfg.data.steps_per_epoch`.

### Config wiring

Everything is composed by explicit `build_*` functions in `src/builder.py` driven by `src/config/default.yaml`. There is no Hydra `instantiate`: swapping the generator or critic class means editing the builder, not the YAML.

`check_channel_consistency(cfg)` asserts `cfg.data.in_channels == cfg.generator.channels[-1] == cfg.critic.channels[0]` and `cfg.generator.channels[0] == cfg.generator.latent_shape[0]` at startup, catching config typos early. RGB is enabled by setting `data.in_channels`, `generator.channels[-1]`, and `critic.channels[0]` all to `3`.

### Generator output mode

`Generator3D.output` selects the final activation: `"tanh"` (matches the [-1, 1] dataset normalization) or `"softmax"` (categorical voxels — requires `channels[-1]` to equal the number of classes).
