# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Training (entry point):
```bash
python train.py                          # uses src/config/cfg_gray.yaml, default image_path
python train.py --path images/foo.png    # override cfg.data.image_path only
```

The config name is hardcoded in `train.py` as `cfg_gray.yaml`. To use a different config, edit the `cfg_name` variable in `train.py:48` (Hydra `compose` resolves it from `src/config/`).

Dependencies: `pip install -r requirements.txt` (PyTorch is pulled from the CUDA 12.6 wheel index). No test suite, no linter config in-repo (a `.ruff_cache/` exists locally but no `pyproject.toml`/`ruff.toml` is checked in).

Outputs land in `runs/<YYYYMMDD-HHMMSS>/`: TensorBoard event files in the root, model weights in `weight/` (`generator.pth`, `critic_{0,1,2}.pth`), and the resolved config copied alongside.

## Architecture

This is a **Slice-GAN-style** trainer: a single 3D generator is supervised by **three independent 2D critics**, one per spatial axis. The generator outputs a 3D volume; the critics each see 2D slices taken along axis 0/1/2. This lets a 3D volume be learned from 2D image data.

Data flow per training step (`src/train/trainer.py`):
1. `axis = step % 3` selects which critic to update.
2. Generator emits `fake_3d` of shape `(B, C, X, Y, Z)`. `fake_3d_transform` rearranges to 2D slices `(B*S, C, H, W)` via einops along the chosen axis.
3. The chosen critic is trained with WGAN-GP loss (`fake_score - real_score + gp`), gradient penalty in `src/train/penalty.py` (the slice batch is randomly subsampled to match the real batch size before mixing).
4. Every `train_gen_interval` steps the generator is updated against the **mean** of fake scores from all three critics (loss `-= fake_score / 3` per axis).

Three `DataLoader`s are constructed — one per `slice_axis` — and each is wrapped in `cycle()` (`src/misc.py`) so `next(loader)` is infinite. `ImageDataset.__len__` is hardcoded to 512; one dataset instance is created per axis but currently `slice_axis` is accepted but unused inside `ImageDataset` (axes are differentiated only via the generator-side rearrange).

### Hydra / OmegaConf wiring

Everything is constructed via `hydra.utils.instantiate` driven by `src/config/cfg_gray.yaml`:
- `src/build.py` is the single composition root: `build_config`, `build_dl`, `build_model` (1 generator + 3 critics), `build_optimizer` (one optimizer per net), `build_trainer`.
- To swap generator/critic/dataset/optimizer, change the `_target_` in the YAML — no code edits needed.
- `cfg.device` is read directly to move models to GPU; default is `"cuda"`.

### Generator output mode

`Generator.otype` selects the final activation: `"reg"` → `tanh` (grayscale, matches `cfg_gray.yaml`'s normalize-to-[-1,1] dataset), `"clf"` → `softmax` along channel dim (categorical voxels). Picking `otype` should match `channels[-1]` and the dataset normalization.

### Known rough edges (worth noting before changes)

- `Generator.forward` does not `return x` (`src/model/generator.py:85`). `generate()` therefore returns `None` — training as-is will crash. Any work touching the generator should fix this.
- `Trainer.__init__` calls `netG.gen_device()` but `Generator` defines `get_device()` (`src/model/generator.py:92`). Same caveat.
- `Critic.init_weights` is defined but never applied (no `self.apply(...)` in `__init__`).
- `bat/run.bat` has placeholder paths (`path1`, `path2`) — it's a template, not runnable.
