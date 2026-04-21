# conditional-slice-gan

A conditional Slice-GAN trainer for 3D volume synthesis from sparse 2D anchor images. A single 3D U-Net generator is supervised by three independent 2D critics — one per spatial axis — trained with WGAN-GP plus an L1 reconstruction loss at anchor positions.

## What This Project Does

### The Problem in Plain English

Imagine you need 64 thin slices of a cake, but you can only physically cut 4 to 8 of them. A machine has to invent the remaining ~60 slices plausibly enough that the whole stack looks real.

The real-world use case: taking cross-sectional images of a material with a scanning electron microscope (SEM) is expensive and slow. Capture a handful of real slices, then let the model reconstruct the rest — and you save weeks of lab time.

### How the Model Learns to "Draw Plausibly"

We don't show the model a "correct answer" and have it imitate. Instead, we stage a **game**:

- **The Generator** fills in the empty slots to produce 64 slices.
- **The Critic** looks at any single slice and guesses: *real photo, or generated?*

The generator's job is to **fool the critic**. As they compete, the generator's outputs become indistinguishable from real slices.

### The Slice-GAN Trick — Three Critics

A cake can be sliced in three directions (top-to-bottom, front-to-back, left-to-right), and each direction shows a different texture. So we hire **three critics**, one per direction.

The generator has to **fool all three simultaneously** → it can't just produce a stack of 2D pictures; it must produce something that looks like a coherent **3D volume**.

This is what lets us train a **3D generator using only 2D image data**.

### How Conditions Are Injected (the core of this project)

We need to tell the generator where the anchor images go — but **not as numeric indices**. Instead, we hand it a **half-empty cake mold**:

```
[A][ ][ ][B][ ][ ][C][ ]
 0  1  2  3  4  5  6  7
```

The generator's task: *take this, fill the blanks.*

- A is **physically seated in slot 0** → the position is the slot itself.
- Empty slots are filled with zeros (black).

Alongside, we hand the generator a **mask** that says which slots are real vs. empty:

```
[1][0][0][1][0][0][1][0]   ← 1 = real anchor, 0 = blank
```

No separate `"A is at index 0, B is at index 3"` list needed. The position **is** the placement.

### Inside the Generator — a 3D U-Net

Problem: when filling blanks, the anchor image at slot 0 must come out **as close to the original as possible**. A naive network would compress and decompress the input, smearing the pixels — so even the L1 loss we apply to anchor positions couldn't recover them.

Solution: a **U-Net**:

- **Encoder**: progressively compresses the half-empty volume (captures big-picture context).
- **Decoder**: expands it back out, reconstructing detail.
- **Skip connections**: link encoder and decoder layers of the same resolution **directly**, so pixel information bypasses the compression bottleneck and flows straight to the output.

Analogy: translating a sentence by reading only its "gist" produces garbled output; translating with the original words alongside produces something faithful. Skip connections provide the original words.

We also inject **noise** at the bottleneck. There's more than one plausible way to fill between anchors, and noise lets the model produce different plausible answers each time.

### One Training Step

1. Sample a random sub-volume from the real 3D dataset.
2. Pick K ∈ {0, 1, …, D} anchor indices at random (details below).
3. Build `(sparse_volume, mask)` from those anchor slices and hand them to the generator.
4. The generator outputs a completed volume.
5. **Two losses**:
   - **Reconstruction (L1)**: at anchor positions, the output must match the original pixel-for-pixel.
   - **Adversarial (WGAN-GP)**: the three critics score how realistic the slices look along each axis. Non-anchor positions are graded **only** by this.
6. Combine the losses and update generator and critics.

### Inference

A user captures a few SEM slices and says:

> "These 4 slices go at indices 0, 20, 40, 60. Fill in the other 60."

The model returns a **fully populated 3D volume**. The 4 measured slices are reproduced as faithfully as training allowed (no post-hoc overwrite — the model is trusted end-to-end); everything else is filled in plausibly.

### One Model, Three Regimes

The same model naturally handles three extremes:

| Input | Interpretation | Behavior |
|---|---|---|
| **0 anchors** | "Invent a plausible volume from scratch" | Matches the original Slice-GAN (noise-only generation) |
| **Sparse anchors** | "Keep these few slices fixed; fill between" | The main use case (SEM reconstruction) |
| **All anchors** | "Reproduce this exact volume" (identity) | Reconstruction loss covers every slice → output approaches input |

During training we mix the three regimes (default: empty 10% / full 10% / sparse 80%) so the inference API can switch between them simply by changing the size of the anchor list. No separate models, no mode flags.

## What You Get

- **Single 3D dataset → 3D volume generator**: one voxel dataset trains a generator that produces `(B, C, X, Y, Z)` volumes; each axis is critiqued independently by a 2D critic that sees slices along that axis.
- **Grayscale and RGB** supported via `data.in_channels` (1 or 3), enforced end-to-end by `validate_config(cfg)`.
- **WGAN-GP** with random fake-slice subsampling in `src/training/penalty.py::gradient_penalty`.
- **Anchor conditioning** via sparse volume + binary mask, with L1 reconstruction loss at anchor positions (no inference-time overwrite).
- **Unified regime handling**: the same model covers unconditional, sparse-conditional, and identity regimes.
- **Inference beyond training size**: fully convolutional generator accepts up to 2× the training volume size.
- Per-run artifacts in `run/<YYYYMMDD_HHMMSS>/`: `config.yaml` snapshot, TensorBoard events in `logs/`, weights in `weights/`.
- Explicit `build_*` composition in `src/builder.py` — no Hydra `instantiate`, no reflection.
- pytest suite (CPU-only, <30s).

## Quick Start

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Place a training volume at the path referenced by `data.volume_path` (supports `.npy` and `.tif`/`.tiff`).

3. Review `src/config/default.yaml`:
    - `data` — `volume_path`, `sub_size`, `in_channels`, `steps_per_epoch`
    - `anchor` — `axis`, `empty_prob`, `full_prob`, `sparse_min`, `sparse_max`
    - `generator` / `critic` — architecture nodes, kernels, strides, paddings
    - `optimizer` — `lr`, `betas`
    - `trainer` — `gp_lambda`, `gen_freq`, `recon_lambda`, `steps`, `save_freq`
    - `device` — `"cuda"` or `"cpu"`

4. Train:
    ```bash
    python run_train.py
    python run_train.py --config src/config/default.yaml --volume-path volumes/foo.npy
    ```

5. Predict:
    ```bash
    python run_predict.py --weights run/<timestamp>/weights --anchors anchors.json --out output.npy
    ```

## Monitoring

```bash
tensorboard --logdir run
```

Scalars logged under `train/`: `critic_real_score`, `critic_fake_score`, `wass_dist`, `gp`, `loss`, and (every `gen_freq` steps) `generator_loss`, `recon_loss`, `adv_loss`.

## Config Conventions

- `generator.channels` is a **node list** of encoder-path channels: `channels[0]` must equal `data.in_channels + 1` (sparse volume + mask); `channels[-1]` is the bottleneck width. The decoder is a mirror image.
- Final output channels equal `data.in_channels`.
- `critic.channels` is a node list: `channels[0]` must equal `data.in_channels`; `channels[-1]` is 1 (scalar score).
- `validate_config(cfg)` runs at startup and refuses to proceed on mismatch.

### Switching to RGB

Set `data.in_channels` to `3`; update `critic.channels[0]` to `3`; adjust `generator.channels[0]` to `4` (3 image + 1 mask).

## Architecture

Data flow per training step (`src/training/trainer.py`):
1. Sample a sub-volume from the `VoxelDataset`.
2. Sample an anchor regime (empty / sparse / full) and pick `K` anchor indices along the configured axis.
3. Build `(sparse_volume, mask)` and feed them to the U-Net generator along with noise at the bottleneck.
4. Compute reconstruction loss (L1) at anchor positions — skipped when `K = 0`.
5. Every step, train one of three critics (round-robin on `global_step % 3`) with WGAN-GP.
6. Every `gen_freq` steps, update the generator against the mean of the three critics' fake scores + `recon_lambda × recon_loss`.

## Project Layout

```
conditional-slice-gan/
├── run_train.py
├── run_predict.py
├── requirements.txt
├── src/
│   ├── builder.py                 # composition root
│   ├── config/default.yaml
│   ├── data/
│   │   ├── dataset.py             # VoxelDataset
│   │   └── anchor_sampling.py     # three-regime anchor sampler
│   ├── model/
│   │   ├── generator.py           # UNet3DGenerator
│   │   └── critic.py              # Critic2D
│   ├── training/
│   │   ├── trainer.py
│   │   └── penalty.py
│   └── inference/
│       └── predictor.py           # Predictor
└── tests/                         # pytest suite (CPU-only)
```

## Tests

```bash
pytest tests/ -v
```
