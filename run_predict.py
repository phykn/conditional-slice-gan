import argparse

from src.inference.io import load_anchor_image, load_anchor_spec, save_volume
from src.inference.predictor import Predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="path to run/<timestamp>/")
    p.add_argument("--anchors", required=True, help="anchor spec YAML file")
    p.add_argument("--output", required=True, help="output .npy or .tif path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_anchor_spec(args.anchors)

    anchor_entries = spec.get("anchors", [])
    anchor_indices = [a["index"] for a in anchor_entries]

    shape = spec.get("shape")
    axis = spec.get("axis")

    predictor = Predictor(run_dir=args.run_dir, device=args.device)
    anchor_images = [
        load_anchor_image(a["path"], predictor.in_channels) for a in anchor_entries
    ]
    out = predictor.predict(
        anchor_images=anchor_images,
        anchor_indices=anchor_indices,
        shape=tuple(shape) if shape is not None else None,
        axis=axis,
        seed=args.seed,
    )

    save_volume(args.output, out)
    print(f"wrote {out.shape} volume to {args.output}")


if __name__ == "__main__":
    main()
