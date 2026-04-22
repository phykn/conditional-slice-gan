import numpy as np
from omegaconf import OmegaConf


def load_anchor_spec(path: str) -> dict:
    spec = OmegaConf.load(path)
    return OmegaConf.to_container(spec, resolve=True)  # type: ignore[return-value]


def save_volume(path: str, volume: np.ndarray) -> None:
    np.save(path, volume)
