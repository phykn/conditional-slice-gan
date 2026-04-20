import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def get_fname(path):
    _, fname = os.path.split(path)
    return fname


def cycle(loader: DataLoader):
    while True:
        for data in loader:
            yield data


def save_yaml(obj, path):
    with open(path, "w") as f:
        OmegaConf.save(obj=obj, f=f)


def load_yaml(path):
    return OmegaConf.load(path)
