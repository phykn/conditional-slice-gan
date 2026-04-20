from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate
from .misc import cycle


def build_config(file):
    with initialize(version_base=None, config_path="config"):
        cfg = compose(file)
        OmegaConf.resolve(cfg)
    return cfg


def build_dl(cfg):
    dataset0 = instantiate(cfg.data, slice_axis=0)
    dataset1 = instantiate(cfg.data, slice_axis=1)
    dataset2 = instantiate(cfg.data, slice_axis=2)

    dl0 = cycle(instantiate(cfg.dl, dataset=dataset0))
    dl1 = cycle(instantiate(cfg.dl, dataset=dataset1))
    dl2 = cycle(instantiate(cfg.dl, dataset=dataset2))

    return dl0, dl1, dl2


def build_model(cfg):
    netG = instantiate(cfg.generator, _recursive_=True)
    netC0 = instantiate(cfg.critic, _recursive_=True)
    netC1 = instantiate(cfg.critic, _recursive_=True)
    netC2 = instantiate(cfg.critic, _recursive_=True)

    netG.to(cfg["device"])
    netC0.to(cfg["device"])
    netC1.to(cfg["device"])
    netC2.to(cfg["device"])

    return netG, netC0, netC1, netC2


def build_optimizer(cfg, netG, netC0, netC1, netC2):
    optG = instantiate(cfg.optimizer, params=netG.parameters())

    optC0 = instantiate(cfg.optimizer, params=netC0.parameters())
    optC1 = instantiate(cfg.optimizer, params=netC1.parameters())
    optC2 = instantiate(cfg.optimizer, params=netC2.parameters())

    return optG, optC0, optC1, optC2


def build_trainer(cfg, loaders, netG, optG, netCs, optCs):
    trainer = instantiate(
        cfg.trainer,
        loaders=loaders,
        netG=netG,
        optG=optG,
        netCs=netCs,
        optCs=optCs,
    )
    return trainer


def build_generator(cfg):
    netG = instantiate(cfg.generator, _recursive_=True)
    netG.to(cfg["device"])
    return netG
