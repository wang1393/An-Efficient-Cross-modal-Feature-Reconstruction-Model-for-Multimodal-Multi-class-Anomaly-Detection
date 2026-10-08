from util.registry import Registry

TRAINER = Registry('Trainer')
from .ecfr import ECFRTrainer

def get_trainer(cfg):
    return TRAINER.get_module(cfg.trainer.name)(cfg)
