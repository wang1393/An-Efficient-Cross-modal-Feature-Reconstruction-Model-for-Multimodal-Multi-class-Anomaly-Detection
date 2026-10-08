"""Model registry and checkpoint loading for cross-modal reconstruction."""
from copy import deepcopy
import torch
from util.registry import Registry
MODEL = Registry('Model')

def get_model(cfg):
    kwargs = deepcopy(cfg.kwargs)
    checkpoint_path = kwargs.pop('checkpoint_path', '')
    strict = kwargs.pop('strict', True)
    if cfg.name.startswith('timm_'):
        import timm
        net = timm.create_model(cfg.name[5:], **kwargs)
    else:
        if cfg.name == 'ecfr_efficientnet':
            from .ablations import efficientnet
        net = MODEL.get_module(cfg.name)(**kwargs)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location='cpu')
        if 'net' in state:
            state = state['net']
        elif 'state_dict' in state:
            state = state['state_dict']
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        net.load_state_dict(state, strict=strict)
    return net

from .ecfr import ecfr
