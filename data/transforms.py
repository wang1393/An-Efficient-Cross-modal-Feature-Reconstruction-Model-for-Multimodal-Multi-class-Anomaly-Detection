"""Torchvision transforms used by ECFR."""
from torchvision import transforms
from . import TRANSFORMS

for name in ('Resize', 'CenterCrop', 'ToTensor', 'Normalize', 'Compose'):
    TRANSFORMS.register_module(getattr(transforms, name), name=name)
