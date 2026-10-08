"""Shared RGB-depth dataset defaults."""
from argparse import Namespace

class DataConfig(Namespace):
    def __init__(self):
        self.data = Namespace(
            sampler='naive', loader_type='pil', loader_type_target='pil_L',
            type='RGBDDataset', root='data/mvtec3d', meta='meta.json', cls_names=[])
