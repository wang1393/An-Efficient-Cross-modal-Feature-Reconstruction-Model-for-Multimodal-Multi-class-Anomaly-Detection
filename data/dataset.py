"""Aligned RGB-depth samples read from a generated meta.json index."""
import json
import random
import numpy as np
import tifffile as tiff
from PIL import Image
import torch.utils.data as data
from util.data import get_img_loader
from . import DATA

@DATA.register_module
class RGBDDataset(data.Dataset):
    def __init__(self, cfg, train=True, transform=None, target_transform=None):
        self.root = cfg.data.root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.loader = get_img_loader(cfg.data.loader_type)
        self.loader_target = get_img_loader(cfg.data.loader_type_target)

        self.data_all = []
        name = self.root.split('/')[-1]
        if name in ['mvtec3d']:
            meta_info = json.load(open(f'{self.root}/{cfg.data.meta}', 'r'))
            if self.train:
                meta_info, meta_info_val = meta_info['train'], meta_info['validation']
                for k in meta_info.keys():
                    meta_info[k].extend(meta_info_val[k])
            else:
                meta_info = meta_info['test']
            self.cls_names = cfg.data.cls_names
            if not isinstance(self.cls_names, list):
                self.cls_names = [self.cls_names]
            self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names

        elif name in ['eyecandies_preprocessed']:
            meta_info = json.load(open(f'{self.root}/{cfg.data.meta}', 'r'))
            if self.train:
                meta_info = meta_info['train']
            else:
                meta_info = meta_info['test']
            self.cls_names = cfg.data.cls_names
            if not isinstance(self.cls_names, list):
                self.cls_names = [self.cls_names]
            self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names

        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        random.shuffle(self.data_all) if self.train else None
        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, xyz_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['xyz_path'], \
            data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
        img_path = f'{self.root}/{img_path}'
        img = self.loader(img_path)
        xyz_path = f'{self.root}/{xyz_path}'
        xyz = tiff.imread(xyz_path)
        xyz = np.array(xyz)

        depth = np.copy(xyz)[:, :, 2]
        depth = np.expand_dims(depth, axis=2)
        depth = np.repeat(depth, 3, axis=2)

        depth = ((depth - np.min(depth)) / max(float(np.max(depth) - np.min(depth)), 1e-8) * 255.0).astype(np.uint8)

        depth = Image.fromarray(depth)
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            img_mask = np.array(self.loader_target(f'{self.root}/{mask_path}')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        img = self.transform(img) if self.transform is not None else img
        depth = self.transform(depth) if self.transform is not None else depth
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'depth': depth, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': img_path}
