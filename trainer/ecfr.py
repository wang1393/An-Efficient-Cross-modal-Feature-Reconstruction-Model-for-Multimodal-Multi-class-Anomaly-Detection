import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import cv2
from PIL import Image

from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from util.data import rgb_vis

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from .base import BaseTrainer
from . import TRAINER

import torch.nn.functional as F
from scipy import stats
import json
from collections import defaultdict
import os

import matplotlib.pyplot as plt
from torchvision import transforms


class ShortcutEvaluator:

    def __init__(self):
        self.sample_results = []
        self.aggregated_results = {
            'normal_cosine_similarity': [],
            'anomaly_cosine_similarity': [],
            'normal_mse': [],
            'anomaly_mse': []
        }

    def compute_cosine_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        feat1_flat = feat1.flatten()
        feat2_flat = feat2.flatten()

        if torch.norm(feat1_flat) < 1e-6 or torch.norm(feat2_flat) < 1e-6:
            return 0.0

        feat1_norm = F.normalize(feat1_flat, dim=0)
        feat2_norm = F.normalize(feat2_flat, dim=0)
        similarity = torch.dot(feat1_norm, feat2_norm)
        return similarity.item()

    def compute_mse(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        mse = F.mse_loss(feat1, feat2)
        return mse.item()

    def align_features_to_original_size(self, teacher_feats, student_feats, original_size=(256, 256)):
        aligned_teacher = []
        aligned_student = []

        for t_feat, s_feat in zip(teacher_feats, student_feats):

            t_upsampled = F.interpolate(t_feat, size=original_size, mode='bilinear', align_corners=False)
            s_upsampled = F.interpolate(s_feat, size=original_size, mode='bilinear', align_corners=False)

            aligned_teacher.append(t_upsampled)
            aligned_student.append(s_upsampled)

        teacher_concat = torch.cat(aligned_teacher, dim=1)
        student_concat = torch.cat(aligned_student, dim=1)

        return teacher_concat, student_concat

    def extract_region_features(self, features: torch.Tensor, mask: torch.Tensor, region_type: str = 'anomaly'):

        if region_type == 'anomaly':
            region_mask = (mask > 0.5).float()
        else:
            region_mask = (mask <= 0.5).float()


        if region_mask.sum() == 0:
            return None, None


        region_features = features * region_mask.unsqueeze(0).unsqueeze(0)

        return region_features, region_mask

    def evaluate_sample(self, teacher_feats, student_feats, mask: torch.Tensor, cls_name: str, sample_id: str):

        teacher_aligned, student_aligned = self.align_features_to_original_size(teacher_feats, student_feats)


        teacher_aligned = teacher_aligned[0]  # [C, H, W]
        student_aligned = student_aligned[0]  # [C, H, W]

        sample_result = {
            'sample_id': sample_id,
            'cls_name': cls_name,
            'mask_size': mask.shape,
            'feature_channels': teacher_aligned.shape[0],
            'regions': {}
        }

        for region_type in ['normal', 'anomaly']:

            teacher_region, region_mask = self.extract_region_features(teacher_aligned, mask, region_type)
            student_region, _ = self.extract_region_features(student_aligned, mask, region_type)


            if teacher_region is None or student_region is None:
                sample_result['regions'][region_type] = {
                    'exists': False,
                    'reason': 'No pixels in this region'
                }
                continue


            valid_indices = region_mask > 0
            region_pixel_count = int(valid_indices.sum().item())

            if region_pixel_count == 0:
                sample_result['regions'][region_type] = {
                    'exists': False,
                    'reason': 'No valid pixels after masking'
                }
                continue


            teacher_valid = teacher_region[:, valid_indices]  # [C, N_valid_pixels]
            student_valid = student_region[:, valid_indices]  # [C, N_valid_pixels]


            cos_sim = self.compute_cosine_similarity(teacher_valid, student_valid)
            mse = self.compute_mse(teacher_valid, student_valid)

            region_result = {
                'exists': True,
                'pixel_count': region_pixel_count,
                'cosine_similarity': cos_sim,
                'mse': mse,
                'feature_stats': {
                    'teacher_mean': float(teacher_valid.mean().item()),
                    'teacher_std': float(teacher_valid.std().item()),
                    'student_mean': float(student_valid.mean().item()),
                    'student_std': float(student_valid.std().item())
                }
            }

            sample_result['regions'][region_type] = region_result


            self.aggregated_results[f'{region_type}_cosine_similarity'].append(cos_sim)
            self.aggregated_results[f'{region_type}_mse'].append(mse)


        self.sample_results.append(sample_result)

        return sample_result

    def compute_statistics(self):
        statistics = {}

        for key, values in self.aggregated_results.items():
            if len(values) > 0:
                values_array = np.array(values)
                statistics[key] = {
                    'mean': float(np.mean(values_array)),
                    'variance': float(np.var(values_array)),
                    'std': float(np.std(values_array)),
                    'count': len(values_array),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array))
                }
            else:
                statistics[key] = {
                    'mean': None,
                    'variance': None,
                    'std': None,
                    'count': 0,
                    'min': None,
                    'max': None
                }

        return statistics


@TRAINER.register_module
class ECFRTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ECFRTrainer, self).__init__(cfg)
        self.shortcut_evaluator = ShortcutEvaluator() if getattr(cfg, 'enable_shortcut_eval', False) else None

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.depths = inputs['depth'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.bs = self.imgs.shape[0]
        self.img_path = inputs['img_path']

    def forward(self):
        self.feats_t, self.depths_t, self.feats_s, self.depths_s = self.net(self.imgs, self.depths)

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
            self.depths, _ = self.mixup_fn(self.depths, torch.ones(self.depths.shape[0], device=self.depths.device))
        with self.amp_autocast():
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.depths_s)
            loss_depth_mse = self.loss_terms['pixel'](self.depths_t, self.feats_s)
            loss = loss_mse + loss_depth_mse
        self.backward_term(loss, self.optim)
        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss, self.world_size).clone().detach().item(), 1,
                        self.master)


    @torch.no_grad()
    def test(self):
        self.reset(isTrain=False)

        mse_stats = {
            'rgb': {'all': [], 'normal': [], 'abnormal': []},
            'depth': {'all': [], 'normal': [], 'abnormal': []}
        }

        imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []

        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)

        if self.master:
            print(f"Starting test on {test_length} batches...")

        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()

            rgb_sample_mse = self.calculate_sample_mse(self.feats_t, self.depths_s)
            depth_sample_mse = self.calculate_sample_mse(self.depths_t, self.feats_s)
            mse_stats['rgb']['all'].extend(rgb_sample_mse)
            mse_stats['depth']['all'].extend(depth_sample_mse)

            for i, is_abnormal in enumerate(self.anomaly):
                if is_abnormal.item() == 0:
                    mse_stats['rgb']['normal'].append(rgb_sample_mse[i])
                    mse_stats['depth']['normal'].append(depth_sample_mse[i])
                else:
                    mse_stats['rgb']['abnormal'].append(rgb_sample_mse[i])
                    mse_stats['depth']['abnormal'].append(depth_sample_mse[i])


            loss_rgb = self.loss_terms['pixel'](self.feats_t, self.depths_s)
            loss_depth = self.loss_terms['pixel'](self.depths_t, self.feats_s)
            loss = loss_rgb + loss_depth
            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss, self.world_size).clone().detach().item(),
                            1, self.master)


            anomaly_map_rgb, _ = self.evaluator.cal_anomaly_map(
                self.feats_t, self.depths_s,
                [self.imgs.shape[2], self.imgs.shape[3]],
                uni_am=False, amap_mode='add', gaussian_sigma=4
            )
            anomaly_map_depth, _ = self.evaluator.cal_anomaly_map(
                self.depths_t, self.feats_s,
                [self.imgs.shape[2], self.imgs.shape[3]],
                uni_am=False, amap_mode='add', gaussian_sigma=4
            )
            anomaly_map = anomaly_map_rgb + anomaly_map_depth

            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))

            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None


            if self.master and batch_idx % 100 == 0:
                msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                log_msg(self.logger, msg)


        if self.master:
            self.print_mse_statistics(mse_stats)


        if self.cfg.dist:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f'{self.tmp_dir}/*.pth')
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except:
                                time.sleep(1)
                        valid_results = True
        else:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)


        if self.master:
            results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None

                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
                                    stralign="center")
            log_msg(self.logger, f'\n{msg}')

            if self.shortcut_evaluator and self.master:
                evaluate_shortcut_issues(self)

    def calculate_sample_mse(self, teacher_features, student_features):

        batch_size = teacher_features[0].shape[0]
        sample_mse_list = []

        for b in range(batch_size):
            sample_total_mse = 0


            for t_feat, s_feat in zip(teacher_features, student_features):

                if t_feat.shape != s_feat.shape:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear')


                scale_mse = torch.mean((t_feat[b] - s_feat[b]) ** 2)
                sample_total_mse += scale_mse

            sample_avg_mse = sample_total_mse / 3
            sample_mse_list.append(sample_avg_mse.item())

        return sample_mse_list

    def print_mse_statistics(self, mse_stats):

        import numpy as np

        def calculate_stats(mse_list, name):
            if len(mse_list) == 0:
                return f"{name}: No samples"

            mse_array = np.array(mse_list)
            mean_val = np.mean(mse_array)
            std_val = np.std(mse_array)
            count = len(mse_list)

            return f"{name}: Mean={mean_val:.6f}, Std={std_val:.6f}, Count={count}"

        def calculate_difference(normal_list, abnormal_list, modal_name):
            if len(normal_list) == 0 or len(abnormal_list) == 0:
                return f"{modal_name} Difference: Cannot calculate (missing samples)"

            normal_mean = np.mean(normal_list)
            abnormal_mean = np.mean(abnormal_list)
            diff = abnormal_mean - normal_mean

            return f"{modal_name} Difference (Abnormal - Normal): {diff:.6f}"

        log_msg(self.logger, "\n" + "=" * 80)
        log_msg(self.logger, "MSE Reconstruction Statistics (Same-Modal)")
        log_msg(self.logger, "=" * 80)

        log_msg(self.logger, "\nRGB Modality:")
        log_msg(self.logger, calculate_stats(mse_stats['rgb']['all'], "  All samples"))
        log_msg(self.logger, calculate_stats(mse_stats['rgb']['normal'], "  Normal samples"))
        log_msg(self.logger, calculate_stats(mse_stats['rgb']['abnormal'], "  Abnormal samples"))
        log_msg(self.logger,
                f"  {calculate_difference(mse_stats['rgb']['normal'], mse_stats['rgb']['abnormal'], 'RGB')}")

        log_msg(self.logger, "\nDepth Modality:")
        log_msg(self.logger, calculate_stats(mse_stats['depth']['all'], "  All samples"))
        log_msg(self.logger, calculate_stats(mse_stats['depth']['normal'], "  Normal samples"))
        log_msg(self.logger, calculate_stats(mse_stats['depth']['abnormal'], "  Abnormal samples"))
        log_msg(self.logger,
                f"  {calculate_difference(mse_stats['depth']['normal'], mse_stats['depth']['abnormal'], 'Depth')}")

        log_msg(self.logger, "=" * 80)


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize_compound(imgs, preds, masks, clsname, batchid):
    max_score = None
    min_score = None
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    for i in range(imgs.shape[0]):
        vis_dir = f'runs/ecfr_vis/vis_{clsname[i]}/'
        os.makedirs(vis_dir, exist_ok=True)

        h, w = imgs[i].shape[:2]
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))
        img = imgs[i].permute(2, 0, 1)
        img = rgb_vis(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        filename = f'{clsname[i]}_{batchid}_{i}'

        scoremap_self = apply_ad_scoremap(img, normalize(pred))
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(img, pred)

        if masks is not None:
            mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            save_path = os.path.join(vis_dir, filename + '.png')
            if mask.sum() == 0:
                scoremap = np.vstack([img, scoremap_global])
            else:
                scoremap = np.vstack([img, mask, scoremap_global, scoremap_self])
        else:
            scoremap = np.vstack([img, scoremap_global, scoremap_self])
            save_path = os.path.join(vis_dir, filename + '.png')

        scoremap = cv2.cvtColor(scoremap.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)


@torch.no_grad()
def evaluate_shortcut_issues(self):


    self.reset(isTrain=False)

    batch_idx = 0
    test_length = min(self.cfg.data.test_size, 100)  # 闄愬埗璇勪及鏍锋湰鏁伴噺
    test_loader = iter(self.test_loader)

    evaluated_samples = 0

    while batch_idx < test_length and evaluated_samples < 200:
        batch_idx += 1
        try:
            test_data = next(test_loader)
        except StopIteration:
            break

        self.set_input(test_data)

        anomaly_indices = torch.where(self.anomaly == 1)[0]
        if len(anomaly_indices) == 0:
            continue

        self.forward()

        for idx in anomaly_indices:
            if evaluated_samples >= 200:
                break

            sample_id = f"batch_{batch_idx}_sample_{idx.item()}"

            teacher_feats_rgb = [feat[idx:idx + 1] for feat in self.feats_t]
            student_feats_depth = [feat[idx:idx + 1] for feat in self.depths_s]
            mask_sample = self.imgs_mask[idx]
            cls_name = self.cls_name[idx]


            result1 = self.shortcut_evaluator.evaluate_sample(
                teacher_feats_rgb, student_feats_depth,
                mask_sample, cls_name, f"{sample_id}_rgb2depth"
            )

            teacher_feats_depth = [feat[idx:idx + 1] for feat in self.depths_t]
            student_feats_rgb = [feat[idx:idx + 1] for feat in self.feats_s]


            result2 = self.shortcut_evaluator.evaluate_sample(
                teacher_feats_depth, student_feats_rgb,
                mask_sample, cls_name, f"{sample_id}_depth2rgb"
            )

            evaluated_samples += 1


    os.makedirs(self.cfg.logdir, exist_ok=True)


    json_path = f'{self.cfg.logdir}/shortcut_detailed_results.json'
    self.shortcut_evaluator.save_detailed_results(json_path)

    csv_path = f'{self.cfg.logdir}/shortcut_summary_results.csv'
    self.shortcut_evaluator.save_summary_csv(csv_path)

    self.shortcut_evaluator.print_simple_summary()
