import argparse
import os
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data_utils_16_15_6_1 import TrainDatasetFromFolder
from img_util import tensor2img


INPUT_DIR = 'dataset_0804/input_gray2/'
GT_DIR = 'dataset_0804/gt_gray3/'
INPUT_DIR_COLOR = 'dataset_0804/input_color/'
GT_DIR_COLOR = 'dataset_0804/gt_color3/'
INPUT_DIR_HW = 'dataset_0804/handwriting_output/gray/1/'
GT_DIR_HW = 'dataset_0804/handwriting_output/gray/2/'
INPUT_DIR_COLOR_HW = 'dataset_0804/handwriting_output/color/1/'
GT_DIR_COLOR_HW = 'dataset_0804/handwriting_output/color/2/'
INPUT_DIR_TA = 'documents_data/input1/'
GT_DIR_TA = 'documents_data/output1/'
PIC_DIR = 'pic_patch/'
NOISE_DIR = 'fg_tiles_512/'
NOISE2_DIR = 'noise_patch2/'
BG_DIR = 'bg2/'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export visualization samples from TrainDatasetFromFolder in data_utils_16_15_6_1.py'
    )
    parser.add_argument('--num_samples', type=int, default=100, help='How many samples to export.')
    parser.add_argument('--output_dir', type=str, default='vis_train_dataset_16_15_6_1', help='Output directory.')
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Base random seed. Each exported sample uses seed + sample_index.',
    )
    parser.add_argument(
        '--save_separate',
        action='store_true',
        help='Also save separate lq/gt images in addition to the comparison image.',
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_dataset():
    return TrainDatasetFromFolder(
        INPUT_DIR,
        GT_DIR,
        PIC_DIR,
        NOISE_DIR,
        NOISE2_DIR,
        BG_DIR,
        INPUT_DIR_COLOR,
        GT_DIR_COLOR,
        INPUT_DIR_HW,
        GT_DIR_HW,
        INPUT_DIR_COLOR_HW,
        GT_DIR_COLOR_HW,
        INPUT_DIR_TA,
        GT_DIR_TA,
    )


def add_label(image, text):
    canvas = image.copy()
    cv2.putText(
        canvas,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def cls_map_to_color(cls_map):
    # 0 keep (white), 1 enhance (blue), 2 remove (red), 255 ignore (gray).
    if torch.is_tensor(cls_map):
        cls_map = cls_map.detach().cpu().numpy()
    cls_map = np.ascontiguousarray(cls_map).astype(np.int64)
    h, w = cls_map.shape[-2:]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[cls_map == 0] = (255, 255, 255)
    vis[cls_map == 1] = (255, 0, 0)
    vis[cls_map == 2] = (0, 0, 255)
    vis[cls_map == 255] = (128, 128, 128)
    return vis


def rec_weight_to_color(rec_weight_map):
    if torch.is_tensor(rec_weight_map):
        rec_weight_map = rec_weight_map.detach().cpu().numpy()
    w = np.ascontiguousarray(rec_weight_map).astype(np.float32)
    w_min = float(w.min())
    w_max = float(w.max())
    if w_max - w_min < 1e-6:
        norm = np.zeros_like(w, dtype=np.uint8)
    else:
        norm = np.clip((w - w_min) / (w_max - w_min) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    label = f'rec_weight [{w_min:.2f}, {w_max:.2f}]'
    cv2.putText(color, label, (16, color.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return color


def gt_mask_to_color(gt_mask):
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.detach().cpu().numpy()
    mask = np.ascontiguousarray(gt_mask).astype(np.float32)
    if mask.ndim == 3:
        if mask.shape[0] in (1, 3):
            mask = np.max(mask, axis=0)
        else:
            mask = np.max(mask, axis=2)
    mask = np.clip(mask, 0.0, 1.0)
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    vis[mask > 0.5] = (0, 255, 0)
    return vis


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    if args.save_separate:
        ensure_dir(os.path.join(args.output_dir, 'lq'))
        ensure_dir(os.path.join(args.output_dir, 'gt'))

    dataset = build_dataset()
    export_count = max(args.num_samples, 0)

    for idx in tqdm(range(export_count), desc='Exporting'):
        sample_seed = args.seed + idx
        set_seed(sample_seed)

        sample = dataset[idx % max(len(dataset), 1)]
        if len(sample) == 2:
            img_lq, img_gt = sample
            img_lq_np = tensor2img(img_lq, rgb2bgr=True)
            img_gt_np = tensor2img(img_gt, rgb2bgr=True)
            compare = np.concatenate(
                [
                    add_label(img_lq_np, 'LQ / Input'),
                    add_label(img_gt_np, 'GT / Target'),
                ],
                axis=1,
            )
        elif len(sample) == 3:
            img_lq, img_gt, gt_mask = sample
            img_lq_np = tensor2img(img_lq, rgb2bgr=True)
            img_gt_np = tensor2img(img_gt, rgb2bgr=True)
            mask_vis = gt_mask_to_color(gt_mask)
            compare = np.concatenate(
                [
                    add_label(img_lq_np, 'LQ / Input'),
                    add_label(img_gt_np, 'GT / Target'),
                    add_label(mask_vis, 'artifact mask'),
                ],
                axis=1,
            )
        elif len(sample) == 4:
            img_lq, img_gt, cls_map, rec_weight_map = sample
            img_lq_np = tensor2img(img_lq, rgb2bgr=True)
            img_gt_np = tensor2img(img_gt, rgb2bgr=True)
            cls_vis = cls_map_to_color(cls_map)
            weight_vis = rec_weight_to_color(rec_weight_map)
            compare = np.concatenate(
                [
                    add_label(img_lq_np, 'LQ / Input'),
                    add_label(img_gt_np, 'GT / Target'),
                    add_label(cls_vis, 'cls: 0=keep 1=enh 2=rm 255=ign'),
                    add_label(weight_vis, 'rec_weight'),
                ],
                axis=1,
            )
        else:
            raise RuntimeError(f'Unexpected sample size: {len(sample)}')

        compare_name = f'{idx:04d}_seed_{sample_seed}_compare.jpg'
        cv2.imwrite(os.path.join(args.output_dir, compare_name), compare)

        if args.save_separate:
            lq_name = f'{idx:04d}_seed_{sample_seed}_lq.png'
            gt_name = f'{idx:04d}_seed_{sample_seed}_gt.png'
            cv2.imwrite(os.path.join(args.output_dir, 'lq', lq_name), img_lq_np)
            cv2.imwrite(os.path.join(args.output_dir, 'gt', gt_name), img_gt_np)

    print(f'Exported {export_count} samples to: {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
