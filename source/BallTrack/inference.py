"""BallTrack inference — run TrackNetV3 on video frames and output ball predictions.

Outputs per-rally CSV to: <data_root>/<sport>/pred_ball/<match>/<rally>/results.csv

Usage:
    python inference.py --cfg configs/tracknetv3_base.py \
                        --ckpt checkpoints/balltrack_best.pth \
                        --data_root ../data --sport badminton --split test
"""

import argparse
import os
import sys
import glob
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.amp
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from mmengine.config import Config
from mmengine.registry import MODELS

import model  # noqa: F401  register TrackNetV3
import dataset  # noqa: F401  register Uball


def remove_ddp_prefix(state_dict):
    return {(k[7:] if k.startswith('module.') else k): v
            for k, v in state_dict.items()}


class BallInferencer:
    def __init__(self, cfg_path, ckpt_path, device='cuda', thre=0.5, batchsize=20):
        self.cfg = Config.fromfile(cfg_path, lazy_import=False)
        self.device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model = self._build_model(ckpt_path)
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.seq_len = self.cfg.seq_len
        self.thre = thre
        self.batchsize = batchsize

    def _build_model(self, ckpt_path):
        mdl = MODELS.build(self.cfg.model)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state:
            state = state['state_dict']
        mdl.load_state_dict(remove_ddp_prefix(state))
        mdl.to(self.device).eval()
        return mdl

    def _load_frames(self, frame_paths):
        def _read(fp):
            return cv2.imread(fp)

        with ThreadPoolExecutor() as ex:
            raw = list(ex.map(_read, frame_paths))

        resized = np.array(
            [cv2.resize(f, (self.width, self.height)) for f in raw],
            dtype=np.float32) / 255.0
        return raw, resized

    def _preprocess_batch(self, frames, start, end, median):
        batch = []
        for i in range(start, end):
            fids = [max(0, j) for j in range(i - self.seq_len, i)]
            seq = frames[fids]
            data = np.concatenate(
                [deepcopy(median), np.moveaxis(seq, -1, 1)], 0
            ).reshape(-1, self.height, self.width)
            batch.append(data)
        return torch.from_numpy(np.array(batch)).float().to(self.device)

    def _predict_location(self, heatmap):
        mask = heatmap > self.thre
        if mask.max() == 0:
            return 0, 0, 0, 0, 0.0
        mask_img = (mask.astype('uint8')) * 255
        cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0, 0, 0, 0, 0.0
        rects = [cv2.boundingRect(c) for c in cnts]
        best = max(rects, key=lambda r: r[2] * r[3])
        x, y, w, h = best
        conf = float(np.mean(heatmap[y:y + h, x:x + w]))
        return x, y, w, h, conf

    def __call__(self, frame_paths, median_path):
        raw_frames, frames = self._load_frames(frame_paths)
        img_h, img_w = raw_frames[0].shape[:2]
        sx, sy = img_w / self.width, img_h / self.height

        median = np.load(median_path)['median']
        median = cv2.resize(median, (self.width, self.height)) / 255.0
        median = np.moveaxis(np.expand_dims(median, 0), -1, 1)

        results = []
        for i in tqdm(range(0, len(frames), self.batchsize), desc='  batches', leave=False):
            end = min(len(frames), i + self.batchsize)
            data = self._preprocess_batch(frames, i, end, median)
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        preds, _, _ = self.model.forward(frames=data)
                else:
                    preds, _, _ = self.model.forward(frames=data)
            preds = preds.detach().cpu().numpy()

            for j in range(preds.shape[0]):
                hm = preds[j][0]
                x, y, w, h, conf = self._predict_location(hm)
                cx = int((x + w / 2) * sx)
                cy = int((y + h / 2) * sy)
                vis = 0 if cx == 0 and cy == 0 else 1
                results.append(dict(
                    Frame=i + j, X=cx, Y=cy, Visibility=vis, Confidence=round(conf, 4)))
        return results


def main():
    parser = argparse.ArgumentParser(description='BallTrack inference')
    parser.add_argument('--cfg', type=str, default='configs/tracknetv3_base.py')
    parser.add_argument('--ckpt', type=str, default='checkpoints/balltrack_best.pth')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--sport', type=str, default='badminton',
                        choices=['badminton', 'tabletennis', 'tennis'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--thre', type=float, default=0.5)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    sport_dir = os.path.join(args.data_root, args.sport)
    anno_path = os.path.join(sport_dir, 'info', f'{args.split}.json')
    anno = json.load(open(anno_path, 'r'))
    if args.debug:
        anno = anno[:1]

    inferencer = BallInferencer(
        args.cfg, args.ckpt, device=args.device,
        thre=args.thre, batchsize=args.batchsize)

    for match_id, round_id in tqdm(anno, desc=f'{args.sport}/{args.split}'):
        frame_dir = os.path.join(sport_dir, 'all', match_id, 'frame', round_id)
        frames = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
        if not frames:
            continue
        median_path = os.path.join(sport_dir, 'all', match_id, 'median.npz')
        if not os.path.exists(median_path):
            median_path = os.path.join(sport_dir, 'all', match_id, round_id, 'median.npz')

        results = inferencer(frames, median_path)

        out_dir = os.path.join(sport_dir, 'pred_ball', match_id, round_id)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(out_dir, 'results.csv'), index=False)
        print(f'  -> {out_dir}/results.csv ({len(results)} frames)')

    print('Done.')


if __name__ == '__main__':
    main()
