"""RacketPose inference — run RTMDet + RTMPose on video frames and output racket predictions.

Uses low-level mmdet/mmpose init_model + inference APIs instead of MMPoseInferencer
to avoid strict config format requirements.

Outputs per-rally JSON to: <data_root>/<sport>/pred_racket/<match>/<rally>/result.json

Usage:
    cd source/RacketPose
    python tools/inference.py \
        --det_config configs/detection/rtmdet_m_racket_infer.py \
        --det_ckpt checkpoints/epoch_300.pth \
        --pose_config configs/pose/rtmpose_m_racket_infer.py \
        --pose_ckpt checkpoints/best_PCK_epoch_90.pth \
        --data_root ../data --sport badminton --split test
"""

import argparse
import functools
import os
import sys
import glob
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

_torch_load = torch.load
torch.load = functools.partial(_torch_load, weights_only=False)

from mmengine.registry import DefaultScope

try:
    from mmdet.apis import init_detector, inference_detector
except ImportError:
    raise ImportError('mmdet is required. Install via: mim install "mmdet>=3.0.0"')

try:
    from mmpose.apis import init_model as init_pose_model
    from mmpose.apis import inference_topdown
except ImportError:
    raise ImportError('mmpose is required. Install via: mim install "mmpose>=1.0.0"')

SPORT_TO_DET_CAT_ID = {
    'badminton': 0,
    'tabletennis': 1,
    'tennis': 2,
}


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def detect_and_estimate_pose(det_model, pose_model, img, cat_id,
                             bbox_thr=0.3, bbox_max_ratio=0.5, max_instances=4):
    """Run detection then pose estimation on a single image."""
    h, w = img.shape[:2]

    with DefaultScope.overwrite_default_scope('mmdet'):
        det_result = inference_detector(det_model, img)
    pred_instances = det_result.pred_instances

    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()

    keep = (labels == cat_id) & (scores >= bbox_thr)
    bboxes = bboxes[keep]
    scores = scores[keep]

    valid_bboxes = []
    valid_scores = []
    for bbox, score in zip(bboxes, scores):
        area_ratio = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)
        if area_ratio < bbox_max_ratio:
            valid_bboxes.append(bbox)
            valid_scores.append(score)
            if len(valid_bboxes) >= max_instances:
                break

    if not valid_bboxes:
        return []

    valid_bboxes = np.array(valid_bboxes)

    pose_results = inference_topdown(pose_model, img, valid_bboxes)

    predictions = []
    for i, pose_result in enumerate(pose_results):
        pred = {
            'bbox': [valid_bboxes[i].tolist()],
            'bbox_score': float(valid_scores[i]),
            'keypoints': pose_result.pred_instances.keypoints[0].tolist(),
            'keypoint_scores': pose_result.pred_instances.keypoint_scores[0].tolist(),
        }
        predictions.append(pred)

    return predictions


def main():
    parser = argparse.ArgumentParser(description='RacketPose inference')
    parser.add_argument('--det_config', type=str,
                        default='configs/detection/rtmdet_m_racket_infer.py')
    parser.add_argument('--det_ckpt', type=str,
                        default='checkpoints/epoch_300.pth')
    parser.add_argument('--pose_config', type=str,
                        default='configs/pose/rtmpose_m_racket_infer.py')
    parser.add_argument('--pose_ckpt', type=str,
                        default='checkpoints/best_PCK_epoch_90.pth')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--sport', type=str, default='badminton',
                        choices=['badminton', 'tabletennis', 'tennis'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--bbox_thr', type=float, default=0.3)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cat_id = SPORT_TO_DET_CAT_ID[args.sport]
    print(f'Building models for {args.sport} (det_cat_id={cat_id}) ...')

    print(f'  Loading detector: {args.det_config}')
    det_model = init_detector(args.det_config, args.det_ckpt, device=args.device)
    det_model.eval()

    print(f'  Loading pose model: {args.pose_config}')
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, device=args.device)
    pose_model.eval()
    print('  Models ready.')

    sport_dir = os.path.join(args.data_root, args.sport)
    anno_path = os.path.join(sport_dir, 'info', f'{args.split}.json')
    anno = json.load(open(anno_path, 'r'))
    if args.debug:
        anno = anno[:1]

    for match_id, round_id in tqdm(anno, desc=f'{args.sport}/{args.split}'):
        frame_dir = os.path.join(sport_dir, 'all', match_id, 'frame', round_id)
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
        if not frame_paths:
            continue

        results_all = []
        for fp in tqdm(frame_paths, desc=f'  {match_id}/{round_id}', leave=False):
            img = cv2.imread(fp)
            preds = detect_and_estimate_pose(
                det_model, pose_model, img, cat_id, bbox_thr=args.bbox_thr)
            results_all.append(preds)

        results_all = convert_to_serializable(results_all)

        out_dir = os.path.join(sport_dir, 'pred_racket', match_id, round_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'result.json')
        with open(out_path, 'w') as f:
            json.dump(results_all, f)
        print(f'  -> {out_path} ({len(results_all)} frames)')

    print('Done.')


if __name__ == '__main__':
    main()
