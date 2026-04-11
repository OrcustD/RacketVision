# RacketPose

Racket detection and 5-keypoint pose estimation module, built on **RTMDet** (detection) and **RTMPose** (pose estimation) from the OpenMMLab ecosystem.

## Architecture

The module runs a two-stage pipeline:

1. **RTMDet-M** detects racket bounding boxes (3 classes: badminton / tabletennis / tennis racket).
2. **RTMPose-M** estimates 5 keypoints per detected racket: `top`, `bottom`, `handle`, `left`, `right`.

## Directory Structure

```
RacketPose/
├── configs/
│   ├── detection/
│   │   ├── rtmdet_m_racket.py          # Full training config
│   │   ├── rtmdet_m_racket_infer.py    # Inference-only config (no _base_)
│   │   └── rtmdet_toy.py              # Toy config for pipeline testing
│   ├── pose/
│   │   ├── rtmpose_m_racket.py         # Full training config
│   │   ├── rtmpose_m_racket_infer.py   # Inference-only config (no _base_)
│   │   └── rtmpose_toy.py             # Toy config for pipeline testing
│   └── _base_/                         # Shared base configs
├── checkpoints/
│   ├── epoch_300.pth                   # RTMDet pre-trained weights
│   └── best_PCK_epoch_90.pth          # RTMPose pre-trained weights
├── datasets/                           # Custom dataset classes
├── evaluation/                         # Custom metrics
├── tools/
    │   ├── train_detection.py             # RTMDet training entry point
    │   ├── train_pose.py                  # RTMPose training entry point
    │   ├── test_detection.py              # RTMDet evaluation entry point
    │   ├── test_pose.py                   # RTMPose evaluation entry point
    │   └── inference.py                   # Batch det+pose inference
└── README.md
```

## Environment

RacketPose requires `mmdet` and `mmpose` (tested with mmengine==0.10.7, mmcv==2.1.0, mmdet==3.3.0, mmpose==1.3.2). Use the same **`uball`** conda environment as the rest of the project; see the repository [README](../../README.md) for installation instructions.

## Training

### Detection (RTMDet)

```bash
conda activate uball
python tools/train_detection.py configs/detection/rtmdet_m_racket.py
```

### Pose Estimation (RTMPose)

```bash
python tools/train_pose.py configs/pose/rtmpose_m_racket.py
```

Both scripts use `data_root` from the config's `_base_/datasets/` files. The default expects COCO-format annotations in `data/info/`.

## Evaluation

### Detection (RTMDet)

```bash
python tools/test_detection.py configs/detection/rtmdet_m_racket.py \
       --checkpoint checkpoints/epoch_300.pth
```

### Pose Estimation (RTMPose)

```bash
python tools/test_pose.py configs/pose/rtmpose_m_racket.py \
       --checkpoint checkpoints/best_PCK_epoch_90.pth
```

Both test scripts load the checkpoint, disable training components, and run `runner.test()`. You can override config values with `--cfg-options`, e.g.:

```bash
python tools/test_detection.py configs/detection/rtmdet_m_racket.py \
       --checkpoint checkpoints/epoch_300.pth \
       --cfg-options test_dataloader.batch_size=16
```

## Inference

Run detection + pose estimation on all clips in a split:

```bash
python tools/inference.py \
    --det_config configs/detection/rtmdet_m_racket_infer.py \
    --det_ckpt checkpoints/epoch_300.pth \
    --pose_config configs/pose/rtmpose_m_racket_infer.py \
    --pose_ckpt checkpoints/best_PCK_epoch_90.pth \
    --data_root ../data \
    --sport badminton \
    --split test \
    --device cuda
```

**Output**: `../data/<sport>/pred_racket/<match>/<rally>/result.json`

Each output JSON is a list of frames. Each frame contains a list of detected rackets:

```json
[
  [
    {
      "bbox": [[x1, y1, x2, y2]],
      "bbox_score": 0.95,
      "keypoints": [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]],
      "keypoint_scores": [0.9, 0.8, 0.7, 0.6, 0.5]
    }
  ],
  [],
  ...
]
```

### Inference Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sport` | `badminton` | Target sport (`badminton` / `tabletennis` / `tennis`) |
| `--bbox_thr` | 0.3 | Detection confidence threshold |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--debug` | off | Process only the first clip |

## Config Notes

- **Training configs** (`rtmdet_m_racket.py`, `rtmpose_m_racket.py`) use `_base_` inheritance and require dataset registration via `import datasets`.
- **Inference configs** (`*_infer.py`) are standalone single-file configs with no `_base_` dependencies. They use `BN` instead of `SyncBN` for compatibility with single-GPU and CPU inference.
- Detection uses `num_classes=3` (one per sport's racket type), not 4.
