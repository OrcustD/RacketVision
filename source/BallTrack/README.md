# BallTrack

Ball detection and tracking module based on **TrackNetV3**. Given a sequence of video frames and a median background image, the model outputs per-frame heatmaps from which ball position is decoded.

## Architecture

TrackNetV3 is an encoder-decoder network that takes `seq_len + 1` concatenated RGB frames (including the background median) as input and produces `seq_len` heatmaps. Ball position is extracted by thresholding and contour detection.

Key parameters (see `configs/tracknetv3_base.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seq_len` | 4 | Number of frames per sample |
| `width` / `height` | 512 / 288 | Input resolution |
| `bg_mode` | `concat` | Background frame concatenated as extra input |
| `sigma` | 3.5 | Gaussian heatmap radius |

## Directory Structure

```
BallTrack/
├── configs/
│   └── tracknetv3_base.py     # Training / evaluation config
├── checkpoints/
│   └── balltrack_best.pth   # Pre-trained weights
├── model/
│   ├── tracknet_v3.py         # TrackNetV3 model definition
│   └── loss_utils.py          # WBCELoss (focal-style)
├── dataset/
│   └── uball.py               # Dataset loader (CSV + extracted frames)
├── metrics/
│   ├── ball_metrics.py        # TP/TN/FP/FN, Precision/Recall/F1
│   └── ball_coco_metrics.py   # mAP-style evaluation
├── hooks/
│   └── visualizer.py          # Heatmap visualisation hook
├── utils/
│   └── general.py             # Utility functions
├── train.py                   # Training entry point
├── test.py                    # Evaluation entry point
└── inference.py               # Batch inference on video frames
```

## Training

```bash
conda activate uball
python train.py --cfg configs/tracknetv3_base.py
```

The config uses `data_root = '../data'` (relative to `BallTrack/`). Adjust `batch_size`, `max_epochs`, and `num_workers` to fit your hardware.

Training logs and checkpoints are saved to the `work_dir` specified in the config.

## Evaluation

Evaluate a trained checkpoint on the test set:

```bash
# Test on all sports (uses ConcatDataset from config)
python test.py --cfg configs/tracknetv3_base.py \
               --ckpt checkpoints/balltrack_best.pth

# Test on a single sport
python test.py --cfg configs/tracknetv3_base.py \
               --ckpt checkpoints/balltrack_best.pth \
               --sport badminton

# Also evaluate on val set
python test.py --cfg configs/tracknetv3_base.py \
               --ckpt checkpoints/balltrack_best.pth \
               --sport badminton --eval-val
```

When `--sport` is specified, a per-sport test dataset is constructed. Without `--sport`, the config's default `ConcatDataset` (all sports) is used.

Metrics reported: **Accuracy**, **Precision**, **Recall**, **F1**, **Miss Rate**, **Mean Distance**.

## Inference

Generate ball predictions for all clips in a split:

```bash
python inference.py \
    --cfg configs/tracknetv3_base.py \
    --ckpt checkpoints/balltrack_best.pth \
    --data_root ../data \
    --sport badminton \
    --split test \
    --device cuda
```

**Output**: `../data/<sport>/pred_ball/<match>/<rally>/results.csv` with columns `Frame, X, Y, Visibility, Confidence`.

### Inference Options

| Flag | Default | Description |
|------|---------|-------------|
| `--thre` | 0.5 | Heatmap confidence threshold |
| `--batchsize` | 20 | Frames per GPU batch |
| `--debug` | off | Process only the first clip |
