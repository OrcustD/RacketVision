# TrajPred

Ball trajectory prediction module. Given a history of ball positions and racket poses, the model predicts future ball trajectory coordinates.

## Models

| Model | Registry name | Description |
|-------|-----------------|-------------|
| **LSTM** | `TPLSTM` | Baseline LSTM (`ball_only` or `ball_racket` concatenation) |
| **CrossLSTM-Attn** | `TPLSTMAttn` | Low-dim cross-attention + gated residual + LSTM (used for released weights) |

Released checkpoints use **CrossLSTM-Attn** (`TPLSTMAttn`).

## Directory Structure

```
TrajPred/
├── configs/
│   ├── crosslstm_long_badminton.py    # CrossLSTM-Attn config (badminton)
│   ├── crosslstm_long_tabletennis.py
│   ├── crosslstm_short_tennis.py
│   └── lstm_toy.py                     # Toy config for pipeline testing
├── checkpoints/
│   ├── crosslstm_long_badminton.pth
│   ├── crosslstm_long_tabletennis.pth
│   └── crosslstm_short_tennis.pth
├── model/
│   ├── lstm.py              # TPLSTM
│   ├── cross_lstm.py        # TPLSTMAttn
│   └── loss.py              # Weighted MSE loss
├── dataset/
│   └── trajectory.py        # BallTraj dataset (loads PKL files)
├── metrics/
│   └── traj_metric.py       # ADE / FDE metrics (pixel-space)
├── hooks/
│   └── traj_visualize.py    # Trajectory visualisation hook
├── train.py                 # Training entry point
├── test.py                  # Evaluation entry point
├── build_dataset.py         # Build PKL datasets from predictions
├── linear_interpolate_ball_traj.py  # Interpolate ball trajectory gaps
└── merge_gt_with_predictions.py     # Merge racket GT with predictions
```

## Data Pipeline

TrajPred requires pre-processed ball and racket predictions. The full pipeline:

```
pred_ball/ ──→ linear_interpolate ──→ interp_ball/
pred_racket/ ──→ merge_gt ──→ merged_racket/
interp_ball/ + merged_racket/ ──→ build_dataset ──→ data_traj/*.pkl
```

### Step 1: Interpolate Ball Trajectories

Fill short gaps (< 5 frames) in ball prediction CSVs with linear interpolation:

```bash
python linear_interpolate_ball_traj.py --data_root ../data --sport badminton
```

### Step 2: Merge Racket Predictions with Ground Truth

Replace racket predictions with ground truth annotations where available, creating "soft labels":

```bash
python merge_gt_with_predictions.py --data_root ../data --sport badminton
```

### Step 3: Build PKL Dataset

Create sliding-window trajectory samples:

```bash
python build_dataset.py \
    --data_root ../data \
    --sport badminton \
    --history 80 --future 20
```

Output: `../data/data_traj/ball_racket_badminton_h80_f20.pkl`

Pre-built PKL files for all sports are provided in `../data/data_traj/`.

## Training

```bash
conda activate uball
python train.py --cfg configs/crosslstm_long_badminton.py
```

Training runs both `train()` and `test()`. Checkpoints are saved based on best `ADE` (average displacement error).

## Evaluation

Evaluate a trained checkpoint on the test set:

```bash
python test.py \
    --cfg configs/crosslstm_long_badminton.py \
    --ckpt checkpoints/crosslstm_long_badminton.pth
```

### Metrics

| Metric | Description |
|--------|-------------|
| **ADE** | Average Displacement Error — mean L2 distance (pixels) across all predicted time steps |
| **FDE** | Final Displacement Error — L2 distance (pixels) at the last predicted time step |

Coordinates are de-normalised to 1920×1080 before computing metrics.

## Model Details (CrossLSTM-Attn)

1. **Embedding**: Ball (2D → 64D) and racket (10D → 64D) are projected to a shared low-dimensional space.
2. **Cross-Attention**: Ball features attend to racket features via multi-head attention, with a learnable gating coefficient `alpha`.
3. **Projection**: Fused features are projected up to the LSTM hidden dimension (512D).
4. **LSTM**: 2-layer LSTM processes the fused sequence.
5. **Decoder**: The last hidden state is mapped to `pred_len × 2` coordinates.

Loss: Weighted MSE with linearly decaying weights from 1.0 to 0.5 across the prediction horizon.
