import torch
import torch.nn as nn


def weighted_mse_loss(prediction, target, weights):
    """Per-step MSE between prediction and target, scaled by `weights` (length pred_len), then averaged."""
    loss_fn = nn.MSELoss(reduction='none')
    elementwise_loss = loss_fn(prediction, target)
    weights_reshaped = weights.view(1, -1, 1).to(prediction.device)
    weighted_loss_elements = elementwise_loss * weights_reshaped
    return weighted_loss_elements.mean()


if __name__ == '__main__':
    batch, pred_len = 4, 20
    pred = torch.randn(batch, pred_len, 2)
    tgt = torch.randn(batch, pred_len, 2)
    w = torch.linspace(1.0, 0.5, pred_len)
    print(weighted_mse_loss(pred, tgt, w).item())
