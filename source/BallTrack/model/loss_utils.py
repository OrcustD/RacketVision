import torch


def WBCELoss(y_pred, y, reduce=True, sigmoid=True):
    """ Weighted Binary Cross Entropy loss function (focal-style) from TrackNetV2.

        Args:
            y_pred (torch.Tensor): Predicted values with shape (N, 1, H, W)
            y (torch.Tensor): Ground truth values with shape (N, 1, H, W)
            reduce (bool): Whether to reduce the loss to a single value or not

        Returns:
            (torch.Tensor): Loss value with shape (1,) if reduce, else (N, 1)
    """
    if not sigmoid:
        y_pred = torch.sigmoid(y_pred)

    loss = (-1) * (torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1))
                   + torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    if reduce:
        return torch.mean(loss)
    else:
        return torch.mean(torch.flatten(loss, start_dim=1), 1)
