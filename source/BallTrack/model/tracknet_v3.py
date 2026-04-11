import torch
import torch.nn as nn
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModel

from model.loss_utils import WBCELoss


class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


@MODELS.register_module(name='TrackNetV3')
class TrackNet(BaseModel):
    def __init__(self, in_dim, out_dim, mixup=False, alpha=0.5, last_only=False, d_model=64):
        super(TrackNet, self).__init__()
        self.alpha = alpha
        self.mixup = mixup
        self.last_only = last_only
        self.down_block_1 = Double2DConv(in_dim, d_model)
        self.down_block_2 = Double2DConv(d_model, d_model * 2)
        self.down_block_3 = Triple2DConv(d_model * 2, d_model * 4)
        self.bottleneck = Triple2DConv(d_model * 4, d_model * 8)
        self.up_block_1 = Triple2DConv(d_model * (8 + 4), d_model * 4)
        self.up_block_2 = Double2DConv(d_model * (4 + 2), d_model * 2)
        self.up_block_3 = Double2DConv(d_model * (2 + 1), d_model)
        self.predictor = nn.Conv2d(d_model, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def _forward(self, x):
        x1 = self.down_block_1(x)                                        # (N,   64, 288, 512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                      # (N,   64, 144, 256)
        x2 = self.down_block_2(x)                                        # (N,  128, 144, 256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                      # (N,  128,  72, 128)
        x3 = self.down_block_3(x)                                        # (N,  256,  72, 128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                      # (N,  256,  36,  64)
        x = self.bottleneck(x)                                            # (N,  512,  36,  64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)       # (N,  768,  72, 128)
        x = self.up_block_1(x)                                            # (N,  256,  72, 128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)       # (N,  384, 144, 256)
        x = self.up_block_2(x)                                            # (N,  128, 144, 256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)       # (N,  192, 288, 512)
        x = self.up_block_3(x)                                            # (N,   64, 288, 512)
        x = self.predictor(x)                                             # (N, out_dim, 288, 512)
        x = self.sigmoid(x)
        return x

    def forward(self, data_idx=None, frames=None, heatmaps=None, coor=None, vis=None, mode='predict'):
        x = self._forward(frames)
        if self.last_only:
            x = x[:, -1:, :, :]
        labels = heatmaps
        if mode == 'loss':
            return {'loss': WBCELoss(x, labels)}
        elif mode == 'predict':
            return x, labels, coor

    def _mixup(self, x, y, alpha=0.5):
        batch_size = x.size()[0]
        lamb = np.random.beta(alpha, alpha, size=batch_size)
        lamb = np.maximum(lamb, 1 - lamb)
        lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
        index = torch.randperm(batch_size)
        x_mix = x * lamb + x[index] * (1 - lamb)
        y_mix = y * lamb + y[index] * (1 - lamb)
        return x_mix, y_mix

    def data_preprocessor(self, data, training=True):
        data = {key: (val.cuda() if isinstance(val, torch.Tensor) else val)
                for key, val in data.items()}
        if not training:
            return data

        if self.mixup:
            frames, heatmaps = self._mixup(data['frames'], data['heatmaps'], self.alpha)
            data['frames'] = frames
            data['heatmaps'] = heatmaps
        return data
