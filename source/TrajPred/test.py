"""TrajPred test — load a trained checkpoint and evaluate on test set.

Usage:
    python test.py --cfg configs/crosslstm_long_badminton.py \
                   --ckpt checkpoints/crosslstm_long_badminton.pth
"""

import functools
import torch

from mmengine.config import Config
from mmengine.runner import Runner
import argparse

_torch_load = torch.load
torch.load = functools.partial(_torch_load, weights_only=False)

import dataset  # noqa: F401
import model  # noqa: F401
import metrics  # noqa: F401
import hooks  # noqa: F401


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg, lazy_import=False)

    cfg.load_from = args.ckpt
    cfg.train_cfg = None
    cfg.train_dataloader = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None

    runner = Runner.from_cfg(cfg)
    runner.test()
