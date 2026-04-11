import functools
import torch

_torch_load = torch.load
torch.load = functools.partial(_torch_load, weights_only=False)

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dist import init_dist
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/crosslstm_long_badminton.py')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', type=str, default=None)
    args = parser.parse_args()
    if args.launcher is not None:
        init_dist(args.launcher)
    print(args)

    cfg = Config.fromfile(args.cfg, lazy_import=False)

    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.test()
