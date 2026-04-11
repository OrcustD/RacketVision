"""RacketPose pose estimation test — load a trained RTMPose checkpoint and evaluate.

Usage:
    python tools/test_pose.py configs/pose/rtmpose_m_racket.py \
           --checkpoint checkpoints/best_PCK_epoch_90.pth
"""

import argparse
import functools
import os.path as osp
import sys
import torch

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import datasets  # noqa: F401  register custom datasets

_torch_load = torch.load
torch.load = functools.partial(_torch_load, weights_only=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test racket pose estimation model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--work-dir', help='directory to save test outputs')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config settings in key=value format')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config, lazy_import=False)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.train_cfg = None
    cfg.train_dataloader = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
