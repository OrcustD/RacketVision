#!/usr/bin/env python3
"""Train racket pose estimation model (RTMPose)."""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import datasets  # noqa: F401  register custom datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Train racket pose estimation model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--work-dir', help='directory to save logs and models')
    parser.add_argument(
        '--resume', nargs='?', type=str, const='auto',
        help='resume from checkpoint')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config settings in key=value format')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config, lazy_import=False)
    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.test()


if __name__ == '__main__':
    main()
