"""BallTrack test — load a trained checkpoint and evaluate on test set.

Usage:
    # Test on all sports (default, uses ConcatDataset from config)
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
"""

import functools
import torch

_torch_load = torch.load
torch.load = functools.partial(_torch_load, weights_only=False)

import argparse
import os
from copy import deepcopy
from mmengine.config import Config
from mmengine.runner import Runner


def prepare_test_cfg(cfg, ckpt, sport=None, eval_val=False):
    """Convert a training config to test-only mode."""
    cfg.load_from = ckpt
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.auto_scale_lr = None
    cfg.param_scheduler = None
    cfg.resume = False

    train_dl = cfg.pop('train_dataloader', None)

    if sport is not None:
        dataset_template = None
        if train_dl is not None:
            ds = train_dl.dataset
            if hasattr(ds, 'datasets') and isinstance(ds.datasets, list):
                dataset_template = ds.datasets[0]
            else:
                dataset_template = ds
        if dataset_template is None:
            dataset_template = cfg.test_dataloader.dataset
            if hasattr(dataset_template, 'datasets'):
                dataset_template = dataset_template.datasets[0]

        data_root = os.path.dirname(dataset_template.get('root_dir', '../data'))
        sport_root = os.path.join(data_root, sport)

        test_ds = deepcopy(dataset_template)
        test_ds['root_dir'] = sport_root
        test_ds['split'] = 'test'

        cfg.test_dataloader = dict(
            dataset=test_ds,
            sampler=dict(type='DefaultSampler', shuffle=False),
            collate_fn=dict(type='default_collate'),
            batch_size=cfg.get('batch_size', 2),
            pin_memory=True,
            num_workers=4)

        if eval_val:
            val_ds = deepcopy(dataset_template)
            val_ds['root_dir'] = sport_root
            val_ds['split'] = 'val'
            cfg.val_dataloader = dict(
                dataset=val_ds,
                sampler=dict(type='DefaultSampler', shuffle=False),
                collate_fn=dict(type='default_collate'),
                batch_size=cfg.get('batch_size', 2),
                pin_memory=True,
                num_workers=4)
        else:
            cfg.val_cfg = None
            cfg.val_dataloader = None
            cfg.val_evaluator = None
    else:
        if not eval_val:
            cfg.val_cfg = None
            cfg.val_dataloader = None
            cfg.val_evaluator = None

    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BallTrack evaluation')
    parser.add_argument('--cfg', type=str, required=True,
                        help='config file path')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='checkpoint file path')
    parser.add_argument('--sport', type=str, default=None,
                        choices=['badminton', 'tabletennis', 'tennis'],
                        help='evaluate on a single sport (default: all)')
    parser.add_argument('--eval-val', action='store_true',
                        help='also evaluate on val set')
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg, lazy_import=False)
    cfg = prepare_test_cfg(cfg, args.ckpt, args.sport, args.eval_val)

    runner = Runner.from_cfg(cfg)
    if args.eval_val:
        runner.val()
    runner.test()
