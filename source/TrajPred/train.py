import functools
import torch

_torch_load = torch.load
torch.load = functools.partial(_torch_load, weights_only=False)

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dist import init_dist
from mmengine.registry import DATASETS, MODELS
from mmengine.analysis import get_model_complexity_info
import argparse
from dataset.trajectory import BallTrajectoryDataset

def debug_dataset(cfg):
    print(cfg.train_dataloader.dataset)
    dataset_train = DATASETS.build(cfg.train_dataloader.dataset)
    sample = dataset_train[1]
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(key, type(val), val.shape, val.dtype)
        else:
            # For non-tensor items like metadata
            if isinstance(val, list):
                print(key, type(val), len(val), "items")
            else:
                # Handle other types of data
                if isinstance(val, dict):
                    print(key, type(val), len(val), "keys")
                else:
                    # Fallback for other types
                    print(key, type(val), val)
    return dataset_train

def debug_model(cfg):
    print(cfg.model)
    model = MODELS.build(cfg.model)
    print(model)
    input_shape = (20, 2)
    analysis_results = get_model_complexity_info(model, input_shape)
    print("Model Flops: {}".format(analysis_results['flops_str']))
    print("Model Parameters: {}".format(analysis_results['params_str']))
    return model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/tp_transformer/short/tabletennis/ball.py')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', type=str, default=None)
    args = parser.parse_args()
    if args.launcher is not None:
        init_dist(args.launcher)
    print(args)
    # Initialize the configuration
    # cfg_file = 'configs/vanilla_tracknetv3.py'
    cfg = Config.fromfile(args.cfg, lazy_import=False)

    # debug_dataset(cfg)
    # debug_model(cfg)

    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.test()
    
