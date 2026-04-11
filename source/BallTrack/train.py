from mmengine.config import Config
from mmengine.runner import Runner
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg, lazy_import=False)
    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.test()
