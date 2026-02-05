# train.py
import os
os.environ["MPLBACKEND"] = "Agg"  
import yaml
from argparse import ArgumentParser
import os
import torch

from src.engine.trainer import Trainer


def main():
    """
    The main entry point for the training script.
    """
    parser = ArgumentParser(description="Unified Training Script for LVEF Estimation")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return

    if 'run' in config and 'gpu_ids' in config['run'] and config['run']['gpu_ids'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['run']['gpu_ids'])
        print(f"[Info] CUDA_VISIBLE_DEVICES set to '{config['run']['gpu_ids']}'")
    else:
        print("[Info] CUDA_VISIBLE_DEVICES not set in config, using default PyTorch visibility.")


    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
