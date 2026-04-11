import sys
import torch
from torch.utils.data import Dataset
import pickle
import os
from mmengine.registry import DATASETS


class _CompatUnpickler(pickle.Unpickler):
    """Handle PKL files created by build_dataset.py running as __main__."""

    def find_class(self, module, name):
        if name == 'BallTrajectoryDataset' and module == '__main__':
            return _DummySampleHolder
        return super().find_class(module, name)


class _DummySampleHolder:
    """Minimal stand-in so pickle can deserialize the old Dataset objects."""

    def __init__(self, samples=None, **kw):
        self.samples = samples or []

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.samples)


@DATASETS.register_module(name='BallTraj')
class BallTrajectoryDataset(Dataset):
    """PyTorch Dataset for ball trajectory prediction."""

    def __init__(self, load_path, split, transform=None):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Dataset file not found: {load_path}")
        with open(load_path, 'rb') as f:
            dataset_dict = _CompatUnpickler(f).load()
        print(f"Dataset loaded from: {load_path}")
        print(f"Metadata: {dataset_dict['metadata']}")

        ds_key = f'{split}_dataset'
        samples_key = f'{split}_samples'
        if samples_key in dataset_dict:
            self.samples = dataset_dict[samples_key]
        elif ds_key in dataset_dict:
            obj = dataset_dict[ds_key]
            self.samples = obj.samples if hasattr(obj, 'samples') else obj
        else:
            raise KeyError(f"Neither '{samples_key}' nor '{ds_key}' found in PKL")
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract history and future trajectories (only X, Y coordinates)
        history = torch.FloatTensor(sample['history'])  # Shape: (history_len, 2)
        future = torch.FloatTensor(sample['future'])    # Shape: (future_len, 2)
        
        # Extract racket sequences
        history_rkt = torch.FloatTensor(sample['history_rkt'])  # Shape: (history_len, 10)
        future_rkt = torch.FloatTensor(sample['future_rkt'])    # Shape: (future_len, 10)
        
        # Extract metadata
        if 'metadata' in sample:
            metadata = sample['metadata']
        else:
            # Fallback to constructing metadata if not present
            metadata = {
                'sport': sample['sport'],
                'match': sample['match'],
                'sequence': sample['sequence'],
                'start_frame': sample['start_frame']
            }
        
        if self.transform:
            history, future, history_rkt, future_rkt = self.transform(history, future, history_rkt, future_rkt)
        
        return {
            'history': history,
            'future': future,
            'history_rkt': history_rkt,
            'future_rkt': future_rkt,
            'metadata': metadata
        }