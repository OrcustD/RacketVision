"""Build trajectory PKL datasets from `interp_ball/` CSV and `merged_racket/` JSON.

CLI:
    python build_dataset.py --data_root ../data --sport badminton --history 80 --future 20
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import pickle
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json

class BallTrajectoryDataset(Dataset):
    """In-memory samples: history/future ball (2D) and racket (10D) windows."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        history = torch.FloatTensor(sample['history'])
        future = torch.FloatTensor(sample['future'])
        history_rkt = torch.FloatTensor(sample['history_rkt'])
        future_rkt = torch.FloatTensor(sample['future_rkt'])

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

class TrajectoryDatasetBuilder:
    """Scan `interp_ball` + `merged_racket`, build sliding windows, optional PKL save."""

    def __init__(self, data_root="data", history_len=80, future_len=20, sport='uball'):
        self.data_root = data_root
        self.history_len = history_len
        self.future_len = future_len
        self.sports = ['tabletennis', 'badminton', 'tennis'] if sport == 'uball' else [sport]
    
    def load_racket_data(self, json_path):
        try:
            with open(json_path, 'r') as f:
                racket_data = json.load(f)
            return racket_data
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            return []
    
    def extract_racket_keypoints(self, frame_rackets):
        keypoints_flat = np.zeros(10, dtype=np.float32)

        if frame_rackets and len(frame_rackets) > 0:
            racket = frame_rackets[0]
            if 'keypoints' in racket and racket['keypoints']:
                keypoints = racket['keypoints']
                for i, kp in enumerate(keypoints[:5]):
                    if i < 5 and len(kp) >= 2:
                        keypoints_flat[i * 2] = kp[0] / 1920.0
                        keypoints_flat[i * 2 + 1] = kp[1] / 1080.0

        return keypoints_flat
    
    def extract_sequences_from_csv(self, csv_path, json_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return []
        
        racket_data = self.load_racket_data(json_path)
        if not racket_data:
            print(f"No racket data found for {json_path}")
            return []
        
        min_len = min(len(df), len(racket_data))
        df = df.iloc[:min_len]
        racket_data = racket_data[:min_len]
        
        non_empty_mask = ~((df['X'] == 0) & (df['Y'] == 0) & (df['Visibility'] == 0))
        
        if non_empty_mask.sum() < self.history_len + self.future_len:
            return []

        sequences = []
        start_idx = None
        
        for i, is_non_empty in enumerate(non_empty_mask):
            if is_non_empty and start_idx is None:
                start_idx = i
            elif not is_non_empty and start_idx is not None:
                ball_sequence = df.iloc[start_idx:i][['X', 'Y']].values.astype(np.float32)
                ball_sequence[:, 0] /= 1920.0
                ball_sequence[:, 1] /= 1080.0

                racket_sequence = []
                for frame_idx in range(start_idx, i):
                    frame_rackets = racket_data[frame_idx]
                    racket_keypoints = self.extract_racket_keypoints(frame_rackets)
                    racket_sequence.append(racket_keypoints)
                racket_sequence = np.array(racket_sequence, dtype=np.float32)
                
                if len(ball_sequence) >= self.history_len + self.future_len:
                    sequences.append((ball_sequence, racket_sequence))
                start_idx = None

        if start_idx is not None:
            ball_sequence = df.iloc[start_idx:][['X', 'Y']].values.astype(np.float32)
            ball_sequence[:, 0] /= 1920.0
            ball_sequence[:, 1] /= 1080.0

            racket_sequence = []
            for frame_idx in range(start_idx, len(racket_data)):
                frame_rackets = racket_data[frame_idx]
                racket_keypoints = self.extract_racket_keypoints(frame_rackets)
                racket_sequence.append(racket_keypoints)
            racket_sequence = np.array(racket_sequence, dtype=np.float32)
            
            if len(ball_sequence) >= self.history_len + self.future_len:
                sequences.append((ball_sequence, racket_sequence))
        
        return sequences
    
    def create_samples_from_sequence(self, ball_sequence, racket_sequence, sport, match, seq_id):
        samples = []
        total_len = self.history_len + self.future_len
        
        for start_idx in range(len(ball_sequence) - total_len + 1):
            history = ball_sequence[start_idx:start_idx + self.history_len]
            future = ball_sequence[start_idx + self.history_len:start_idx + total_len]
            history_rkt = racket_sequence[start_idx:start_idx + self.history_len]
            future_rkt = racket_sequence[start_idx + self.history_len:start_idx + total_len]
            
            sample = {
                'history': history,
                'future': future,
                'history_rkt': history_rkt,
                'future_rkt': future_rkt,
                'sport': sport,
                'match': match,
                'sequence': seq_id,
                'start_frame': start_idx
            }
            samples.append(sample)
        
        return samples
    
    def build_dataset(self, save_path=None):
        all_samples = []
        
        for sport in self.sports:
            csv_pattern = f"{self.data_root}/{sport}/interp_ball/match*/*/results.csv"
            csv_files = glob.glob(csv_pattern)
            
            print(f"Processing {len(csv_files)} files for {sport}...")
            
            for csv_path in tqdm(csv_files, desc=f"Processing {sport}"):
                csv_path = csv_path.replace('\\', '/')
                path_parts = csv_path.split('/')
                match = path_parts[-3]
                seq_name = path_parts[-2]
                
                json_path = csv_path.replace('/interp_ball/', '/merged_racket/').replace('/results.csv', '/result.json')

                if not os.path.exists(json_path):
                    print(f"Warning: Racket file not found: {json_path}")
                    continue
                
                sequences = self.extract_sequences_from_csv(csv_path, json_path)

                for seq_idx, (ball_sequence, racket_sequence) in enumerate(sequences):
                    seq_id = f"{seq_name}"
                    samples = self.create_samples_from_sequence(ball_sequence, racket_sequence, sport, match, seq_id)
                    all_samples.extend(samples)
        
        print(f"Total samples created: {len(all_samples)}")
        
        if len(all_samples) == 0:
            raise ValueError("No samples were created. Check your data paths and parameters.")
        
        train_samples, test_samples = train_test_split(
            all_samples, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {len(train_samples)}")
        print(f"Testing samples: {len(test_samples)}")
        
        train_dataset = BallTrajectoryDataset(train_samples)
        test_dataset = BallTrajectoryDataset(test_samples)
        
        dataset_dict = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'metadata': {
                'history_len': self.history_len,
                'future_len': self.future_len,
                'sports': self.sports,
                'total_samples': len(all_samples),
                'train_size': len(train_samples),
                'test_size': len(test_samples)
            }
        }
        
        if save_path:
            self.save_dataset(dataset_dict, save_path)
        
        return dataset_dict
    
    def save_dataset(self, dataset_dict, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset_dict, f)
        print(f"Dataset saved to: {save_path}")
    
    @staticmethod
    def load_dataset(load_path):
        with open(load_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        print(f"Dataset loaded from: {load_path}")
        print(f"Metadata: {dataset_dict['metadata']}")
        return dataset_dict

def create_data_loaders(dataset_dict, batch_size=32, num_workers=4):
    train_loader = DataLoader(
        dataset_dict['train_dataset'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset_dict['test_dataset'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build trajectory PKL from interp_ball + merged_racket.")
    parser.add_argument('--history', type=int, default=80)
    parser.add_argument('--future', type=int, default=20)
    parser.add_argument('--sport', type=str, default='uball',
                        help='uball (all sports) or tabletennis / badminton / tennis')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='../data/data_traj')
    args = parser.parse_args()

    builder = TrajectoryDatasetBuilder(
        data_root=args.data_root,
        history_len=args.history,
        future_len=args.future,
        sport=args.sport
    )

    pkl_path = os.path.join(args.output_dir, f'ball_racket_{args.sport}_h{args.history}_f{args.future}.pkl')
    builder.build_dataset(save_path=pkl_path)