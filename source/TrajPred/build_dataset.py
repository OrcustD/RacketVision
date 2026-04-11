"""
Ball Trajectory Dataset Builder

Usage:
python TrajPred2/build_dataset.py --history 20 --future 5 --sport tabletennis

# Build and save dataset
builder = TrajectoryDatasetBuilder(history_len=80, future_len=20)
dataset_dict = builder.build_dataset("datasets/ball_trajectory_dataset.pkl")

# Load existing dataset
dataset_dict = TrajectoryDatasetBuilder.load_dataset("datasets/ball_trajectory_dataset.pkl")

# Create data loaders
train_loader, test_loader = create_data_loaders(dataset_dict, batch_size=16)
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
    """
    PyTorch Dataset for ball trajectory prediction.
    
    Args:
        samples: List of trajectory samples
        transform: Optional transform to be applied on a sample
    """
    
    def __init__(self, samples, transform=None):
        self.samples = samples
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
    """
    Builder class for creating ball trajectory datasets from CSV files.
    """
    
    def __init__(self, data_root="data", history_len=80, future_len=20, sport='uball'):
        self.data_root = data_root
        self.history_len = history_len
        self.future_len = future_len
        self.sports = ['tabletennis', 'badminton', 'tennis'] if sport == 'uball' else [sport]
    
    def load_racket_data(self, json_path):
        """
        Load racket pose data from JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of racket poses for each frame
        """
        try:
            with open(json_path, 'r') as f:
                racket_data = json.load(f)
            return racket_data
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            return []
    
    def extract_racket_keypoints(self, frame_rackets):
        """
        Extract racket keypoints from a single frame.
        
        Args:
            frame_rackets: List of racket detections for one frame
            
        Returns:
            Numpy array of shape (10,) containing flattened keypoints [x1,y1,x2,y2,...]
        """
        # Initialize with zeros
        keypoints_flat = np.zeros(10, dtype=np.float32)
        
        if frame_rackets and len(frame_rackets) > 0:
            # Use the first racket if multiple detected
            racket = frame_rackets[0]
            if 'keypoints' in racket and racket['keypoints']:
                keypoints = racket['keypoints']
                # Flatten the keypoints: [[x1,y1], [x2,y2], ...] -> [x1,y1,x2,y2,...]
                # Flatten the keypoints: [[x1,y1], [x2,y2], ...] -> [x1,y1,x2,y2,...]
                for i, kp in enumerate(keypoints[:5]):  # Only use first 5 keypoints
                    if i < 5 and len(kp) >= 2:
                        keypoints_flat[i*2] = kp[0] / 1920.0      # Normalize X by image width
                        keypoints_flat[i*2+1] = kp[1] / 1080.0   # Normalize Y by image height
        
        return keypoints_flat
    
    def extract_sequences_from_csv(self, csv_path, json_path):
        """
        Extract continuous non-empty frame sequences from CSV file and corresponding racket data.
        
        Args:
            csv_path: Path to the ball trajectory CSV file
            json_path: Path to the racket pose JSON file
            
        Returns:
            List of tuples (ball_sequence, racket_sequence)
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return []
        
        # Load racket data
        racket_data = self.load_racket_data(json_path)
        if not racket_data:
            print(f"No racket data found for {json_path}")
            return []
        
        # Ensure same length
        min_len = min(len(df), len(racket_data))
        df = df.iloc[:min_len]
        racket_data = racket_data[:min_len]
        
        # Identify non-empty frames
        non_empty_mask = ~((df['X'] == 0) & (df['Y'] == 0) & (df['Visibility'] == 0))
        
        if non_empty_mask.sum() < self.history_len + self.future_len:
            return []  # Not enough non-empty frames
        
        # Find continuous segments of non-empty frames
        sequences = []
        start_idx = None
        
        for i, is_non_empty in enumerate(non_empty_mask):
            if is_non_empty and start_idx is None:
                start_idx = i
            elif not is_non_empty and start_idx is not None:
                # End of sequence
                # Extract ball trajectory
                ball_sequence = df.iloc[start_idx:i][['X', 'Y']].values.astype(np.float32)
                ball_sequence[:, 0] /= 1920.0  # Normalize X
                ball_sequence[:, 1] /= 1080.0  # Normalize Y
                
                # Extract racket sequence
                racket_sequence = []
                for frame_idx in range(start_idx, i):
                    frame_rackets = racket_data[frame_idx]
                    racket_keypoints = self.extract_racket_keypoints(frame_rackets)
                    racket_sequence.append(racket_keypoints)
                racket_sequence = np.array(racket_sequence, dtype=np.float32)
                
                if len(ball_sequence) >= self.history_len + self.future_len:
                    sequences.append((ball_sequence, racket_sequence))
                start_idx = None
        
        # Handle case where sequence goes to the end
        if start_idx is not None:
            # Extract ball trajectory
            ball_sequence = df.iloc[start_idx:][['X', 'Y']].values.astype(np.float32)
            ball_sequence[:, 0] /= 1920.0  # Normalize X
            ball_sequence[:, 1] /= 1080.0  # Normalize Y
            
            # Extract racket sequence
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
        """
        Create training samples from continuous sequences.
        
        Args:
            ball_sequence: Numpy array of shape (seq_len, 2) containing normalized [X, Y]
            racket_sequence: Numpy array of shape (seq_len, 10) containing racket keypoints
            sport: Sport name
            match: Match identifier
            seq_id: Sequence identifier
            
        Returns:
            List of samples
        """
        samples = []
        total_len = self.history_len + self.future_len
        
        for start_idx in range(len(ball_sequence) - total_len + 1):
            # Ball trajectory
            history = ball_sequence[start_idx:start_idx + self.history_len]
            future = ball_sequence[start_idx + self.history_len:start_idx + total_len]
            
            # Racket trajectory
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
        """
        Build the complete dataset from all CSV and JSON files.
        
        Args:
            save_path: Optional path to save the dataset
            
        Returns:
            Dictionary containing train and test datasets
        """
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
                
                # Check if JSON file exists
                if not os.path.exists(json_path):
                    print(f"Warning: Racket file not found: {json_path}")
                    continue
                
                # Extract sequences from CSV and JSON
                sequences = self.extract_sequences_from_csv(csv_path, json_path)
                
                # Create samples from each sequence
                for seq_idx, (ball_sequence, racket_sequence) in enumerate(sequences):
                    seq_id = f"{seq_name}"
                    samples = self.create_samples_from_sequence(ball_sequence, racket_sequence, sport, match, seq_id)
                    all_samples.extend(samples)
        
        print(f"Total samples created: {len(all_samples)}")
        
        if len(all_samples) == 0:
            raise ValueError("No samples were created. Check your data paths and parameters.")
        
        # Split into train and test (80-20 split)
        train_samples, test_samples = train_test_split(
            all_samples, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {len(train_samples)}")
        print(f"Testing samples: {len(test_samples)}")
        
        # Create dataset objects
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
        
        # Save if path is provided
        if save_path:
            self.save_dataset(dataset_dict, save_path)
        
        return dataset_dict
    
    def save_dataset(self, dataset_dict, save_path):
        """
        Save the dataset to a pickle file.
        
        Args:
            dataset_dict: Dataset dictionary to save
            save_path: Path to save the pickle file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset_dict, f)
        print(f"Dataset saved to: {save_path}")
    
    @staticmethod
    def load_dataset(load_path):
        """
        Load a dataset from a pickle file.
        
        Args:
            load_path: Path to the pickle file
            
        Returns:
            Dataset dictionary
        """
        with open(load_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        print(f"Dataset loaded from: {load_path}")
        print(f"Metadata: {dataset_dict['metadata']}")
        return dataset_dict

def create_data_loaders(dataset_dict, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders from the dataset.
    
    Args:
        dataset_dict: Dataset dictionary
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
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

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and load ball trajectory dataset.")
    parser.add_argument('--history', type=int, default=80, help='Length of history trajectory')
    parser.add_argument('--future', type=int, default=20, help='Length of future trajectory')
    parser.add_argument('--sport', type=str, default='uball', help='Sport type (uball, tabletennis, badminton, tennis)')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory of data')
    parser.add_argument('--output_dir', type=str, default='../data/data_traj', help='Output directory for PKL files')
    args = parser.parse_args()

    history_len = args.history
    future_len = args.future
    sport = args.sport
    
    builder = TrajectoryDatasetBuilder(
        data_root=args.data_root,
        history_len=history_len,
        future_len=future_len,
        sport=sport
    )

    pkl_path = os.path.join(args.output_dir, f'ball_racket_{sport}_h{history_len}_f{future_len}.pkl')
    
    # Create and save dataset
    dataset_dict = builder.build_dataset(save_path=pkl_path)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(dataset_dict, batch_size=16)
    
    # Test the data loader
    for batch in train_loader:
        print("Batch shapes:")
        print(f"History: {batch['history'].shape}")          # Should be (batch_size, 80, 2)
        print(f"Future: {batch['future'].shape}")            # Should be (batch_size, 20, 2)
        print(f"History Racket: {batch['history_rkt'].shape}") # Should be (batch_size, 80, 10)
        print(f"Future Racket: {batch['future_rkt'].shape}")   # Should be (batch_size, 20, 10)
        print(f"Sports in batch: {set(batch['metadata']['sport'])}")
        
        # Check ranges
        print(f"Ball history range: X[{batch['history'][:,:,0].min():.3f}, {batch['history'][:,:,0].max():.3f}], Y[{batch['history'][:,:,1].min():.3f}, {batch['history'][:,:,1].max():.3f}]")
        print(f"Racket history range: [{batch['history_rkt'].min():.3f}, {batch['history_rkt'].max():.3f}]")
        break
    
    # Save and load example
    builder.save_dataset(dataset_dict=dataset_dict, save_path=pkl_path)
    loaded_dataset = builder.load_dataset(pkl_path)