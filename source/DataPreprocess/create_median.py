"""
Create median background frame for each match.

Computes the pixel-wise median across all frames in each round of a match,
then saves as both .npz and .png at the match level.

Output:
    {sport}/all/{match_id}/median.npz   (numpy archive with key 'median')
    {sport}/all/{match_id}/median.png   (visual preview)

Usage:
    python create_median.py --data_root ../data
    python create_median.py --data_root ../data --sports tennis --max_frames 100
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def compute_median_from_frames(frame_dir, max_frames=None, img_format='jpg'):
    """Compute median image from a directory of frames."""
    frame_files = sorted(glob.glob(os.path.join(frame_dir, f'*.{img_format}')))
    if len(frame_files) == 0:
        return None

    if max_frames and len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]

    frames = []
    for ff in frame_files:
        img = cv2.imread(ff)
        if img is not None:
            frames.append(img)

    if len(frames) == 0:
        return None

    median = np.median(np.array(frames), axis=0).astype(np.uint8)
    return median


def main():
    parser = argparse.ArgumentParser(description='Create median background frames')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--sports', nargs='+', default=['badminton', 'tabletennis', 'tennis'])
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to sample for median (None = use all)')
    parser.add_argument('--img_format', type=str, default='jpg')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)

    for sport in args.sports:
        sport_dir = os.path.join(data_root, sport, 'all')
        if not os.path.isdir(sport_dir):
            print(f"Skipping {sport}: no 'all' directory")
            continue

        match_dirs = sorted([d for d in os.listdir(sport_dir)
                            if os.path.isdir(os.path.join(sport_dir, d))])
        print(f"\n[{sport}] Found {len(match_dirs)} matches")

        for match_id in tqdm(match_dirs, desc=f'{sport}'):
            match_dir = os.path.join(sport_dir, match_id)
            npz_path = os.path.join(match_dir, 'median.npz')
            png_path = os.path.join(match_dir, 'median.png')

            if os.path.exists(npz_path):
                print(f"  Skipping {match_id}: median already exists")
                continue

            frame_base = os.path.join(match_dir, 'frame')
            if not os.path.isdir(frame_base):
                print(f"  Skipping {match_id}: no frame directory (run extract_frames.py first)")
                continue

            all_frames = []
            for round_dir in sorted(os.listdir(frame_base)):
                round_path = os.path.join(frame_base, round_dir)
                if os.path.isdir(round_path):
                    frames = sorted(glob.glob(os.path.join(round_path, f'*.{args.img_format}')))
                    all_frames.extend(frames)

            if len(all_frames) == 0:
                print(f"  Skipping {match_id}: no frames found")
                continue

            if args.max_frames and len(all_frames) > args.max_frames:
                indices = np.linspace(0, len(all_frames) - 1, args.max_frames, dtype=int)
                all_frames = [all_frames[i] for i in indices]

            frame_arrays = []
            for ff in all_frames:
                img = cv2.imread(ff)
                if img is not None:
                    frame_arrays.append(img)

            if len(frame_arrays) == 0:
                continue

            median = np.median(np.array(frame_arrays), axis=0).astype(np.uint8)
            np.savez(npz_path, median=median)
            cv2.imwrite(png_path, median)
            print(f"  {match_id}: median computed from {len(frame_arrays)} frames → {npz_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
