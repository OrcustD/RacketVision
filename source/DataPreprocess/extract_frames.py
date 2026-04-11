"""
Extract video frames to JPG images.

For each video {sport}/videos/{match_id}_{round_id}.mp4, extracts frames to:
    {sport}/all/{match_id}/frame/{round_id}/0000.jpg, 0001.jpg, ...

Usage:
    python extract_frames.py --data_root ../data
    python extract_frames.py --data_root ../data --sports tennis
"""

import os
import glob
import argparse
import cv2
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, img_format='jpg'):
    """Extract all frames from a video file."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f'{frame_idx:04d}.{img_format}')
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    return frame_idx


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--sports', nargs='+', default=['badminton', 'tabletennis', 'tennis'])
    parser.add_argument('--img_format', type=str, default='jpg')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)

    for sport in args.sports:
        video_dir = os.path.join(data_root, sport, 'videos')
        if not os.path.isdir(video_dir):
            print(f"Skipping {sport}: no videos directory")
            continue

        video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
        print(f"\n[{sport}] Found {len(video_files)} videos")

        for vf in tqdm(video_files, desc=f'{sport}'):
            basename = os.path.splitext(os.path.basename(vf))[0]
            parts = basename.rsplit('_', 1)
            if len(parts) != 2:
                print(f"  Warning: cannot parse {basename}, skipping")
                continue
            match_id, round_id = parts
            output_dir = os.path.join(data_root, sport, 'all', match_id, 'frame', round_id)

            if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
                print(f"  Skipping {basename}: frames already exist ({len(os.listdir(output_dir))} files)")
                continue

            n_frames = extract_frames_from_video(vf, output_dir, args.img_format)
            print(f"  {basename}: extracted {n_frames} frames → {output_dir}")

    print("\nDone!")


if __name__ == '__main__':
    main()
