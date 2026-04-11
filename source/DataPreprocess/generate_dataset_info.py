"""
Generate dataset metadata files from raw data:
  - {sport}/info/train.json, val.json, test.json  (per-sport split files)
  - {sport}/info/metainfo.json                     (image shape info)
  - annotations/dataset_info.json                  (global dataset info)

Usage:
    python generate_dataset_info.py --data_root ../data
"""

import os
import json
import glob
import argparse
import cv2


def get_video_info(video_path):
    """Get video metadata: frame count, fps, width, height."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, width, height


def discover_clips(data_root, sports):
    """Discover all clips from the data directory structure.
    
    Expected structure:
        data_root/{sport}/videos/{match_id}_{round_id}.mp4
        data_root/{sport}/all/{match_id}/csv/{round_id}_ball.csv
        data_root/{sport}/all/{match_id}/racket/{round_id}/*.json
    """
    clips = []
    for sport in sports:
        video_dir = os.path.join(data_root, sport, 'videos')
        if not os.path.isdir(video_dir):
            print(f"Warning: video dir not found: {video_dir}")
            continue
        video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
        for vf in video_files:
            basename = os.path.splitext(os.path.basename(vf))[0]
            parts = basename.rsplit('_', 1)
            if len(parts) != 2:
                print(f"Warning: cannot parse video filename: {basename}")
                continue
            match_id, round_id = parts

            csv_path = os.path.join(data_root, sport, 'all', match_id, 'csv', f'{round_id}_ball.csv')
            racket_dir = os.path.join(data_root, sport, 'all', match_id, 'racket', round_id)

            ball_frames = 0
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                ball_frames = int(df['Visibility'].sum()) if 'Visibility' in df.columns else len(df)

            racket_frames = 0
            if os.path.isdir(racket_dir):
                racket_frames = len(glob.glob(os.path.join(racket_dir, '*.json')))

            frame_count, fps, width, height = get_video_info(vf)
            video_length = round(frame_count / fps, 2) if fps > 0 else 0

            clip = {
                'sport': sport,
                'match_id': match_id,
                'round_id': round_id,
                'video_file_path': f'{sport}/videos/{basename}.mp4',
                'frame_files_path': f'{sport}/all/{match_id}/frame/{round_id}/',
                'frame_number': frame_count,
                'video_length': video_length,
                'fps': fps,
                'width': width,
                'height': height,
                'ball_annotated_frames': ball_frames,
                'racket_annotated_frames': racket_frames,
                'median_path': f'{sport}/all/{match_id}/median.npz',
                'median_img_path': f'{sport}/all/{match_id}/median.png',
            }
            clips.append(clip)
    return clips


def generate_split_files(data_root, clips, split_rule):
    """Generate per-sport train/val/test split JSON files.
    
    split_rule: dict mapping match_id pattern to split name.
        e.g. {'match1': 'train', 'match2': 'val_test'}
        'val_test' means the clip appears in both val and test.
    """
    sport_splits = {}
    for i, clip in enumerate(clips):
        sport = clip['sport']
        match_id = clip['match_id']
        round_id = clip['round_id']
        if sport not in sport_splits:
            sport_splits[sport] = {'train': [], 'val': [], 'test': []}

        assigned = False
        for pattern, split_name in split_rule.items():
            if match_id == pattern or match_id.startswith(pattern):
                if split_name == 'val_test':
                    sport_splits[sport]['val'].append([match_id, round_id])
                    sport_splits[sport]['test'].append([match_id, round_id])
                else:
                    sport_splits[sport][split_name].append([match_id, round_id])
                assigned = True
                break
        if not assigned:
            sport_splits[sport]['train'].append([match_id, round_id])

    for sport, splits in sport_splits.items():
        info_dir = os.path.join(data_root, sport, 'info')
        os.makedirs(info_dir, exist_ok=True)
        for split_name, entries in splits.items():
            split_path = os.path.join(info_dir, f'{split_name}.json')
            with open(split_path, 'w') as f:
                json.dump(entries, f, indent=4)
            print(f"  Written {split_path} ({len(entries)} clips)")


def generate_metainfo(data_root, clips):
    """Generate per-sport metainfo.json with image shape."""
    sport_shapes = {}
    for clip in clips:
        sport = clip['sport']
        if sport not in sport_shapes:
            sport_shapes[sport] = (clip['height'], clip['width'], 3)

    for sport, shape in sport_shapes.items():
        info_dir = os.path.join(data_root, sport, 'info')
        os.makedirs(info_dir, exist_ok=True)
        metainfo = {'image_shape': list(shape)}
        metainfo_path = os.path.join(info_dir, 'metainfo.json')
        with open(metainfo_path, 'w') as f:
            json.dump(metainfo, f, indent=4)
        print(f"  Written {metainfo_path}")


def generate_dataset_info(data_root, clips, split_rule):
    """Generate global annotations/dataset_info.json."""
    train_indices, val_indices, test_indices = [], [], []
    sports_set = set()

    for i, clip in enumerate(clips):
        clip['clip_uid'] = i
        clip['clip_id'] = f"{clip['sport']}_{clip['match_id']}_{clip['round_id']}"
        sports_set.add(clip['sport'])

        match_id = clip['match_id']
        for pattern, split_name in split_rule.items():
            if match_id == pattern or match_id.startswith(pattern):
                if split_name == 'val_test':
                    val_indices.append(i)
                    test_indices.append(i)
                elif split_name == 'train':
                    train_indices.append(i)
                elif split_name == 'val':
                    val_indices.append(i)
                elif split_name == 'test':
                    test_indices.append(i)
                break

    dataset_info = {
        'dataset_name': 'UnifiedRacketSports',
        'dataset_version': '1.0',
        'description': 'Toy dataset for ball tracking, racket pose estimation, and trajectory prediction',
        'splits': {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
        },
        'sports': sorted(sports_set),
        'clips': clips,
        'data_statistics': {
            'total_clips': len(clips),
            'total_sports': len(sports_set),
        }
    }

    anno_dir = os.path.join(data_root, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    info_path = os.path.join(anno_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"  Written {info_path} ({len(clips)} clips)")


def main():
    parser = argparse.ArgumentParser(description='Generate dataset metadata files')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root directory of the dataset')
    parser.add_argument('--sports', nargs='+', default=['badminton', 'tabletennis', 'tennis'],
                        help='List of sports to process')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print(f"Data root: {data_root}")
    print(f"Sports: {args.sports}")

    # Split rule: match1 → train, match2 → val & test
    split_rule = {
        'match1': 'train',
        'match2': 'val_test',
    }

    print("\n[1/4] Discovering clips...")
    clips = discover_clips(data_root, args.sports)
    print(f"  Found {len(clips)} clips")
    for c in clips:
        print(f"    {c['clip_id'] if 'clip_id' in c else c['sport'] + '_' + c['match_id'] + '_' + c['round_id']}: "
              f"{c['frame_number']} frames, {c['ball_annotated_frames']} ball GT, "
              f"{c['racket_annotated_frames']} racket GT")

    print("\n[2/4] Generating per-sport split files...")
    generate_split_files(data_root, clips, split_rule)

    print("\n[3/4] Generating per-sport metainfo.json...")
    generate_metainfo(data_root, clips)

    print("\n[4/4] Generating global dataset_info.json...")
    generate_dataset_info(data_root, clips, split_rule)

    print("\nDone!")


if __name__ == '__main__':
    main()
