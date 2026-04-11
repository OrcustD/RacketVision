#!/usr/bin/env python3
"""
Script to merge ground truth racket pose annotations with prediction results.
Replaces predicted bbox and pose with ground truth annotations for corresponding frames.

Usage:
    # Process all sports, all matches, all rounds
    python merge_gt_with_predictions.py --all
    
    # Process specific sport, all matches, all rounds
    python merge_gt_with_predictions.py --sport tabletennis
    
    # Process specific sport and match, all rounds
    python merge_gt_with_predictions.py --sport tabletennis --match_id match1
    
    # Process specific sport, match and round (original behavior)
    python merge_gt_with_predictions.py --sport tabletennis --match_id match1 --round_id 000

The script will:
1. Load prediction results from: data/{sport}/pred_racket/{match_id}/{round_id}/result.json
2. Find corresponding clip_uid from dataset_info.json
3. Load ground truth annotations from: data/annotations/clip_annot/{clip_uid}.json
4. Replace prediction bbox and pose with ground truth for annotated frames
5. Save merged results to: data/{sport}/merged_racket/{match_id}/{round_id}/result.json
"""

import json
import argparse
import os
import copy
from pathlib import Path
import glob

def load_dataset_info(dataset_info_path):
    """Load dataset info and create mapping from (sport, match_id, round_id) to clip_uid"""
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    mapping = {}
    for clip in dataset_info['clips']:
        key = (clip['sport'], clip['match_id'], clip['round_id'])
        mapping[key] = clip['clip_uid']
    
    return mapping

def discover_prediction_files(data_root, sport=None, match_id=None, round_id=None):
    """Discover all prediction files that match the given criteria"""
    pred_files = []
    
    # Build the search pattern
    sports = [sport] if sport else ['tabletennis', 'tennis', 'badminton']
    
    for s in sports:
        sport_path = data_root / s / 'pred_racket'
        if not sport_path.exists():
            print(f"Warning: Sport directory not found: {sport_path}")
            continue
            
        if match_id:
            match_paths = [sport_path / match_id] if (sport_path / match_id).exists() else []
        else:
            match_paths = [p for p in sport_path.iterdir() if p.is_dir()]
        
        for match_path in match_paths:
            match_name = match_path.name
            
            if round_id:
                round_paths = [match_path / round_id] if (match_path / round_id).exists() else []
            else:
                round_paths = [p for p in match_path.iterdir() if p.is_dir()]
            
            for round_path in round_paths:
                round_name = round_path.name
                result_file = round_path / 'result.json'
                
                if result_file.exists():
                    pred_files.append((s, match_name, round_name, result_file))
                else:
                    print(f"Warning: Result file not found: {result_file}")
    
    return pred_files

def load_predictions(pred_path):
    """Load prediction results"""
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    return predictions

def load_ground_truth(gt_path):
    """Load ground truth annotations"""
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    return gt_data

def convert_gt_format(gt_racket_data):
    """Convert ground truth format to prediction format"""
    converted_data = []
    
    for racket in gt_racket_data:
        if not racket:  # Skip empty racket data
            continue
            
        # Extract bbox from bbox_xywh format [x, y, w, h] to [[x1, y1, x2, y2]]
        bbox_xywh = racket['bbox_xywh']
        x, y, w, h = bbox_xywh
        bbox = [[x, y, x + w, y + h]]
        
        # Extract keypoints - format: [[x, y, visibility], ...]
        keypoints = []
        keypoint_scores = []
        
        for kp in racket['keypoints']:
            x_kp, y_kp, vis = kp
            if vis > 0:  # Only include visible keypoints
                keypoints.append([x_kp, y_kp])
                keypoint_scores.append(1.0)  # Use confidence 1.0 for ground truth
            else:
                keypoints.append([0.0, 0.0])  # Use [0, 0] for invisible keypoints
                keypoint_scores.append(0.0)
        
        converted_racket = {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "bbox": bbox,
            "bbox_score": 1.0  # Use confidence 1.0 for ground truth
        }
        
        converted_data.append(converted_racket)
    
    return converted_data

def merge_predictions_with_gt(predictions, gt_data):
    """Merge predictions with ground truth data"""
    merged_results = copy.deepcopy(predictions)
    
    # Get annotated frames with racket data
    racket_frames = gt_data.get('racket_visible_frames', [])
    annotations = gt_data.get('annotation', {})
    
    replaced_count = 0
    for frame_idx in racket_frames:
        # Convert to 0-based indexing (GT uses 1-based frame numbers)
        frame_key = str(frame_idx)
        
        if frame_key in annotations:
            frame_annotation = annotations[frame_key]
            
            if 'racket' in frame_annotation and frame_annotation['racket']:
                # Convert ground truth format to prediction format
                gt_racket_data = convert_gt_format(frame_annotation['racket'])
                
                # Ensure we have enough frames in predictions
                while len(merged_results) <= frame_idx:
                    merged_results.append([])
                
                # Replace prediction with ground truth
                merged_results[frame_idx] = gt_racket_data
                replaced_count += 1
    
    return merged_results, replaced_count

def save_merged_results(merged_results, output_path):
    """Save merged results to file"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged_results, f, indent=2)

def process_single_file(sport, match_id, round_id, pred_path, mapping, data_root):
    """Process a single prediction file"""
    # Find clip_uid for the given sport, match_id, round_id
    key = (sport, match_id, round_id)
    if key not in mapping:
        print(f"  Warning: Could not find clip_uid for {key}")
        return False
    
    clip_uid = mapping[key]
    
    # Define ground truth path
    gt_path = data_root / 'annotations' / 'clip_annot' / f'{clip_uid}.json'
    
    if not gt_path.exists():
        print(f"  Warning: Ground truth file not found: {gt_path}")
        return False
    
    # Load data
    try:
        predictions = load_predictions(pred_path)
        gt_data = load_ground_truth(gt_path)
    except Exception as e:
        print(f"  Error loading data: {e}")
        return False
    
    # Merge predictions with ground truth
    merged_results, replaced_count = merge_predictions_with_gt(predictions, gt_data)
    
    # Save merged results
    output_path = data_root / sport / 'merged_racket' / match_id / round_id / 'result.json'
    try:
        save_merged_results(merged_results, output_path)
        print(f"  ✓ Processed {sport}/{match_id}/{round_id} - replaced {replaced_count} frames")
        return True
    except Exception as e:
        print(f"  Error saving results: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Merge ground truth with prediction results')
    parser.add_argument('--sport', choices=['tabletennis', 'tennis', 'badminton'],
                        help='Sport type (if not specified, process all sports)')
    parser.add_argument('--match_id', help='Match ID (e.g., match1, if not specified, process all matches)')
    parser.add_argument('--round_id', help='Round ID (e.g., 000, if not specified, process all rounds)')
    parser.add_argument('--data_root', default='data', help='Data root directory')
    parser.add_argument('--all', action='store_true', help='Process all sports, matches, and rounds')
    
    args = parser.parse_args()
    
    # Define paths
    data_root = Path(args.data_root)
    dataset_info_path = data_root / 'annotations' / 'dataset_info.json'
    
    # Load dataset info to get clip_uid mapping
    print("Loading dataset info...")
    try:
        mapping = load_dataset_info(dataset_info_path)
        print(f"Loaded mapping for {len(mapping)} clips")
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return
    
    # Discover prediction files
    print("Discovering prediction files...")
    pred_files = discover_prediction_files(data_root, args.sport, args.match_id, args.round_id)
    
    if not pred_files:
        print("No prediction files found matching the criteria")
        return
    
    print(f"Found {len(pred_files)} prediction files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for sport, match_id, round_id, pred_path in pred_files:
        print(f"Processing {sport}/{match_id}/{round_id}...")
        
        if process_single_file(sport, match_id, round_id, pred_path, mapping, data_root):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(pred_files)}")

if __name__ == "__main__":
    main() 