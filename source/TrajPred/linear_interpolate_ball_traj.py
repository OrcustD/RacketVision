import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def interpolate_ball_trajectory(csv_file, threshold=5):
    """
    Interpolate empty frames in ball trajectory data.
    
    Args:
        csv_file: Path to the CSV file
        threshold: Maximum number of continuous empty frames to interpolate
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Identify empty frames (where X, Y, Visibility are all 0)
    empty_frames = (df['X'] == 0) & (df['Y'] == 0) & (df['Visibility'] == 0)
    
    # Find first and last non-empty frames
    non_empty_indices = df[~empty_frames].index
    if len(non_empty_indices) == 0:
        print("No non-empty frames found!")
        return df
    
    first_non_empty = non_empty_indices[0]
    last_non_empty = non_empty_indices[-1]
    
    print(f"First non-empty frame: {first_non_empty}")
    print(f"Last non-empty frame: {last_non_empty}")
    
    # Create a copy of the dataframe for modifications
    result_df = df.copy()
    
    # Find continuous empty frame segments within the range
    i = first_non_empty
    while i <= last_non_empty:
        if empty_frames[i]:
            # Found start of empty segment
            segment_start = i
            segment_end = i
            
            # Find end of continuous empty segment
            while segment_end <= last_non_empty and empty_frames[segment_end]:
                segment_end += 1
            segment_end -= 1  # Last empty frame in segment
            
            segment_length = segment_end - segment_start + 1
            
            print(f"Empty segment: frames {segment_start}-{segment_end}, length: {segment_length}")
            
            # Only interpolate if segment length < threshold
            if segment_length < threshold:
                # Find previous and next non-empty frames
                prev_frame = segment_start - 1
                next_frame = segment_end + 1
                
                # Make sure we have valid boundary frames
                if prev_frame >= 0 and next_frame < len(df) and not empty_frames[prev_frame] and not empty_frames[next_frame]:
                    print(f"Interpolating between frame {prev_frame} and {next_frame}")
                    
                    # Linear interpolation for X, Y coordinates
                    x_start, y_start = df.loc[prev_frame, 'X'], df.loc[prev_frame, 'Y']
                    x_end, y_end = df.loc[next_frame, 'X'], df.loc[next_frame, 'Y']
                    
                    # Interpolate confidence as average of boundary frames
                    conf_start = df.loc[prev_frame, 'Confidence']
                    conf_end = df.loc[next_frame, 'Confidence']
                    
                    for j, frame_idx in enumerate(range(segment_start, segment_end + 1)):
                        # Linear interpolation factor
                        t = (j + 1) / (segment_length + 1)
                        
                        # Interpolate coordinates
                        result_df.loc[frame_idx, 'X'] = int(x_start + t * (x_end - x_start))
                        result_df.loc[frame_idx, 'Y'] = int(y_start + t * (y_end - y_start))
                        result_df.loc[frame_idx, 'Visibility'] = 1
                        result_df.loc[frame_idx, 'Confidence'] = conf_start + t * (conf_end - conf_start)
                else:
                    print(f"Cannot interpolate segment {segment_start}-{segment_end}: invalid boundary frames")
            else:
                print(f"Skipping interpolation for segment {segment_start}-{segment_end}: length {segment_length} >= threshold {threshold}")
            
            i = segment_end + 1
        else:
            i += 1
    
    return result_df

def run(input_file, output_file, threshold=5):
    # Process the results.csv file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Processing ball trajectory interpolation...")
    interpolated_df = interpolate_ball_trajectory(input_file, threshold=threshold)
    
    # Save the result
    interpolated_df.to_csv(output_file, index=False)
    print(f"Interpolated data saved to: {output_file}")
    
    # Show some statistics
    original_df = pd.read_csv(input_file)
    empty_original = ((original_df['X'] == 0) & (original_df['Y'] == 0) & (original_df['Visibility'] == 0)).sum()
    empty_interpolated = ((interpolated_df['X'] == 0) & (interpolated_df['Y'] == 0) & (interpolated_df['Visibility'] == 0)).sum()
    
    print(f"\nStatistics:")
    print(f"Original empty frames: {empty_original}")
    print(f"Remaining empty frames: {empty_interpolated}")
    print(f"Interpolated frames: {empty_original - empty_interpolated}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--sport', type=str, default=None,
                        help='Process single sport. Default: all sports.')
    parser.add_argument('--threshold', type=int, default=5)
    args = parser.parse_args()

    sports = [args.sport] if args.sport else ['tabletennis', 'badminton', 'tennis']
    for sport in sports:
        pattern = os.path.join(args.data_root, sport, 'pred_ball', 'match*', '*', 'results.csv')
        csv_files = glob.glob(pattern)
        for input_file in tqdm(csv_files, desc=f"Processing {sport}"):
            output_file = input_file.replace('pred_ball', 'interp_ball')
            run(input_file, output_file, threshold=args.threshold)