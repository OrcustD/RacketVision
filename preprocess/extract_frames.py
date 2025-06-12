import os
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def extract_frames_ffmpeg(video_path, output_dir):
    """
    Extract frames from a video file using ffmpeg
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the frames
    """
    # Create output directory
    create_directory(output_dir)
    
    # Construct ffmpeg command
    # -i: input file
    # -q:v: quality of output (2 is high quality, range is 2-31, lower is better)
    # -frame_pts: add presentation timestamp to frame filename
    # -start_number: start frame number
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',
        '-frame_pts', '1',
        '-start_number', '0',
        os.path.join(output_dir, '%04d.jpg')
    ]
    
    try:
        # Run ffmpeg command
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully extracted frames from {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_path}")
        print(f"Error message: {e.stderr.decode()}")
    except Exception as e:
        print(f"Unexpected error processing {video_path}: {str(e)}")

def process_videos(dataset_info_path):
    """
    Process all videos listed in the dataset_info.json file
    
    Args:
        dataset_info_path (str): Path to the dataset_info.json file
    """
    # Load dataset info
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Process each clip
    for clip_info in tqdm(dataset_info['clips'][:1], desc="Processing clips", unit="clip"):
        # Get video path
        video_path = clip_info['video_file_path']
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue
        
        # Get frame output path
        frame_path = clip_info['frame_files_path']
        
        # Extract frames
        print(f"\nProcessing video: {video_path}")
        print(f"Output directory: {frame_path}")
        extract_frames_ffmpeg(video_path, frame_path)
        # Verify number of extracted frames matches dataset info
        expected_frames = clip_info['frame_number']
        extracted_frames = len([f for f in os.listdir(frame_path) if f.endswith('.jpg')])
        
        if extracted_frames != expected_frames:
            print(f"Warning: Number of extracted frames ({extracted_frames}) does not match expected ({expected_frames})")
            print(f"Video: {video_path}")

def main():
    # Path to dataset_info.json
    dataset_info_path = 'annotations/dataset_info.json'
    # Process all videos
    process_videos(dataset_info_path)

if __name__ == '__main__':
    main()