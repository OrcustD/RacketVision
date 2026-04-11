"""Download pre-trained checkpoints from Hugging Face and place them in the correct directories.

Usage:
    python download_checkpoints.py
    python download_checkpoints.py --module BallTrack
    python download_checkpoints.py --module TrajPred --sport badminton
"""

import argparse
import os
import shutil

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError(
        "huggingface_hub is required. Install via: pip install huggingface_hub"
    )

HF_REPO_ID = "linfeng302/RacketVision-Models"

CHECKPOINTS = {
    "BallTrack": [
        {"filename": "balltrack_best.pth", "dest": "BallTrack/checkpoints/balltrack_best.pth"},
    ],
    "RacketPose": [
        {"filename": "epoch_300.pth", "dest": "RacketPose/checkpoints/epoch_300.pth"},
        {"filename": "best_PCK_epoch_90.pth", "dest": "RacketPose/checkpoints/best_PCK_epoch_90.pth"},
    ],
    "TrajPred": [
        {"filename": "crosslstm_long_badminton.pth", "dest": "TrajPred/checkpoints/crosslstm_long_badminton.pth",
         "sport": "badminton"},
        {"filename": "crosslstm_long_tabletennis.pth", "dest": "TrajPred/checkpoints/crosslstm_long_tabletennis.pth",
         "sport": "tabletennis"},
        {"filename": "crosslstm_short_tennis.pth", "dest": "TrajPred/checkpoints/crosslstm_short_tennis.pth",
         "sport": "tennis"},
    ],
}


def download(module=None, sport=None, root="."):
    for mod_name, files in CHECKPOINTS.items():
        if module and mod_name != module:
            continue
        for entry in files:
            if sport and entry.get("sport") and entry["sport"] != sport:
                continue

            dest = os.path.join(root, entry["dest"])
            if os.path.exists(dest):
                print(f"  [skip] {dest} (already exists)")
                continue

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f"  Downloading {entry['filename']} -> {dest}")
            cached = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                filename=f"checkpoints/{entry['filename']}",
            )
            shutil.copy2(cached, dest)
            print(f"  [done] {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained checkpoints from Hugging Face")
    parser.add_argument("--module", type=str, default=None,
                        choices=["BallTrack", "RacketPose", "TrajPred"],
                        help="Download checkpoints for a specific module only")
    parser.add_argument("--sport", type=str, default=None,
                        choices=["badminton", "tabletennis", "tennis"],
                        help="Download TrajPred checkpoint for a specific sport only")
    parser.add_argument("--root", type=str, default=".",
                        help="Repository root directory (default: current directory)")
    args = parser.parse_args()

    print(f"Downloading checkpoints from {HF_REPO_ID} ...")
    download(module=args.module, sport=args.sport, root=args.root)
    print("All done.")


if __name__ == "__main__":
    main()
