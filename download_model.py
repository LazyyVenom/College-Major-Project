"""Download pretrained seatbelt detection model from Kaggle.

Source: https://www.kaggle.com/datasets/sachinmlwala/seatbelt3
Model: YOLOv5 trained on seatbelt dataset
Classes: 0 = "No Seatbelt", 1 = "Seatbelt Worn"

Prerequisites:
    uv add kaggle
    Then set up your Kaggle API credentials:
    - Go to kaggle.com -> Account -> Create New Token
    - Place kaggle.json in ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys
import zipfile
import shutil
import glob


DATASET = "sachinmlwala/seatbelt3"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
TARGET_PATH = os.path.join(MODELS_DIR, "seatbelt.pt")


def main():
    if os.path.exists(TARGET_PATH):
        print(f"Model already exists at {TARGET_PATH}")
        return

    # Check kaggle CLI
    if not shutil.which("kaggle"):
        print("Kaggle CLI not found. Install it with: uv add kaggle")
        print("Then set up credentials: https://www.kaggle.com/docs/api")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp_dir = os.path.join(MODELS_DIR, "_tmp_download")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        print(f"Downloading dataset: {DATASET} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", tmp_dir],
            check=True,
        )

        # Extract zip
        zip_files = glob.glob(os.path.join(tmp_dir, "*.zip"))
        if zip_files:
            print("Extracting...")
            with zipfile.ZipFile(zip_files[0], "r") as zf:
                zf.extractall(tmp_dir)

        # Find best.pt in extracted files
        best_pt = None
        for root, dirs, files in os.walk(tmp_dir):
            for f in files:
                if f == "best.pt":
                    best_pt = os.path.join(root, f)
                    break
            if best_pt:
                break

        if best_pt is None:
            print("Error: best.pt not found in downloaded dataset.")
            print("Contents:")
            for root, dirs, files in os.walk(tmp_dir):
                for f in files:
                    print(f"  {os.path.join(root, f)}")
            sys.exit(1)

        shutil.move(best_pt, TARGET_PATH)
        print(f"Model saved to {TARGET_PATH}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
