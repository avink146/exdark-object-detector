"""
Dataset preparation script for ExDark.
Extracts archive.zip to dataset/raw/ExDark (non-destructively)
and prepares dataset/processed/ExDark with verified configurations.
"""
import os
import zipfile
import shutil
from pathlib import Path

def prepare_exdark(zip_path: str = "archive.zip", 
                   raw_dir: str = "dataset/raw/ExDark", 
                   processed_dir: str = "dataset/processed/ExDark"):
    zip_p = Path(zip_path).resolve()
    raw_p = Path(raw_dir).resolve()
    proc_p = Path(processed_dir).resolve()

    print(f"Checking archive: {zip_p}")
    if not zip_p.exists():
        raise FileNotFoundError(f"Archive not found: {zip_p}")

    # Extract to raw directory if not already extracted
    raw_p.mkdir(parents=True, exist_ok=True)
    if not (raw_p / "train" / "images").exists():
        print(f"Extracting {zip_p.name} to {raw_p}...")
        with zipfile.ZipFile(zip_p, 'r') as z:
            z.extractall(raw_p)
        print("Extraction complete.")
    else:
        print(f"Raw dataset already present in {raw_p}.")

    # Set up processed directory
    proc_p.mkdir(parents=True, exist_ok=True)
    
    # Check if raw files exist
    for split in ['train', 'valid', 'test']:
        split_raw_img = raw_p / split / "images"
        split_raw_lbl = raw_p / split / "labels"
        split_proc_img = proc_p / split / "images"
        split_proc_lbl = proc_p / split / "labels"

        split_proc_img.mkdir(parents=True, exist_ok=True)
        split_proc_lbl.mkdir(parents=True, exist_ok=True)

        existing_proc = list(split_proc_img.glob("*.*"))
        if len(existing_proc) == 0:
            print(f"Populating {split} in processed directory...")
            for img in split_raw_img.glob("*.*"):
                shutil.copy2(img, split_proc_img / img.name)
            for lbl in split_raw_lbl.glob("*.txt"):
                shutil.copy2(lbl, split_proc_lbl / lbl.name)
            print(f"Done populating {split}.")

    classes = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Cat', 'Cup', 'Motorbike', 'People', 'Table', 'car', 'chair', 'dog']
    yaml_lines = [
        f"path: '{proc_p.as_posix()}'",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(classes)}",
        f"names: {classes}\n"
    ]
    yaml_content = "\n".join(yaml_lines)

    yaml_out_proc = proc_p / "data.yaml"
    with open(yaml_out_proc, 'w') as f:
        f.write(yaml_content)
    print(f"Saved dataset yaml to {yaml_out_proc}")

    config_dir = Path("config").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_out_cfg = config_dir / "data.yaml"
    with open(yaml_out_cfg, 'w') as f:
        f.write(yaml_content)
    print(f"Saved dataset yaml to {yaml_out_cfg}")

if __name__ == "__main__":
    prepare_exdark()
