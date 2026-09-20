"""
Dataset Inspector and Exploratory Data Analysis for ExDark Object Detection.
Performs thorough dataset verification, data integrity validation, and statistical analysis.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Cat', 'Cup', 
    'Motorbike', 'People', 'Table', 'car', 'chair', 'dog'
]

def inspect_dataset(data_dir: str = "dataset/processed/ExDark", output_report: str = "reports/eda_summary.json"):
    data_path = Path(data_dir).resolve()
    report_path = Path(output_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_name": "ExDark (Exclusively Dark)",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "splits": {}
    }

    total_images_all = 0
    total_objects_all = 0
    class_totals_all = defaultdict(int)

    for split in ["train", "valid", "test"]:
        img_dir = data_path / split / "images"
        lbl_dir = data_path / split / "labels"

        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist yet.")
            continue

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        labels = sorted(list(lbl_dir.glob("*.txt")))

        split_class_counts = defaultdict(int)
        objects_per_img = []
        empty_labels_count = 0
        corrupt_labels_count = 0
        missing_labels_count = 0

        for img in images:
            lbl_file = lbl_dir / f"{img.stem}.txt"
            if not lbl_file.exists():
                missing_labels_count += 1
                objects_per_img.append(0)
                continue

            with open(lbl_file, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                empty_labels_count += 1
                objects_per_img.append(0)
                continue

            valid_objs = 0
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        # Verify coordinate bounds
                        if 0 <= xc <= 1.0 and 0 <= yc <= 1.0 and 0 <= w <= 1.0 and 0 <= h <= 1.0 and 0 <= cls_id < len(CLASS_NAMES):
                            cname = CLASS_NAMES[cls_id]
                            split_class_counts[cname] += 1
                            class_totals_all[cname] += 1
                            valid_objs += 1
                        else:
                            corrupt_labels_count += 1
                    except ValueError:
                        corrupt_labels_count += 1
                else:
                    corrupt_labels_count += 1
            objects_per_img.append(valid_objs)

        num_imgs = len(images)
        total_objs = sum(split_class_counts.values())
        avg_objs = round(total_objs / num_imgs, 2) if num_imgs > 0 else 0

        summary["splits"][split] = {
            "num_images": num_imgs,
            "num_label_files": len(labels),
            "total_objects": total_objs,
            "avg_objects_per_image": avg_objs,
            "empty_labels": empty_labels_count,
            "missing_labels": missing_labels_count,
            "corrupt_annotations": corrupt_labels_count,
            "class_distribution": dict(split_class_counts)
        }

        total_images_all += num_imgs
        total_objects_all += total_objs

    summary["total_images"] = total_images_all
    summary["total_objects"] = total_objects_all
    summary["total_class_distribution"] = dict(class_totals_all)

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Successfully generated inspection report: {report_path}")
    print(f"Total images analyzed: {total_images_all}")
    print(f"Total objects analyzed: {total_objects_all}")
    return summary

if __name__ == "__main__":
    inspect_dataset()
