"""
One-time final test split evaluation of the frozen final model (models/final_best.pt).
Executed ONLY after model freeze.
Dataset split: test (734 images).
"""

import sys
import os
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
torch.set_num_threads(10)

from ultralytics import YOLO

def main():
    frozen_model_path = PROJECT_ROOT / "models" / "final_best.pt"
    assert frozen_model_path.exists(), f"Frozen model missing at {frozen_model_path}"

    print("=" * 75)
    print("EXECUTING ONE-TIME FINAL TEST EVALUATION OF FROZEN MODEL")
    print(f"Checkpoint: {frozen_model_path}")
    print("Dataset Split: TEST (734 images)")
    print("=" * 75)

    model = YOLO(str(frozen_model_path))

    t0 = time.time()
    val_results = model.val(
        data=str(PROJECT_ROOT / "config" / "data.yaml"),
        split="test",
        imgsz=416,
        batch=8,
        device="cpu",
        project="runs/detect",
        name="final_test_evaluation",
        save_json=False,
        plots=True,
        verbose=True
    )
    total_eval_time = round(time.time() - t0, 2)

    results_dict = val_results.results_dict
    test_map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))
    test_map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0))
    test_precision = float(results_dict.get("metrics/precision(B)", 0.0))
    test_recall = float(results_dict.get("metrics/recall(B)", 0.0))

    # Per-class metrics
    class_names = model.names
    per_class_results = {}
    if hasattr(val_results, "box") and hasattr(val_results.box, "maps"):
        maps50 = val_results.box.maps
        for idx, name in class_names.items():
            per_class_results[name] = round(float(maps50[idx]), 4) if idx < len(maps50) else 0.0

    eval_record = {
        "model_path": "models/final_best.pt",
        "frozen_sha256": "700b816735dbd9734b8e883fe2c1edbaf98ab7ac9be4f70977bbea932de16031",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "test_image_count": 734,
        "test_mAP50": round(test_map50, 4),
        "test_mAP50_95": round(test_map50_95, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "evaluation_time_seconds": total_eval_time,
        "inference_speed_ms": {
            "preprocess": round(val_results.speed.get("preprocess", 0.0), 2),
            "inference": round(val_results.speed.get("inference", 0.0), 2),
            "loss": round(val_results.speed.get("loss", 0.0), 2),
            "postprocess": round(val_results.speed.get("postprocess", 0.0), 2)
        },
        "per_class_mAP50": per_class_results
    }

    out_json = PROJECT_ROOT / "reports" / "final_test_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)

    print(f"\nSaved final test evaluation record to: {out_json}")
    print(f"Test mAP@50:    {test_map50:.4f}")
    print(f"Test mAP@50-95: {test_map50_95:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Evaluation completed in {total_eval_time} seconds.")

if __name__ == "__main__":
    main()
