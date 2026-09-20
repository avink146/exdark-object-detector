"""
Systematic Controlled Experiment Runner for ExDark Object Detection.
Manages EXP-001 through EXP-004 execution, tracking, and evaluation.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import train_low_light_detector
from src.evaluation.evaluator import evaluate_model

EXPERIMENT_TRACKER_PATH = PROJECT_ROOT / "reports" / "experiments_summary.json"

def load_experiments_summary() -> Dict[str, Any]:
    if EXPERIMENT_TRACKER_PATH.exists():
        with open(EXPERIMENT_TRACKER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "benchmark": "ExDark (Exclusively Dark)",
        "hardware": "AMD Ryzen 5 7530U CPU (Multi-threaded)",
        "experiments": []
    }

def save_experiment_record(record: Dict[str, Any]):
    summary = load_experiments_summary()
    # Replace existing experiment with same id or append
    existing_idx = next((i for i, e in enumerate(summary["experiments"]) if e["id"] == record["id"]), None)
    if existing_idx is not None:
        summary["experiments"][existing_idx] = record
    else:
        summary["experiments"].append(record)
    
    EXPERIMENT_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_TRACKER_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment record: {record['id']} to {EXPERIMENT_TRACKER_PATH}")

def run_experiment_2_extended():
    """EXP-002: Extended YOLOv8n fine-tuning with cosine LR scheduler."""
    print("=" * 70)
    print("STARTING EXP-002: Extended YOLOv8n Fine-Tuning")
    print("=" * 70)
    t0 = time.time()
    
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(PROJECT_ROOT / "config" / "data.yaml"),
        epochs=6,
        imgsz=416,
        batch=8,
        fraction=0.15,
        device="cpu",
        project="runs/detect",
        name="exp002_extended",
        cos_lr=True,
        lr0=0.005,
        lrf=0.01,
        patience=4,
        workers=0,
        verbose=True,
        hsv_h=0.015,
        hsv_s=0.60,
        hsv_v=0.40,
        mosaic=0.50
    )
    total_time = time.time() - t0
    
    # Save checkpoint
    weights_dir = Path(model.trainer.save_dir) / "weights"
    out_pt = PROJECT_ROOT / "models" / "exp002_best.pt"
    if (weights_dir / "best.pt").exists():
        import shutil
        shutil.copy2(weights_dir / "best.pt", out_pt)
        print(f"Copied EXP-002 best checkpoint to {out_pt}")

    # Validate on full val split
    val_metrics = model.val(data=str(PROJECT_ROOT / "config" / "data.yaml"), split="val", imgsz=416, batch=8, device="cpu")
    val_map50 = float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    val_map50_95 = float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
    val_prec = float(val_metrics.results_dict.get("metrics/precision(B)", 0.0))
    val_rec = float(val_metrics.results_dict.get("metrics/recall(B)", 0.0))

    record = {
        "id": "EXP-002",
        "name": "Extended YOLOv8n (Cosine LR)",
        "model": "YOLOv8n",
        "imgsz": 416,
        "batch": 8,
        "epochs": 6,
        "fraction": 0.15,
        "augmentation": "hsv_v=0.40, hsv_s=0.60, mosaic=0.50, cos_lr=True",
        "val_mAP50": round(val_map50, 4),
        "val_mAP50_95": round(val_map50_95, 4),
        "val_precision": round(val_prec, 4),
        "val_recall": round(val_rec, 4),
        "train_time_sec": round(total_time, 1),
        "checkpoint": "models/exp002_best.pt",
        "status": "COMPLETED"
    }
    save_experiment_record(record)
    return record

def run_experiment_3_augmentation():
    """EXP-003: Targeted Low-Light Augmentations (Contrast, wider HSV, tuned mosaic)."""
    print("=" * 70)
    print("STARTING EXP-003: Targeted Low-Light Augmentation")
    print("=" * 70)
    t0 = time.time()
    
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(PROJECT_ROOT / "config" / "data.yaml"),
        epochs=6,
        imgsz=416,
        batch=8,
        fraction=0.15,
        device="cpu",
        project="runs/detect",
        name="exp003_augmentation",
        cos_lr=True,
        lr0=0.005,
        lrf=0.01,
        patience=4,
        workers=0,
        verbose=True,
        # Targeted low-light augmentations:
        hsv_h=0.02,
        hsv_s=0.70,        # Higher saturation variation for color casts
        hsv_v=0.50,        # Wider value variation for extreme dark-to-light
        mosaic=0.30,       # Reduced mosaic to preserve natural continuous shadows
        mixup=0.15,        # Ghost silhouettes
        degrees=8.0
    )
    total_time = time.time() - t0
    
    weights_dir = Path(model.trainer.save_dir) / "weights"
    out_pt = PROJECT_ROOT / "models" / "exp003_best.pt"
    if (weights_dir / "best.pt").exists():
        import shutil
        shutil.copy2(weights_dir / "best.pt", out_pt)
        print(f"Copied EXP-003 best checkpoint to {out_pt}")

    val_metrics = model.val(data=str(PROJECT_ROOT / "config" / "data.yaml"), split="val", imgsz=416, batch=8, device="cpu")
    val_map50 = float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    val_map50_95 = float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
    val_prec = float(val_metrics.results_dict.get("metrics/precision(B)", 0.0))
    val_rec = float(val_metrics.results_dict.get("metrics/recall(B)", 0.0))

    record = {
        "id": "EXP-003",
        "name": "Targeted Low-Light Augmentation",
        "model": "YOLOv8n",
        "imgsz": 416,
        "batch": 8,
        "epochs": 6,
        "fraction": 0.15,
        "augmentation": "hsv_v=0.50, hsv_s=0.70, mosaic=0.30, mixup=0.15",
        "val_mAP50": round(val_map50, 4),
        "val_mAP50_95": round(val_map50_95, 4),
        "val_precision": round(val_prec, 4),
        "val_recall": round(val_rec, 4),
        "train_time_sec": round(total_time, 1),
        "checkpoint": "models/exp003_best.pt",
        "status": "COMPLETED"
    }
    save_experiment_record(record)
    return record

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="exp002", choices=["exp002", "exp003"])
    args = parser.parse_args()
    if args.exp == "exp002":
        run_experiment_2_extended()
    elif args.exp == "exp003":
        run_experiment_3_augmentation()
