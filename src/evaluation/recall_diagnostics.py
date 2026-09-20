"""
Recall & Failure Mode Diagnostics Engine.
Investigates the root causes of low recall under low-light conditions:
1. Confidence threshold sensitivity sweep (0.05 to 0.50)
2. Object scale vs. false negative distribution (Small vs Medium vs Large)
3. Scene luminance vs. detection probability
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_confidence_threshold_sweep(
    model_path: str = "models/baseline_best.pt",
    data_yaml: str = "config/data.yaml",
    split: str = "val",
    imgsz: int = 416,
    thresholds: list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    output_dir: str = "reports"
):
    from ultralytics import YOLO

    m_path = Path(model_path).resolve()
    assert m_path.exists(), f"Model not found: {m_path}"
    model = YOLO(str(m_path))

    sweep_results = []
    print(f"--- Running Confidence Threshold Sweep on {split.upper()} split ---")

    for conf in thresholds:
        t0 = time.time()
        metrics = model.val(
            data=str(Path(data_yaml).resolve()),
            split=split,
            imgsz=imgsz,
            batch=8,
            conf=conf,
            device="cpu",
            verbose=False,
            plots=False
        )
        elapsed = time.time() - t0
        
        prec = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
        rec = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
        map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
        
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        sweep_results.append({
            "Confidence Threshold": conf,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "mAP@50": round(map50, 4),
            "mAP@50-95": round(map50_95, 4),
            "Latency (s)": round(elapsed, 1)
        })
        print(f"  Conf: {conf:4.2f} | Precision: {prec:6.4f} | Recall: {rec:6.4f} | F1: {f1:6.4f} | mAP@50: {map50:6.4f}")

    sweep_df = pd.DataFrame(sweep_results)
    
    # Save plot
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(9, 5))
    plt.plot(sweep_df["Confidence Threshold"], sweep_df["Precision"], "o-", label="Precision", color="#38bdf8", linewidth=2)
    plt.plot(sweep_df["Confidence Threshold"], sweep_df["Recall"], "s-", label="Recall", color="#f87171", linewidth=2)
    plt.plot(sweep_df["Confidence Threshold"], sweep_df["F1 Score"], "^-", label="F1 Score", color="#34d399", linewidth=2)
    plt.plot(sweep_df["Confidence Threshold"], sweep_df["mAP@50"], "d--", label="mAP@50", color="#fbbf24", linewidth=2)
    plt.xlabel("Inference Confidence Threshold")
    plt.ylabel("Metric Score")
    plt.title("Precision-Recall-F1 Tradeoff Across Confidence Thresholds")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "confidence_threshold_sweep.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved threshold sweep plot to: {plot_path}")

    # Save json
    json_path = out_dir / "confidence_sweep_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Saved sweep metrics to: {json_path}")

    return sweep_df

if __name__ == "__main__":
    run_confidence_threshold_sweep()
