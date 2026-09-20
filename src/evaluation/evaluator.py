"""
Evaluation and error analysis engine for Low-Light Object Detection.
Computes COCO mAP@50, mAP@50-95, per-class metrics, and analyzes failure modes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

CLASS_NAMES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Cat', 'Cup', 
    'Motorbike', 'People', 'Table', 'car', 'chair', 'dog'
]

def evaluate_model(
    model_path: str = "models/best.pt",
    data_yaml: str = "config/data.yaml",
    split: str = "test",
    imgsz: int = 416,
    batch: int = 4,
    device: str = "cpu",
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Evaluates fine-tuned model checkpoint on test or validation split.
    """
    from ultralytics import YOLO

    m_path = Path(model_path).resolve()
    if not m_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {m_path}")

    print("=" * 70)
    print(f"EVALUATING MODEL ON {split.upper()} SPLIT")
    print(f"Checkpoint: {m_path}")
    print(f"Data:       {data_yaml}")
    print(f"Resolution: {imgsz}x{imgsz}")
    print("=" * 70)

    model = YOLO(str(m_path))

    metrics = model.val(
        data=str(Path(data_yaml).resolve()),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=True,
        plots=True
    )

    # Extract COCO metrics
    res_dict = metrics.results_dict
    map50 = float(res_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(res_dict.get("metrics/mAP50-95(B)", 0.0))
    precision = float(res_dict.get("metrics/precision(B)", 0.0))
    recall = float(res_dict.get("metrics/recall(B)", 0.0))

    # Per-class AP@50
    per_class_ap50 = {}
    per_class_ap = {}
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        for idx, ap in enumerate(metrics.box.ap50):
            cname = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            per_class_ap50[cname] = round(float(ap), 4)
    if hasattr(metrics.box, 'ap') and metrics.box.ap is not None:
        for idx, ap in enumerate(metrics.box.ap):
            cname = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            per_class_ap[cname] = round(float(ap), 4)

    evaluation_summary = {
        "model_checkpoint": str(m_path.name),
        "split_evaluated": split,
        "primary_metric_mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "overall_precision": round(precision, 4),
        "overall_recall": round(recall, 4),
        "per_class_mAP50": per_class_ap50,
        "per_class_mAP50_95": per_class_ap
    }

    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / f"{split}_metrics.json"

    with open(out_file, "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    print(f"\nSaved evaluation metrics to: {out_file}")
    print(f"Primary Metric (mAP@50): {map50:.4f}")
    print(f"Secondary Metric (mAP@50-95): {map50_95:.4f}")
    print(f"Overall Precision: {precision:.4f} | Overall Recall: {recall:.4f}")

    return evaluation_summary

if __name__ == "__main__":
    evaluate_model()
