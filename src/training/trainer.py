"""
Training manager with live progress callbacks, memory monitoring, and checkpoint management.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

class TrainingProgressCallback:
    """Ultralytics callback to display live formatted progress and metrics."""
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.epoch_start_time = 0
        self.train_start_time = time.time()

    def on_train_epoch_start(self, trainer):
        self.epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {trainer.epoch + 1}/{self.total_epochs}")
        print(f"{'='*70}")

    def on_train_epoch_end(self, trainer):
        elapsed_epoch = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.train_start_time
        current_ep = trainer.epoch + 1
        avg_epoch_time = total_elapsed / current_ep
        remaining_epochs = self.total_epochs - current_ep
        eta_seconds = remaining_epochs * avg_epoch_time

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        # Losses
        loss_items = getattr(trainer, 'loss_items', None)
        loss_str = ""
        if loss_items is not None:
            try:
                if isinstance(loss_items, dict):
                    loss_val = sum(float(v) for v in loss_items.values())
                elif hasattr(loss_items, 'sum'):
                    loss_val = float(loss_items.sum())
                elif isinstance(loss_items, (list, tuple)):
                    loss_val = sum(float(v) for v in loss_items)
                else:
                    loss_val = float(loss_items)
                loss_str = f"Loss: {loss_val:.4f}"
            except Exception:
                loss_str = ""

        # Progress bar (ASCII-safe for Windows console)
        pct = int((current_ep / self.total_epochs) * 20)
        bar = "=" * pct + "-" * (20 - pct)

        print(f"[{bar}] {int((current_ep / self.total_epochs) * 100)}%")
        print(f"{loss_str} | Elapsed: {elapsed_str} | ETA: {eta_str}")

    def on_val_end(self, validator):
        metrics = getattr(validator, 'metrics', None)
        if metrics is not None:
            try:
                map50 = getattr(metrics, 'box.map50', None) or validator.metrics.results_dict.get('metrics/mAP50(B)', 0.0)
                map50_95 = getattr(metrics, 'box.map', None) or validator.metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
                prec = validator.metrics.results_dict.get('metrics/precision(B)', 0.0)
                rec = validator.metrics.results_dict.get('metrics/recall(B)', 0.0)
                print(f"Validation Metrics -> Precision: {prec:.4f} | Recall: {rec:.4f} | mAP@50: {map50:.4f} | mAP@50-95: {map50_95:.4f}")
            except Exception:
                pass

def train_low_light_detector(
    data_yaml: str = "config/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 416,
    batch: int = 4,
    device: str = "cpu",
    project: str = "runs/detect",
    name: str = "train_baseline",
    patience: int = 10,
    save_period: int = 5,
    fraction: float = 1.0,
    workers: int = 0
) -> Dict[str, Any]:
    """
    Executes YOLOv8 fine-tuning on the ExDark dataset with live progress tracking.
    """
    from ultralytics import YOLO

    print("=" * 70)
    print("STARTING LOW-LIGHT OBJECT DETECTION TRAINING")
    print("=" * 70)
    print(f"Base Pretrained Model: {model_name}")
    print(f"Dataset Config:        {data_yaml}")
    print(f"Input Image Size:      {imgsz}x{imgsz}")
    print(f"Batch Size:            {batch}")
    print(f"Target Epochs:         {epochs}")
    print(f"Dataset Fraction:      {fraction}")
    print(f"Device:                {device}")
    print(f"Run Output:            {project}/{name}")
    print("=" * 70)

    # Initialize model
    model = YOLO(model_name)

    # Register custom live callback
    cb = TrainingProgressCallback(total_epochs=epochs)
    model.add_callback("on_train_epoch_start", cb.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
    model.add_callback("on_val_end", cb.on_val_end)

    # Launch training
    results = model.train(
        data=str(Path(data_yaml).resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=True,
        save_period=save_period,
        workers=workers,
        pretrained=True,
        verbose=True,
        fraction=fraction,
        # Low-light hyperparameter overrides
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1
    )

    # Copy best.pt and last.pt to models/ directory for clean checkpointing
    models_dir = Path("models").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(project) / name))
    weights_candidates = [
        save_dir / "weights",
        Path(project) / name / "weights",
        Path("runs/detect/runs/detect") / name / "weights",
    ]

    for wdir in weights_candidates:
        if (wdir / "best.pt").exists():
            import shutil
            shutil.copy2(wdir / "best.pt", models_dir / "best.pt")
            print(f"Copied best weights from {wdir} to {models_dir / 'best.pt'}")
            if (wdir / "last.pt").exists():
                shutil.copy2(wdir / "last.pt", models_dir / "last.pt")
                print(f"Copied last weights from {wdir} to {models_dir / 'last.pt'}")
            break

    return results

if __name__ == "__main__":
    train_low_light_detector()
