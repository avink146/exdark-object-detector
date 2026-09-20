"""
EXP-002-CONTROLLED: Strictly Controlled Cosine LR Scheduler Experiment.
Reference: EXP-001 (train_baseline)
Only Difference: cos_lr: False -> True
All other hyperparameters, seeds, dataset fractions, augmentations, and resolutions are 100% identical.
"""

import sys
import os
import time
import json
import csv
import yaml
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

EXP001_ARGS_PATH = PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "train_baseline" / "args.yaml"

def verify_and_diff_config():
    if not EXP001_ARGS_PATH.exists():
        raise FileNotFoundError(f"EXP-001 args.yaml not found at {EXP001_ARGS_PATH}")
    
    with open(EXP001_ARGS_PATH, "r", encoding="utf-8") as f:
        exp001_cfg = yaml.safe_load(f)

    # Intended target config for EXP-002-CONTROLLED
    target_args = {
        "data": str(PROJECT_ROOT / "config" / "data.yaml"),
        "epochs": exp001_cfg["epochs"],             # 3
        "imgsz": exp001_cfg["imgsz"],               # 416
        "batch": exp001_cfg["batch"],               # 8
        "device": exp001_cfg["device"],             # cpu
        "project": "runs/detect",
        "name": "exp002_controlled",
        "patience": exp001_cfg["patience"],         # 3
        "save": True,
        "save_period": exp001_cfg["save_period"],   # 5
        "workers": exp001_cfg["workers"],           # 0
        "pretrained": True,
        "verbose": True,
        "fraction": exp001_cfg["fraction"],         # 0.1
        "seed": exp001_cfg["seed"],                 # 0
        "deterministic": exp001_cfg["deterministic"], # True
        "lr0": exp001_cfg["lr0"],                   # 0.01
        "lrf": exp001_cfg["lrf"],                   # 0.01
        "momentum": exp001_cfg["momentum"],         # 0.937
        "weight_decay": exp001_cfg["weight_decay"], # 0.0005
        "warmup_epochs": exp001_cfg["warmup_epochs"], # 3.0
        "warmup_momentum": exp001_cfg["warmup_momentum"], # 0.8
        "warmup_bias_lr": exp001_cfg["warmup_bias_lr"], # 0.1
        # Augmentations (must match EXP-001 exactly)
        "hsv_h": exp001_cfg["hsv_h"],               # 0.015
        "hsv_s": exp001_cfg["hsv_s"],               # 0.6
        "hsv_v": exp001_cfg["hsv_v"],               # 0.4
        "degrees": exp001_cfg["degrees"],           # 5.0
        "translate": exp001_cfg["translate"],       # 0.1
        "scale": exp001_cfg["scale"],               # 0.3
        "fliplr": exp001_cfg["fliplr"],             # 0.5
        "mosaic": exp001_cfg["mosaic"],             # 0.5
        "mixup": exp001_cfg["mixup"],               # 0.1
        # The single independent experimental variable under test:
        "cos_lr": True                              # Changed from False
    }

    # Verify differences
    print("=" * 70)
    print("CONFIGURATION AUDIT & DIFF: EXP-001 vs EXP-002-CONTROLLED")
    print("=" * 70)
    
    critical_keys = [
        "model", "data", "epochs", "imgsz", "batch", "device", "workers",
        "fraction", "seed", "deterministic", "lr0", "lrf", "momentum",
        "weight_decay", "warmup_epochs", "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "fliplr", "mosaic", "mixup", "cos_lr"
    ]

    differences = []
    for k in critical_keys:
        val1 = exp001_cfg.get(k)
        if k == "model":
            val2 = "yolov8n.pt"
        else:
            val2 = target_args.get(k)
        
        # Normalize data path comparison
        if k == "data":
            match = Path(val1).resolve() == Path(val2).resolve()
            if not match:
                differences.append((k, val1, val2))
            continue
            
        if val1 != val2:
            differences.append((k, val1, val2))

    print(f"{'PARAMETER':<20} | {'EXP-001 VALUE':<25} | {'EXP-002-CONTROLLED':<25}")
    print("-" * 75)
    for k in critical_keys:
        val1 = str(exp001_cfg.get(k))
        val2 = "yolov8n.pt" if k == "model" else str(target_args.get(k))
        marker = " <--- CHANGED" if val1 != val2 else ""
        print(f"{k:<20} | {val1:<25} | {val2:<25}{marker}")

    print("-" * 75)
    print(f"Total differences found: {len(differences)}")
    for d in differences:
        print(f"  DIFF: {d[0]} changed from '{d[1]}' to '{d[2]}'")

    if len(differences) != 1 or differences[0][0] != "cos_lr":
        raise ValueError(
            f"FATAL: Experiment is NOT strictly controlled! Expected only 'cos_lr' to differ, but found {differences}"
        )
    
    print("\n>>> CONTROLLED EXPERIMENT VERIFICATION PASSED: ONLY cos_lr DIFFERS! <<<")
    print("=" * 70)
    return target_args

def update_status(stage, exp_id, status, epoch_str, current_best, latest_epoch, elapsed, checkpoint, notes):
    status_file = PROJECT_ROOT / "reports" / "STATUS.md"
    content = f"""# Low-Light Object Detection — Research Status Report

**Last Updated:** {time.strftime("%Y-%m-%dT%H:%M:%S+05:30")}  
**Project Objective:** Build the strongest possible ExDark low-light object detector supported by real experiments, real validation evidence, and reproducible artifacts.

---

### CURRENT STAGE
`{stage}`

### CURRENT EXPERIMENT
`{exp_id}`

### STATUS
`{status}`

### CURRENT EPOCH
`{epoch_str}`

### CURRENT BEST
{current_best}

### LATEST EPOCH
{latest_epoch}

### TRAINING TIME
- Elapsed: {elapsed}

### CHECKPOINT
- Exact Path: `{checkpoint}`

### WHAT CHANGED
- Single isolated variable: `cos_lr: false -> true`
- Reference baseline: `EXP-001`
- All other hyperparameters strictly fixed: `fraction=0.10`, `epochs=3`, `batch=8`, `imgsz=416`, `seed=0`, `workers=0`, `lr0=0.01`, `lrf=0.01`, augmentations identical.

### WHAT WAS VERIFIED
1. Dataset split isolation: Train (5,142), Validation (1,469), Test (734) images. Total: 7,345 images. 0 corrupt labels.
2. Protected test set integrity: Zero test data leakage verified. Test split is strictly forbidden during model development.
3. Pre-training configuration diff between EXP-001 and EXP-002-CONTROLLED strictly verified: only `cos_lr` differs.
4. Model weights integrity: Checkpoints from EXP-001 (`models/baseline_best.pt`) and EXP-002-PRELIM (`models/exp002_best.pt`) preserved without overwriting.

### PROBLEMS FOUND
1. **Confounding in initial EXP-002**: Previous EXP-002 changed data fraction (0.10 -> 0.15), epochs (3 -> 6), lr0 (0.01 -> 0.005), and cos_lr simultaneously. It cannot support a causal conclusion regarding cosine scheduling.
2. **Premature final model promotion**: `models/final_best.pt` was populated from preliminary EXP-002 without full-data training or valid controlled comparison.
3. **Overclaimed causal language in reports**: Previous reports used terms like "confirmed bottleneck" and "true cause", which have been audited and replaced with cautious scientific language.

### FIXES APPLIED
1. Reclassified previous EXP-002 as `EXP-002-PRELIM` (Scientific validity: `CONFOUNDED`).
2. Frozen and invalidated `models/final_best.pt` until full-data training and formal freeze criteria are satisfied.
3. Formally launched `EXP-002-CONTROLLED` with strict 1-variable isolation against EXP-001.

### NEXT ACTION
{notes}
"""
    with open(status_file, "w", encoding="utf-8") as f:
        f.write(content)

class SurveillanceCallback:
    """Surveillance callback to capture live metrics and update STATUS.md per epoch."""
    def __init__(self, total_epochs=3):
        self.total_epochs = total_epochs
        self.t0 = time.time()
        self.epoch_t0 = time.time()
        self.best_map50 = -1.0
        self.best_epoch = -1
        self.best_row = {}

    def on_train_epoch_start(self, trainer):
        self.epoch_t0 = time.time()
        ep = trainer.epoch + 1
        print(f"\n{'='*70}\n[EXP-002-CONTROLLED] Epoch {ep}/{self.total_epochs} Starting\n{'='*70}")
        update_status(
            stage="PHASE 1 — CONTROLLED EXPERIMENTATION",
            exp_id="EXP-002-CONTROLLED",
            status="RUNNING",
            epoch_str=f"Epoch {ep} / {self.total_epochs} (In Progress)",
            current_best=f"- Best Epoch: {self.best_epoch if self.best_epoch > 0 else 'N/A'}\n- Best Val mAP@50: {self.best_map50:.4f}" if self.best_epoch > 0 else "- Best Epoch: Pending\n- Best Val mAP@50: Pending",
            latest_epoch=f"Epoch {ep} running...",
            elapsed=f"{round(time.time() - self.t0, 1)} s",
            checkpoint="models/exp002_controlled_best.pt (in progress)",
            notes=f"Training epoch {ep} of 3 on CPU."
        )

    def on_train_epoch_end(self, trainer):
        ep = trainer.epoch + 1
        dt = time.time() - self.epoch_t0
        print(f"[EXP-002-CONTROLLED] Epoch {ep} train completed in {dt:.1f} s")

    def on_val_end(self, validator):
        metrics = getattr(validator, 'metrics', None)
        if metrics is not None:
            results = validator.metrics.results_dict
            map50 = float(results.get('metrics/mAP50(B)', 0.0))
            map50_95 = float(results.get('metrics/mAP50-95(B)', 0.0))
            prec = float(results.get('metrics/precision(B)', 0.0))
            rec = float(results.get('metrics/recall(B)', 0.0))
            
            # Epoch index
            ep = getattr(validator, 'epoch', 0) + 1
            if map50 > self.best_map50:
                self.best_map50 = map50
                self.best_epoch = ep
                self.best_row = {
                    "epoch": ep, "mAP50": map50, "mAP50_95": map50_95, "precision": prec, "recall": rec
                }
            
            print(f"\n>>> [SURVEILLANCE EPOCH {ep}] Val Precision: {prec:.4f} | Recall: {rec:.4f} | mAP@50: {map50:.4f} | mAP@50-95: {map50_95:.4f} <<<")
            
            update_status(
                stage="PHASE 1 — CONTROLLED EXPERIMENTATION",
                exp_id="EXP-002-CONTROLLED",
                status="RUNNING",
                epoch_str=f"Epoch {ep} / {self.total_epochs} (Val Completed)",
                current_best=f"- Best Epoch: {self.best_epoch}\n- Best Val mAP@50: {self.best_map50:.4f}\n- Best Val mAP@50-95: {self.best_row.get('mAP50_95', 0.0):.4f}\n- Precision: {self.best_row.get('precision', 0.0):.4f}\n- Recall: {self.best_row.get('recall', 0.0):.4f}",
                latest_epoch=f"- Epoch: {ep}\n- Val mAP@50: {map50:.4f}\n- Val mAP@50-95: {map50_95:.4f}\n- Precision: {prec:.4f}\n- Recall: {rec:.4f}",
                elapsed=f"{round(time.time() - self.t0, 1)} s",
                checkpoint="models/exp002_controlled_best.pt (in progress)",
                notes=f"Epoch {ep} complete. Best so far: Epoch {self.best_epoch} (mAP@50: {self.best_map50:.4f})."
            )

def main():
    target_args = verify_and_diff_config()
    
    t_start = time.time()
    model = YOLO("yolov8n.pt")
    
    cb = SurveillanceCallback(total_epochs=3)
    model.add_callback("on_train_epoch_start", cb.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
    model.add_callback("on_val_end", cb.on_val_end)

    print("\nStarting Ultralytics YOLO training for EXP-002-CONTROLLED...")
    results = model.train(**target_args)
    total_train_time = round(time.time() - t_start, 2)
    print(f"\nTraining completed in {total_train_time} seconds.")

    # Locate saved weights
    save_dir = Path(model.trainer.save_dir)
    print(f"Model save directory: {save_dir}")
    weights_dir = save_dir / "weights"
    
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    best_dst = models_dir / "exp002_controlled_best.pt"
    last_dst = models_dir / "exp002_controlled_last.pt"
    
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"Preserved best checkpoint: {best_dst} ({best_dst.stat().st_size} bytes)")
    else:
        raise FileNotFoundError(f"Missing best.pt at {best_src}")

    if last_src.exists():
        shutil.copy2(last_src, last_dst)
        print(f"Preserved last checkpoint: {last_dst} ({last_dst.stat().st_size} bytes)")

    # Read the real results.csv
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        # Check subdirectories
        for p in save_dir.glob("**/results.csv"):
            results_csv = p
            break

    print(f"\nReading actual generated results from: {results_csv}")
    rows = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Strip whitespace in keys
            clean_r = {k.strip(): v.strip() for k, v in r.items()}
            rows.append(clean_r)

    print(f"\n{'Epoch':<6} | {'Box Loss':<10} | {'Cls Loss':<10} | {'DFL Loss':<10} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}")
    print("-" * 88)
    best_map50 = -1.0
    best_epoch_data = {}
    for r in rows:
        ep = r.get("epoch")
        box = float(r.get("train/box_loss", 0.0))
        cls_l = float(r.get("train/cls_loss", 0.0))
        dfl = float(r.get("train/dfl_loss", 0.0))
        p = float(r.get("metrics/precision(B)", 0.0))
        rec = float(r.get("metrics/recall(B)", 0.0))
        map50 = float(r.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(r.get("metrics/mAP50-95(B)", 0.0))
        print(f"{ep:<6} | {box:<10.4f} | {cls_l:<10.4f} | {dfl:<10.4f} | {p:<10.4f} | {rec:<10.4f} | {map50:<10.4f} | {map50_95:<10.4f}")
        if map50 > best_map50:
            best_map50 = map50
            best_epoch_data = {
                "epoch": int(ep),
                "box_loss": box,
                "cls_loss": cls_l,
                "dfl_loss": dfl,
                "precision": p,
                "recall": rec,
                "mAP50": map50,
                "mAP50_95": map50_95
            }

    # Record in experiment registry
    reg_entry = {
        "experiment_id": "EXP-002-CONTROLLED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parent_experiment": "EXP-001",
        "status": "COMPLETED",
        "scientific_validity": "VALID CONTROLLED",
        "model": "YOLOv8n",
        "pretrained_weights": "yolov8n.pt",
        "dataset": "ExDark",
        "dataset_version": "processed_v1",
        "train_count": 514,
        "val_count": 1469,
        "test_count": 734,
        "fraction": 0.10,
        "imgsz": 416,
        "batch": 8,
        "epochs": 3,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": True,
        "augmentations": "hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.3, fliplr=0.5, mosaic=0.5, mixup=0.1",
        "seed": 0,
        "workers": 0,
        "device": "cpu",
        "best_epoch": best_epoch_data["epoch"],
        "best_val_mAP50": round(best_epoch_data["mAP50"], 4),
        "best_val_mAP50_95": round(best_epoch_data["mAP50_95"], 4),
        "best_precision": round(best_epoch_data["precision"], 4),
        "best_recall": round(best_epoch_data["recall"], 4),
        "training_time": total_train_time,
        "checkpoint": "models/exp002_controlled_best.pt",
        "notes": "Strictly controlled scheduler experiment against EXP-001. Only cos_lr changed from False to True."
    }

    # Update experiment_registry.json & csv
    reg_json_path = PROJECT_ROOT / "reports" / "experiment_registry.json"
    with open(reg_json_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    # Replace if exists or append
    idx = next((i for i, x in enumerate(registry) if x["experiment_id"] == "EXP-002-CONTROLLED"), None)
    if idx is not None:
        registry[idx] = reg_entry
    else:
        registry.append(reg_entry)

    with open(reg_json_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    reg_csv_path = PROJECT_ROOT / "reports" / "experiment_registry.csv"
    with open(reg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registry[0].keys()))
        writer.writeheader()
        writer.writerows(registry)

    # Update STATUS.md
    update_status(
        stage="PHASE 1 — CONTROLLED EXPERIMENTATION",
        exp_id="EXP-002-CONTROLLED",
        status="COMPLETED",
        epoch_str=f"Epoch 3 / 3 (Completed)",
        current_best=f"- Best Epoch: {best_epoch_data['epoch']}\n- Best Val mAP@50: {best_epoch_data['mAP50']:.4f}\n- Best Val mAP@50-95: {best_epoch_data['mAP50_95']:.4f}\n- Precision: {best_epoch_data['precision']:.4f}\n- Recall: {best_epoch_data['recall']:.4f}",
        latest_epoch=f"- Epoch: 3\n- Val mAP@50: {rows[-1].get('metrics/mAP50(B)')}\n- Precision: {rows[-1].get('metrics/precision(B)')}\n- Recall: {rows[-1].get('metrics/recall(B)')}",
        elapsed=f"{total_train_time} s",
        checkpoint="models/exp002_controlled_best.pt",
        notes="EXP-002-CONTROLLED completed. Next: Compute direct comparison with EXP-001."
    )
    print("\nEXP-002-CONTROLLED finished and logged successfully.")

if __name__ == "__main__":
    main()
