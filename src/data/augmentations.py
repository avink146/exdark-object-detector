"""
Low-light specific data augmentation configurations and transformations.

In low-light object detection, key visual challenges are:
1. Low contrast and underexposure
2. Non-uniform lighting and heavy shadows
3. High sensor noise (shot noise, read noise in high-ISO low-light captures)
4. Motion and optical blur due to slow shutter speeds
5. Color casts and white balance shifts

This module configures realistic low-light augmentation parameters for YOLOv8 fine-tuning
without distorting geometric structure.
"""

import yaml
from pathlib import Path

# Hyperparameters tailored specifically for fine-tuning on low-light underexposed datasets
LOW_LIGHT_HYPERPARAMETERS = {
    # Optimizer & Learning Rate
    "lr0": 0.005,           # Initial learning rate (conservative for transfer learning)
    "lrf": 0.01,            # Final learning rate fraction (cosine schedule: lr0 * lrf)
    "momentum": 0.937,       # SGD / Adam momentum
    "weight_decay": 0.0005,  # Optimizer weight decay
    "warmup_epochs": 3.0,    # Warmup epochs
    "warmup_momentum": 0.8,  # Warmup initial momentum
    "warmup_bias_lr": 0.1,   # Warmup initial bias lr
    
    # Loss gains
    "box": 7.5,              # Box loss gain
    "cls": 0.5,              # Class loss gain
    "dfl": 1.5,              # DFL loss gain
    
    # Low-Light Specific Color / Photometric Augmentation
    "hsv_h": 0.015,          # Image HSV-Hue augmentation (fraction)
    "hsv_s": 0.6,            # Image HSV-Saturation augmentation (simulates varying color saturation under night lights)
    "hsv_v": 0.4,            # Image HSV-Value/Brightness augmentation (critical for low-light/night variability)
    
    # Geometric Augmentation
    "degrees": 5.0,          # Image rotation (+/- deg)
    "translate": 0.1,        # Image translation (+/- fraction)
    "scale": 0.3,            # Image scale (+/- gain)
    "shear": 0.0,            # Image shear (+/- deg)
    "perspective": 0.0,      # Image perspective (+/- fraction)
    "flipud": 0.0,           # Probability of flipping upside-down
    "fliplr": 0.5,           # Probability of flipping left-right
    
    # Low-light Mosaic & Composition
    "mosaic": 0.5,           # Probability of mosaic augmentation (helps detect small/distant objects in dark scenes)
    "mixup": 0.1,            # Probability of mixup augmentation
    "copy_paste": 0.0,       # Probability of copy-paste
    "auto_augment": "randaugment", # Policy for photometric transforms
    "erasing": 0.2           # Random erasing probability (simulates occlusion by darkness/shadows)
}

def save_hyp_yaml(output_path: str = "config/hyp.yaml"):
    """Exports low-light augmentation hyperparameters to a YAML config file."""
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(LOW_LIGHT_HYPERPARAMETERS, f, default_flow_style=False, sort_keys=False)
    print(f"Saved low-light hyperparameter configuration to: {p}")
    return p

if __name__ == "__main__":
    save_hyp_yaml()
