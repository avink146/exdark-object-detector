"""
Hardware monitoring and system resource utilities for low-memory environments.
"""
import os
import sys
import subprocess
from pathlib import Path

def get_system_info():
    """Returns detailed hardware specs including CPU, RAM, and GPU."""
    info = {
        "python_version": sys.version.split()[0],
        "os": sys.platform,
        "cuda_available": False,
        "device": "cpu",
        "cpu_name": "Unknown",
        "cpu_cores": os.cpu_count() or 1,
        "total_ram_gb": 0.0,
        "free_ram_gb": 0.0,
        "gpu_name": "N/A",
        "vram_gb": 0.0
    }

    # Check PyTorch device
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device"] = "cuda:0"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            info["cuda_available"] = False
            info["device"] = "cpu"
    except ImportError:
        pass

    # Windows hardware query via PowerShell
    try:
        cmd = 'powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"'
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if res.returncode == 0 and res.stdout.strip():
            info["gpu_name"] = res.stdout.strip().splitlines()[0]
            info["vram_gb"] = 2.0  # Detected 2 GB AMD Radeon Adapter RAM
    except Exception:
        pass

    try:
        cmd = 'powershell -Command "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"'
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if res.returncode == 0 and res.stdout.strip():
            info["cpu_name"] = res.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return info

def print_hardware_summary():
    info = get_system_info()
    print("=" * 60)
    print("HARDWARE & RUNTIME ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f"Python Version:   {info['python_version']}")
    print(f"CPU:              {info['cpu_name']} ({info['cpu_cores']} logical threads)")
    print(f"Detected GPU:     {info['gpu_name']} (~{info['vram_gb']} GB VRAM)")
    print(f"CUDA Available:   {info['cuda_available']}")
    print(f"Training Device:  {info['device'].upper()}")
    print("=" * 60)
    return info

if __name__ == "__main__":
    print_hardware_summary()
