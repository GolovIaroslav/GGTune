"""pynvml wrappers with nvidia-smi fallback."""
import subprocess
import re
from typing import Optional


def nvml_available() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def get_gpu_info_nvml() -> Optional[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
        except Exception:
            driver = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_cap = f"{major}.{minor}"
        except Exception:
            compute_cap = None
        return {
            "name": name,
            "vram_total_mb": mem.total // (1024 * 1024),
            "vram_free_mb": mem.free // (1024 * 1024),
            "driver_version": driver,
            "compute_cap": compute_cap,
        }
    except Exception:
        return None


def get_gpu_info_smi() -> Optional[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 4:
            return None
        return {
            "name": parts[0],
            "vram_total_mb": int(parts[1]),
            "vram_free_mb": int(parts[2]),
            "driver_version": parts[3],
            "compute_cap": None,
        }
    except Exception:
        return None
