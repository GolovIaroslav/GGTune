"""Module 1: Hardware Scanner."""
import hashlib
import platform
import os
import psutil

from ggtune.models.hardware import HardwareProfile, Backend
from ggtune.utils.gpu import get_gpu_info_nvml, get_gpu_info_smi


def _detect_backend_and_gpu() -> tuple[Backend, dict]:
    gpu = get_gpu_info_nvml()
    if gpu:
        return Backend.CUDA, gpu

    try:
        import rocm_smi  # noqa: F401
        gpu = get_gpu_info_smi()
        if gpu:
            return Backend.ROCM, gpu
    except ImportError:
        pass

    if platform.system() == "Darwin":
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        name = "Apple GPU"
        for line in result.stdout.splitlines():
            if "Chipset Model" in line:
                name = line.split(":", 1)[-1].strip()
                break
        return Backend.METAL, {
            "name": name,
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "driver_version": None,
            "compute_cap": None,
        }

    gpu = get_gpu_info_smi()
    if gpu:
        return Backend.CUDA, gpu

    return Backend.CPU, {
        "name": "CPU",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "driver_version": None,
        "compute_cap": None,
    }


def _detect_shell() -> str:
    shell_path = os.environ.get("SHELL", "")
    if "zsh" in shell_path:
        return "zsh"
    if "bash" in shell_path:
        return "bash"
    if platform.system() == "Windows":
        return "cmd"
    return "bash"


def _hw_fingerprint(gpu_name: str, cpu_name: str, vram_total_mb: int) -> str:
    raw = f"{gpu_name}|{cpu_name}|{vram_total_mb}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def scan() -> HardwareProfile:
    backend, gpu = _detect_backend_and_gpu()

    cpu_name = platform.processor() or platform.machine()
    cores_physical = psutil.cpu_count(logical=False) or 1
    cores_logical = psutil.cpu_count(logical=True) or cores_physical

    vm = psutil.virtual_memory()
    ram_total_gb = vm.total / (1024 ** 3)
    ram_available_gb = vm.available / (1024 ** 3)

    sys_map = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}
    os_name = sys_map.get(platform.system(), "linux")

    fingerprint = _hw_fingerprint(gpu["name"], cpu_name, gpu["vram_total_mb"])

    return HardwareProfile(
        gpu_name=gpu["name"],
        vram_total_mb=gpu["vram_total_mb"],
        vram_free_mb=gpu["vram_free_mb"],
        backend=backend,
        driver_version=gpu["driver_version"],
        compute_cap=gpu["compute_cap"],
        cores_physical=cores_physical,
        cores_logical=cores_logical,
        cpu_name=cpu_name,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        os=os_name,
        shell=_detect_shell(),
        hw_fingerprint=fingerprint,
    )
