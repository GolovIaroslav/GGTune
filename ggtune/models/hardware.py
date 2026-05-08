from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Backend(str, Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    CPU = "cpu"


@dataclass
class HardwareProfile:
    gpu_name: str
    vram_total_mb: int
    vram_free_mb: int
    backend: Backend
    driver_version: Optional[str]
    compute_cap: Optional[str]
    cores_physical: int
    cores_logical: int
    cpu_name: str
    ram_total_gb: float
    ram_available_gb: float
    os: str
    shell: str
    hw_fingerprint: str

    def __str__(self) -> str:
        gpu = f"{self.gpu_name} {self.vram_total_mb // 1024}GB"
        return f"{gpu} · {self.ram_total_gb:.0f}GB RAM · {self.cores_physical} cores"
