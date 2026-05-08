from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class StoredProfile:
    version: str
    created_at: str
    profile_id: str
    model_path: str
    model_name: str
    model_is_moe: bool
    model_quantization: str
    model_file_size_gb: float
    hw_gpu_name: str
    hw_vram_total_mb: int
    hw_cpu_cores_physical: int
    hw_ram_total_gb: float
    best_params: dict
    tg_tokens_per_sec: float
    pp_tokens_per_sec: float
    tg_std: float
    stability_cv: float
    bottleneck: str
    optuna_trials: int
    total_bench_time_min: float
    llama_cpp_build: str
    optimal_context: int
