"""Module 7: Profile Storage — cache benchmark results per model+hardware."""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import filelock

from ggtune.config import CONFIG_DIR, PROFILES_DIR, LLAMA_CPP_PINNED_BUILD
from ggtune.models.hardware import HardwareProfile
from ggtune.models.model_profile import ModelProfile
from ggtune.models.profile import StoredProfile

PROFILE_VERSION = "1.0"
STALE_DAYS = 30


def _model_hash(model_path: str) -> str:
    with open(model_path, "rb") as f:
        data = f.read(1_048_576)
    return hashlib.sha256(data).hexdigest()[:16]


def compute_profile_id(model_path: str, hw: HardwareProfile) -> str:
    model_hash = _model_hash(model_path)
    hw_hash = hashlib.sha256(hw.hw_fingerprint.encode()).hexdigest()[:8]
    return f"{model_hash}_{hw_hash}"


def _profile_path(profile_id: str) -> Path:
    return PROFILES_DIR / f"{profile_id}.json"


def load(model_path: str, hw: HardwareProfile) -> Optional[StoredProfile]:
    """Return cached profile or None if missing/stale/invalid."""
    try:
        profile_id = compute_profile_id(model_path, hw)
    except Exception:
        return None

    path = _profile_path(profile_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    created_at = datetime.fromisoformat(data.get("created_at", "2000-01-01"))
    if datetime.now() - created_at > timedelta(days=STALE_DAYS):
        return None

    if data.get("llama_cpp_build") != LLAMA_CPP_PINNED_BUILD:
        return None

    try:
        return StoredProfile(**data)
    except Exception:
        return None


def save(
    model: ModelProfile,
    hw: HardwareProfile,
    result: dict,
    bottleneck: str,
    total_bench_time_min: float,
) -> StoredProfile:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_id = compute_profile_id(model.path, hw)
    path = _profile_path(profile_id)

    profile = StoredProfile(
        version=PROFILE_VERSION,
        created_at=datetime.now().isoformat(),
        profile_id=profile_id,
        model_path=model.path,
        model_name=model.name,
        model_is_moe=model.is_moe,
        model_quantization=model.quantization,
        model_file_size_gb=model.file_size_gb,
        hw_gpu_name=hw.gpu_name,
        hw_vram_total_mb=hw.vram_total_mb,
        hw_cpu_cores_physical=hw.cores_physical,
        hw_ram_total_gb=hw.ram_total_gb,
        best_params=result["best_params"],
        tg_tokens_per_sec=result["tg_tokens_per_sec"],
        pp_tokens_per_sec=result.get("pp_tokens_per_sec", 0.0),
        tg_std=result.get("tg_std", 0.0),
        stability_cv=result.get("stability_cv", 0.0),
        bottleneck=bottleneck,
        optuna_trials=result.get("optuna_trials", 0),
        total_bench_time_min=total_bench_time_min,
        llama_cpp_build=LLAMA_CPP_PINNED_BUILD,
        optimal_context=result.get("optimal_ctx", 8192),
    )

    lock_path = str(path) + ".lock"
    with filelock.FileLock(lock_path):
        path.write_text(json.dumps(profile.__dict__, indent=2))

    return profile


def list_all() -> list[StoredProfile]:
    if not PROFILES_DIR.exists():
        return []
    profiles = []
    for p in PROFILES_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            profiles.append(StoredProfile(**data))
        except Exception:
            pass
    return sorted(profiles, key=lambda x: x.created_at, reverse=True)


def delete(model_path: str, hw: HardwareProfile) -> bool:
    try:
        profile_id = compute_profile_id(model_path, hw)
        path = _profile_path(profile_id)
        if path.exists():
            path.unlink()
            return True
    except Exception:
        pass
    return False


def delete_all() -> int:
    if not PROFILES_DIR.exists():
        return 0
    count = 0
    for p in PROFILES_DIR.glob("*.json"):
        p.unlink()
        count += 1
    return count
