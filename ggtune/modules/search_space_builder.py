"""Module 5: Search Space Builder."""
import math
from typing import List

from ggtune.config import CONTEXT_CANDIDATES
from ggtune.models.hardware import HardwareProfile, Backend
from ggtune.models.model_profile import ModelProfile
from ggtune.models.search_space import SearchSpace


def _estimate_kv_mb_per_token(model: ModelProfile) -> float:
    """Very rough KV estimate: 2 * layers * heads * head_dim * 2 bytes (f16)."""
    head_dim = 128
    return 2 * model.n_layers * model.n_heads * head_dim * 2 / (1024 * 1024)


def _thread_candidates(hw: HardwareProfile) -> List[int]:
    center = hw.cores_physical
    candidates = sorted({
        max(1, center - 2),
        center,
        min(hw.cores_logical, center + 2),
        min(hw.cores_logical, center + 4),
    })
    return candidates


def build(hw: HardwareProfile, model: ModelProfile) -> SearchSpace:
    if model.file_size_gb > hw.ram_available_gb * 0.9:
        raise RuntimeError(
            f"Model {model.file_size_gb:.1f}GB won't fit in available RAM "
            f"({hw.ram_available_gb:.1f}GB). Free up memory or use a smaller model."
        )

    threads = _thread_candidates(hw)
    flash_attn = hw.backend in (Backend.CUDA, Backend.ROCM, Backend.METAL)

    vram_for_kv = max(0, hw.vram_free_mb - model.file_size_gb * 1024)
    kv_per_token = _estimate_kv_mb_per_token(model)
    ctx_in_vram_limit = vram_for_kv / kv_per_token if kv_per_token > 0 else 0
    need_nkvo = ctx_in_vram_limit < 32768

    kv_quants = ["f16", "q8_0", "q4_0"]
    nkvo_options = [True, False] if need_nkvo else [False]

    ctx_candidates = [c for c in CONTEXT_CANDIDATES if c <= model.context_length_max]
    if not ctx_candidates:
        ctx_candidates = [min(CONTEXT_CANDIDATES)]

    if model.is_moe and model.n_experts_total and model.n_experts_used:
        n_min = model.n_experts_used
        n_max = model.n_experts_total
        step = max(1, model.n_experts_used)
        ncmoe_range = range(n_min, n_max + 1, step)
        return SearchSpace(
            ncmoe_range=ncmoe_range,
            thread_candidates=threads,
            kv_quant_options=kv_quants,
            nkvo_options=nkvo_options,
            context_candidates=ctx_candidates,
            flash_attn=flash_attn,
        )
    else:
        # Dense model: search ngl (GPU layers)
        ngl_range = range(0, model.n_layers + 1)
        return SearchSpace(
            ngl_range=ngl_range,
            thread_candidates=threads,
            kv_quant_options=kv_quants,
            nkvo_options=[False],
            context_candidates=ctx_candidates,
            flash_attn=flash_attn,
        )
