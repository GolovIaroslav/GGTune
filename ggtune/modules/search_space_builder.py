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
    # Use total RAM for the check — available RAM fluctuates with running processes
    if model.file_size_gb > hw.ram_total_gb * 0.85:
        raise RuntimeError(
            f"Model {model.file_size_gb:.1f}GB won't fit in RAM "
            f"({hw.ram_total_gb:.1f}GB total). Use a smaller model."
        )

    threads = _thread_candidates(hw)
    flash_attn = hw.backend in (Backend.CUDA, Backend.ROCM, Backend.METAL)
    kv_quants = ["f16", "q8_0", "q4_0"]

    ctx_candidates = [c for c in CONTEXT_CANDIDATES if c <= model.context_length_max]
    if not ctx_candidates:
        ctx_candidates = [min(CONTEXT_CANDIDATES)]

    if model.is_moe and model.n_experts_total and model.n_experts_used:
        # ncmoe = number of experts offloaded to CPU RAM.
        # ncmoe=N means N experts in CPU, (total-N) experts in GPU VRAM.
        # To fit in limited VRAM: search from high ncmoe (few in GPU) downward.
        #
        # Safe minimum: total - used (keeps exactly used# experts in GPU)
        # Safe maximum: total - 1 (1 expert in GPU, all others in CPU)
        n_total = model.n_experts_total
        n_used = model.n_experts_used
        ncmoe_min = max(n_total - n_used, n_used)  # at least n_used in GPU
        ncmoe_max = n_total - 1
        step = max(1, n_used)
        ncmoe_range = range(ncmoe_min, ncmoe_max + 1, step)
        return SearchSpace(
            ncmoe_range=ncmoe_range,
            thread_candidates=threads,
            kv_quant_options=kv_quants,
            nkvo_options=[False],
            context_candidates=ctx_candidates,
            flash_attn=flash_attn,
        )
    else:
        ngl_range = range(0, model.n_layers + 1)
        return SearchSpace(
            ngl_range=ngl_range,
            thread_candidates=threads,
            kv_quant_options=kv_quants,
            nkvo_options=[False],
            context_candidates=ctx_candidates,
            flash_attn=flash_attn,
        )
