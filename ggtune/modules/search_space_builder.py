"""Module 5: Search Space Builder."""
from typing import List

from ggtune.config import CONTEXT_CANDIDATES
from ggtune.models.hardware import HardwareProfile, Backend
from ggtune.models.model_profile import ModelProfile
from ggtune.models.search_space import SearchSpace


def kv_mb_per_token(model: ModelProfile) -> float:
    """Estimate KV cache RAM per token in MB (f16, using actual KV heads if known)."""
    kv_heads = model.n_kv_heads or max(1, model.n_heads // 8)
    head_dim = 128
    # 2 (K+V) × layers × kv_heads × head_dim × 2 bytes (f16)
    return 2 * model.n_layers * kv_heads * head_dim * 2 / (1024 * 1024)


def max_practical_ctx(hw: HardwareProfile, model: ModelProfile) -> int:
    """Max context that fits within ~70% of available RAM for KV cache."""
    mb_per_tok = kv_mb_per_token(model)
    if mb_per_tok <= 0:
        return max(CONTEXT_CANDIDATES)
    # RAM budget: total minus model (conservatively all in RAM) minus 5GB for OS/apps
    ram_for_kv_mb = (hw.ram_total_gb - model.file_size_gb - 5.0) * 1024 * 0.85
    if ram_for_kv_mb <= 0:
        return CONTEXT_CANDIDATES[0]
    return int(ram_for_kv_mb / mb_per_tok)


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
    if model.file_size_gb > hw.ram_total_gb * 0.85:
        raise RuntimeError(
            f"Model {model.file_size_gb:.1f}GB won't fit in RAM "
            f"({hw.ram_total_gb:.1f}GB total). Use a smaller model."
        )

    threads = _thread_candidates(hw)
    flash_attn = hw.backend in (Backend.CUDA, Backend.ROCM, Backend.METAL)
    kv_quants = ["f16", "q8_0", "q4_0"]

    ctx_cap = max_practical_ctx(hw, model)
    ctx_candidates = [
        c for c in CONTEXT_CANDIDATES
        if c <= model.context_length_max and c <= ctx_cap
    ]
    if not ctx_candidates:
        ctx_candidates = [min(c for c in CONTEXT_CANDIDATES if c <= model.context_length_max)
                          if any(c <= model.context_length_max for c in CONTEXT_CANDIDATES)
                          else CONTEXT_CANDIDATES[0]]

    if model.is_moe and model.n_experts_total and model.n_experts_used:
        # ncmoe=N: N experts per layer stay in CPU RAM, (total-N) go to GPU VRAM.
        # Low ncmoe = most experts on GPU = fast (might OOM; benchmark handles crashes).
        # High ncmoe = most experts in CPU = uses less VRAM, much slower.
        n_total = model.n_experts_total
        n_used = model.n_experts_used

        # Search from n_used (only the "active" quota in CPU, rest on GPU)
        # up to n_total - n_used (keep exactly n_used experts on GPU as minimum).
        ncmoe_min = n_used
        ncmoe_max = n_total - n_used

        # ~8 probe points across the range so Optuna covers both extremes
        step = max(n_used, (ncmoe_max - ncmoe_min) // 8)
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
