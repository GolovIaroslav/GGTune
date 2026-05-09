"""Tests for search space builder."""
import pytest
from unittest.mock import patch
from ggtune.modules import search_space_builder
from ggtune.models.hardware import HardwareProfile, Backend
from ggtune.models.model_profile import ModelProfile


def _hw(vram_free_mb=5800, cores_physical=8, ram_available_gb=28.0):
    return HardwareProfile(
        gpu_name="RTX 3060", vram_total_mb=6144, vram_free_mb=vram_free_mb,
        backend=Backend.CUDA, driver_version="535", compute_cap="8.6",
        cores_physical=cores_physical, cores_logical=cores_physical * 2,
        cpu_name="i7-12700H", ram_total_gb=32.0, ram_available_gb=ram_available_gb,
        os="linux", shell="zsh",
        hw_fingerprint="abc123",
    )


def _model_moe(size_gb=12.0):
    return ModelProfile(
        path="/tmp/model.gguf", name="Qwen3.6-35B", architecture="qwen_moe",
        is_moe=True, n_layers=64, n_heads=32, context_length_max=131072,
        file_size_gb=size_gb, quantization="Q2_K_XL",
        n_experts_total=128, n_experts_used=8,
    )


def _model_dense(size_gb=8.0):
    return ModelProfile(
        path="/tmp/model.gguf", name="Llama-3.1-8B", architecture="llama",
        is_moe=False, n_layers=32, n_heads=32, context_length_max=131072,
        file_size_gb=size_gb, quantization="Q8_0",
    )


def test_moe_search_space():
    space = search_space_builder.build(_hw(), _model_moe())
    assert space.ncmoe_range is not None
    assert space.ncmoe_range.start == 8
    assert space.ncmoe_range.stop == 121  # range(8, n_total-n_used+1, step) = range(8, 121, 14)
    assert len(space.thread_candidates) >= 2
    assert space.flash_attn is True


def test_dense_search_space():
    space = search_space_builder.build(_hw(), _model_dense())
    assert space.ngl_range is not None
    assert space.ncmoe_range is None


def test_oom_model_raises():
    # ram_total_gb=8 → threshold=6.8 GB, model=12 GB → should raise
    hw_small = HardwareProfile(
        gpu_name="RTX 3060", vram_total_mb=6144, vram_free_mb=5800,
        backend=Backend.CUDA, driver_version="535", compute_cap="8.6",
        cores_physical=8, cores_logical=16,
        cpu_name="i7-12700H", ram_total_gb=8.0, ram_available_gb=5.0,
        os="linux", shell="zsh", hw_fingerprint="abc123",
    )
    with pytest.raises(RuntimeError, match="won't fit"):
        search_space_builder.build(hw_small, _model_moe(size_gb=12.0))
