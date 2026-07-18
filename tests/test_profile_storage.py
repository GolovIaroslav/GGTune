"""Tests for profile cache — regressions from the 2026-07 audit.

Previously profile_storage.load() was never called from the orchestrator
(the cache was write-only) and cache validity was checked against a
hardcoded config constant instead of the actually-installed llama.cpp build.
"""
from ggtune.modules import profile_storage
from ggtune.models.hardware import HardwareProfile, Backend


def _hw():
    return HardwareProfile(
        gpu_name="RTX 3060", vram_total_mb=6144, vram_free_mb=5800,
        backend=Backend.CUDA, driver_version="535", compute_cap="8.6",
        cores_physical=8, cores_logical=16,
        cpu_name="i7-12700H", ram_total_gb=32.0, ram_available_gb=28.0,
        os="windows", shell="bash",
        hw_fingerprint="abc123",
    )


def _model(tmp_path):
    from ggtune.models.model_profile import ModelProfile
    p = tmp_path / "model.gguf"
    p.write_bytes(b"fake gguf content")
    return ModelProfile(
        path=str(p), name="Test-Model", architecture="llama",
        is_moe=False, n_layers=32, n_heads=32, context_length_max=8192,
        file_size_gb=0.01, quantization="Q8_0",
    )


def _result():
    return {
        "best_params": {"ngl": 32, "threads": 8, "ctk": "q8_0", "flash_attn": True},
        "tg_tokens_per_sec": 25.0,
        "pp_tokens_per_sec": 500.0,
        "tg_std": 0.5,
        "stability_cv": 0.02,
        "optimal_ctx": 8192,
    }


def test_save_then_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_storage, "PROFILES_DIR", tmp_path)
    hw = _hw()
    model = _model(tmp_path)

    profile_storage.save(model, hw, _result(), "well_balanced", 5.0, "b10066")
    loaded = profile_storage.load(model.path, hw, "b10066")

    assert loaded is not None
    assert loaded.tg_tokens_per_sec == 25.0
    assert loaded.pp_tokens_per_sec == 500.0
    assert loaded.best_params["ngl"] == 32


def test_load_invalidated_by_build_mismatch(tmp_path, monkeypatch):
    """A cached profile from an older llama.cpp build must not be served as current."""
    monkeypatch.setattr(profile_storage, "PROFILES_DIR", tmp_path)
    hw = _hw()
    model = _model(tmp_path)

    profile_storage.save(model, hw, _result(), "well_balanced", 5.0, "b9089")
    loaded = profile_storage.load(model.path, hw, "b10066")

    assert loaded is None


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_storage, "PROFILES_DIR", tmp_path)
    hw = _hw()
    model = _model(tmp_path)

    assert profile_storage.load(model.path, hw, "b10066") is None
