"""Tests for hardware scanner — mocked so no real GPU needed."""
from unittest.mock import patch, MagicMock
from ggtune.modules import hardware_scanner
from ggtune.models.hardware import Backend


def _mock_gpu():
    return {
        "name": "RTX 3060 Laptop GPU",
        "vram_total_mb": 6144,
        "vram_free_mb": 5800,
        "driver_version": "535.0",
        "compute_cap": "8.6",
    }


def test_scan_returns_profile():
    with patch("ggtune.modules.hardware_scanner._detect_backend_and_gpu",
               return_value=(Backend.CUDA, _mock_gpu())):
        hw = hardware_scanner.scan()
    assert hw.gpu_name == "RTX 3060 Laptop GPU"
    assert hw.vram_total_mb == 6144
    assert hw.backend == Backend.CUDA
    assert hw.cores_physical >= 1
    assert hw.ram_total_gb > 0


def test_hw_fingerprint_is_stable():
    with patch("ggtune.modules.hardware_scanner._detect_backend_and_gpu",
               return_value=(Backend.CUDA, _mock_gpu())):
        hw1 = hardware_scanner.scan()
        hw2 = hardware_scanner.scan()
    assert hw1.hw_fingerprint == hw2.hw_fingerprint


def test_cpu_fallback():
    cpu_gpu = {"name": "CPU", "vram_total_mb": 0, "vram_free_mb": 0,
               "driver_version": None, "compute_cap": None}
    with patch("ggtune.modules.hardware_scanner._detect_backend_and_gpu",
               return_value=(Backend.CPU, cpu_gpu)):
        hw = hardware_scanner.scan()
    assert hw.backend == Backend.CPU
