"""Tests for advisor launch-command generation — regressions from the 2026-07 audit."""
from ggtune.modules import advisor
from ggtune.modules.env_manager import EnvConfig
from ggtune.models.hardware import Backend
from ggtune.models.model_profile import ModelProfile


def _env_cfg():
    return EnvConfig(
        llama_bench_path="/path/to/llama-bench",
        llama_cli_path="/path/to/llama-cli",
        llama_server_path="/path/to/llama-server",
        bin_dir="/path/to",
        build="b10066",
        backend=Backend.CUDA,
        env_dict={},
    )


def _dense_model():
    return ModelProfile(
        path="/tmp/model.gguf", name="Llama-3.1-8B", architecture="llama",
        is_moe=False, n_layers=32, n_heads=32, context_length_max=131072,
        file_size_gb=8.0, quantization="Q8_0",
    )


def test_launch_argv_uses_searched_ngl_not_hardcoded_999():
    """Regression: generate_launch_cmd used to always say -ngl 999, ignoring
    the ngl value actually found by the search (which may be a partial
    offload on a VRAM-constrained system)."""
    params = {"ngl": 17, "threads": 8, "ctk": "q8_0", "flash_attn": True}
    argv = advisor.build_launch_argv(_dense_model(), params, 8192, _env_cfg())
    assert argv[argv.index("-ngl") + 1] == "17"


def test_launch_argv_respects_flash_attn_value():
    """Regression: -fa on was hardcoded regardless of the benchmarked value."""
    off_params = {"ngl": 32, "threads": 8, "flash_attn": False}
    argv = advisor.build_launch_argv(_dense_model(), off_params, 8192, _env_cfg())
    assert argv[argv.index("-fa") + 1] == "off"


def test_generate_launch_cmd_matches_argv():
    params = {"ngl": 32, "threads": 8, "ctk": "q8_0", "flash_attn": True}
    env_cfg = _env_cfg()
    argv = advisor.build_launch_argv(_dense_model(), params, 8192, env_cfg)
    cmd_str = advisor.generate_launch_cmd(_dense_model(), params, 8192, env_cfg)
    for tok in argv:
        assert tok in cmd_str
