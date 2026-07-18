"""Tests for benchmark command construction — regressions from the 2026-07 audit."""
from ggtune.modules.benchmark_engine import _build_cmd, _ALL_FLAGS
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


def _moe_model():
    return ModelProfile(
        path="/tmp/model.gguf", name="Qwen3.6-35B-A3B", architecture="qwen_moe",
        is_moe=True, n_layers=40, n_heads=16, context_length_max=131072,
        file_size_gb=12.0, quantization="Q2_K_XL",
        n_experts_total=256, n_experts_used=8,
    )


def test_build_cmd_uses_depth_not_ctx_size():
    """llama-bench has no -c/--ctx-size flag; context is simulated via -d."""
    cmd = _build_cmd(_env_cfg(), _dense_model(), {"threads": 8, "flash_attn": True}, ctx=8192)
    assert "-d" in cmd
    assert cmd[cmd.index("-d") + 1] == "8192"
    assert "-c" not in cmd


def test_build_cmd_always_passes_fa_explicitly():
    """Newer llama-bench defaults to 'auto' (not off) when -fa is omitted."""
    cmd_on = _build_cmd(_env_cfg(), _dense_model(), {"flash_attn": True}, ctx=0)
    cmd_off = _build_cmd(_env_cfg(), _dense_model(), {"flash_attn": False}, ctx=0)
    assert cmd_on[cmd_on.index("-fa") + 1] == "1"
    assert cmd_off[cmd_off.index("-fa") + 1] == "0"


def test_build_cmd_respects_avail_flags():
    """A llama-bench build that lacks -fa/-ctk must not be passed those flags."""
    cmd = _build_cmd(_env_cfg(), _dense_model(), {"flash_attn": True, "ctk": "q8_0"}, ctx=0, avail_flags=frozenset())
    assert "-fa" not in cmd
    assert "-ctk" not in cmd
    assert "-ctv" not in cmd


def test_build_cmd_dense_uses_searched_ngl():
    cmd = _build_cmd(_env_cfg(), _dense_model(), {"ngl": 17, "threads": 8}, ctx=0)
    assert cmd[cmd.index("-ngl") + 1] == "17"


def test_build_cmd_moe_uses_ncmoe_not_experts():
    cmd = _build_cmd(_env_cfg(), _moe_model(), {"ncmoe": 12, "threads": 8}, ctx=0, avail_flags=_ALL_FLAGS)
    assert "-ncmoe" in cmd
    assert cmd[cmd.index("-ncmoe") + 1] == "12"


def test_build_cmd_no_dead_pg_flag():
    """The -pg 512,32 run was always discarded by parse_bench_output — should be gone."""
    cmd = _build_cmd(_env_cfg(), _dense_model(), {"threads": 8}, ctx=0)
    assert "-pg" not in cmd
