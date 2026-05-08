"""Tests for GGUF reader — using a real tiny model if available, else mock."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


TINY_GGUF = Path(__file__).parent / "fixtures" / "tiny_model.gguf"


def _make_mock_reader(arch="llama", n_layers=32, n_heads=32, ctx=4096,
                      n_experts=None, n_experts_used=None):
    def make_field(val):
        f = MagicMock()
        if isinstance(val, int):
            f.parts = [val]
        else:
            encoded = MagicMock()
            encoded.tobytes.return_value = val.encode()
            f.parts = [encoded]
        return f

    fields = {
        "general.architecture": make_field(arch),
        "general.name": make_field("test-model"),
        f"{arch}.block_count": make_field(n_layers),
        f"{arch}.attention.head_count": make_field(n_heads),
        f"{arch}.context_length": make_field(ctx),
    }
    if n_experts:
        fields[f"{arch}.expert_count"] = make_field(n_experts)
    if n_experts_used:
        fields[f"{arch}.expert_used_count"] = make_field(n_experts_used)

    reader = MagicMock()
    reader.fields.values.return_value = [
        MagicMock(name=k, **{"name": k}) for k in fields
    ]
    # Override __iter__ via dict
    reader.fields = {k: v for k, v in fields.items()}
    reader.tensors = []
    return reader


def test_moe_detection():
    from ggtune.modules.gguf_reader import _is_moe
    assert _is_moe("qwen_moe", 128) is True
    assert _is_moe("llama", None) is False
    assert _is_moe("mixtral", 8) is True
    assert _is_moe("deepseek2", 64) is True


def test_quantization_unknown_with_no_tensors():
    from ggtune.modules.gguf_reader import _extract_quantization
    reader = MagicMock()
    reader.tensors = []
    result = _extract_quantization(reader)
    assert result == "unknown"
