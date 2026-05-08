"""Module 2: GGUF Reader — reads model metadata without loading the model."""
import os
from pathlib import Path
from typing import Optional

from ggtune.models.model_profile import ModelProfile

MOE_ARCHITECTURES = {"deepseek2", "qwen_moe", "mixtral", "deepseek_v2", "deepseek_v3"}
MOE_KEYWORDS = ("moe", "mixture")


def _is_moe(arch: str, n_experts_total: Optional[int]) -> bool:
    if n_experts_total and n_experts_total > 0:
        return True
    arch_lower = arch.lower()
    if arch_lower in MOE_ARCHITECTURES:
        return True
    return any(kw in arch_lower for kw in MOE_KEYWORDS)


def _extract_quantization(reader) -> str:
    """Determine quantization from the first attention weight tensor."""
    try:
        from gguf import GGUFValueType
        quant_map = {
            0: "F32", 1: "F16",
            2: "Q4_0", 3: "Q4_1",
            6: "Q5_0", 7: "Q5_1",
            8: "Q8_0", 9: "Q8_1",
            10: "Q2_K", 11: "Q3_K",
            12: "Q4_K", 13: "Q5_K",
            14: "Q6_K", 15: "Q8_K",
            16: "IQ2_XXS", 17: "IQ2_XS",
            18: "IQ3_XXS", 19: "IQ1_S",
            20: "IQ4_NL", 21: "IQ3_S",
            22: "IQ2_S", 23: "IQ4_XS",
            24: "IQ1_M", 25: "BF16",
            26: "Q4_0_4_4", 27: "Q4_0_4_8", 28: "Q4_0_8_8",
        }
        for tensor in reader.tensors:
            name = tensor.name
            if "attn_q" in name or "attention.wq" in name:
                return quant_map.get(tensor.tensor_type, f"type{tensor.tensor_type}")
        # fallback: first tensor
        if reader.tensors:
            t = reader.tensors[0]
            return quant_map.get(t.tensor_type, f"type{t.tensor_type}")
    except Exception:
        pass
    return "unknown"


def read(model_path: str) -> ModelProfile:
    try:
        from gguf import GGUFReader
    except ImportError:
        raise RuntimeError("gguf package not installed. Run: pip install gguf")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    reader = GGUFReader(str(path))
    fields = {f.name: f for f in reader.fields.values()}

    def get_str(key: str) -> Optional[str]:
        f = fields.get(key)
        if f is None:
            return None
        try:
            parts = f.parts
            if parts:
                val = parts[-1]
                if hasattr(val, "tobytes"):
                    return val.tobytes().decode("utf-8", errors="replace")
                return str(val)
        except Exception:
            pass
        return None

    def get_int(key: str) -> Optional[int]:
        f = fields.get(key)
        if f is None:
            return None
        try:
            return int(f.parts[-1])
        except Exception:
            return None

    arch = get_str("general.architecture") or "unknown"
    name = get_str("general.name") or path.stem

    prefix = arch
    n_layers = get_int(f"{prefix}.block_count") or 32
    n_heads = get_int(f"{prefix}.attention.head_count") or 32
    context_length_max = get_int(f"{prefix}.context_length") or 4096
    n_experts_total = get_int(f"{prefix}.expert_count")
    n_experts_used = get_int(f"{prefix}.expert_used_count")

    is_moe = _is_moe(arch, n_experts_total)
    quantization = _extract_quantization(reader)
    file_size_gb = os.path.getsize(model_path) / (1024 ** 3)

    return ModelProfile(
        path=str(path.resolve()),
        name=name,
        architecture=arch,
        is_moe=is_moe,
        n_layers=n_layers,
        n_heads=n_heads,
        context_length_max=context_length_max,
        file_size_gb=file_size_gb,
        quantization=quantization,
        n_experts_total=n_experts_total,
        n_experts_used=n_experts_used,
    )
