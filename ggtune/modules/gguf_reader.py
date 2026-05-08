"""Module 2: GGUF Reader — reads model metadata without loading the model."""
import os
import re
from pathlib import Path
from typing import Optional

from ggtune.models.model_profile import ModelProfile

MOE_ARCHITECTURES = {"deepseek2", "qwen_moe", "mixtral", "deepseek_v2", "deepseek_v3"}
MOE_KEYWORDS = ("moe", "mixture")

# Matches quant names in filenames: Q2_K, Q4_K_M, IQ3_XXS, IQ4_XS, BF16, Q2_K_XL …
_FILENAME_QUANT_RE = re.compile(
    r'(?<![A-Za-z])'
    r'((?:IQ\d_[A-Z0-9]+|Q\d_[A-Z0-9]+(?:_[A-Z0-9]+)*|BF16|F16|F32))'
    r'(?![A-Za-z0-9])',
    re.IGNORECASE,
)


def _is_moe(arch: str, n_experts_total: Optional[int]) -> bool:
    if n_experts_total and n_experts_total > 0:
        return True
    arch_lower = arch.lower()
    if arch_lower in MOE_ARCHITECTURES:
        return True
    return any(kw in arch_lower for kw in MOE_KEYWORDS)


_QUANT_MAP = {
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

# High-precision types used for output/embed layers even in "low-quant" models
_HIGH_PREC = {"F32", "F16", "BF16", "Q8_0", "Q8_K", "Q6_K"}


def _quant_from_filename(stem: str) -> Optional[str]:
    """Parse quantization name from model filename (most reliable for Unsloth/bartowski)."""
    matches = _FILENAME_QUANT_RE.findall(stem)
    if matches:
        return matches[-1].upper()
    return None


def _extract_quantization(reader, filename_stem: str = "") -> str:
    """Determine the dominant quantization of the model.

    Priority: filename → dominant bulk tensor type (ignoring high-prec outliers).
    Unsloth UD models keep output/embed/attention at Q5_K or Q8_0 while the bulk
    FFN experts are Q2_K — the filename is the authoritative source in that case.
    """
    if filename_stem:
        q = _quant_from_filename(filename_stem)
        if q:
            return q

    try:
        from collections import Counter
        counts: Counter = Counter()
        for tensor in reader.tensors:
            qt = _QUANT_MAP.get(tensor.tensor_type, f"type{tensor.tensor_type}")
            counts[qt] += 1

        if not counts:
            return "unknown"

        # Most common type overall
        dominant = counts.most_common(1)[0][0]

        # If dominant is high-prec (embed/output layers inflate it), find the most
        # common non-high-prec type — that reflects the actual bulk quantization.
        if dominant in _HIGH_PREC:
            bulk = [(q, n) for q, n in counts.most_common() if q not in _HIGH_PREC]
            if bulk:
                dominant = bulk[0][0]

        return dominant
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
            val = f.parts[-1]
            # numpy memmap or array
            if hasattr(val, "flat"):
                return int(val.flat[0])
            return int(val)
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
    n_kv_heads = get_int(f"{prefix}.attention.head_count_kv")

    is_moe = _is_moe(arch, n_experts_total)
    quantization = _extract_quantization(reader, path.stem)
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
        n_kv_heads=n_kv_heads,
    )
