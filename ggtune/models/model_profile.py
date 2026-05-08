from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelProfile:
    path: str
    name: str
    architecture: str
    is_moe: bool
    n_layers: int
    n_heads: int
    context_length_max: int
    file_size_gb: float
    quantization: str
    n_experts_total: Optional[int] = None
    n_experts_used: Optional[int] = None
    n_kv_heads: Optional[int] = None

    def __str__(self) -> str:
        moe = f" MoE {self.n_experts_total}e" if self.is_moe else ""
        return f"{self.name} ({self.quantization}{moe}, {self.file_size_gb:.1f}GB)"
