from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SearchSpace:
    ncmoe_range: Optional[range] = None
    ngl_range: Optional[range] = None
    thread_candidates: List[int] = field(default_factory=list)
    kv_quant_options: List[str] = field(default_factory=list)
    nkvo_options: List[bool] = field(default_factory=list)
    context_candidates: List[int] = field(default_factory=list)
    flash_attn: bool = True

    def total_combinations(self) -> int:
        """Total parameter combinations in Optuna search space."""
        ncmoe_count = len(list(self.ncmoe_range)) if self.ncmoe_range else 1
        return ncmoe_count * len(self.thread_candidates) * len(self.kv_quant_options) * len(self.nkvo_options)

    def estimated_quick_probe_runs(self) -> int:
        """Matches the actual [:2] slicing done in quick_probe()."""
        ncmoe_vals = 3 if self.ncmoe_range else 1
        threads = min(2, len(self.thread_candidates))
        kv_quants = min(2, len(self.kv_quant_options))
        return ncmoe_vals * threads * kv_quants
