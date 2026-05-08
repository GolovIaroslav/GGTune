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

    def estimated_quick_probe_runs(self) -> int:
        ncmoe_vals = 3 if self.ncmoe_range else 1
        return ncmoe_vals * len(self.thread_candidates) * len(self.kv_quant_options)
