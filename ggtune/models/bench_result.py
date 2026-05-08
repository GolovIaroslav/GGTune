from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchResult:
    params: dict = field(default_factory=dict)
    tg_tokens_per_sec: float = 0.0
    pp_tokens_per_sec: float = 0.0
    context: int = 0
    crashed: bool = False
    error: Optional[str] = None
    duration_sec: float = 0.0

    @property
    def valid(self) -> bool:
        return not self.crashed and self.tg_tokens_per_sec > 0
