from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelRecommendation:
    model_id: str
    filename: str
    size_gb: float
    quantization: str
    fits_vram: bool
    fits_vram_ncmoe: bool
    min_vram_gb: float
    is_moe: bool
    score: float
    hf_url: str
    download_cmd: str
    downloads: int = 0
