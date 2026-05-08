from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelRecommendation:
    model_id: str
    filename: str
    size_gb: float
    quantization: str
    fits_vram: bool       # fits fully in VRAM
    fits_vram_ncmoe: bool # fits with ncmoe (MoE only)
    min_vram_gb: float    # estimated minimum VRAM needed
    is_moe: bool
    score: float
    hf_url: str
    download_cmd: str
