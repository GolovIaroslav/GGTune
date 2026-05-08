from dataclasses import dataclass


@dataclass
class ModelRecommendation:
    model_id: str
    filename: str
    size_gb: float
    quantization: str
    fits_vram: bool
    score: float
    hf_url: str
    download_cmd: str
