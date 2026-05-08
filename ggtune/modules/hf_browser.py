"""Module 11: HuggingFace Model Browser."""
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

import requests
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, BarColumn, TextColumn

from ggtune.config import HF_API, RECOMMENDED_AUTHORS
from ggtune.models.hardware import HardwareProfile
from ggtune.models.hf_model import ModelRecommendation

console = Console()

QUANT_PRIORITY = {
    "Q8_0": 10, "Q6_K": 9,
    "Q5_K": 8, "Q5_K_M": 8, "Q5_K_S": 8,
    "Q4_K": 7, "Q4_K_M": 7, "Q4_K_S": 7,
    "Q4_0": 6,
    "Q3_K": 5, "Q3_K_M": 5,
    "Q2_K": 4, "Q2_K_XL": 4,
    "IQ4_XS": 7, "IQ3_XXS": 5, "IQ2_XXS": 3,
    "UD-Q4_K_XL": 8, "UD-Q2_K": 4,
}


def extract_quantization(filename: str) -> str:
    patterns = [
        r"UD-Q\d+_K(?:_[A-Z]+)?",
        r"Q\d+_K(?:_[A-Z]+)?",
        r"Q\d+_\d+",
        r"Q\d+[A-Z]",
        r"IQ\d+_[A-Z]+",
        r"F16", r"F32", r"BF16",
    ]
    name = filename.upper()
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return m.group(0)
    return "unknown"


def _quant_score(quant: str, size_gb: float, vram_gb: float, fits_vram: bool) -> float:
    priority = QUANT_PRIORITY.get(quant, 5)
    vram_bonus = 2.0 if fits_vram else 0.0
    size_penalty = max(0, size_gb - vram_gb) * 0.1
    return priority + vram_bonus - size_penalty


def _fetch_models(author: str, limit: int = 30) -> list:
    try:
        resp = requests.get(
            f"{HF_API}/models",
            params={"author": author, "filter": "gguf", "sort": "downloads",
                    "direction": -1, "limit": limit},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def _fetch_files(model_id: str) -> list:
    try:
        resp = requests.get(f"{HF_API}/models/{model_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("siblings", [])
    except Exception:
        pass
    return []


def recommend(hw: HardwareProfile, author: str = "unsloth", vram_gb: Optional[float] = None) -> List[ModelRecommendation]:
    vram = vram_gb or hw.vram_free_mb / 1024

    models = _fetch_models(author)
    if not models:
        return []

    recommendations: List[ModelRecommendation] = []

    for model in models:
        model_id = model.get("modelId") or model.get("id", "")
        files = _fetch_files(model_id)

        for f in files:
            fname = f.get("rfilename", "")
            if not fname.endswith(".gguf") or "mmproj" in fname.lower():
                continue

            size_bytes = f.get("size", 0)
            size_gb = size_bytes / 1e9
            if size_gb == 0:
                continue
            if size_gb > hw.ram_available_gb * 0.85:
                continue

            quant = extract_quantization(fname)
            fits_vram = size_gb < vram * 0.9
            score = _quant_score(quant, size_gb, vram, fits_vram)

            recommendations.append(ModelRecommendation(
                model_id=model_id,
                filename=fname,
                size_gb=size_gb,
                quantization=quant,
                fits_vram=fits_vram,
                score=score,
                hf_url=f"https://huggingface.co/{model_id}/blob/main/{fname}",
                download_cmd=f"huggingface-cli download {model_id} {fname}",
            ))

    return sorted(recommendations, key=lambda x: x.score, reverse=True)[:10]


def print_table(recs: List[ModelRecommendation], hw: HardwareProfile) -> None:
    table = Table(
        title=f"Models for {hw.gpu_name} {hw.vram_total_mb // 1024}GB + {hw.ram_total_gb:.0f}GB RAM",
        box=box.ROUNDED,
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", min_width=35)
    table.add_column("Size", justify="right")
    table.add_column("VRAM", justify="center")
    table.add_column("Quant")

    for i, r in enumerate(recs, 1):
        vram_sym = "[green]✓[/]" if r.fits_vram else "[yellow]part[/]"
        table.add_row(
            str(i),
            f"{r.model_id}  [dim]{r.filename[:30]}[/]",
            f"{r.size_gb:.1f} GB",
            vram_sym,
            r.quantization,
        )

    console.print(table)


def download(rec: ModelRecommendation, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / rec.filename

    if shutil.which("huggingface-cli"):
        subprocess.run(
            ["huggingface-cli", "download", rec.model_id, rec.filename,
             "--local-dir", str(dest_dir)],
            check=True,
        )
    else:
        url = f"https://huggingface.co/{rec.model_id}/resolve/main/{rec.filename}"
        _download_with_progress(url, dest)

    return dest


def _download_with_progress(url: str, dest: Path) -> None:
    import requests as req

    with req.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(), DownloadColumn(), TransferSpeedColumn(),
        ) as p:
            task = p.add_task(f"Downloading {dest.name}", total=total)
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    p.advance(task, len(chunk))


def interactive_browse(hw: HardwareProfile, author: str = "unsloth", vram_gb: Optional[float] = None) -> Optional[Path]:
    console.print(f"[dim]Fetching models from {author}...[/]")
    recs = recommend(hw, author, vram_gb)

    if not recs:
        console.print("[red]No models found.[/]")
        return None

    print_table(recs, hw)
    console.print("\n[dim]Enter number to download, or [q] to quit[/]")

    choice = input("> ").strip().lower()
    if choice == "q" or not choice:
        return None

    try:
        idx = int(choice) - 1
        rec = recs[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice.[/]")
        return None

    dest_dir = Path.home() / "models"
    dest = download(rec, dest_dir)
    console.print(f"[green]✓ Downloaded to {dest}[/]")
    return dest
