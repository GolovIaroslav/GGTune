"""Module 11: HuggingFace Model Browser."""
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, BarColumn, TextColumn

from ggtune.config import HF_API, RECOMMENDED_AUTHORS, MODELS_DIR
from ggtune.models.hardware import HardwareProfile
from ggtune.models.hf_model import ModelRecommendation

console = Console()

QUANT_PRIORITY = {
    "Q8_0": 10, "Q6_K": 9,
    "Q5_K_M": 8, "Q5_K_S": 8, "Q5_K": 8,
    "Q4_K_M": 7, "Q4_K_S": 7, "Q4_K": 7, "IQ4_XS": 7,
    "Q4_0": 6,
    "Q3_K_M": 5, "Q3_K": 5, "IQ3_XXS": 5,
    "Q2_K_XL": 4, "Q2_K": 4, "UD-Q2_K": 4, "IQ2_XXS": 3,
    "UD-Q4_K_XL": 8, "UD-Q5_K_XL": 9,
}


def extract_quantization(filename: str) -> str:
    patterns = [
        r"UD-Q\d+_K(?:_[A-Z]+)?",
        r"Q\d+_K(?:_[A-Z]+)?",
        r"Q\d+_\d+",
        r"IQ\d+_[A-Z]+",
        r"Q\d+[A-Z]",
        r"BF16", r"F16", r"F32",
    ]
    name = filename.upper()
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return m.group(0)
    return "unknown"


def _parse_moe_info(model_id: str, tags: list) -> Tuple[bool, float]:
    """
    Returns (is_moe, active_fraction).
    active_fraction = active_params / total_params, used to estimate min VRAM.

    Examples: "35B-A3B" → 3/35 ≈ 0.086
              "671B-A37B" → 37/671 ≈ 0.055
              "22B-A3B" → 3/22 ≈ 0.136
    """
    name = model_id.upper()
    tag_set = {t.lower() for t in tags}

    m = re.search(r'(\d+(?:\.\d+)?)B-A(\d+(?:\.\d+)?)B', name)
    if m:
        total = float(m.group(1))
        active = float(m.group(2))
        return True, active / total

    is_moe = "moe" in tag_set or "mixture-of-experts" in tag_set
    # MoE but can't determine ratio — assume 25% active (conservative)
    return is_moe, 0.25


def _min_vram_gb(size_gb: float, active_fraction: float) -> float:
    """
    Estimate minimum VRAM needed with ncmoe = n_experts_used.

    MoE model structure:
      ~30% non-expert weights (attention, norms) → always in VRAM
      ~70% expert weights → only active fraction in VRAM with ncmoe

    Formula: size * (0.30 + 0.70 * active_fraction) * 1.15 overhead
    """
    return size_gb * (0.30 + 0.70 * active_fraction) * 1.15


def _score(quant: str, size_gb: float, vram_gb: float, fits_vram: bool, fits_ncmoe: bool) -> float:
    priority = QUANT_PRIORITY.get(quant, 5)
    vram_bonus = 2.0 if fits_vram else (1.0 if fits_ncmoe else 0.0)
    size_penalty = max(0, size_gb - vram_gb) * 0.05
    return priority + vram_bonus - size_penalty


def _fetch_models(author: str, limit: int = 50) -> list:
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
        resp = requests.get(f"{HF_API}/models/{model_id}", params={"blobs": "true"}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("siblings", [])
    except Exception:
        pass
    return []


def recommend(hw: HardwareProfile, author: str = "unsloth", vram_gb: Optional[float] = None) -> List[ModelRecommendation]:
    # use total VRAM for browse — free VRAM changes constantly and misleads filtering
    vram = vram_gb or hw.vram_total_mb / 1024

    console.print("[dim]Fetching model list from HuggingFace — this may take 10–20 seconds...[/]")
    models = _fetch_models(author)
    if not models:
        return []

    recommendations: List[ModelRecommendation] = []

    for model in models:
        model_id = model.get("modelId") or model.get("id", "")
        tags = model.get("tags", [])
        dl_count = model.get("downloads", 0) or 0
        is_moe, active_fraction = _parse_moe_info(model_id, tags)

        files = _fetch_files(model_id)

        for f in files:
            fname = f.get("rfilename", "")
            if not fname.endswith(".gguf") or "mmproj" in fname.lower():
                continue
            if re.search(r"-\d{5}-of-\d{5}\.gguf$", fname):
                continue

            size_bytes = f.get("size") or f.get("lfs", {}).get("size", 0)
            size_gb = (size_bytes or 0) / 1e9
            if size_gb == 0:
                continue
            if size_gb > hw.ram_total_gb * 0.85:
                continue

            quant = extract_quantization(fname)
            fits_vram = size_gb < vram * 0.9

            if is_moe:
                min_vram = _min_vram_gb(size_gb, active_fraction)
                fits_ncmoe = (not fits_vram) and min_vram < vram * 0.9
            else:
                min_vram = size_gb
                fits_ncmoe = False

            if not fits_vram and not fits_ncmoe:
                continue

            recommendations.append(ModelRecommendation(
                model_id=model_id,
                filename=fname,
                size_gb=size_gb,
                quantization=quant,
                fits_vram=fits_vram,
                fits_vram_ncmoe=fits_ncmoe,
                min_vram_gb=min_vram,
                is_moe=is_moe,
                score=_score(quant, size_gb, vram, fits_vram, fits_ncmoe),
                hf_url=f"https://huggingface.co/{model_id}/blob/main/{fname}",
                download_cmd=f"huggingface-cli download {model_id} {fname}",
                downloads=dl_count,
            ))

    # Deduplicate: one entry per model_id — keep best score per group
    def _best_per_model(recs: List[ModelRecommendation]) -> List[ModelRecommendation]:
        best: dict = {}
        for r in recs:
            prev = best.get(r.model_id)
            if prev is None or r.score > prev.score:
                best[r.model_id] = r
        return sorted(best.values(), key=lambda x: (x.downloads, x.score), reverse=True)

    full_fit = _best_per_model([r for r in recommendations if r.fits_vram])[:10]
    ncmoe_fit = _best_per_model([r for r in recommendations if r.fits_vram_ncmoe])[:10]
    return full_fit + ncmoe_fit


def print_table(recs: List[ModelRecommendation], hw: HardwareProfile) -> None:
    title = f"Models for {hw.gpu_name} {hw.vram_total_mb // 1024}GB + {hw.ram_total_gb:.0f}GB RAM"

    full = [r for r in recs if r.fits_vram]
    ncmoe = [r for r in recs if r.fits_vram_ncmoe]

    def _fmt_dl(n: int) -> str:
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n//1_000}K"
        return str(n)

    def _make_table() -> Table:
        t = Table(title=title, box=box.ROUNDED, show_header=True)
        t.add_column("#", style="dim", justify="right")
        t.add_column("Model", min_width=24)
        t.add_column("File", min_width=20)
        t.add_column("Size", justify="right")
        t.add_column("VRAM", justify="center")
        t.add_column("Quant")
        t.add_column("↓ DLs", justify="right", style="dim")
        return t

    table = _make_table()
    idx = 1

    if full:
        table.add_section()
        for r in full:
            table.add_row(
                str(idx), r.model_id, r.filename, f"{r.size_gb:.1f} GB",
                "[green]✓ full[/]", r.quantization, _fmt_dl(r.downloads),
            )
            idx += 1

    if ncmoe:
        table.add_section()
        for r in ncmoe:
            table.add_row(
                str(idx), f"{r.model_id} [dim](MoE)[/]", r.filename,
                f"{r.size_gb:.1f} GB",
                f"[yellow]ncmoe ~{r.min_vram_gb:.1f}GB[/]",
                r.quantization, _fmt_dl(r.downloads),
            )
            idx += 1

    console.print(table)
    console.print(
        "[dim]✓ full = fits in VRAM entirely  "
        "| [yellow]ncmoe[/dim] [dim]= MoE model, only active experts in VRAM  "
        "| sorted by popularity (↓ DLs)[/]\n"
    )


def download(rec: ModelRecommendation, dest_dir: Optional[Path] = None) -> Path:
    dest_dir = dest_dir or MODELS_DIR
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

    from ggtune.modules.model_tracker import register
    register(dest, rec.model_id)
    return dest


def parse_model_input(text: str) -> Optional[str]:
    """Parse 'author/model' or HF URL → model_id, or None if invalid."""
    text = text.strip()
    if text.startswith("https://huggingface.co/"):
        parts = text.removeprefix("https://huggingface.co/").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    if "/" in text and len(text.split("/")) == 2:
        a, m = text.split("/", 1)
        if a and m:
            return text
    return None


def fetch_gguf_files(model_id: str) -> tuple:
    """Return (main_files, mmproj_files) — each is list of {filename, size_gb}."""
    files = _fetch_files(model_id)
    main, mmproj = [], []
    for f in files:
        fname = f.get("rfilename", "")
        if not fname.lower().endswith(".gguf"):
            continue
        size_bytes = f.get("size") or f.get("lfs", {}).get("size", 0) or 0
        entry = {"filename": fname, "size_gb": size_bytes / 1e9}
        if "mmproj" in fname.lower():
            mmproj.append(entry)
        else:
            main.append(entry)
    return main, mmproj


def download_by_id(model_id: str, filename: str) -> Path:
    """Download a specific file from a model repo to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # filename may include a subpath (e.g. "gguf/file.gguf") — flatten to MODELS_DIR root
    dest = MODELS_DIR / Path(filename).name

    if shutil.which("huggingface-cli"):
        # huggingface-cli respects the subpath inside the repo
        result = subprocess.run(
            ["huggingface-cli", "download", model_id, filename,
             "--local-dir", str(MODELS_DIR)],
            check=True,
        )
        # huggingface-cli may place file in a subdir — move to root if needed
        hf_dest = MODELS_DIR / filename
        if hf_dest.exists() and hf_dest != dest:
            hf_dest.rename(dest)
    else:
        url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download_with_progress(url, dest)

    from ggtune.modules.model_tracker import register
    register(dest, model_id)
    return dest


def _download_with_progress(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=30) as resp:
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
    recs = recommend(hw, author, vram_gb)

    if not recs:
        console.print("[red]No models found.[/]")
        return None

    print_table(recs, hw)
    console.print("[dim]Enter number to download, or [b] to go back[/]")

    choice = input("> ").strip().lower()
    if choice in ("b", "q", ""):
        return None

    try:
        idx = int(choice) - 1
        rec = recs[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice.[/]")
        return None

    dest = download(rec)
    console.print(f"[green]✓ Downloaded to {dest}[/]")
    return dest
