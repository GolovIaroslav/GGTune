"""Tracks downloaded and locally-found GGUF models."""
import json
from datetime import date
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict, field

from ggtune.config import MODELS_INDEX, SCAN_DIRS_FILE


@dataclass
class TrackedModel:
    path: str
    filename: str
    size_gb: float
    source: str
    downloaded_at: str
    mmproj_path: Optional[str] = None
    source_type: str = "downloaded"  # "downloaded" | "local"


def _load() -> List[dict]:
    if not MODELS_INDEX.exists():
        return []
    try:
        return json.loads(MODELS_INDEX.read_text())
    except Exception:
        return []


def _save(entries: List[dict]) -> None:
    MODELS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    MODELS_INDEX.write_text(json.dumps(entries, indent=2))


def register(
    path: Path,
    source: str,
    mmproj_path: Optional[Path] = None,
    source_type: str = "downloaded",
) -> None:
    entries = _load()
    p = str(path.resolve())
    entries = [e for e in entries if e["path"] != p]
    size_gb = path.stat().st_size / 1e9 if path.exists() else 0.0
    entry: dict = {
        "path": p,
        "filename": path.name,
        "size_gb": round(size_gb, 2),
        "source": source,
        "downloaded_at": str(date.today()),
        "mmproj_path": str(mmproj_path.resolve()) if mmproj_path else None,
        "source_type": source_type,
    }
    entries.append(entry)
    _save(entries)


def set_mmproj(model_path: str, mmproj_path: Optional[str]) -> None:
    entries = _load()
    for e in entries:
        if e["path"] == model_path:
            e["mmproj_path"] = mmproj_path
    _save(entries)


def list_all() -> List[TrackedModel]:
    entries = _load()
    result = []
    dirty = False
    for e in entries:
        if Path(e["path"]).exists():
            result.append(TrackedModel(
                path=e["path"],
                filename=e["filename"],
                size_gb=e.get("size_gb", 0.0),
                source=e.get("source", ""),
                downloaded_at=e.get("downloaded_at", ""),
                mmproj_path=e.get("mmproj_path"),
                source_type=e.get("source_type", "downloaded"),
            ))
        else:
            dirty = True
    if dirty:
        _save([asdict(m) for m in result])
    return result


def find_by_path(model_path: str) -> Optional[TrackedModel]:
    for m in list_all():
        if m.path == model_path:
            return m
    return None


def remove(path: str, delete_file: bool = True) -> bool:
    p = Path(path)
    deleted = False
    if delete_file and p.exists():
        p.unlink()
        deleted = True
    entries = [e for e in _load() if e["path"] != str(p.resolve())]
    _save(entries)
    return deleted


# ── Scan dirs ─────────────────────────────────────────────────────────────

def load_scan_dirs() -> List[str]:
    if not SCAN_DIRS_FILE.exists():
        return []
    try:
        return json.loads(SCAN_DIRS_FILE.read_text())
    except Exception:
        return []


def add_scan_dir(path: str) -> None:
    dirs = load_scan_dirs()
    p = str(Path(path).expanduser().resolve())
    if p not in dirs:
        dirs.append(p)
    SCAN_DIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_DIRS_FILE.write_text(json.dumps(dirs, indent=2))


def remove_scan_dir(path: str) -> None:
    dirs = [d for d in load_scan_dirs() if d != path]
    SCAN_DIRS_FILE.write_text(json.dumps(dirs, indent=2))


def rescan_dirs() -> int:
    """Scan all custom dirs for .gguf files and register new ones. Returns count of new files found."""
    dirs = load_scan_dirs()
    existing = {m.path for m in list_all()}
    found = 0
    for d in dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        for f in dp.rglob("*.gguf"):
            if not f.is_file() or f.stat().st_size < 50 * 1024 * 1024:
                continue
            if str(f.resolve()) in existing:
                continue
            # Auto-detect mmproj in same directory
            mmproj = _find_mmproj_near(f)
            register(f, source=d, mmproj_path=mmproj, source_type="local")
            existing.add(str(f.resolve()))
            found += 1
    return found


def _find_mmproj_near(model_path: Path) -> Optional[Path]:
    """Look for mmproj .gguf files in the same directory."""
    for f in model_path.parent.glob("mmproj*.gguf"):
        return f
    for f in model_path.parent.glob("*mmproj*.gguf"):
        return f
    return None
