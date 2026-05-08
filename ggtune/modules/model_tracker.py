"""Tracks downloaded GGUF models in ~/.llamatune/models.json."""
import json
from datetime import date
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

from ggtune.config import MODELS_INDEX


@dataclass
class TrackedModel:
    path: str
    filename: str
    size_gb: float
    source: str
    downloaded_at: str


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


def register(path: Path, source: str) -> None:
    entries = _load()
    p = str(path.resolve())
    entries = [e for e in entries if e["path"] != p]
    size_gb = path.stat().st_size / 1e9 if path.exists() else 0.0
    entries.append({
        "path": p,
        "filename": path.name,
        "size_gb": round(size_gb, 2),
        "source": source,
        "downloaded_at": str(date.today()),
    })
    _save(entries)


def list_all() -> List[TrackedModel]:
    entries = _load()
    result = []
    dirty = False
    for e in entries:
        if Path(e["path"]).exists():
            result.append(TrackedModel(**e))
        else:
            dirty = True
    if dirty:
        _save([asdict(m) for m in result])
    return result


def remove(path: str) -> bool:
    p = Path(path)
    deleted = False
    if p.exists():
        p.unlink()
        deleted = True
    entries = [e for e in _load() if e["path"] != str(p.resolve())]
    _save(entries)
    return deleted
