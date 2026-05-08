"""Module 3+9: Environment Manager — detects, installs, and configures llama.cpp."""
import io
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

from ggtune.config import (
    LLAMA_CPP_PINNED_BUILD, LLAMA_CPP_REPO,
    ENV_FILE, CONFIG_DIR, LLAMA_INSTALL_DIR,
)
from ggtune.models.hardware import Backend, HardwareProfile
from ggtune.utils.shell import make_env_with_lib


_SYSTEM = platform.system()   # "Linux", "Darwin", "Windows"


def _exe(name: str) -> str:
    return f"{name}.exe" if _SYSTEM == "Windows" else name


def _get_platform_search_paths() -> list[Path]:
    """Well-known llama.cpp binary locations per OS."""
    home = Path.home()
    paths: list[Path] = [LLAMA_INSTALL_DIR / "build" / "bin"]

    if _SYSTEM == "Windows":
        local_app = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
        prog = Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        prog86 = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
        paths += [
            local_app / "llama.cpp" / "build" / "bin" / "Release",
            local_app / "llama.cpp" / "build" / "Release",
            local_app / "llama.cpp" / "build" / "bin",
            prog / "llama.cpp" / "build" / "bin",
            prog86 / "llama.cpp" / "build" / "bin",
            home / "llama.cpp" / "build" / "bin" / "Release",
            home / "llama.cpp" / "build" / "Release",
            home / "llama.cpp" / "build" / "bin",
            # Pre-built extract location
            LLAMA_INSTALL_DIR.parent / "llama.cpp.prebuilt",
        ]
    elif _SYSTEM == "Darwin":
        paths += [
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
            Path("/usr/local/opt/llama.cpp/bin"),
            home / ".local" / "bin",
            home / "llama.cpp" / "build" / "bin",
            home / ".local" / "llama.cpp" / "build" / "bin",
        ]
    else:  # Linux
        paths += [
            home / ".local" / "bin",
            home / ".local" / "llama.cpp" / "build" / "bin",
            home / "llama.cpp" / "build" / "bin",
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/opt/llama.cpp/build/bin"),
            Path("/snap/bin"),
        ]

    return paths


def _find_via_system_search(name: str) -> Optional[Path]:
    """Deep OS-level search (find / where / PowerShell). Last resort."""
    exe_name = _exe(name)
    home = Path.home()

    if _SYSTEM == "Windows":
        try:
            r = subprocess.run(["where", exe_name], capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines():
                p = Path(line.strip())
                if p.exists():
                    return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        for root in filter(None, [
            str(home),
            os.environ.get("LOCALAPPDATA", ""),
            os.environ.get("PROGRAMFILES", ""),
        ]):
            try:
                cmd = (
                    f'Get-ChildItem -Path "{root}" -Recurse '
                    f'-Filter "{exe_name}" -ErrorAction SilentlyContinue '
                    f'| Select-Object -First 1 -ExpandProperty FullName'
                )
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    capture_output=True, text=True, timeout=30,
                )
                line = r.stdout.strip()
                if line:
                    p = Path(line)
                    if p.exists():
                        return p
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None

    search_roots = [str(home), "/usr/local", "/opt"]
    if _SYSTEM == "Darwin":
        search_roots.append("/opt/homebrew")
    try:
        r = subprocess.run(
            ["find"] + search_roots + ["-name", exe_name, "-type", "f"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=30,
        )
        for line in r.stdout.splitlines():
            p = Path(line.strip())
            if p.exists() and os.access(str(p), os.X_OK):
                return p
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _locate_binary(name: str) -> Optional[Path]:
    """Use `locate` DB (Linux/macOS only)."""
    if _SYSTEM == "Windows":
        return None
    exe_name = _exe(name)
    try:
        r = subprocess.run(
            ["locate", "-r", f"/{exe_name}$"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            p = Path(line.strip())
            if p.name == exe_name and p.exists() and os.access(str(p), os.X_OK):
                return p
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _cached_bin_dir() -> Optional[Path]:
    data = _load_env_json()
    if data and "bin_dir" in data:
        p = Path(data["bin_dir"])
        if (p / _exe("llama-bench")).exists():
            return p
    return None


def _find_bench_with_via() -> tuple[Optional[Path], str]:
    """Find llama-bench, return (path, how_it_was_found)."""
    exe_name = _exe("llama-bench")

    p = _cached_bin_dir()
    if p:
        return p / exe_name, "cache"

    for base in _get_platform_search_paths():
        candidate = base / exe_name
        if candidate.exists():
            return candidate, "search paths"

    found = shutil.which("llama-bench")
    if found:
        return Path(found), "PATH"

    p = _locate_binary("llama-bench")
    if p:
        return p, "locate"

    p = _find_via_system_search("llama-bench")
    if p:
        return p, "find"

    return None, ""


@dataclass
class EnvConfig:
    llama_bench_path: str
    llama_cli_path: str
    llama_server_path: str
    bin_dir: str
    build: str
    backend: Backend
    env_dict: dict
    found_via: str = "unknown"


@dataclass
class LlamaInstall:
    bin_dir: str
    build: str
    backend: Backend
    found_via: str


def _get_build_version(llama_cli: Path, env: dict) -> Optional[str]:
    try:
        r = subprocess.run(
            [str(llama_cli), "--version"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        combined = r.stdout + r.stderr
        for token in combined.split():
            if token.startswith("b") and token[1:].isdigit():
                return token
        m = re.search(r'(?:version|build)[:\s]+(\d{3,6})', combined, re.IGNORECASE)
        if m:
            return f"b{m.group(1)}"
        m = re.search(r'\b(\d{4,5})\b', combined)
        if m:
            return f"b{m.group(1)}"
    except Exception:
        pass
    return None


def _has_cuda(llama_cli: Path, env: dict) -> bool:
    try:
        r = subprocess.run(
            [str(llama_cli), "--list-devices"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        return "CUDA" in r.stdout or "CUDA" in r.stderr
    except Exception:
        return False


def _load_env_json() -> Optional[dict]:
    if ENV_FILE.exists():
        try:
            return json.loads(ENV_FILE.read_text())
        except Exception:
            pass
    return None


def _save_env_json(bin_dir: Path, build: str, backend: Backend) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_env_json() or {}
    # preserve previous for rollback
    if "bin_dir" in data and str(data["bin_dir"]) != str(bin_dir):
        data["previous_bin_dir"] = data["bin_dir"]
        data["previous_build"] = data.get("build", "unknown")
    data["build"] = build
    data["bin_dir"] = str(bin_dir)
    data["backend"] = backend.value
    ENV_FILE.write_text(json.dumps(data, indent=2))


def clear_cache() -> None:
    """Remove cached llama.cpp path so next detect() does a fresh search."""
    if ENV_FILE.exists():
        try:
            data = json.loads(ENV_FILE.read_text())
            data.pop("bin_dir", None)
            ENV_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


def detect(hw: Optional[HardwareProfile] = None) -> EnvConfig:
    """Find llama.cpp binaries; cache discovered path for future calls."""
    bench, found_via = _find_bench_with_via()

    if not bench:
        raise RuntimeError(
            "llama.cpp not found. Run: ggtune update  (auto-installs with CUDA)\n"
            "Or build manually: cmake -B build -DGGML_CUDA=ON && cmake --build build"
        )

    bin_dir = bench.parent
    cli = bin_dir / _exe("llama-cli")
    server = bin_dir / _exe("llama-server")

    if not cli.exists():
        raise RuntimeError(
            f"llama-bench found at {bench} but llama-cli is missing in the same directory."
        )

    env = make_env_with_lib(str(bin_dir))
    build = _get_build_version(cli, env) or "unknown"

    backend = Backend.CPU
    if hw:
        backend = hw.backend
    elif _has_cuda(cli, env):
        backend = Backend.CUDA

    if hw and hw.backend == Backend.CUDA and not _has_cuda(cli, env):
        from ggtune.utils.formatting import warn
        warn(
            "GPU found but llama.cpp has no CUDA support.\n"
            "  Fix: run [bold]ggtune update[/] to rebuild with -DGGML_CUDA=ON"
        )

    if found_via != "cache":
        _save_env_json(bin_dir, build, backend)

    return EnvConfig(
        llama_bench_path=str(bench),
        llama_cli_path=str(cli),
        llama_server_path=str(server) if server.exists() else str(cli),
        bin_dir=str(bin_dir),
        build=build,
        backend=backend,
        env_dict=env,
        found_via=found_via,
    )


# ── Installation scan ──────────────────────────────────────────────────────

def _probe_install(base: Path, hw: Optional[HardwareProfile], via: str) -> Optional[LlamaInstall]:
    """Check if a directory has a valid llama.cpp install and return LlamaInstall."""
    bench = base / _exe("llama-bench")
    cli = base / _exe("llama-cli")
    if not bench.exists() or not cli.exists():
        return None
    env = make_env_with_lib(str(base))
    build = _get_build_version(cli, env) or "unknown"
    if hw:
        backend = hw.backend
    elif _has_cuda(cli, env):
        backend = Backend.CUDA
    else:
        backend = Backend.CPU
    return LlamaInstall(bin_dir=str(base), build=build, backend=backend, found_via=via)


def scan_all(hw: Optional[HardwareProfile] = None) -> list[LlamaInstall]:
    """Fast scan: known paths + PATH. Returns all found installations."""
    found: list[LlamaInstall] = []
    seen: set[str] = set()

    def _add(base: Path, via: str) -> None:
        key = str(base.resolve())
        if key in seen:
            return
        seen.add(key)
        inst = _probe_install(base, hw, via)
        if inst:
            found.append(inst)

    # Cached path first (if exists)
    cached = _cached_bin_dir()
    if cached:
        _add(cached, "cache")

    for base in _get_platform_search_paths():
        _add(base, "known path")

    which_bench = shutil.which(_exe("llama-bench"))
    if which_bench:
        _add(Path(which_bench).parent, "PATH")

    return found


def scan_deep(hw: Optional[HardwareProfile] = None) -> list[LlamaInstall]:
    """Deep scan: fast scan + locate/find/where across all drives."""
    found = scan_all(hw)
    seen = {str(Path(i.bin_dir).resolve()) for i in found}

    def _add(p: Path, via: str) -> None:
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        inst = _probe_install(p, hw, via)
        if inst:
            found.append(inst)

    if _SYSTEM == "Windows":
        import string
        for drive in string.ascii_uppercase:
            root = Path(f"{drive}:\\")
            if not root.exists():
                continue
            try:
                r = subprocess.run(
                    ["where", "/R", str(root), _exe("llama-bench")],
                    capture_output=True, text=True, timeout=30,
                )
                for line in r.stdout.splitlines():
                    p = Path(line.strip())
                    if p.exists():
                        _add(p.parent, f"drive {drive}:")
            except Exception:
                pass
    else:
        # locate (fast, index-based)
        p = _locate_binary("llama-bench")
        if p:
            _add(p.parent, "locate")
        # find (thorough)
        p2 = _find_via_system_search("llama-bench")
        if p2:
            _add(p2.parent, "find")

    return found


# ── Active install management ──────────────────────────────────────────────

def set_active(install: LlamaInstall) -> None:
    """Set the active llama.cpp installation (with rollback support)."""
    _save_env_json(Path(install.bin_dir), install.build, install.backend)


def get_previous() -> Optional[tuple[str, str]]:
    """Return (bin_dir, build) of the previous installation for rollback, or None."""
    data = _load_env_json() or {}
    prev_dir = data.get("previous_bin_dir")
    prev_build = data.get("previous_build", "unknown")
    if not prev_dir:
        return None
    p = Path(prev_dir)
    bench = p / _exe("llama-bench")
    cli = p / _exe("llama-cli")
    if bench.exists() and cli.exists():
        return prev_dir, prev_build
    return None


def rollback() -> bool:
    """Switch to the previously active installation. Returns True on success."""
    data = _load_env_json() or {}
    prev_dir = data.get("previous_bin_dir")
    prev_build = data.get("previous_build", "unknown")
    if not prev_dir:
        return False
    p = Path(prev_dir)
    if not (p / _exe("llama-bench")).exists() or not (p / _exe("llama-cli")).exists():
        return False
    curr_dir = data.get("bin_dir")
    curr_build = data.get("build", "unknown")
    data["previous_bin_dir"] = curr_dir
    data["previous_build"] = curr_build
    data["bin_dir"] = prev_dir
    data["build"] = prev_build
    ENV_FILE.write_text(json.dumps(data, indent=2))
    return True


# ── Pre-built binary download ──────────────────────────────────────────────

def _pick_prebuilt_asset(assets: list, backend: Backend) -> tuple[Optional[str], Optional[str]]:
    """Choose the best pre-built binary for current OS/backend from release assets."""
    sys_key = {"Windows": "win", "Darwin": "macos", "Linux": "ubuntu"}.get(_SYSTEM, "")
    arch = "arm64" if _SYSTEM == "Darwin" and platform.machine() == "arm64" else "x64"

    scored: list[tuple[int, str, str]] = []
    for asset in assets:
        name = asset.get("name", "").lower()
        url = asset.get("browser_download_url", "")
        if not url or sys_key not in name:
            continue
        score = 0
        if _SYSTEM == "Windows":
            if backend == Backend.CUDA and "cuda" in name and arch in name:
                score = 10
            elif "avx2" in name and arch in name:
                score = 5
            elif "avx" in name and arch in name:
                score = 3
            elif arch in name and ("zip" in name or name.endswith(".zip")):
                score = 1
        elif _SYSTEM == "Darwin":
            if arch in name:
                score = 10
            elif "universal" in name:
                score = 5
        elif _SYSTEM == "Linux":
            if arch in name:
                score = 5
        if score > 0:
            scored.append((score, url, asset["name"]))

    if not scored:
        return None, None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], scored[0][2]


def prebuilt_available() -> bool:
    """True if pre-built binaries are published for this OS."""
    return _SYSTEM in ("Windows", "Darwin")


def download_prebuilt(backend: Backend, build: str, install_dir: Path) -> Optional[Path]:
    """Download a pre-built llama.cpp binary from GitHub releases.

    Returns the bin_dir (where llama-bench lives) or None on failure.
    Linux CUDA has no pre-built binaries — returns None always on Linux+CUDA.
    """
    from rich.console import Console
    console = Console()

    try:
        resp = requests.get(
            f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{build}",
            timeout=10,
        )
        if resp.status_code != 200:
            console.print(f"  [red]GitHub API error {resp.status_code} for {build}[/]")
            return None
        assets = resp.json().get("assets", [])
    except Exception as e:
        console.print(f"  [red]Failed to fetch release info: {e}[/]")
        return None

    asset_url, asset_name = _pick_prebuilt_asset(assets, backend)
    if not asset_url:
        console.print(
            f"  [yellow]No pre-built binary found for {_SYSTEM}/{backend.value} in release {build}.[/]\n"
            "  Try building from source instead."
        )
        return None

    console.print(f"  Downloading [bold]{asset_name}[/] ({build})...")
    try:
        r = requests.get(asset_url, stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        data_bytes = bytearray()
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), DownloadColumn()) as prog:
            task = prog.add_task(asset_name, total=total or None)
            for chunk in r.iter_content(65536):
                data_bytes.extend(chunk)
                prog.advance(task, len(chunk))
    except Exception as e:
        console.print(f"  [red]Download failed: {e}[/]")
        return None

    if install_dir.exists():
        shutil.rmtree(str(install_dir))
    install_dir.mkdir(parents=True, exist_ok=True)

    try:
        if asset_name.lower().endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(io.BytesIO(data_bytes)) as zf:
                zf.extractall(install_dir)
        else:
            import tarfile
            with tarfile.open(fileobj=io.BytesIO(data_bytes)) as tf:
                tf.extractall(install_dir)
    except Exception as e:
        console.print(f"  [red]Extract failed: {e}[/]")
        return None

    # Locate the directory containing llama-bench
    exe_name = _exe("llama-bench")
    for root, _dirs, files in os.walk(install_dir):
        if exe_name in files:
            return Path(root)

    console.print("  [red]Extracted but llama-bench not found in archive.[/]")
    return None


# ── Build from source ──────────────────────────────────────────────────────

def _find_bin_dir(root: Path) -> Path:
    """Find the directory containing llama-bench after a build."""
    exe = _exe("llama-bench")
    candidates = [
        "build/bin/Release", "build/bin", "build/Release", "build", ".",
    ]
    for rel in candidates:
        p = root / rel
        if (p / exe).exists():
            return p
    # deep search as last resort
    for p in root.rglob(exe):
        return p.parent
    raise RuntimeError(
        f"Build finished but {exe} not found under {root}. "
        "Check cmake output above for errors."
    )


def _build_llama_cpp(backend: Backend, target_build: str, install_dir: Path) -> Path:
    """Build llama.cpp from source. Always fresh clone → swap, so no stale cmake cache."""
    cmake_flags = {
        Backend.CUDA:  ["-DGGML_CUDA=ON"],
        Backend.ROCM:  ["-DGGML_HIPBLAS=ON"],
        Backend.METAL: ["-DGGML_METAL=ON"],
        Backend.CPU:   [],
    }[backend]

    def _run(*cmd, **kwargs):
        result = subprocess.run(list(cmd), **kwargs)
        if result.returncode != 0:
            raise RuntimeError(
                f"{cmd[0]} failed (exit {result.returncode}) — "
                "scroll up to see cmake/compiler errors"
            )

    tmp_dir = install_dir.parent / "llama.cpp.building"
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    from rich.console import Console as _Con
    con = _Con()

    con.print(f"  [dim]Cloning llama.cpp {target_build}...[/]")
    _run("git", "clone", "--depth=1", "--branch", target_build,
         LLAMA_CPP_REPO, str(tmp_dir), timeout=600)

    con.print("  [dim]Running cmake configure...[/]")
    _run("cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", *cmake_flags,
         cwd=tmp_dir, timeout=120)

    con.print(f"  [dim]Compiling with {os.cpu_count()} threads — this takes 5–20 min...[/]")
    _run("cmake", "--build", "build", "--config", "Release",
         f"-j{os.cpu_count()}", cwd=tmp_dir, timeout=1800)

    # Detect where binaries actually landed (location changed across llama.cpp versions)
    bin_dir_in_tmp = _find_bin_dir(tmp_dir)
    bin_dir_rel = bin_dir_in_tmp.relative_to(tmp_dir)

    con.print("  [dim]Swapping in new build...[/]")
    if install_dir.exists():
        shutil.rmtree(str(install_dir))
    shutil.move(str(tmp_dir), str(install_dir))

    result_bin_dir = install_dir / bin_dir_rel
    con.print(f"  [green]✓ Binaries at: {result_bin_dir}[/]")
    return result_bin_dir


def install(hw: HardwareProfile, build: str = LLAMA_CPP_PINNED_BUILD) -> EnvConfig:
    """Build llama.cpp from source at the given build tag."""
    bin_dir = _build_llama_cpp(hw.backend, build, LLAMA_INSTALL_DIR)
    _save_env_json(bin_dir, build, hw.backend)

    bench = bin_dir / _exe("llama-bench")
    cli = bin_dir / _exe("llama-cli")
    server = bin_dir / _exe("llama-server")
    env = make_env_with_lib(str(bin_dir))

    # Read actual built version (may differ if checkout had issues)
    actual_build = _get_build_version(cli, env) or build

    return EnvConfig(
        llama_bench_path=str(bench),
        llama_cli_path=str(cli),
        llama_server_path=str(server) if server.exists() else str(cli),
        bin_dir=str(bin_dir),
        build=actual_build,
        backend=hw.backend,
        env_dict=env,
        found_via="installed",
    )
