"""Module 3+9: Environment Manager — detects, installs, and configures llama.cpp."""
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

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
    """Well-known llama.cpp binary locations per OS — no hardcoded user paths."""
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
            home / "llama.cpp" / "build" / "bin",
            home / ".local" / "llama.cpp" / "build" / "bin",
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/opt/llama.cpp/build/bin"),
            Path("/snap/bin"),
        ]

    return paths


def _find_via_system_search(name: str) -> Optional[Path]:
    """Deep OS-level search (find / where / PowerShell). Used only as last resort."""
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
        # PowerShell search in personal + program dirs
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

    # Linux / macOS
    search_roots = [str(home), "/usr/local", "/opt"]
    if _SYSTEM == "Darwin":
        search_roots.append("/opt/homebrew")
    try:
        r = subprocess.run(
            ["find"] + search_roots + ["-name", exe_name, "-type", "f"],
            capture_output=True, text=True, timeout=30, stderr=subprocess.DEVNULL,
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
    data = {
        "build": build,
        "bin_dir": str(bin_dir),
        "backend": backend.value,
    }
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

    # Cache the discovered path (skip if already came from cache)
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


def _build_llama_cpp(backend: Backend, target_build: str, install_dir: Path) -> Path:
    from rich.console import Console
    console = Console()

    cmake_flags = {
        Backend.CUDA:  ["-DGGML_CUDA=ON"],
        Backend.ROCM:  ["-DGGML_HIPBLAS=ON"],
        Backend.METAL: ["-DGGML_METAL=ON"],
        Backend.CPU:   [],
    }[backend]

    def run(*cmd, **kwargs):
        result = subprocess.run(list(cmd), **kwargs)
        if result.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn()) as p:
        t = p.add_task("Setting up llama.cpp...", total=4)

        if install_dir.exists():
            run("git", "-C", str(install_dir), "fetch", "--tags", timeout=120)
            run("git", "-C", str(install_dir), "checkout", target_build, timeout=30)
        else:
            install_dir.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--depth=1", "--branch", target_build,
                LLAMA_CPP_REPO, str(install_dir), timeout=300)
        p.advance(t)

        run("cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", *cmake_flags,
            cwd=install_dir, timeout=120)
        p.advance(t)

        run("cmake", "--build", "build", "--config", "Release",
            f"-j{os.cpu_count()}", cwd=install_dir, timeout=1800)
        p.advance(t)

        bin_dir = install_dir / "build" / "bin"
        p.advance(t)

    return bin_dir


def install(hw: HardwareProfile) -> EnvConfig:
    """Install llama.cpp at pinned build with correct backend."""
    bin_dir = _build_llama_cpp(hw.backend, LLAMA_CPP_PINNED_BUILD, LLAMA_INSTALL_DIR)
    _save_env_json(bin_dir, LLAMA_CPP_PINNED_BUILD, hw.backend)

    bench = bin_dir / _exe("llama-bench")
    cli = bin_dir / _exe("llama-cli")
    server = bin_dir / _exe("llama-server")
    env = make_env_with_lib(str(bin_dir))

    return EnvConfig(
        llama_bench_path=str(bench),
        llama_cli_path=str(cli),
        llama_server_path=str(server) if server.exists() else str(cli),
        bin_dir=str(bin_dir),
        build=LLAMA_CPP_PINNED_BUILD,
        backend=hw.backend,
        env_dict=env,
        found_via="installed",
    )
