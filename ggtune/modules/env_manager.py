"""Module 3+9: Environment Manager — detects, installs, and configures llama.cpp."""
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ggtune.config import (
    LLAMA_CPP_PINNED_BUILD, LLAMA_CPP_REPO,
    ENV_FILE, CONFIG_DIR, LLAMA_INSTALL_DIR,
)
from ggtune.models.hardware import Backend, HardwareProfile
from ggtune.utils.shell import make_env_with_lib


SEARCH_PATHS = [
    LLAMA_INSTALL_DIR / "build" / "bin",
    Path.home() / ".local" / "llama.cpp" / "build" / "bin",
    Path("/usr/local/bin"),
    Path("/usr/bin"),
]


@dataclass
class EnvConfig:
    llama_bench_path: str
    llama_cli_path: str
    llama_server_path: str
    bin_dir: str
    build: str
    backend: Backend
    env_dict: dict


def _find_binary(name: str) -> Optional[Path]:
    for base in SEARCH_PATHS:
        p = base / name
        if p.exists():
            return p
    found = shutil.which(name)
    return Path(found) if found else None


def _get_build_version(llama_cli: Path, env: dict) -> Optional[str]:
    try:
        result = subprocess.run(
            [str(llama_cli), "--version"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        for token in result.stdout.split() + result.stderr.split():
            if token.startswith("b") and token[1:].isdigit():
                return token
    except Exception:
        pass
    return None


def _has_cuda(llama_cli: Path, env: dict) -> bool:
    try:
        result = subprocess.run(
            [str(llama_cli), "--list-devices"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        return "CUDA" in result.stdout or "CUDA" in result.stderr
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


def detect(hw: Optional[HardwareProfile] = None) -> EnvConfig:
    """Find llama.cpp binaries and validate CUDA support."""
    bench = _find_binary("llama-bench")
    cli = _find_binary("llama-cli")
    server = _find_binary("llama-server")

    if not bench or not cli:
        raise RuntimeError(
            "llama.cpp not found. Run: ggtune update  (auto-installs with CUDA)\n"
            "Or build manually: cmake -B build -DGGML_CUDA=ON && cmake --build build"
        )

    bin_dir = str(bench.parent)
    env = make_env_with_lib(bin_dir)
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

    return EnvConfig(
        llama_bench_path=str(bench),
        llama_cli_path=str(cli),
        llama_server_path=str(server) if server else str(cli),
        bin_dir=bin_dir,
        build=build,
        backend=backend,
        env_dict=env,
    )


def install(hw: HardwareProfile) -> EnvConfig:
    """Install llama.cpp at pinned build with correct backend."""
    bin_dir = _build_llama_cpp(hw.backend, LLAMA_CPP_PINNED_BUILD, LLAMA_INSTALL_DIR)
    _save_env_json(bin_dir, LLAMA_CPP_PINNED_BUILD, hw.backend)

    bench = bin_dir / "llama-bench"
    cli = bin_dir / "llama-cli"
    server = bin_dir / "llama-server"
    env = make_env_with_lib(str(bin_dir))

    return EnvConfig(
        llama_bench_path=str(bench),
        llama_cli_path=str(cli),
        llama_server_path=str(server) if server.exists() else str(cli),
        bin_dir=str(bin_dir),
        build=LLAMA_CPP_PINNED_BUILD,
        backend=hw.backend,
        env_dict=env,
    )
