import subprocess
import os
from typing import Optional


def run(
    cmd: list[str],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: int = 300,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env or os.environ.copy(),
        timeout=timeout,
        capture_output=capture_output,
        text=True,
    )


def run_checked(cmd: list[str], **kwargs) -> str:
    result = run(cmd, capture_output=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command {cmd[0]} failed:\n{result.stderr}")
    return result.stdout


def make_env_with_lib(bin_dir: str) -> dict:
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir
    return env
