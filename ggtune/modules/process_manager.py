"""Module 4: Process Manager — find and kill GPU-hungry processes."""
import subprocess
import time
from typing import Dict, List, Tuple

import psutil
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

HUNGRY_PROCESSES = {
    "llama-server": "llama-server instance",
    "llama-cli":    "llama-cli",
    "ollama":       "Ollama (holds GPU)",
    "chrome":       "Google Chrome",
    "chromium":     "Chromium",
    "firefox":      "Firefox",
    "code":         "VS Code",
    "electron":     "Electron app",
    "stable-diffusion": "Stable Diffusion",
    "comfyui":      "ComfyUI",
}

WARN_UNSAVED = {"chrome", "chromium", "firefox", "code"}


def _gpu_mem_by_pid() -> Dict[int, int]:
    """Returns {pid: vram_mb} for all processes using GPU via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        result = {}
        for line in r.stdout.splitlines():
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    result[int(parts[0].strip())] = int(parts[1].strip())
                except ValueError:
                    pass
        return result
    except Exception:
        return {}


# Each group: (group_key, display_name, description, pids, ram_mb, vram_mb, has_gpu)
_Group = Tuple[str, str, str, List[int], float, int, bool]


def _find_hungry() -> List[_Group]:
    """Returns grouped process entries sorted by VRAM desc then RAM desc."""
    gpu_mem = _gpu_mem_by_pid()
    gpu_pids = set(gpu_mem)
    groups: Dict[str, dict] = {}

    try:
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            pid = proc.info["pid"]
            raw_name = proc.info["name"] or ""
            name_lower = raw_name.lower()
            ram_mb = (proc.info["memory_info"].rss / (1024 * 1024)
                      if proc.info["memory_info"] else 0)
            has_gpu = pid in gpu_pids

            # Find matching HUNGRY key
            group_key = None
            for key in HUNGRY_PROCESSES:
                if key in name_lower:
                    group_key = key
                    break

            if not has_gpu and group_key is None:
                continue

            if group_key is None:
                group_key = f"_gpu_{name_lower}"

            if group_key not in groups:
                groups[group_key] = {
                    "name": raw_name,
                    "desc": HUNGRY_PROCESSES.get(group_key, raw_name),
                    "pids": [],
                    "ram_mb": 0.0,
                    "vram_mb": 0,
                    "has_gpu": False,
                }

            groups[group_key]["pids"].append(pid)
            groups[group_key]["ram_mb"] += ram_mb
            if has_gpu:
                groups[group_key]["has_gpu"] = True
                groups[group_key]["vram_mb"] += gpu_mem.get(pid, 0)
    except Exception:
        pass

    result: List[_Group] = []
    for key, g in groups.items():
        result.append((key, g["name"], g["desc"], g["pids"],
                       g["ram_mb"], g["vram_mb"], g["has_gpu"]))

    # Sort: VRAM-using first (by vram_mb desc), then RAM desc
    result.sort(key=lambda g: (-g[5], not g[6], -g[4]))
    return result


def prompt_and_kill() -> None:
    """Show resource-hungry process groups and let user choose which to kill."""
    groups = _find_hungry()
    if not groups:
        return

    table = Table(title="Processes using GPU/RAM", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3)
    table.add_column("App")
    table.add_column("Description")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("RAM (MB)", justify="right")
    table.add_column("Procs", justify="right", style="dim")

    for i, (key, name, desc, pids, ram, vram, has_gpu) in enumerate(groups, 1):
        vram_str = f"[yellow]{vram}[/]" if vram > 0 else "[dim]—[/]"
        desc_str = f"[yellow]{desc}[/]" if has_gpu else desc
        table.add_row(str(i), name, desc_str, vram_str, f"{ram:.0f}", str(len(pids)))

    console.print(table)

    for _, name, _, _, _, _, _ in groups:
        if any(w in name.lower() for w in WARN_UNSAVED):
            console.print(f"[yellow]⚠  {name} may have unsaved data.[/]")

    console.print("\nClose these? [[bold]a[/]]ll / [[bold]n[/]]one / comma-separated numbers")
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    to_kill: List[Tuple[List[int], str]] = []
    if choice == "a":
        to_kill = [(pids, name) for _, name, _, pids, _, _, _ in groups]
    elif choice == "n" or not choice:
        return
    else:
        indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
        to_kill = [(groups[i][3], groups[i][1]) for i in indices if 0 <= i < len(groups)]

    for pids, name in to_kill:
        killed = 0
        for pid in pids:
            try:
                psutil.Process(pid).terminate()
                killed += 1
            except Exception:
                pass
        n = f"{killed} process{'es' if killed != 1 else ''}"
        console.print(f"[green]Terminated {name} ({n})[/]")

    if to_kill:
        time.sleep(2)
