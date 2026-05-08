"""Module 4: Process Manager — find and kill GPU-hungry processes."""
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import psutil
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

HUNGRY_PROCESSES = {
    "llama-server": "llama-server",
    "llama-cli":    "llama-cli",
    "ollama":       "Ollama (holds GPU)",
    "google-chrome": "Google Chrome",
    "chromium":     "Chromium",
    "firefox":      "Firefox",
    "code":         "VS Code",
    "electron":     "Electron app",
    "stable-diffusion": "Stable Diffusion",
    "comfyui":      "ComfyUI",
}

# Exact-name processes to skip (crash handlers, sandboxes, etc.)
SKIP_NAMES = {
    "chrome_crashpad_handler", "chromium-sandbox", "nacl_helper",
    "zygote_host_exec", "chrome-sandbox",
}

# Exe path substrings → HUNGRY_PROCESSES key
# Catches multiprocess apps (Firefox Web Content, Chromium renderers, etc.)
EXE_KEY_MAP = {
    "firefox": "firefox",
    "chromium": "chromium",
    "google-chrome": "google-chrome",
    "chrome": "google-chrome",
}

WARN_UNSAVED = {"google-chrome", "chromium", "firefox", "code"}


def _input_timed(timeout: int = 60) -> Optional[str]:
    """Read a line from stdin with a timeout. Returns None on timeout."""
    try:
        import select as _sel
        sys.stdout.flush()
        ready, _, _ = _sel.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip().lower()
        print(f"\n  [auto-skipped after {timeout}s]")
        return None
    except Exception:
        # Windows or unusual environment — fall back to plain input (no timeout)
        try:
            return input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None


def _gpu_mem_by_pid() -> Dict[int, int]:
    """Returns {pid: vram_mb} for all GPU processes via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        result: Dict[int, int] = {}
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

            if name_lower in SKIP_NAMES:
                continue

            ram_mb = (proc.info["memory_info"].rss / (1024 * 1024)
                      if proc.info["memory_info"] else 0)
            has_gpu = pid in gpu_pids

            # 1) Exact / prefix name match against HUNGRY_PROCESSES keys
            group_key = None
            for key in HUNGRY_PROCESSES:
                if name_lower == key or name_lower.startswith(key):
                    group_key = key
                    break

            # 2) Exe-path match (catches Firefox Web Content, Chromium renderers…)
            if group_key is None:
                try:
                    exe = proc.exe().lower()
                    for pattern, key in EXE_KEY_MAP.items():
                        if pattern in exe:
                            group_key = key
                            break
                except Exception:
                    pass

            # 3) Unknown GPU process — still show it
            if group_key is None and has_gpu:
                group_key = f"_gpu_{name_lower}"

            if group_key is None:
                continue

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

    # Sort: VRAM desc first, then RAM desc
    result.sort(key=lambda g: (-g[5], -g[4]))
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

    console.print()

    killed_any = False
    remaining = list(groups)  # mutable copy so we can shrink it

    while remaining:
        console.print(
            f"  Enter the [bold]number[/] of the process to close "
            f"(1–{len(remaining)}), [bold]n[/] to skip, or wait 60s to auto-skip:"
        )
        console.print("[bold]> [/]", end="")
        choice = _input_timed(60)
        if choice is None:
            break

        if choice in ("n", "q", ""):
            break

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(remaining):
                key, name, desc, pids, ram, vram, has_gpu = remaining[idx]
                killed = 0
                for pid in pids:
                    try:
                        psutil.Process(pid).terminate()
                        killed += 1
                    except Exception:
                        pass
                n_str = f"{killed} process{'es' if killed != 1 else ''}"
                console.print(f"  [green]Terminated {name} ({n_str})[/]")
                remaining.pop(idx)
                killed_any = True
                # Reprint remaining table
                if remaining:
                    console.print()
                    new_table = Table(box=box.ROUNDED)
                    new_table.add_column("#", style="dim", width=3)
                    new_table.add_column("App")
                    new_table.add_column("VRAM (MB)", justify="right")
                    new_table.add_column("RAM (MB)", justify="right")
                    for j, (_, nm, _, _, rm, vm, hg) in enumerate(remaining, 1):
                        v = f"[yellow]{vm}[/]" if vm > 0 else "[dim]—[/]"
                        new_table.add_row(str(j), nm, v, f"{rm:.0f}")
                    console.print(new_table)
            else:
                console.print(f"  [red]Enter 1–{len(remaining)} or n.[/]")
        else:
            console.print(f"  [red]Enter a number or n.[/]")

    if killed_any:
        time.sleep(2)
