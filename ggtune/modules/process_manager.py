"""Module 4: Process Manager — find and kill GPU-hungry processes."""
import time
from typing import List, Tuple

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


def _find_hungry() -> List[Tuple[int, str, str, float]]:
    """Returns list of (pid, name, description, ram_mb)."""
    found = []
    try:
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            name = (proc.info["name"] or "").lower()
            for key, desc in HUNGRY_PROCESSES.items():
                if key in name:
                    ram_mb = proc.info["memory_info"].rss / (1024 * 1024) if proc.info["memory_info"] else 0
                    found.append((proc.info["pid"], proc.info["name"], desc, ram_mb))
                    break
    except Exception:
        pass
    return found


def prompt_and_kill() -> None:
    """Show hungry processes and let user choose which to kill."""
    procs = _find_hungry()
    if not procs:
        return

    table = Table(title="Processes using GPU/RAM", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("PID")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("RAM (MB)", justify="right")

    for i, (pid, name, desc, ram) in enumerate(procs, 1):
        table.add_row(str(i), str(pid), name, desc, f"{ram:.0f}")

    console.print(table)

    for pid, name, _, _ in procs:
        if any(w in name.lower() for w in WARN_UNSAVED):
            console.print(f"[yellow]⚠ {name} may have unsaved data.[/]")

    console.print("\nClose these? [[bold]a[/]]ll / [[bold]n[/]]one / comma-separated numbers")
    choice = input("> ").strip().lower()

    to_kill: List[Tuple[int, str]] = []
    if choice == "a":
        to_kill = [(pid, name) for pid, name, _, _ in procs]
    elif choice == "n" or not choice:
        return
    else:
        indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
        to_kill = [(procs[i][0], procs[i][1]) for i in indices if 0 <= i < len(procs)]

    for pid, name in to_kill:
        try:
            p = psutil.Process(pid)
            p.terminate()
            console.print(f"[green]Terminated {name} (PID {pid})[/]")
        except Exception as e:
            console.print(f"[red]Could not terminate {name}: {e}[/]")

    if to_kill:
        time.sleep(3)
