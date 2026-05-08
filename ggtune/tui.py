"""Interactive TUI — full guided flow for GGTune."""
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

console = Console()
WIDTH = 70


def _clear() -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")


def _ask(prompt: str = "> ") -> str:
    try:
        console.print(f"[bold cyan]{prompt}[/]", end="")
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return "q"


def _banner(subtitle: str = "") -> None:
    sub = f"  [dim]{subtitle}[/]" if subtitle else ""
    console.print(Panel(
        f"[bold cyan]GGTune[/] — llama.cpp auto-optimizer{sub}",
        border_style="dim cyan",
        padding=(0, 2),
    ))
    console.print()


def _sep() -> None:
    console.print(Rule(style="dim"))


# ── Main menu ──────────────────────────────────────────────────────────────

def main_menu() -> None:
    """Entry point. Loops until user exits."""
    while True:
        _clear()
        _banner()
        console.print(Panel(
            "  [bold cyan][1][/]  Scan for .gguf models on this machine\n"
            "  [bold cyan][2][/]  Enter model path manually\n"
            "  [bold cyan][3][/]  Browse HuggingFace (find & download models)\n"
            "  [bold cyan][4][/]  Show saved benchmark profiles\n"
            "  [bold cyan][5][/]  Hardware info\n"
            "  [bold cyan][q][/]  Exit",
            title="[bold]What do you want to do?[/]",
            border_style="cyan",
            padding=(1, 4),
            width=WIDTH,
        ))

        choice = _ask("Choice: ")

        if choice == "1":
            path = _screen_scan()
            if path:
                _screen_model_and_run(path)
        elif choice == "2":
            path = _screen_manual_path()
            if path:
                _screen_model_and_run(path)
        elif choice == "3":
            path = _screen_browse_hf()
            if path:
                _screen_model_and_run(path)
        elif choice == "4":
            _screen_profiles()
        elif choice == "5":
            _screen_hardware()
        elif choice.lower() in ("q", "quit", "exit", ""):
            _clear()
            console.print("[dim]Bye![/]")
            break


# ── Screen 1: Scan ────────────────────────────────────────────────────────

def _scan_gguf_files() -> List[Tuple[str, int]]:
    """Return list of (path, size_bytes) for found .gguf models."""
    found: dict[str, int] = {}

    def _add(p: Path) -> None:
        n = p.name.lower()
        if p.suffix != ".gguf" or not p.is_file():
            return
        if "mmproj" in n or "ggml-vocab" in n:
            return
        if p.stat().st_size < 50 * 1024 * 1024:
            return
        found[str(p)] = p.stat().st_size

    home = Path.home()
    for d in [
        home / "models", home / "Downloads",
        home / ".cache" / "lm-studio" / "models",
        home / ".local" / "share" / "models",
        Path("/data"),
    ]:
        if d.exists():
            for f in d.rglob("*.gguf"):
                _add(f)

    if platform.system() in ("Linux", "Darwin"):
        try:
            r = subprocess.run(
                ["locate", "-r", r"\.gguf$"],
                capture_output=True, text=True, timeout=12,
            )
            for line in r.stdout.splitlines():
                _add(Path(line.strip()))
        except Exception:
            pass

    return sorted(found.items(), key=lambda x: -x[1])


def _screen_scan() -> Optional[str]:
    _clear()
    _banner("Scan for Models")
    console.print("[dim]Searching standard locations + locate index...[/]\n")

    items = _scan_gguf_files()

    if not items:
        console.print("[yellow]No .gguf models found.[/]")
        console.print("[dim]Try option 2 (manual path) or 3 (HuggingFace browse).[/]")
        _ask("\nPress Enter to go back")
        return None

    _clear()
    _banner("Scan Results")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Size", justify="right", width=9)
    table.add_column("Path")

    for i, (fpath, size) in enumerate(items, 1):
        table.add_row(str(i), f"{size / 1e9:.1f} GB", fpath)

    console.print(table)
    console.print(f"\n  Found [bold]{len(items)}[/] model(s).")
    console.print("  Enter a [bold]number[/] to select, or [bold]b[/] to go back.\n")

    while True:
        choice = _ask("Select model #: ")
        if choice.lower() in ("b", ""):
            return None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx][0]
        console.print(f"  [red]Invalid choice.[/] Enter 1–{len(items)} or b.")


# ── Screen 2: Manual path ─────────────────────────────────────────────────

def _screen_manual_path() -> Optional[str]:
    _clear()
    _banner("Enter Model Path")
    console.print("  Paste or type the full path to your .gguf file.\n")
    console.print("  [dim]Example:[/]  /home/user/models/llama-3-8b-q4.gguf")
    console.print("  [dim]Enter [bold]b[/] to go back.[/]\n")

    while True:
        raw = _ask("Path: ").strip("'\"")
        if raw.lower() in ("b", ""):
            return None
        p = Path(raw)
        if p.exists() and p.suffix == ".gguf":
            return str(p)
        if p.exists():
            console.print(f"  [red]Not a .gguf file.[/]")
        else:
            console.print(f"  [red]File not found:[/] {raw}")


# ── Screen 3: HuggingFace browse ──────────────────────────────────────────

def _screen_browse_hf() -> Optional[str]:
    _clear()
    _banner("Browse HuggingFace")

    author = _ask("Author to browse (default: unsloth): ") or "unsloth"
    if author.lower() == "b":
        return None

    _clear()
    _banner(f"HuggingFace — {author}")

    try:
        from ggtune.modules import hardware_scanner, hf_browser
        hw = hardware_scanner.scan()
        path = hf_browser.interactive_browse(hw, author=author)
        return str(path) if path else None
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        _ask("\nPress Enter to go back")
        return None


# ── Screen 4: Saved profiles ──────────────────────────────────────────────

def _screen_profiles() -> None:
    _clear()
    _banner("Saved Benchmark Profiles")

    from ggtune.modules import profile_storage
    profiles = profile_storage.list_all()

    if not profiles:
        console.print("  [dim]No profiles yet. Run a benchmark first.[/]")
    else:
        table = Table(box=box.ROUNDED, header_style="bold")
        table.add_column("Model")
        table.add_column("Quant")
        table.add_column("TG speed", justify="right")
        table.add_column("Context", justify="right")
        table.add_column("GPU")
        table.add_column("Date", style="dim")

        for p in profiles:
            table.add_row(
                p.model_name,
                p.model_quantization,
                f"[green]{p.tg_tokens_per_sec:.1f} t/s[/]",
                f"{p.optimal_context:,}",
                p.hw_gpu_name,
                p.created_at[:10],
            )
        console.print(table)

    _ask("\nPress Enter to go back")


# ── Screen 5: Hardware ────────────────────────────────────────────────────

def _screen_hardware() -> None:
    _clear()
    _banner("Hardware Info")

    from ggtune.modules import hardware_scanner
    hw = hardware_scanner.scan()

    console.print(f"  [bold]GPU[/]")
    console.print(f"    Name:     {hw.gpu_name}")
    console.print(f"    VRAM:     {hw.vram_total_mb // 1024}GB total / {hw.vram_free_mb // 1024}GB free")
    console.print(f"    Backend:  [bold]{hw.backend.value}[/]")
    if hw.driver_version:
        console.print(f"    Driver:   {hw.driver_version}")
    if hw.compute_cap:
        console.print(f"    CUDA:     compute {hw.compute_cap}")
    console.print()
    console.print(f"  [bold]CPU[/]")
    console.print(f"    Name:     {hw.cpu_name}")
    console.print(f"    Cores:    {hw.cores_physical} physical / {hw.cores_logical} logical")
    console.print()
    console.print(f"  [bold]RAM[/]    {hw.ram_total_gb:.1f}GB total / {hw.ram_available_gb:.1f}GB available")
    console.print(f"  [bold]OS[/]     {hw.os}  shell={hw.shell}")

    _ask("\nPress Enter to go back")


# ── Model info + benchmark mode ───────────────────────────────────────────

def _screen_model_and_run(model_path: str) -> None:
    """Show model info, pick benchmark mode, run, then offer alias."""
    _clear()
    _banner("Model Info")

    # Read model
    from ggtune.modules import gguf_reader, hardware_scanner
    try:
        model = gguf_reader.read(model_path)
    except Exception as e:
        console.print(f"[red]Could not read model: {e}[/]")
        _ask("\nPress Enter to go back")
        return

    hw = hardware_scanner.scan()

    # Model info panel
    info_lines = [
        f"  [bold]Name:[/]         {model.name}",
        f"  [bold]Architecture:[/] {model.architecture}",
        f"  [bold]Quantization:[/] {model.quantization}",
        f"  [bold]File size:[/]    {model.file_size_gb:.2f} GB",
        f"  [bold]Layers:[/]       {model.n_layers}    "
        f"[bold]Heads:[/] {model.n_heads}"
        + (f"  [bold]KV heads:[/] {model.n_kv_heads}" if model.n_kv_heads else ""),
        f"  [bold]Max context:[/]  {model.context_length_max:,} tokens",
    ]
    if model.is_moe:
        info_lines.insert(3,
            f"  [bold]MoE:[/]          {model.n_experts_total} experts total, "
            f"{model.n_experts_used} active per token"
        )

    vram_gb = hw.vram_total_mb / 1024
    if model.file_size_gb > vram_gb:
        info_lines.append(
            f"\n  [yellow]⚠  Model ({model.file_size_gb:.1f}GB) > VRAM ({vram_gb:.1f}GB) — "
            "will use RAM offload.[/]"
        )
    else:
        info_lines.append(f"\n  [green]✓ Fits in VRAM ({vram_gb:.1f}GB).[/]")

    console.print(Panel(
        "\n".join(info_lines),
        title=f"[bold]{model.name}[/]",
        border_style="cyan",
        padding=(1, 2),
        width=WIDTH,
    ))

    # Benchmark mode selection
    from ggtune.config import OPTUNA_TRIALS, STABILITY_RUNS
    from ggtune.modules import search_space_builder
    try:
        space = search_space_builder.build(hw, model)
        probe_runs = space.estimated_quick_probe_runs()
        ctx_runs = len(space.context_candidates)
        total_runs = probe_runs + OPTUNA_TRIALS + ctx_runs + STABILITY_RUNS
        full_est = f"~{total_runs} runs"
        ctx_list = ", ".join(f"{c // 1024}k" for c in space.context_candidates)
        ctx_note = f"context search: {ctx_list}"
    except Exception:
        full_est = "~70 runs"
        ctx_note = ""

    console.print(Panel(
        f"  [bold cyan][1][/]  Full benchmark   [dim]({full_est}, {ctx_note})[/]\n"
        f"  [bold cyan][2][/]  Quick benchmark  [dim](~35 runs, skips context search)[/]\n"
        f"  [bold cyan][b][/]  Back",
        title="[bold]Benchmark Mode[/]",
        border_style="dim",
        padding=(1, 4),
        width=WIDTH,
    ))

    choice = _ask("Choice: ")
    if choice.lower() in ("b", ""):
        return

    quick = (choice == "2")
    _clear()

    from ggtune.orchestrator import run as _run
    _run(model_path, quick=quick)

    _ask("\nPress Enter to return to main menu")
