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
            "  [bold cyan][1][/]  Scan for models on this machine\n"
            "  [bold cyan][2][/]  Enter model path manually\n"
            "  [bold cyan][3][/]  Browse HuggingFace (find & download)\n"
            "  [bold cyan][4][/]  Saved benchmark profiles\n"
            "  [bold cyan][5][/]  Hardware info\n"
            "  [bold cyan][6][/]  llama.cpp — version / update\n"
            "  [bold cyan][7][/]  Compatibility test\n"
            "  [bold cyan][8][/]  Clear all profiles\n"
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
        elif choice == "6":
            _screen_llama_update()
        elif choice == "7":
            _screen_compat()
        elif choice == "8":
            _screen_clear_profiles()
        elif choice.lower() in ("q", "quit", "exit", ""):
            _clear()
            console.print("[dim]Bye![/]")
            break


# ── Screen 1: Scan ────────────────────────────────────────────────────────

MODEL_EXTENSIONS = {".gguf", ".bin"}


def _scan_gguf_files(search_method: str = "locate") -> List[Tuple[str, int]]:
    """Return list of (path, size_bytes) for found GGUF-compatible models.

    search_method: "locate" (fast, index-based) or "find" (thorough, real-time)
    """
    found: dict[str, int] = {}

    def _add(p: Path) -> None:
        if not p.is_file():
            return
        if p.suffix.lower() not in MODEL_EXTENSIONS:
            return
        n = p.name.lower()
        if "mmproj" in n or "ggml-vocab" in n:
            return
        if p.stat().st_size < 50 * 1024 * 1024:
            return
        found[str(p)] = p.stat().st_size

    home = Path.home()
    sys = platform.system()

    # Common directories across all platforms
    common_dirs = [
        home / "models",
        home / "Downloads",
        home / ".cache" / "lm-studio" / "models",
        home / ".local" / "share" / "models",
    ]

    if sys == "Windows":
        common_dirs += [
            home / "Documents" / "models",
            home / "Desktop",
            home / "AppData" / "Local" / "LM Studio" / "models",
            home / ".ollama" / "models",
        ]
        # Also scan drive roots D:\, E:\ (C:\ is slow and mostly system files)
        import string
        for drive in string.ascii_uppercase[3:8]:  # D: through H:
            d = Path(f"{drive}:\\")
            if d.exists():
                common_dirs.append(d / "models")
                common_dirs.append(d / "AI" / "models")
    else:
        common_dirs.append(Path("/data"))
        common_dirs.append(home / ".ollama" / "models")

    for d in common_dirs:
        if d.exists():
            for ext in MODEL_EXTENSIONS:
                for f in d.rglob(f"*{ext}"):
                    _add(f)

    if sys in ("Linux", "Darwin"):
        if search_method == "find":
            try:
                args = ["find", str(home), "-type", "f"]
                for ext in MODEL_EXTENSIONS:
                    args += ["-name", f"*{ext}", "-o"]
                args.pop()  # remove trailing -o
                r = subprocess.run(args, capture_output=True, text=True, timeout=120)
                for line in r.stdout.splitlines():
                    _add(Path(line.strip()))
            except Exception:
                pass
        else:  # locate
            for ext in MODEL_EXTENSIONS:
                try:
                    r = subprocess.run(
                        ["locate", "-r", rf"\{ext}$"],
                        capture_output=True, text=True, timeout=12,
                    )
                    for line in r.stdout.splitlines():
                        _add(Path(line.strip()))
                except Exception:
                    pass

    return sorted(found.items(), key=lambda x: -x[1])


def _truncate_path(path: str, max_len: int = 55) -> str:
    """Show only the rightmost part of a path if it's too long."""
    if len(path) <= max_len:
        return path
    return "…" + path[-(max_len - 1):]


def _screen_scan() -> Optional[str]:
    _clear()
    _banner("Scan for Models")

    # On Linux/macOS let user pick locate vs find
    search_method = "locate"
    if platform.system() in ("Linux", "Darwin"):
        console.print(Panel(
            "  [bold cyan][1][/]  [bold]locate[/]  [dim]— fast (uses file index, may miss recently downloaded files)[/]\n"
            "  [bold cyan][2][/]  [bold]find[/]    [dim]— thorough (scans your home folder in real time, slower ~30s)[/]",
            title="[bold]Search method[/]",
            border_style="dim",
            padding=(1, 4),
            width=WIDTH,
        ))
        m = _ask("Choice [1/2]: ")
        if m == "2":
            search_method = "find"
            console.print("[dim]Running find (this may take up to 30 seconds)...[/]\n")
        else:
            console.print("[dim]Searching via locate index...[/]\n")
    else:
        console.print("[dim]Scanning standard locations...[/]\n")

    items = _scan_gguf_files(search_method)

    if not items:
        console.print("[yellow]No models found.[/]")
        if search_method == "locate":
            console.print("[dim]Tip: if you just downloaded a model, try scan again with [bold]find[/] (option 2).[/]")
        console.print("[dim]Or use option 2 (manual path) or 3 (HuggingFace browse).[/]")
        _ask("\nPress Enter to go back")
        return None

    _clear()
    _banner("Scan Results")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Size", justify="right", width=9)
    table.add_column("Path")

    for i, (fpath, size) in enumerate(items, 1):
        table.add_row(str(i), f"{size / 1e9:.1f} GB", _truncate_path(fpath))

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
        if p.exists() and p.is_file():
            if p.suffix.lower() not in MODEL_EXTENSIONS:
                console.print(
                    f"  [yellow]Warning: extension '{p.suffix}' is unusual.[/] "
                    "GGTune supports GGUF format. Trying anyway..."
                )
            return str(p)
        elif p.exists():
            console.print(f"  [red]Not a file:[/] {raw}")
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


# ── Screen 6: llama.cpp version / update ─────────────────────────────────────

def _screen_llama_update() -> None:
    _clear()
    _banner("llama.cpp")

    from ggtune.modules import hardware_scanner, env_manager
    from ggtune.modules.compat_guard import get_latest_build, check_for_changes

    hw = hardware_scanner.scan()
    env = None
    current_build = "not found"

    try:
        console.print("  [dim]Searching for llama.cpp...[/]")
        env = env_manager.detect(hw)
        current_build = env.build
        console.print(f"  Installed:  [bold]{env.build}[/]  ({env.backend.value})")
        console.print(f"  Location:   {env.bin_dir}")
        console.print(f"  Found via:  [dim]{env.found_via}[/]")
    except RuntimeError:
        console.print(f"  [yellow]llama.cpp not installed or not found.[/]")

    console.print()
    console.print("  [dim]Checking latest release...[/]")
    latest = get_latest_build()

    if latest and current_build != "not found":
        try:
            cur_num = int(current_build.lstrip("b"))
            lat_num = int(latest.lstrip("b"))
            diff = lat_num - cur_num
            if diff <= 0:
                console.print(f"  [green]✓ Up to date[/] ({current_build})")
            else:
                console.print(
                    f"  [yellow]Latest: {latest}[/]  "
                    f"[dim](you're {diff} build{'s' if diff != 1 else ''} behind)[/]"
                )
                changes = check_for_changes(current_build)
                breaking = [c for c in changes if c.is_breaking]
                if breaking:
                    console.print(f"\n  [red]⚠ {len(breaking)} breaking change(s):[/]")
                    for c in breaking[:5]:
                        console.print(f"    b{c.build}  {c.title}")
        except ValueError:
            if latest:
                console.print(f"  Latest: {latest}")
    elif latest:
        console.print(f"  Latest available: [bold]{latest}[/]")
    else:
        console.print("  [dim]Could not reach GitHub.[/]")

    console.print()
    menu = "  [bold cyan][1][/]  Install / rebuild llama.cpp"
    if env:
        menu += f"  [dim](currently {current_build}, ~10 min build from source)[/]"
    menu += "\n  [bold cyan][2][/]  Re-scan  [dim](clear cached path, search again)[/]"
    menu += "\n  [bold cyan][b][/]  Back"
    console.print(Panel(menu, border_style="dim", padding=(1, 4), width=WIDTH))

    choice = _ask("Choice: ")
    if choice == "1":
        console.print("[dim]Building llama.cpp — this may take 5–20 minutes...[/]")
        try:
            env_manager.install(hw)
            console.print("[green]✓ Done![/]")
        except Exception as e:
            console.print(f"[red]Build failed: {e}[/]")
    elif choice == "2":
        env_manager.clear_cache()
        console.print("  [dim]Cache cleared. Searching...[/]")
        try:
            env2 = env_manager.detect(hw)
            console.print(f"  [green]✓ Found:[/] {env2.bin_dir}  [dim](via {env2.found_via})[/]")
        except RuntimeError as e:
            console.print(f"  [red]Not found: {e}[/]")
    _ask("\nPress Enter to go back")


# ── Screen 7: Compatibility test ──────────────────────────────────────────────

def _screen_compat() -> None:
    _clear()
    _banner("Compatibility Test")

    from ggtune.modules import hardware_scanner, env_manager
    from ggtune.modules import compat_guard

    hw = hardware_scanner.scan()
    try:
        env = env_manager.detect(hw)
    except RuntimeError as e:
        console.print(f"[red]llama.cpp not found: {e}[/]")
        _ask("\nPress Enter to go back")
        return

    console.print(f"[dim]Testing {env.bin_dir}...[/]\n")
    try:
        report = compat_guard.run_tests(env.bin_dir)
        compat_guard.print_report(report)
        if report.all_critical_passed:
            console.print("\n[green]✓ All critical tests passed.[/]")
        else:
            console.print("\n[red]✗ Critical tests failed — run option 6 to rebuild.[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")

    _ask("\nPress Enter to go back")


# ── Screen 8: Clear all profiles ──────────────────────────────────────────────

def _screen_clear_profiles() -> None:
    _clear()
    _banner("Clear All Profiles")

    from ggtune.modules import profile_storage
    profiles = profile_storage.list_all()

    if not profiles:
        console.print("  [dim]No profiles found.[/]")
        _ask("\nPress Enter to go back")
        return

    console.print(f"  This will delete [bold]{len(profiles)}[/] saved profile(s).")
    confirm = _ask("  Are you sure? [y/N]: ")
    if confirm.lower() == "y":
        try:
            deleted = profile_storage.delete_all()
            console.print(f"  [green]✓ Deleted {deleted} profile(s).[/]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/]")
    else:
        console.print("  [dim]Cancelled.[/]")
    _ask("\nPress Enter to go back")
