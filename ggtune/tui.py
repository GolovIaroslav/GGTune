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


# ── Setup wizard ───────────────────────────────────────────────────────────

_DRIVER_MIN = {"Windows": 527, "Darwin": 0, "Linux": 525}


def _driver_ok(version_str: Optional[str]) -> bool:
    if not version_str:
        return False
    try:
        return int(version_str.split(".")[0]) >= _DRIVER_MIN.get(platform.system(), 525)
    except (ValueError, IndexError):
        return True


def _has_bin(name: str) -> bool:
    import shutil as _shutil
    return _shutil.which(name) is not None


def _screen_setup(hw) -> None:
    """First-run wizard: detects missing deps and offers to install llama.cpp."""
    from ggtune.modules import env_manager

    _clear()
    _banner("Setup")

    has_gpu = hw.backend.value != "cpu"
    drv_ok = _driver_ok(hw.driver_version)
    drv_str = hw.driver_version or "not detected"
    minimum = _DRIVER_MIN.get(platform.system(), 525)

    # System status table
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", min_width=10)
    t.add_column(min_width=28)
    t.add_column()

    gpu_label = f"{hw.gpu_name}  ({hw.vram_total_mb // 1024} GB)" if has_gpu else "none detected"
    t.add_row("GPU:", gpu_label, "[green]✓[/]" if has_gpu else "[dim]-[/]")

    if has_gpu:
        drv_status = "[green]✓[/]" if drv_ok else f"[red]✗ need ≥{minimum}[/]"
        t.add_row("Driver:", drv_str, drv_status)

    t.add_row("llama.cpp:", "[red]not found[/]", "[red]✗[/]")
    console.print(t)
    console.print()

    # Driver warning with fix instructions
    if has_gpu and not drv_ok:
        console.print(f"  [yellow]NVIDIA driver too old for CUDA 12.4 (need ≥{minimum}).[/]")
        if platform.system() == "Linux":
            console.print("  Fix:  [dim]sudo apt install nvidia-driver-550[/]")
            console.print("        [dim](or: ubuntu-drivers autoinstall)[/]")
        elif platform.system() == "Windows":
            console.print("  Fix:  [dim]nvidia.com/drivers — download and run installer[/]")
        console.print("  You can still build CPU-only and use GPU after updating drivers.\n")

    # Check build prerequisites
    has_git   = _has_bin("git")
    has_cmake = _has_bin("cmake")
    has_cxx   = _has_bin("g++") or _has_bin("clang++") or _has_bin("cl")
    prereqs   = has_git and has_cmake and has_cxx

    console.print("  Build prerequisites:")
    console.print(f"    git     {'[green]✓[/]' if has_git   else '[red]✗[/]'}")
    console.print(f"    cmake   {'[green]✓[/]' if has_cmake else '[red]✗[/]'}")
    console.print(f"    c++     {'[green]✓[/]' if has_cxx   else '[red]✗[/]'}")
    console.print()

    if not prereqs:
        console.print("  [yellow]Install missing tools first:[/]")
        if platform.system() == "Linux":
            console.print("    [dim]sudo apt install git cmake build-essential[/]  (Debian/Ubuntu)")
            console.print("    [dim]sudo dnf install git cmake gcc-c++[/]  (Fedora/RHEL)")
        elif platform.system() == "Darwin":
            console.print("    [dim]brew install git cmake[/]")
            console.print("    [dim]xcode-select --install[/]")
        elif platform.system() == "Windows":
            console.print("    [dim]Install: Visual Studio Build Tools + CMake + Git[/]")
        console.print()

    can_cuda = has_gpu and drv_ok and prereqs
    can_cpu  = prereqs

    lines = []
    if can_cuda:
        lines.append("  [bold cyan][1][/]  Build with CUDA  [dim](GPU — recommended, ~10 min)[/]")
    elif has_gpu and not drv_ok:
        lines.append("  [dim][1]  Build with CUDA — update drivers first[/]")
    if can_cpu:
        lines.append("  [bold cyan][2][/]  Build CPU-only  [dim](works without GPU, ~5 min)[/]")
    lines.append("  [bold cyan][3][/]  Already installed — re-scan  [dim](PATH / custom location)[/]")
    lines.append("  [bold cyan][s][/]  Skip  [dim](main menu, benchmarks won't work)[/]")

    console.print(Panel("\n".join(lines), border_style="dim", padding=(1, 4), width=WIDTH))
    choice = _ask("Choice: ")

    if choice == "1" and can_cuda:
        console.print("\n  [dim]Building llama.cpp with CUDA — may take 5–20 min...[/]")
        try:
            env_manager.install(hw)
            console.print("  [green]✓ Done! llama.cpp installed.[/]")
        except Exception as e:
            console.print(f"  [red]Build failed: {e}[/]")
        _ask("\nPress Enter to continue")

    elif choice == "2" and can_cpu:
        hw.backend = __import__("ggtune.models.hardware", fromlist=["Backend"]).Backend.CPU
        console.print("\n  [dim]Building CPU-only llama.cpp — ~5 min...[/]")
        try:
            env_manager.install(hw)
            console.print("  [green]✓ Done! CPU-only llama.cpp installed.[/]")
        except Exception as e:
            console.print(f"  [red]Build failed: {e}[/]")
        _ask("\nPress Enter to continue")

    elif choice == "3":
        env_manager.clear_cache()
        console.print("  [dim]Scanning...[/]")
        try:
            found = env_manager.detect(hw)
            console.print(f"  [green]✓ Found:[/] {found.bin_dir}  [dim](via {found.found_via})[/]")
        except RuntimeError:
            console.print(
                "  [red]Not found.[/]\n"
                "  Make sure [bold]llama-bench[/] is in your PATH, or add its directory to PATH and re-run."
            )
        _ask("\nPress Enter to continue")
    # "s" or anything else → fall through to main menu


def _maybe_run_setup() -> None:
    """Run setup wizard once if llama.cpp is not found."""
    from ggtune.modules import hardware_scanner, env_manager
    hw = hardware_scanner.scan()
    try:
        env_manager.detect(hw)
    except RuntimeError:
        _screen_setup(hw)


# ── Screen 1: Models (unified: view / scan / manage / launch) ─────────────

def _screen_models() -> Optional[str]:
    from ggtune.modules import model_tracker

    session_found: dict = {}  # path -> size_bytes, from disk scan this session

    while True:
        _clear()
        _banner("Models")

        tracked = model_tracker.list_all()
        tracked_paths = {m.path for m in tracked}

        rows: list = []
        for m in tracked:
            vision = " [cyan]+V[/]" if m.mmproj_path and Path(m.mmproj_path).exists() else ""
            tag = "HF" if m.source_type == "downloaded" else "local"
            rows.append({"path": m.path, "filename": m.filename,
                         "size_gb": m.size_gb, "tag": tag, "vision": vision})
        for path, size in session_found.items():
            if path not in tracked_paths:
                rows.append({"path": path, "filename": Path(path).name,
                             "size_gb": size / 1e9, "tag": "found", "vision": ""})

        if rows:
            table = Table(box=box.ROUNDED, header_style="bold", width=WIDTH)
            table.add_column("#", style="dim", justify="right", width=3)
            table.add_column("File")
            table.add_column("Size", justify="right", width=8)
            table.add_column("", width=6, style="dim")
            for i, r in enumerate(rows, 1):
                table.add_row(str(i), f"[bold]{r['filename']}[/bold]" + r["vision"],
                              f"{r['size_gb']:.1f} GB", r["tag"])
            console.print(table)

            scan_dirs = model_tracker.load_scan_dirs()
            if scan_dirs:
                console.print(f"  [dim]Watch folders: {', '.join(scan_dirs)}[/]")
        else:
            console.print(
                "  [dim]No models yet. Press [bold]s[/] to scan disk "
                "or [bold]3[/] in the main menu to download one.[/]"
            )

        console.print(
            "\n  [dim][bold]number[/bold] — benchmark  "
            "[bold]s[/bold] — quick scan  [bold]f[/bold] — thorough scan  "
            "[bold]d[/bold] — delete  [bold]a[/bold] — add watch folder  "
            "[bold]r[/bold] — rescan folders  [bold]b[/bold] — back[/dim]"
        )
        choice = _ask("").strip().lower()

        if choice in ("b", ""):
            return None

        elif choice == "s":
            console.print("[dim]Scanning via locate...[/]")
            items = _scan_gguf_files("locate")
            new = sum(1 for p, _ in items if p not in session_found and p not in tracked_paths)
            session_found.update({p: sz for p, sz in items})
            console.print(f"[dim]Found {len(items)} model(s), {new} new.[/]")

        elif choice == "f":
            console.print("[dim]Running find — up to 30 seconds...[/]")
            items = _scan_gguf_files("find")
            new = sum(1 for p, _ in items if p not in session_found and p not in tracked_paths)
            session_found.update({p: sz for p, sz in items})
            console.print(f"[dim]Found {len(items)} model(s), {new} new.[/]")

        elif choice == "d":
            if not rows:
                continue
            n = _ask("Delete model #: ")
            try:
                target = rows[int(n) - 1]
            except (ValueError, IndexError):
                console.print("[red]Invalid.[/]")
                continue
            tm = next((m for m in tracked if m.path == target["path"]), None)
            if tm:
                if tm.source_type == "local":
                    action = _ask("  [r] remove from list  [x] delete file: ").lower()
                    if action == "r":
                        model_tracker.remove(tm.path, delete_file=False)
                        console.print("[green]✓ Removed from list.[/]")
                    elif action == "x":
                        if _ask(f"  Delete {tm.filename}? [y/N]: ").lower() == "y":
                            model_tracker.remove(tm.path, delete_file=True)
                            console.print("[green]✓ Deleted.[/]")
                else:
                    if _ask(f"  Delete {target['filename']}? [y/N]: ").lower() == "y":
                        model_tracker.remove(tm.path)
                        console.print("[green]✓ Deleted.[/]")
            else:
                session_found.pop(target["path"], None)
                console.print("[green]✓ Removed from list.[/]")

        elif choice == "a":
            folder = _ask("Folder path: ")
            if folder:
                p = Path(folder).expanduser()
                if p.is_dir():
                    model_tracker.add_scan_dir(str(p))
                    console.print(f"[green]✓ Added. Press [bold]r[/] to scan it.[/]")
                else:
                    console.print("[red]Directory not found.[/]")

        elif choice == "r":
            console.print("[dim]Rescanning watch folders...[/]")
            n = model_tracker.rescan_dirs()
            console.print(f"[green]✓ Found {n} new model(s).[/]")

        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(rows):
                    return rows[idx]["path"]
                console.print(f"[red]Enter 1–{len(rows)}.[/]")
            except ValueError:
                console.print("[red]Unknown command.[/]")


# ── Main menu ──────────────────────────────────────────────────────────────

def main_menu() -> None:
    """Entry point. Loops until user exits."""
    _maybe_run_setup()

    while True:
        _clear()
        _banner()

        # Status line: show llama.cpp build if found
        from ggtune.modules import env_manager as _em
        try:
            _env = _em.detect()
            _status = f"  [dim]llama.cpp {_env.build}  ·  {_env.backend.value}  ·  {_env.bin_dir}[/]\n"
        except RuntimeError:
            _status = "  [yellow]⚠ llama.cpp not found — go to option 6 to install[/]\n"
        console.print(_status)

        console.print(Panel(
            "  [bold cyan][1][/]  Models\n"
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
            path = _screen_models()
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


def _format_path(path: str, max_len: int = 55) -> str:
    """Show path truncated with filename bolded."""
    p = Path(path)
    name = p.name
    parent = str(p.parent)
    prefix = parent + "/"
    full = prefix + name
    if len(full) > max_len:
        keep = max_len - len(name) - 2  # 2 for "…/"
        prefix = "…" + prefix[-keep:] if keep > 0 else "…/"
    return f"{prefix}[bold]{name}[/bold]"


def _truncate_path(path: str, max_len: int = 55) -> str:
    if len(path) <= max_len:
        return path
    return "…" + path[-(max_len - 1):]


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

_HF_SEARCH_URL = "https://huggingface.co/models?filter=gguf&sort=downloads"

_MANUAL_GUIDE = (
    "Enter a model ID or HuggingFace URL:\n\n"
    "  [bold]unsloth/Llama-3.3-70B-Instruct-GGUF[/]\n"
    "  [bold]https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF[/]\n\n"
    f"  Browse all GGUF models: [cyan]{_HF_SEARCH_URL}[/]"
)


def _screen_browse_manual() -> Optional[str]:
    from ggtune.modules import hf_browser
    while True:
        _clear()
        _banner("Download by Model ID")
        console.print(Panel(_MANUAL_GUIDE, border_style="dim", padding=(1, 2), width=WIDTH))
        raw = _ask("\nModel ID / URL (or [b] to go back): ")
        if raw.lower() in ("b", ""):
            return None

        model_id = hf_browser.parse_model_input(raw)
        if not model_id:
            console.print("[red]Not a valid model ID or URL. Try: author/model-name[/]")
            _ask("Press Enter to try again")
            continue

        console.print(f"[dim]Fetching file list for {model_id} — this may take a few seconds...[/]")
        main_files, mmproj_files = hf_browser.fetch_gguf_files(model_id)
        if not main_files:
            console.print(f"[red]No GGUF files found in {model_id}. Check the model ID.[/]")
            _ask("Press Enter to try again")
            continue

        _clear()
        _banner(model_id)
        table = Table(box=box.ROUNDED, header_style="bold", width=WIDTH)
        table.add_column("#", style="dim", justify="right")
        table.add_column("File")
        table.add_column("Size", justify="right")
        for i, f in enumerate(main_files, 1):
            table.add_row(str(i), f["filename"], f"{f['size_gb']:.1f} GB")
        console.print(table)

        choice = _ask("Number to download (or [b] to go back): ")
        if choice.lower() in ("b", ""):
            continue
        try:
            idx = int(choice) - 1
            chosen = main_files[idx]
        except (ValueError, IndexError):
            console.print("[red]Invalid choice.[/]")
            _ask("Press Enter to try again")
            continue

        console.print(f"[dim]Downloading {chosen['filename']}...[/]")
        try:
            dest = hf_browser.download_by_id(model_id, chosen["filename"])
            console.print(f"[green]✓ Saved to {dest}[/]")

            # Offer mmproj if model has vision files
            mmproj_dest = None
            if mmproj_files:
                console.print(f"\n  [cyan]This model has {len(mmproj_files)} vision file(s) (mmproj).[/]")
                table2 = Table(box=box.ROUNDED, header_style="bold", width=WIDTH)
                table2.add_column("#", style="dim", justify="right")
                table2.add_column("File")
                table2.add_column("Size", justify="right")
                for i, f in enumerate(mmproj_files, 1):
                    table2.add_row(str(i), f["filename"], f"{f['size_gb']:.1f} GB")
                console.print(table2)
                mp_choice = _ask("Download vision file? Enter number or [n] to skip: ")
                if mp_choice.lower() not in ("n", ""):
                    try:
                        mp_idx = int(mp_choice) - 1
                        mp_file = mmproj_files[mp_idx]
                        console.print(f"[dim]Downloading {mp_file['filename']}...[/]")
                        mmproj_dest = hf_browser.download_by_id(model_id, mp_file["filename"])
                        console.print(f"[green]✓ Vision file saved to {mmproj_dest}[/]")
                        from ggtune.modules.model_tracker import set_mmproj
                        set_mmproj(str(dest.resolve()), str(mmproj_dest.resolve()))
                    except (ValueError, IndexError):
                        console.print("[dim]Skipping vision file.[/]")

            _ask("\nPress Enter to continue")
            return str(dest)
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/]")
            _ask("Press Enter to go back")
            return None


def _screen_browse_hf() -> Optional[str]:
    _clear()
    _banner("Browse HuggingFace")

    console.print(Panel(
        f"  [bold cyan][1][/]  Popular models (unsloth, filtered by your VRAM)\n"
        f"  [bold cyan][2][/]  Enter model ID or URL manually\n"
        f"  [bold cyan][b][/]  Back\n\n"
        f"  [dim]All GGUF models: [cyan]{_HF_SEARCH_URL}[/dim]",
        title="[bold]Get a Model[/]",
        border_style="dim",
        padding=(1, 4),
        width=WIDTH,
    ))

    choice = _ask("Choice: ")
    if choice.lower() in ("b", ""):
        return None

    if choice == "1":
        try:
            from ggtune.modules import hardware_scanner, hf_browser
            hw = hardware_scanner.scan()
            path = hf_browser.interactive_browse(hw)
            return str(path) if path else None
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            _ask("\nPress Enter to go back")
            return None

    if choice == "2":
        return _screen_browse_manual()

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

    # Vision / mmproj
    mmproj_path = _resolve_mmproj(model_path)

    from ggtune.orchestrator import run as _run
    _run(model_path, quick=quick, mmproj_path=mmproj_path)

    _ask("\nPress Enter to return to main menu")


def _resolve_mmproj(model_path: str) -> Optional[str]:
    """Ask user about vision (mmproj) before benchmark. Returns mmproj path or None."""
    from ggtune.modules import model_tracker

    # Check tracker first
    tracked = model_tracker.find_by_path(model_path)
    if tracked and tracked.mmproj_path and Path(tracked.mmproj_path).exists():
        use = _ask(f"  Vision file detected ({Path(tracked.mmproj_path).name}). Include in launch command? [y/N]: ")
        return tracked.mmproj_path if use.lower() == "y" else None

    # Check same directory for mmproj files
    mp = model_tracker._find_mmproj_near(Path(model_path))
    if mp:
        use = _ask(f"  Vision file found nearby ({mp.name}). Include in launch command? [y/N]: ")
        if use.lower() == "y":
            model_tracker.set_mmproj(model_path, str(mp))
            return str(mp)

    return None


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
