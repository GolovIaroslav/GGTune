"""Interactive TUI — full guided flow for GGTune."""
import os
import platform
import shutil
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


def _ask(prompt: str = "") -> str:
    try:
        if prompt:
            from rich.markup import escape as _esc
            import sys
            sys.stdout.write(f"\033[1;36m{_esc(prompt)}\033[0m")
            sys.stdout.flush()
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
    from ggtune.config import LLAMA_INSTALL_DIR, LLAMA_CPP_PINNED_BUILD

    _clear()
    _banner("Setup")

    sys = platform.system()
    has_gpu = hw.backend.value != "cpu"
    drv_ok  = _driver_ok(hw.driver_version)
    drv_str = hw.driver_version or "not detected"
    minimum = _DRIVER_MIN.get(sys, 525)

    # ── Status table ───────────────────────────────────────────────────────
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

    if has_gpu and not drv_ok:
        console.print(f"  [yellow]NVIDIA driver too old for CUDA 12.4 (need ≥{minimum}).[/]")
        if sys == "Linux":
            console.print("  Fix:  [dim]sudo apt install nvidia-driver-550[/]")
            console.print("        [dim](or: ubuntu-drivers autoinstall)[/]")
        elif sys == "Windows":
            console.print("  Fix:  [dim]nvidia.com/drivers — download and run installer[/]")
        console.print("  You can still use CPU-only and switch to GPU after updating drivers.\n")

    # ── Build tool detection ───────────────────────────────────────────────
    # On Windows cmake often lives inside VS Build Tools, not on PATH.
    # _find_cmake_windows() checks ~24 known VS / standalone cmake paths.
    has_git = _has_bin("git")
    if sys == "Windows":
        cmake_path = env_manager._find_cmake_windows()
        has_cmake  = cmake_path is not None
        cmake_tag  = " [dim](VS Build Tools)[/]" if has_cmake and not _has_bin("cmake") else ""
        # cmake from VS Build Tools bundles cl.exe; cmake finds it via vswhere automatically
        has_cxx = has_cmake or _has_bin("g++") or _has_bin("clang++") or _has_bin("cl")
    else:
        cmake_path = None
        cmake_tag  = ""
        has_cmake  = _has_bin("cmake")
        has_cxx    = _has_bin("g++") or _has_bin("clang++") or _has_bin("cl")

    prereqs = has_git and has_cmake and has_cxx

    console.print("  Build prerequisites:")
    console.print(f"    git     {'[green]✓[/]' if has_git   else '[red]✗[/]'}")
    console.print(f"    cmake   {('[green]✓[/]' + cmake_tag) if has_cmake else '[red]✗[/]'}")
    console.print(f"    c++     {'[green]✓[/]' if has_cxx   else '[red]✗[/]'}")
    console.print()

    if not prereqs:
        if sys == "Windows":
            console.print("  [yellow]Build tools not found — use option [1] to download a pre-built binary.[/]")
            console.print("  [dim]To build from source: Visual Studio Build Tools (C++ workload + CMake) + Git[/]")
        elif sys == "Linux":
            console.print("  [yellow]Install missing tools first:[/]")
            console.print("    [dim]sudo apt install git cmake build-essential[/]  (Debian/Ubuntu)")
            console.print("    [dim]sudo dnf install git cmake gcc-c++[/]  (Fedora/RHEL)")
        elif sys == "Darwin":
            console.print("  [yellow]Install missing tools first:[/]")
            console.print("    [dim]brew install git cmake[/]")
            console.print("    [dim]xcode-select --install[/]")
        console.print()

    # ── Menu ──────────────────────────────────────────────────────────────
    can_prebuilt = sys == "Windows"           # GitHub releases always have Windows ZIPs
    can_cuda     = has_gpu and drv_ok and prereqs
    can_cpu      = prereqs

    # option_map tracks which action each digit actually triggers (varies by platform/state)
    option_map: dict[str, str] = {}
    lines: list[str] = []
    n = 1  # next available number

    if can_prebuilt:
        flavor = "CUDA" if (has_gpu and drv_ok) else "CPU/AVX2"
        lines.append(
            f"  [bold cyan][{n}][/]  Download pre-built [dim]({flavor} — no build tools needed)[/]"
        )
        option_map[str(n)] = "prebuilt"
        n += 1

    if can_cuda:
        lines.append(f"  [bold cyan][{n}][/]  Build with CUDA  [dim](GPU — ~10–20 min)[/]")
        option_map[str(n)] = "cuda"
        n += 1
    elif has_gpu and not drv_ok:
        lines.append(f"  [dim][{n}]  Build with CUDA — update drivers first[/]")
        n += 1
    elif has_gpu:
        lines.append(f"  [dim][{n}]  Build with CUDA — install build tools first[/]")
        n += 1

    if can_cpu:
        lines.append(f"  [bold cyan][{n}][/]  Build CPU-only  [dim](~5–10 min)[/]")
        option_map[str(n)] = "cpu"
        n += 1
    elif not can_prebuilt:  # on Linux/Mac, show the hint even when tools missing
        lines.append(f"  [dim][{n}]  Build CPU-only — install build tools first[/]")
        n += 1

    lines.append(f"  [bold cyan][{n}][/]  Already installed — re-scan  [dim](PATH / custom location)[/]")
    option_map[str(n)] = "rescan"
    lines.append("  [bold cyan][s][/]  Skip  [dim](main menu, benchmarks won't work)[/]")

    console.print(Panel("\n".join(lines), border_style="dim", padding=(1, 4), width=WIDTH))
    choice = _ask("Choice: ")

    Backend = __import__("ggtune.models.hardware", fromlist=["Backend"]).Backend
    action  = option_map.get(choice)

    if action == "prebuilt":
        _do_install(LLAMA_CPP_PINNED_BUILD, hw, env_manager, LLAMA_INSTALL_DIR)
        _ask("\nPress Enter to continue")

    elif action == "cuda":
        console.print("\n  [dim]Building llama.cpp with CUDA — may take 10–20 min...[/]")
        try:
            env_manager.install(hw)
            console.print("  [green]✓ Done! llama.cpp installed.[/]")
        except Exception as e:
            console.print(f"  [red]Build failed: {e}[/]")
        _ask("\nPress Enter to continue")

    elif action == "cpu":
        hw.backend = Backend.CPU
        console.print("\n  [dim]Building CPU-only llama.cpp — ~5–10 min...[/]")
        try:
            env_manager.install(hw)
            console.print("  [green]✓ Done! CPU-only llama.cpp installed.[/]")
        except Exception as e:
            console.print(f"  [red]Build failed: {e}[/]")
        _ask("\nPress Enter to continue")

    elif action == "rescan":
        env_manager.clear_cache()
        console.print("  [dim]Scanning...[/]")
        try:
            found = env_manager.detect(hw)
            console.print(f"  [green]✓ Found:[/] {found.bin_dir}  [dim](via {found.found_via})[/]")
        except RuntimeError:
            console.print(
                "  [red]Not found.[/]\n"
                "  Make sure [bold]llama-bench[/] is in your PATH, "
                "or add its directory to PATH and re-run."
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
            if platform.system() == "Windows":
                console.print("[dim]Scanning common locations...[/]")
            else:
                console.print("[dim]Scanning via locate...[/]")
            items = _scan_gguf_files("locate")
            new = sum(1 for p, _ in items if p not in session_found and p not in tracked_paths)
            session_found.update({p: sz for p, sz in items})
            console.print(f"[dim]Found {len(items)} model(s), {new} new.[/]")

        elif choice == "f":
            if platform.system() == "Windows":
                console.print("[dim]Scanning home folder and other drives — up to 60 sec...[/]")
            else:
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
                        if _ask(f"  Delete {tm.filename}? (y/n): ").lower() == "y":
                            model_tracker.remove(tm.path, delete_file=True)
                            console.print("[green]✓ Deleted.[/]")
                else:
                    if _ask(f"  Delete {target['filename']}? (y/n): ").lower() == "y":
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

    if sys == "Windows":
        if search_method == "find":
            # Walk user home + secondary drives, skipping known noisy directories
            import string as _str
            _skip = {
                "appdata\\local\\temp",
                "appdata\\local\\microsoft",
                "appdata\\roaming\\microsoft",
                "appdata\\local\\google",
                "appdata\\local\\mozilla",
                "appdata\\local\\nvidia",
                "appdata\\locallow",
                ".git", "node_modules", "__pycache__", ".venv", "venv",
            }

            def _skip_dir(parent: str, d: str) -> bool:
                full = (parent + "\\" + d).lower()
                return d.startswith("$") or any(s in full for s in _skip)

            walk_roots = [home]
            for _drive in _str.ascii_uppercase[3:8]:  # D: through H:
                _dp = Path(f"{_drive}:\\")
                if _dp.exists():
                    walk_roots.append(_dp)

            for walk_root in walk_roots:
                try:
                    for dirpath, dirnames, filenames in os.walk(str(walk_root), topdown=True):
                        dirnames[:] = [d for d in dirnames if not _skip_dir(dirpath, d)]
                        for fname in filenames:
                            if Path(fname).suffix.lower() in MODEL_EXTENSIONS:
                                try:
                                    _add(Path(dirpath) / fname)
                                except Exception:
                                    pass
                except Exception:
                    pass

    elif sys in ("Linux", "Darwin"):
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
        use = _ask(f"  Vision file detected ({Path(tracked.mmproj_path).name}). Include in launch command? (y/n): ")
        return tracked.mmproj_path if use.lower() == "y" else None

    # Check same directory for mmproj files
    mp = model_tracker._find_mmproj_near(Path(model_path))
    if mp:
        use = _ask(f"  Vision file found nearby ({mp.name}). Include in launch command? (y/n): ")
        if use.lower() == "y":
            model_tracker.set_mmproj(model_path, str(mp))
            return str(mp)

    return None


# ── llama.cpp install helper ──────────────────────────────────────────────────

def _do_install(target: str, hw, env_manager, install_dir) -> None:
    """Download pre-built (Win/Mac) or build from source (Linux). Activates on success."""
    from ggtune.utils.shell import make_env_with_lib as _menv

    if env_manager.prebuilt_available():
        console.print(f"  [dim]Downloading pre-built {target}...[/]")
        dl_dir = install_dir.parent / f"llama.cpp.{target}"
        bin_dir = env_manager.download_prebuilt(hw.backend, target, dl_dir)
        if bin_dir:
            cli_p = bin_dir / env_manager._exe("llama-cli")
            actual = env_manager._get_build_version(cli_p, _menv(str(bin_dir))) or target
            inst = env_manager.LlamaInstall(
                bin_dir=str(bin_dir), build=actual,
                backend=hw.backend, found_via="downloaded",
            )
            env_manager.set_active(inst)
            console.print(f"  [green]✓ Downloaded and activated {actual}.[/]")
            return
        console.print(
            f"  [yellow]No pre-built binary for {target} on this platform.[/]\n"
            "  Falling back to build from source..."
        )

    # Build from source (Linux always; Win/Mac as fallback)
    console.print(f"  [dim]Building {target} from source — 5–20 min...[/]")
    try:
        env_cfg = env_manager.install(hw, target)
        console.print(f"  [green]✓ Built and activated {env_cfg.build}.[/]")
    except Exception as e:
        console.print(f"  [red]Build failed: {e}[/]")


# ── Screen 6: llama.cpp manager ───────────────────────────────────────────────

def _screen_llama_update() -> None:
    from ggtune.modules import hardware_scanner, env_manager
    from ggtune.modules.compat_guard import get_latest_build
    from ggtune.config import LLAMA_CPP_PINNED_BUILD, LLAMA_INSTALL_DIR

    hw = hardware_scanner.scan()
    installs: list = []
    latest: Optional[str] = None

    def _refresh() -> None:
        nonlocal installs, latest
        installs = env_manager.scan_all(hw)
        latest = get_latest_build()

    _refresh()

    while True:
        _clear()
        _banner("llama.cpp")

        # Active install
        active_dir: Optional[str] = None
        try:
            env = env_manager.detect(hw)
            active_dir = str(Path(env.bin_dir).resolve())
        except RuntimeError:
            pass

        # Show installations table (full terminal width)
        tbl_w = max(80, console.width - 2)
        path_max = tbl_w - 28  # 3 + 7 + 8 + borders/padding
        if installs:
            table = Table(box=box.ROUNDED, header_style="bold", width=tbl_w)
            table.add_column("#", style="dim", justify="right", width=3)
            table.add_column("Build", width=7)
            table.add_column("Backend", width=8)
            table.add_column("Path")
            for i, inst in enumerate(installs, 1):
                is_active = active_dir and (
                    str(Path(inst.bin_dir).resolve()) == active_dir
                )
                path_str = _truncate_path(inst.bin_dir, path_max)
                tag = "  [green]← active[/]" if is_active else ""
                table.add_row(str(i), inst.build, inst.backend.value, path_str + tag)
            console.print(table)
        else:
            console.print("  [yellow]No llama.cpp installations found.[/]")

        # Latest release + diff
        console.print()
        if latest:
            if active_dir and installs:
                active_inst = next(
                    (x for x in installs if str(Path(x.bin_dir).resolve()) == active_dir),
                    installs[0],
                )
                try:
                    diff = int(latest.lstrip("b")) - int(active_inst.build.lstrip("b"))
                    if diff <= 0:
                        console.print(f"  [green]✓ Up to date[/]  ({latest})")
                    else:
                        console.print(
                            f"  Latest: [bold]{latest}[/]  "
                            f"[dim](active is {diff} version{'s' if diff != 1 else ''} behind)[/]"
                        )
                except ValueError:
                    console.print(f"  Latest: [bold]{latest}[/]")
            else:
                console.print(f"  Latest: [bold]{latest}[/]")
        else:
            console.print("  [dim]Could not reach GitHub.[/]")

        # Rollback info
        prev = env_manager.get_previous()

        # Two-line hint (to fit everything)
        row1 = "[bold]#[/bold] use"
        if installs:
            row1 += "  [bold]x[/bold] delete"
        row1 += "  [bold]a[/bold] add path  [bold]s[/bold] deep scan  [bold]b[/bold] back"
        row2 = ""
        if latest:
            row2 += f"[bold]u[/bold] update→{latest}  "
        row2 += "[bold]i[/bold] install version"
        if prev:
            row2 += f"  [bold]r[/bold] rollback→{prev[1]}"
        console.print(f"\n  [dim]{row1}[/dim]")
        console.print(f"  [dim]{row2}[/dim]")

        choice = _ask("").strip().lower()

        if choice in ("b", ""):
            return

        elif choice == "a":
            raw = _ask("Path to directory with llama-bench: ").strip("'\"")
            p = Path(raw).expanduser()
            bench = p / env_manager._exe("llama-bench")
            cli_p = p / env_manager._exe("llama-cli")
            if not bench.exists() or not cli_p.exists():
                console.print(f"  [red]llama-bench or llama-cli not found in: {p}[/]")
            else:
                from ggtune.utils.shell import make_env_with_lib as _menv
                env_d = _menv(str(p))
                build_v = env_manager._get_build_version(cli_p, env_d) or "unknown"
                inst = env_manager.LlamaInstall(
                    bin_dir=str(p), build=build_v,
                    backend=hw.backend, found_via="manual",
                )
                env_manager.set_active(inst)
                console.print(f"  [green]✓ Activated: {build_v} at {p}[/]")
            _ask("\nPress Enter to continue")

        elif choice == "x":
            if not installs:
                continue
            n = _ask("Delete installation #: ")
            try:
                idx = int(n) - 1
                inst = installs[idx]
            except (ValueError, IndexError):
                console.print("  [red]Invalid number.[/]")
                _ask("\nPress Enter to continue")
                continue
            is_active = active_dir and str(Path(inst.bin_dir).resolve()) == active_dir
            if is_active:
                console.print("  [yellow]This is the active installation — activate another first.[/]")
                _ask("\nPress Enter to continue")
                continue
            console.print(f"  [dim]{inst.bin_dir}[/]")
            confirm = _ask(f"  Delete {inst.build} from disk? (y/n): ")
            if confirm.lower() == "y":
                try:
                    shutil.rmtree(inst.bin_dir)
                    console.print(f"  [green]✓ Deleted {inst.build}.[/]")
                    _refresh()
                except Exception as e:
                    console.print(f"  [red]Delete failed: {e}[/]")
            else:
                console.print("  [dim]Cancelled.[/]")
            _ask("\nPress Enter to continue")

        elif choice == "u":
            _do_install(latest or LLAMA_CPP_PINNED_BUILD, hw, env_manager, LLAMA_INSTALL_DIR)
            _refresh()
            _ask("\nPress Enter to continue")

        elif choice == "i":
            raw = _ask(
                f"Build number (e.g. b9072 or 9072, Enter = stable {LLAMA_CPP_PINNED_BUILD}): "
            ).strip()
            if not raw:
                target = LLAMA_CPP_PINNED_BUILD
            else:
                target = raw if raw.startswith("b") else f"b{raw}"
            _do_install(target, hw, env_manager, LLAMA_INSTALL_DIR)
            _refresh()
            _ask("\nPress Enter to continue")

        elif choice == "r":
            if prev:
                if env_manager.rollback():
                    console.print(f"  [green]✓ Rolled back to {prev[1]}.[/]")
                    _refresh()
                else:
                    console.print("  [red]Rollback failed — path no longer exists.[/]")
            else:
                console.print("  [yellow]No previous installation recorded.[/]")
            _ask("\nPress Enter to continue")

        elif choice == "s":
            console.print("  [dim]Deep scanning — may take 30–60 seconds...[/]")
            installs = env_manager.scan_deep(hw)
            if installs:
                console.print(f"  [green]Found {len(installs)} installation(s).[/]")
            else:
                console.print("  [yellow]No llama.cpp installations found on this system.[/]")
            _ask("  Press Enter to continue")

        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(installs):
                    inst = installs[idx]
                    env_manager.set_active(inst)
                    console.print(
                        f"  [green]✓ Activated: {inst.build} ({inst.backend.value})[/]\n"
                        f"  [dim]{inst.bin_dir}[/]"
                    )
                    _refresh()
                    _ask("\nPress Enter to continue")
                else:
                    console.print(f"  [red]Enter 1–{len(installs)}, a, d, i, r, s, or b.[/]")
            except ValueError:
                console.print("  [red]Unknown command.[/]")


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
            console.print("\n[red]✗ Critical tests failed.[/]")
            from ggtune.config import LLAMA_CPP_PINNED_BUILD
            fix = _ask(f"  Install stable build ({LLAMA_CPP_PINNED_BUILD}, tested)? (y/n): ")
            if fix.lower() == "y":
                console.print(f"[dim]Building {LLAMA_CPP_PINNED_BUILD} — 5–20 min...[/]")
                try:
                    env_manager.install(hw)
                    console.print(f"[green]✓ Stable build {LLAMA_CPP_PINNED_BUILD} installed.[/]")
                except Exception as be:
                    console.print(f"[red]Build failed: {be}[/]")
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
    confirm = _ask("  Are you sure? (y/n): ")
    if confirm.lower() == "y":
        try:
            deleted = profile_storage.delete_all()
            console.print(f"  [green]✓ Deleted {deleted} profile(s).[/]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/]")
    else:
        console.print("  [dim]Cancelled.[/]")
    _ask("\nPress Enter to go back")
