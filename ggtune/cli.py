"""CLI entry point — ggtune <command>."""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="ggtune",
    help="Stop guessing llama.cpp parameters. Benchmark and find the fastest settings.",
    no_args_is_help=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Launch interactive TUI when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from ggtune.tui import main_menu
        main_menu()


@app.command(name="help")
def help_cmd(ctx: typer.Context) -> None:
    """Show this help message."""
    import click
    root = ctx
    while root.parent:
        root = root.parent
    click.echo(root.get_help())


@app.command()
def run(
    model: str = typer.Argument(..., help="Path to .gguf model file"),
    force: bool = typer.Option(False, "--force", "-f", help="Ignore cached profile"),
    auto_build: bool = typer.Option(False, "--auto-build", help="Auto-install llama.cpp if missing"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick tune (~5 min, no context search)"),
) -> None:
    """Benchmark and find optimal llama.cpp parameters for a model."""
    from ggtune.orchestrator import run as _run
    _run(model, force=force, auto_build=auto_build, quick=quick)


@app.command()
def quick(
    model: str = typer.Argument(..., help="Path to .gguf model file"),
) -> None:
    """Quick tune — probe + Optuna only (~5 minutes)."""
    from ggtune.orchestrator import run as _run
    _run(model, quick=True)


@app.command()
def serve(
    model: str = typer.Argument(..., help="Path to .gguf model file"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the web UI once the server is ready"),
) -> None:
    """Launch llama-server with the best known settings (run 'ggtune run' first)."""
    import subprocess
    import time
    import webbrowser
    from ggtune.modules import hardware_scanner, env_manager, profile_storage, gguf_reader, model_tracker, advisor
    from ggtune.utils.formatting import error, info

    hw = hardware_scanner.scan()
    try:
        env_cfg = env_manager.detect(hw)
    except RuntimeError as e:
        error(str(e))
        raise typer.Exit(1)

    cached = profile_storage.load(model, hw, env_cfg.build)
    if not cached:
        error(
            "No benchmark profile for this model on this hardware/llama.cpp build.\n"
            "  Run [bold]ggtune run[/] (or [bold]ggtune quick[/]) first."
        )
        raise typer.Exit(1)

    model_profile = gguf_reader.read(model)
    mmproj_path = None
    tracked = model_tracker.find_by_path(model)
    if tracked and tracked.mmproj_path and Path(tracked.mmproj_path).exists():
        mmproj_path = tracked.mmproj_path

    argv = advisor.build_launch_argv(
        model_profile, cached.best_params, cached.optimal_context, env_cfg, mmproj_path,
    )
    console.print(f"[dim]{' '.join(argv)}[/]")
    info(f"Starting llama-server ({cached.tg_tokens_per_sec:.1f} t/s expected)...")

    proc = subprocess.Popen(argv, env=env_cfg.env_dict)
    try:
        if open_browser:
            time.sleep(4)
            webbrowser.open("http://localhost:8080")
        proc.wait()
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping llama-server...[/]")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.command()
def export(
    model: str = typer.Argument(..., help="Path to .gguf model file"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output script path (default: alongside the model)"),
) -> None:
    """Save the best known launch command as a runnable script (.bat / .sh)."""
    import platform
    import stat
    from ggtune.modules import hardware_scanner, env_manager, profile_storage, gguf_reader, model_tracker, advisor
    from ggtune.utils.formatting import error, success

    hw = hardware_scanner.scan()
    try:
        env_cfg = env_manager.detect(hw)
    except RuntimeError as e:
        error(str(e))
        raise typer.Exit(1)

    cached = profile_storage.load(model, hw, env_cfg.build)
    if not cached:
        error(
            "No benchmark profile for this model on this hardware/llama.cpp build.\n"
            "  Run [bold]ggtune run[/] (or [bold]ggtune quick[/]) first."
        )
        raise typer.Exit(1)

    model_profile = gguf_reader.read(model)
    mmproj_path = None
    tracked = model_tracker.find_by_path(model)
    if tracked and tracked.mmproj_path and Path(tracked.mmproj_path).exists():
        mmproj_path = tracked.mmproj_path

    argv = advisor.build_launch_argv(
        model_profile, cached.best_params, cached.optimal_context, env_cfg, mmproj_path,
    )

    is_windows = platform.system() == "Windows"
    ext = ".bat" if is_windows else ".sh"
    out_path = Path(out) if out else Path(model).with_suffix("").with_name(Path(model).stem + "-serve" + ext)

    if is_windows:
        quoted = " ".join(f'"{a}"' if " " in a else a for a in argv)
        script = f"@echo off\r\n{quoted}\r\n"
    else:
        import shlex
        quoted = shlex.join(argv)
        script = (
            "#!/bin/sh\n"
            f'export LD_LIBRARY_PATH="{env_cfg.bin_dir}:$LD_LIBRARY_PATH"\n'
            f"{quoted}\n"
        )

    out_path.write_text(script)
    if not is_windows:
        out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    success(f"Saved to {out_path}")


@app.command()
def browse(
    author: str = typer.Option("unsloth", "--author", help="HuggingFace author to search"),
    vram: Optional[float] = typer.Option(None, "--vram", help="Override VRAM GB"),
) -> None:
    """Find and download models that fit your hardware."""
    from ggtune.modules import hardware_scanner, hf_browser
    hw = hardware_scanner.scan()
    path = hf_browser.interactive_browse(hw, author=author, vram_gb=vram)
    if path:
        if typer.confirm(f"\nRun benchmark on {path.name}?"):
            from ggtune.orchestrator import run as _run
            _run(str(path))


@app.command()
def show(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model path"),
) -> None:
    """Show saved benchmark profiles."""
    from ggtune.modules import profile_storage
    profiles = profile_storage.list_all()
    if not profiles:
        console.print("[dim]No saved profiles.[/]")
        return
    for p in profiles:
        match = not model or model in p.model_path
        if match:
            console.print(
                f"[bold]{p.model_name}[/] ({p.model_quantization})  "
                f"[green]{p.tg_tokens_per_sec:.1f} t/s[/]  "
                f"ctx={p.optimal_context:,}  "
                f"[dim]{p.created_at[:10]}  {p.hw_gpu_name}[/]"
            )


@app.command()
def clear(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Clear specific model cache"),
    all_: bool = typer.Option(False, "--all", help="Clear all cached profiles"),
) -> None:
    """Clear cached benchmark profiles."""
    from ggtune.modules import profile_storage
    if all_:
        n = profile_storage.delete_all()
        console.print(f"[green]Cleared {n} profiles.[/]")
    elif model:
        from ggtune.modules import hardware_scanner
        hw = hardware_scanner.scan()
        ok = profile_storage.delete(model, hw)
        console.print("[green]Profile deleted.[/]" if ok else "[yellow]No matching profile.[/]")
    else:
        console.print("[red]Specify --model or --all.[/]")


@app.command()
def hw() -> None:
    """Show hardware information."""
    from ggtune.modules import hardware_scanner
    i = hardware_scanner.scan()
    console.print(f"[bold]GPU:[/]  {i.gpu_name}  {i.vram_total_mb // 1024}GB total / {i.vram_free_mb // 1024}GB free  ({i.backend.value})")
    console.print(f"[bold]CPU:[/]  {i.cpu_name}  {i.cores_physical} physical / {i.cores_logical} logical cores")
    console.print(f"[bold]RAM:[/]  {i.ram_total_gb:.1f}GB total / {i.ram_available_gb:.1f}GB available")
    console.print(f"[bold]OS:[/]   {i.os}  shell={i.shell}")
    if i.driver_version:
        console.print(f"[bold]Driver:[/] {i.driver_version}")
    if i.compute_cap:
        console.print(f"[bold]CUDA:[/]  compute {i.compute_cap}")


@app.command()
def info(
    model: str = typer.Argument(..., help="Path to .gguf model file"),
) -> None:
    """Show GGUF model metadata."""
    from ggtune.modules import gguf_reader
    m = gguf_reader.read(model)
    console.print(f"[bold]Name:[/]         {m.name}")
    console.print(f"[bold]Architecture:[/] {m.architecture}")
    console.print(f"[bold]MoE:[/]          {m.is_moe}")
    if m.is_moe:
        console.print(f"[bold]Experts:[/]      {m.n_experts_total} total / {m.n_experts_used} used")
    console.print(f"[bold]Layers:[/]       {m.n_layers}")
    console.print(f"[bold]Heads:[/]        {m.n_heads}  KV: {m.n_kv_heads or '?'}")
    console.print(f"[bold]Max context:[/]  {m.context_length_max:,}")
    console.print(f"[bold]Quantization:[/] {m.quantization}")
    console.print(f"[bold]File size:[/]    {m.file_size_gb:.2f} GB")


@app.command()
def scan(
    path: Optional[str] = typer.Argument(None, help="Directory to search (default: common locations)"),
) -> None:
    """Find .gguf models on your system."""
    from ggtune.tui import _scan_gguf_files
    import platform

    if path:
        from pathlib import Path as _Path
        found = {}
        def _add(p):
            n = p.name.lower()
            if p.suffix != ".gguf" or not p.is_file(): return
            if "mmproj" in n or "ggml-vocab" in n: return
            if p.stat().st_size < 50 * 1024 * 1024: return
            found[str(p)] = p.stat().st_size
        for f in _Path(path).rglob("*.gguf"):
            _add(f)
        items = sorted(found.items(), key=lambda x: -x[1])
    else:
        items = _scan_gguf_files()

    if not items:
        console.print("[yellow]No .gguf files found.[/]")
        return
    for i, (fpath, size) in enumerate(items, 1):
        print(f"  {i:2}.  {size / 1e9:5.1f} GB  {fpath}")
    print(f"\nFound {len(items)} models. Run with: ggtune run <path>")


@app.command()
def compat(
    report: bool = typer.Option(False, "--report", "-r", help="Show detailed report"),
    debug: bool = typer.Option(False, "--debug", help="Show raw test output"),
) -> None:
    """Run compatibility tests against installed llama.cpp."""
    from ggtune.modules import env_manager, compat_guard
    try:
        env_cfg = env_manager.detect()
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    compat_report = compat_guard.run_tests(env_cfg.bin_dir, env_cfg.build)
    compat_guard.print_report(compat_report)

    if debug:
        for r in compat_report.results:
            if not r.passed:
                console.print(f"\n[bold]{r.test.name} output:[/]\n{r.output}")

    if not compat_report.all_critical_passed:
        raise typer.Exit(1)


@app.command()
def update(
    check: bool = typer.Option(False, "--check", help="Check for relevant llama.cpp changes (no update)"),
    to: Optional[str] = typer.Option(None, "--to", help="Target build (e.g. b9014)"),
) -> None:
    """Update llama.cpp (safe: compat tests before swap)."""
    from ggtune.modules import env_manager, compat_guard
    from ggtune.config import LLAMA_CPP_PINNED_BUILD
    import shutil
    from ggtune.config import LLAMA_INSTALL_DIR

    try:
        env_cfg = env_manager.detect()
        current = env_cfg.build
    except RuntimeError:
        current = LLAMA_CPP_PINNED_BUILD

    if check:
        changes = compat_guard.check_for_changes(current)
        compat_guard.print_changes(changes, current)
        return

    target = to or compat_guard.get_latest_build() or LLAMA_CPP_PINNED_BUILD
    console.print(f"Updating llama.cpp to [bold]{target}[/]...")
    from ggtune.modules.hardware_scanner import scan
    hw = scan()

    temp_dir = LLAMA_INSTALL_DIR.parent / "llama.cpp.new"
    old_dir = LLAMA_INSTALL_DIR.parent / "llama.cpp.old"

    try:
        new_bin = None
        if env_manager.prebuilt_available():
            console.print(f"  [dim]Downloading pre-built {target}...[/]")
            new_bin = env_manager.download_prebuilt(hw.backend, target, temp_dir)
            if new_bin is None:
                console.print("  [yellow]No pre-built binary available — building from source instead.[/]")
        if new_bin is None:
            new_bin = env_manager._build_llama_cpp(hw.backend, target, temp_dir)

        report = compat_guard.run_tests(str(new_bin), target)
        if report.all_critical_passed:
            if LLAMA_INSTALL_DIR.exists():
                shutil.rmtree(str(old_dir), ignore_errors=True)
                shutil.move(str(LLAMA_INSTALL_DIR), str(old_dir))
            # Move just the bin directory into place — temp_dir may also
            # hold source/checkout cruft (source-build case) we don't keep.
            shutil.move(str(new_bin), str(LLAMA_INSTALL_DIR))
            shutil.rmtree(str(temp_dir), ignore_errors=True)
            env_manager._save_env_json(LLAMA_INSTALL_DIR, target, hw.backend)
            console.print(f"[green]✓ Updated to {target}[/]")
        else:
            console.print(f"[red]Compat tests failed for {target}. Keeping {current}.[/]")
            shutil.rmtree(str(temp_dir), ignore_errors=True)
    except Exception as e:
        console.print(f"[red]Update failed: {e}[/]")
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
