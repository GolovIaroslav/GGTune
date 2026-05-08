"""Module 8: Advisor & Output — interpret results, generate alias, print report."""
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Confirm

from ggtune.models.hardware import HardwareProfile, Backend
from ggtune.models.model_profile import ModelProfile
from ggtune.models.profile import StoredProfile
from ggtune.modules.env_manager import EnvConfig

console = Console()


def determine_bottleneck(hw: HardwareProfile, model: ModelProfile, best_params: dict, tg: float) -> str:
    if model.file_size_gb > hw.vram_free_mb / 1024 * 0.9:
        return "vram_limited"
    if tg < 10:
        return "memory_bandwidth"
    if best_params.get("threads", 0) > hw.cores_physical:
        return "cpu_bound"
    return "well_balanced"


def _bottleneck_text(bottleneck: str) -> str:
    return {
        "vram_limited":       "VRAM (model partially in RAM)",
        "memory_bandwidth":   "Memory bandwidth",
        "cpu_bound":          "CPU bound",
        "well_balanced":      "Well balanced",
    }.get(bottleneck, bottleneck)


def generate_alias_cmd(
    alias_name: str,
    model: ModelProfile,
    params: dict,
    optimal_ctx: int,
    env_cfg: EnvConfig,
) -> str:
    flags = [
        f"-m {model.path}",
        "-ngl 999",
    ]
    if model.is_moe and "ncmoe" in params:
        flags.append(f"-ncmoe {params['ncmoe']}")
    flags += [
        "-fa on",
        f"-t {params.get('threads', 8)}",
        f"-ctk {params.get('ctk', 'f16')}",
        f"-ctv {params.get('ctk', 'f16')}",
    ]
    if params.get("nkvo"):
        flags.append("-nkvo")
    flags += [
        f"-c {optimal_ctx}",
        "--jinja",
        "--port 8080",
    ]

    flags_str = " \\\n  ".join(flags)
    ld = f"LD_LIBRARY_PATH={env_cfg.bin_dir}:$LD_LIBRARY_PATH"
    cmd = f"{ld} \\\n  {env_cfg.llama_server_path} \\\n  {flags_str}"
    return f"alias {alias_name}='{cmd} & sleep 3 && xdg-open http://localhost:8080'"


def _write_alias(alias_line: str, shell: str, model_name: str) -> bool:
    rc_map = {"zsh": "~/.zshrc", "bash": "~/.bashrc"}
    rc_file = Path(os.path.expanduser(rc_map.get(shell, "~/.bashrc")))
    try:
        with rc_file.open("a") as f:
            f.write(f"\n# GGTune — {model_name} — {datetime.now().date()}\n")
            f.write(alias_line + "\n")
        return True
    except Exception:
        return False


def print_report(
    model: ModelProfile,
    hw: HardwareProfile,
    result: dict,
    env_cfg: EnvConfig,
    alias_name: str,
    total_min: float,
) -> str:
    params = result["best_params"]
    tg = result["tg_tokens_per_sec"]
    std = result.get("tg_std", 0.0)
    cv = result.get("stability_cv", 0.0)
    ctx = result.get("optimal_ctx", 8192)
    bottleneck = determine_bottleneck(hw, model, params, tg)

    param_parts = []
    if model.is_moe and "ncmoe" in params:
        param_parts.append(f"-ncmoe {params['ncmoe']}")
    param_parts += [
        f"-t {params.get('threads', 8)}",
        f"-ctk {params.get('ctk', 'f16')}",
        f"-ctv {params.get('ctk', 'f16')}",
        "-fa on",
    ]
    if params.get("nkvo"):
        param_parts.append("-nkvo")
    param_str = "  ".join(param_parts)

    alias_line = generate_alias_cmd(alias_name, model, params, ctx, env_cfg)

    lines = [
        f"[bold]Model:[/]    {model}",
        f"[bold]Hardware:[/] {hw}",
        f"[bold]Duration:[/] {total_min:.0f} min",
        "",
        "[bold cyan]RESULTS[/]",
        f"  TG speed:   [bold green]{tg:.1f} t/s[/]  (±{std:.1f}, CV={cv:.1%})",
        f"  Context:    [bold]{ctx:,} tokens[/]",
        f"  Bottleneck: {_bottleneck_text(bottleneck)}",
        "",
        "[bold cyan]PARAMETERS[/]",
        f"  {param_str}",
        f"  -c {ctx}",
        "",
        "[bold cyan]ALIAS[/]",
        f"  [dim]{alias_line[:80]}{'...' if len(alias_line) > 80 else ''}[/]",
    ]

    if params.get("nkvo"):
        lines.insert(-1, "")
        lines.insert(-1, "[yellow]⚠  KV cache in RAM. Speed decreases as context grows.[/]")

    console.print(Panel("\n".join(lines), title="GGTune Results", border_style="cyan"))

    if Confirm.ask(f"\nWrite alias '{alias_name}' to ~/{'.zshrc' if hw.shell == 'zsh' else '.bashrc'}?"):
        ok = _write_alias(alias_line, hw.shell, model.name)
        if ok:
            console.print(f"[green]✓ Alias written. Type '[bold]{alias_name}[/]' to start.[/]")
        else:
            console.print("[red]Could not write alias. Add manually:[/]")
            console.print(alias_line)

    return bottleneck


def print_cached(profile: StoredProfile) -> None:
    console.print(Panel(
        f"[bold]Model:[/] {profile.model_name} ({profile.model_quantization})\n"
        f"[bold]GPU:[/]   {profile.hw_gpu_name}\n\n"
        f"[bold cyan]Cached results[/] (from {profile.created_at[:10]})\n"
        f"  TG speed:  [bold green]{profile.tg_tokens_per_sec:.1f} t/s[/]\n"
        f"  Context:   {profile.optimal_context:,} tokens\n"
        f"  Params:    {json_params(profile.best_params)}\n\n"
        "[dim]Use --force to re-run benchmark.[/]",
        title="GGTune — Cached Profile",
        border_style="cyan",
    ))


def json_params(params: dict) -> str:
    parts = []
    for k, v in params.items():
        if k in ("flash_attn",):
            continue
        parts.append(f"-{k} {v}")
    return "  ".join(parts)
