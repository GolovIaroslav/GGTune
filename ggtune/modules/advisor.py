"""Module 8: Advisor & Output — interpret results, print report."""
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from typing import Optional

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


def generate_launch_cmd(
    model: ModelProfile,
    params: dict,
    optimal_ctx: int,
    env_cfg: EnvConfig,
    mmproj_path: Optional[str] = None,
) -> str:
    flags = [
        f"-m {model.path}",
        "-ngl 999",
    ]
    if mmproj_path:
        flags.append(f"--mmproj {mmproj_path}")
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
    return f"{ld} \\\n  {env_cfg.llama_server_path} \\\n  {flags_str}"


def _build_diagnostics(
    hw: HardwareProfile,
    model: ModelProfile,
    result: dict,
    env_cfg: EnvConfig,
) -> list:
    """Return list of Rich-formatted diagnostic lines."""
    from ggtune.modules.search_space_builder import kv_mb_per_token

    params = result["best_params"]
    tg = result["tg_tokens_per_sec"]
    ctx = result.get("optimal_ctx", 8192)
    cv = result.get("stability_cv", 0.0)
    max_cpu = result.get("thermal_max_cpu", 0.0)
    max_gpu = result.get("thermal_max_gpu", 0.0)

    lines = ["[bold cyan]DIAGNOSTICS[/]"]

    # Speed assessment
    if tg < 5:
        lines.append("  [red]✗ TG < 5 t/s — too slow for real-time use. "
                     "Try a smaller/lower-quant model.[/]")
    elif tg < 15:
        lines.append(f"  [yellow]⚠  TG {tg:.1f} t/s — usable but slow. "
                     "Consider smaller quantization or fewer context tokens.[/]")
    else:
        lines.append(f"  [green]✓ TG {tg:.1f} t/s — good for real-time generation.[/]")

    # Stability
    if cv > 0.15:
        lines.append(f"  [yellow]⚠  High variance (CV={cv:.0%}) — background load interferes. "
                     "Close other apps and re-run.[/]")

    # VRAM vs model size
    vram_gb = hw.vram_total_mb / 1024
    if model.file_size_gb > vram_gb:
        offload_gb = model.file_size_gb - vram_gb
        lines.append(f"  [yellow]⚠  Model {model.file_size_gb:.1f}GB > VRAM {vram_gb:.1f}GB — "
                     f"{offload_gb:.1f}GB in RAM (via ncmoe). "
                     "Speed limited by RAM bandwidth.[/]")
        lines.append("     → Upgrade: larger VRAM GPU or smaller model quant (Q4_K_M).")

    # KV cache RAM info
    mb_per_tok = kv_mb_per_token(model)
    kv_gb = ctx * mb_per_tok / 1024
    total_needed_gb = model.file_size_gb + kv_gb + 4.0  # model + KV + OS overhead
    ram_free_at_ctx = hw.ram_total_gb - total_needed_gb
    if ram_free_at_ctx < 2:
        lines.append(f"  [red]✗ KV cache at {ctx:,} tokens ≈ {kv_gb:.1f}GB — "
                     f"only {ram_free_at_ctx:.1f}GB RAM left for OS! Reduce context.[/]")
        safe_ctx = int((hw.ram_total_gb - model.file_size_gb - 4 - 3) * 0.9 * 1024 / mb_per_tok) if mb_per_tok > 0 else ctx
        safe_ctx = max(4096, (safe_ctx // 4096) * 4096)
        lines.append(f"     → Safer: -c {safe_ctx:,}")
    elif ram_free_at_ctx < 6:
        lines.append(f"  [yellow]⚠  KV cache at {ctx:,} tokens ≈ {kv_gb:.1f}GB. "
                     f"{ram_free_at_ctx:.1f}GB RAM left — tight, close other apps.[/]")
    else:
        lines.append(f"  [dim]ℹ  Context {ctx:,} → KV cache ≈ {kv_gb:.1f}GB RAM "
                     f"({ram_free_at_ctx:.0f}GB remaining for OS)[/]")

    # Quantization advice
    q = model.quantization.upper()
    if "Q2" in q or "IQ1" in q or "IQ2" in q:
        lines.append("  [yellow]⚠  Very low quantization (Q2/IQ2) — quality is reduced. "
                     "Q4_K_M offers better balance.[/]")

    # MoE ncmoe info
    if model.is_moe and "ncmoe" in params:
        ncmoe = params["ncmoe"]
        n_total = model.n_experts_total or 256
        in_gpu = n_total - ncmoe
        lines.append(f"  [dim]ℹ  MoE: {in_gpu} experts in VRAM, {ncmoe} in RAM "
                     f"(-ncmoe {ncmoe})[/]")

    # Thermal
    if max_cpu > 90:
        lines.append(f"  [red]🌡 CPU throttled! Peak {max_cpu:.0f}°C — "
                     "results are pessimistic. Improve cooling.[/]")
    elif max_cpu > 80:
        lines.append(f"  [yellow]🌡 CPU warm: peak {max_cpu:.0f}°C — "
                     "check thermal paste / fans.[/]")
    elif max_cpu > 0:
        lines.append(f"  [dim]🌡 CPU peak: {max_cpu:.0f}°C (GPU: {max_gpu:.0f}°C)[/]")

    if max_gpu > 85:
        lines.append(f"  [red]🌡 GPU throttled! Peak {max_gpu:.0f}°C.[/]")

    # CUDA / driver
    if hw.backend.value == "cuda":
        lines.append(f"  [dim]ℹ  CUDA backend · driver {hw.driver_version or 'unknown'} "
                     f"· llama.cpp {env_cfg.build}[/]")
        lines.append("     → Run [bold]ggtune update --check[/] to see relevant llama.cpp changes.")
    else:
        lines.append(f"  [yellow]⚠  Backend: {hw.backend.value} (not CUDA). "
                     "CUDA gives 2–5× better TG speed.[/]")

    # Unsloth model
    if "unsloth" in model.path.lower():
        lines.append("  [dim]✓ Unsloth model — GGUFs are well-optimized.[/]")

    return lines


def print_report(
    model: ModelProfile,
    hw: HardwareProfile,
    result: dict,
    env_cfg: EnvConfig,
    total_min: float,
    mmproj_path: Optional[str] = None,
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

    launch_cmd = generate_launch_cmd(model, params, ctx, env_cfg, mmproj_path)

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
        "[bold cyan]LAUNCH COMMAND[/]",
        f"  [bold]{launch_cmd}[/]",
    ]

    if params.get("nkvo"):
        lines.insert(-1, "")
        lines.insert(-1, "[yellow]⚠  KV cache in RAM. Speed decreases as context grows.[/]")

    console.print(Panel("\n".join(lines), title="GGTune Results", border_style="cyan"))
    console.print("\n".join(_build_diagnostics(hw, model, result, env_cfg)))

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
