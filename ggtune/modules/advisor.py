"""Module 8: Advisor & Output — interpret results, print report."""
import os
import platform
from pathlib import Path

_SYSTEM = platform.system()

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from typing import List, Optional

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


def build_launch_argv(
    model: ModelProfile,
    params: dict,
    optimal_ctx: int,
    env_cfg: EnvConfig,
    mmproj_path: Optional[str] = None,
) -> List[str]:
    """Argv for actually spawning llama-server — each arg its own list item

    (safe for paths with spaces, unlike the display string from
    generate_launch_cmd which is meant for copy-paste into a shell).
    """
    argv = [
        env_cfg.llama_server_path,
        "-m", model.path,
        "-ngl", str(params["ngl"] if "ngl" in params else 999),
    ]
    if mmproj_path:
        argv += ["--mmproj", mmproj_path]
    if model.is_moe and "ncmoe" in params:
        argv += ["-ncmoe", str(params["ncmoe"])]
    argv += [
        "-fa", "on" if params.get("flash_attn", True) else "off",
        "-t", str(params.get("threads", 8)),
        "-ctk", params.get("ctk", "f16"),
        "-ctv", params.get("ctk", "f16"),
    ]
    if params.get("nkvo"):
        argv.append("-nkvo")
    argv += [
        "-c", str(optimal_ctx),
        "--jinja",
        "--port", "8080",
    ]
    return argv


def generate_launch_cmd(
    model: ModelProfile,
    params: dict,
    optimal_ctx: int,
    env_cfg: EnvConfig,
    mmproj_path: Optional[str] = None,
) -> str:
    """Human-readable command string for copy-paste into a shell."""
    argv = build_launch_argv(model, params, optimal_ctx, env_cfg, mmproj_path)

    parts = []
    i = 1  # argv[0] is the binary path, handled separately below
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            parts.append(f"{tok} {argv[i + 1]}")
            i += 2
        else:
            parts.append(tok)
            i += 1

    if _SYSTEM == "Windows":
        return f"{argv[0]} {' '.join(parts)}"
    else:
        ld = f"LD_LIBRARY_PATH={env_cfg.bin_dir}:$LD_LIBRARY_PATH"
        return f"{ld} {argv[0]} {' '.join(parts)}"


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

    # MoE ncmoe info — ncmoe is a LAYER count (llama.cpp keeps that many
    # layers' MoE weights in CPU RAM), not an expert count.
    if model.is_moe and "ncmoe" in params:
        ncmoe = params["ncmoe"]
        layers_on_gpu = model.n_layers - ncmoe
        lines.append(f"  [dim]ℹ  MoE: {layers_on_gpu}/{model.n_layers} layers' experts in VRAM, "
                     f"{ncmoe} layers in RAM (-ncmoe {ncmoe})[/]")

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
    pp = result.get("pp_tokens_per_sec", 0.0)
    std = result.get("tg_std", 0.0)
    cv = result.get("stability_cv", 0.0)
    ctx = result.get("optimal_ctx", 8192)
    bottleneck = determine_bottleneck(hw, model, params, tg)

    param_parts = []
    if model.is_moe and "ncmoe" in params:
        param_parts.append(f"-ncmoe {params['ncmoe']}")
    elif "ngl" in params:
        param_parts.append(f"-ngl {params['ngl']}")
    param_parts += [
        f"-t {params.get('threads', 8)}",
        f"-ctk {params.get('ctk', 'f16')}",
        f"-ctv {params.get('ctk', 'f16')}",
        f"-fa {'on' if params.get('flash_attn', True) else 'off'}",
    ]
    if params.get("nkvo"):
        param_parts.append("-nkvo")
    param_str = "  ".join(param_parts)

    launch_cmd = generate_launch_cmd(model, params, ctx, env_cfg, mmproj_path)

    results_lines = [f"  TG speed:   [bold green]{tg:.1f} t/s[/]  (±{std:.1f}, CV={cv:.1%})"]
    if pp > 0:
        results_lines.append(f"  PP speed:   [bold green]{pp:.1f} t/s[/]")
    results_lines += [
        f"  Context:    [bold]{ctx:,} tokens[/]",
        f"  Bottleneck: {_bottleneck_text(bottleneck)}",
    ]

    lines = [
        f"[bold]Model:[/]    {model}",
        f"[bold]Hardware:[/] {hw}",
        f"[bold]Duration:[/] {total_min:.0f} min",
        "",
        "[bold cyan]RESULTS[/]",
        *results_lines,
        "",
        "[bold cyan]PARAMETERS[/]",
        f"  {param_str}",
        f"  -c {ctx}",
        "",
    ]

    if params.get("nkvo"):
        lines.append("")
        lines.append("[yellow]⚠  KV cache in RAM. Speed decreases as context grows.[/]")

    console.print(Panel("\n".join(lines), title="GGTune Results", border_style="cyan"))
    console.print("\n".join(_build_diagnostics(hw, model, result, env_cfg)))
    console.print()
    console.print(Rule("Launch Command", style="bold cyan"))
    console.print(launch_cmd)
    console.print()

    return bottleneck


def print_cached(profile: StoredProfile) -> None:
    pp_line = (f"  PP speed:  [bold green]{profile.pp_tokens_per_sec:.1f} t/s[/]\n"
               if profile.pp_tokens_per_sec > 0 else "")
    console.print(Panel(
        f"[bold]Model:[/] {profile.model_name} ({profile.model_quantization})\n"
        f"[bold]GPU:[/]   {profile.hw_gpu_name}\n\n"
        f"[bold cyan]Cached results[/] (from {profile.created_at[:10]})\n"
        f"  TG speed:  [bold green]{profile.tg_tokens_per_sec:.1f} t/s[/]\n"
        f"{pp_line}"
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
