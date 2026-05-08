"""Orchestrator — ties all modules together for the 'run' flow."""
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from ggtune.modules import (
    hardware_scanner,
    gguf_reader,
    env_manager,
    process_manager,
    search_space_builder,
    benchmark_engine,
    profile_storage,
    advisor,
)
from ggtune.utils.formatting import info, warn, error

console = Console()


class _ThermalWatcher:
    """Polls CPU/GPU temperatures in background during benchmark."""

    def __init__(self) -> None:
        self.max_cpu: float = 0.0
        self.max_gpu: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _poll(self) -> None:
        import psutil
        while not self._stop.is_set():
            try:
                for entries in psutil.sensors_temperatures().values():
                    for e in entries:
                        if e.current:
                            self.max_cpu = max(self.max_cpu, e.current)
            except Exception:
                pass
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=3,
                )
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        self.max_gpu = max(self.max_gpu, float(line))
            except Exception:
                pass
            self._stop.wait(5)

    def report(self) -> None:
        if self.max_cpu > 90:
            warn(f"CPU peaked at {self.max_cpu:.0f}°C during benchmark — possible throttling")
        elif self.max_cpu > 0:
            console.print(f"  [dim]CPU peak temp: {self.max_cpu:.0f}°C[/]")
        if self.max_gpu > 85:
            warn(f"GPU peaked at {self.max_gpu:.0f}°C during benchmark — possible throttling")
        elif self.max_gpu > 0:
            console.print(f"  [dim]GPU peak temp: {self.max_gpu:.0f}°C[/]")


def run(
    model_path: str,
    force: bool = False,
    auto_build: bool = False,
    quick: bool = False,
    mmproj_path: Optional[str] = None,
) -> None:
    path = Path(model_path)
    if not path.exists():
        error(f"Model not found: {model_path}")
        raise SystemExit(1)

    # Hardware
    info("Scanning hardware...")
    hw = hardware_scanner.scan()
    console.print(f"  [bold]{hw}[/]")

    # Model metadata
    info("Reading model...")
    model = gguf_reader.read(model_path)
    console.print(f"  [bold]{model}[/]")

    # Sanity check — use total RAM, not available (fluctuates with running processes)
    if model.file_size_gb > hw.ram_total_gb * 0.85:
        error(
            f"Model ({model.file_size_gb:.1f}GB) is too large for this system "
            f"({hw.ram_total_gb:.1f}GB RAM total)."
        )
        raise SystemExit(1)

    # llama.cpp environment
    info("Checking llama.cpp environment...")
    try:
        env_cfg = env_manager.detect(hw)
        console.print(f"  llama.cpp [bold]{env_cfg.build}[/] ({env_cfg.backend.value})")
    except RuntimeError as e:
        if auto_build:
            info("Installing llama.cpp...")
            env_cfg = env_manager.install(hw)
        else:
            error(str(e))
            raise SystemExit(1)

    # Free up resources
    process_manager.prompt_and_kill()

    # Build search space
    info("Building search space...")
    try:
        space = search_space_builder.build(hw, model)
    except RuntimeError as e:
        error(str(e))
        raise SystemExit(1)

    quick_runs = space.estimated_quick_probe_runs()
    from ggtune.config import OPTUNA_TRIALS, STABILITY_RUNS
    ctx_runs = len(space.context_candidates)
    total_est = quick_runs + OPTUNA_TRIALS + ctx_runs + STABILITY_RUNS
    console.print(f"  ~{total_est} benchmark runs")

    # Benchmark + thermal monitoring
    thermal = _ThermalWatcher()
    thermal.start()
    t0 = time.time()
    try:
        if quick:
            info("Quick mode: probe + Optuna only (no context search)")
            probe = benchmark_engine.quick_probe(env_cfg, model, space)
            best_params, peak_tg = benchmark_engine.optuna_search(env_cfg, model, space, probe)
            mean_tg, std_tg, cv = benchmark_engine.stability_pass(env_cfg, model, best_params, 8192)
            result = {
                "best_params": best_params,
                "tg_tokens_per_sec": mean_tg,
                "tg_std": std_tg,
                "stability_cv": cv,
                "optimal_ctx": 8192,
                "optuna_best": peak_tg,
                "optuna_trials": OPTUNA_TRIALS,
            }
        else:
            result = benchmark_engine.run_full(env_cfg, model, space)
            result["optuna_trials"] = OPTUNA_TRIALS
    finally:
        thermal.stop()

    total_min = (time.time() - t0) / 60
    result["thermal_max_cpu"] = thermal.max_cpu
    result["thermal_max_gpu"] = thermal.max_gpu

    # Determine bottleneck
    bottleneck = advisor.determine_bottleneck(hw, model, result["best_params"], result["tg_tokens_per_sec"])

    # Save profile
    profile_storage.save(model, hw, result, bottleneck, total_min)

    advisor.print_report(model, hw, result, env_cfg, total_min, mmproj_path=mmproj_path)
