"""Module 6: Benchmark Engine — phases 6a (quick probe), 6b (Optuna TPE),
6c (context search), 6d (stability pass)."""
import csv
import statistics
import subprocess
import time
from typing import List, Optional, Tuple

import optuna
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

from ggtune.config import (
    BENCH_WARMUP_RUNS, BENCH_MEASUREMENT_RUNS,
    OPTUNA_TRIALS, OPTUNA_TIMEOUT_SEC,
    MIN_ACCEPTABLE_TG_RATIO, STABILITY_RUNS, STABILITY_CV_WARN,
)
from ggtune.models.bench_result import BenchResult
from ggtune.models.model_profile import ModelProfile
from ggtune.models.search_space import SearchSpace
from ggtune.modules.env_manager import EnvConfig
from ggtune.utils.formatting import warn, info

optuna.logging.set_verbosity(optuna.logging.WARNING)
console = Console()


# ── llama-bench output parser ──────────────────────────────────────────────

def parse_bench_output(stdout: str) -> Tuple[float, float]:
    """Parse llama-bench output and return (tg_tps, pp_tps).

    New format (build 9000+): CSV with header starting 'build_commit,...'
    Test type determined by n_prompt/n_gen columns; speed in avg_ts.
    Legacy format: header 'model,...', test type in parts[11], speed in parts[12].
    """
    tg = 0.0
    pp = 0.0
    lines = stdout.splitlines()

    # New format: header starts with build_commit
    for i, line in enumerate(lines):
        if not line.startswith("build_commit,"):
            continue
        header = line.split(",")
        try:
            n_prompt_idx = header.index("n_prompt")
            n_gen_idx = header.index("n_gen")
            avg_ts_idx = header.index("avg_ts")
        except ValueError:
            break
        for row in csv.reader(lines[i + 1:]):
            if len(row) <= max(n_prompt_idx, n_gen_idx, avg_ts_idx):
                continue
            try:
                n_prompt = int(row[n_prompt_idx])
                n_gen = int(row[n_gen_idx])
                ts = float(row[avg_ts_idx])
            except ValueError:
                continue
            if n_prompt > 0 and n_gen == 0:
                pp = max(pp, ts)
            elif n_gen > 0 and n_prompt == 0:
                tg = max(tg, ts)
        if tg > 0 or pp > 0:
            return tg, pp
        break

    # Legacy format: header starts with "model"
    for line in lines:
        s = line.strip()
        if not s or s.startswith("model,") or s.startswith("|"):
            continue
        parts = s.split(",")
        if len(parts) < 13:
            continue
        test_type = parts[11].strip().strip('"')
        try:
            tps = float(parts[12].strip().strip('"'))
        except ValueError:
            continue
        if test_type.startswith("tg"):
            tg = max(tg, tps)
        elif test_type.startswith("pp"):
            pp = max(pp, tps)

    if tg > 0 or pp > 0:
        return tg, pp

    # Markdown fallback
    for line in lines:
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        for i, col in enumerate(cols):
            if col.startswith("tg") and i + 1 < len(cols):
                try:
                    tg = max(tg, float(cols[i + 1]))
                except ValueError:
                    pass
            elif col.startswith("pp") and i + 1 < len(cols):
                try:
                    pp = max(pp, float(cols[i + 1]))
                except ValueError:
                    pass

    return tg, pp


# ── core runner ────────────────────────────────────────────────────────────

def _build_cmd(
    env_cfg: EnvConfig,
    model: ModelProfile,
    params: dict,
    ctx: int = 8192,
) -> List[str]:
    cmd = [
        env_cfg.llama_bench_path,
        "-m", model.path,
        "-ngl", "999",
        "-t", str(params.get("threads", 8)),
        "-pg", "512,128",
        "-r", str(BENCH_MEASUREMENT_RUNS),
        "-o", "csv",
    ]

    if params.get("flash_attn", True):
        cmd += ["-fa", "1"]

    ctk = params.get("ctk", "f16")
    cmd += ["-ctk", ctk, "-ctv", ctk]

    if model.is_moe and "ncmoe" in params:
        cmd += ["-ncmoe", str(params["ncmoe"])]

    if not model.is_moe and "ngl" in params:
        cmd[cmd.index("999")] = str(params["ngl"])

    if params.get("nkvo", False):
        cmd += ["-nkvo"]

    return cmd


def _kill_llama_procs() -> None:
    """Kill any lingering llama-bench or llama-cli processes."""
    import psutil
    for proc in psutil.process_iter(["pid", "name"]):
        name = (proc.info["name"] or "").lower()
        if any(x in name for x in ("llama-bench", "llama-cli")):
            try:
                proc.kill()
            except Exception:
                pass


def run_bench(
    env_cfg: EnvConfig,
    model: ModelProfile,
    params: dict,
    ctx: int = 8192,
    timeout: int = 180,
) -> BenchResult:
    cmd = _build_cmd(env_cfg, model, params, ctx)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env_cfg.env_dict,
        )
    except subprocess.TimeoutExpired:
        _kill_llama_procs()
        return BenchResult(params=params, crashed=True, error="timeout", context=ctx)
    except Exception as e:
        _kill_llama_procs()
        return BenchResult(params=params, crashed=True, error=str(e), context=ctx)

    duration = time.time() - t0

    if result.returncode != 0:
        _kill_llama_procs()
        error_text = result.stderr[:500] if result.stderr else "non-zero exit"
        return BenchResult(params=params, crashed=True, error=error_text, context=ctx, duration_sec=duration)

    tg, pp = parse_bench_output(result.stdout)
    if tg == 0 and pp == 0:
        _kill_llama_procs()
        return BenchResult(params=params, crashed=True, error="no output parsed", context=ctx, duration_sec=duration)

    return BenchResult(
        params=params,
        tg_tokens_per_sec=tg,
        pp_tokens_per_sec=pp,
        context=ctx,
        crashed=False,
        duration_sec=duration,
    )


# ── Phase 6a: Quick Probe ──────────────────────────────────────────────────

def quick_probe(
    env_cfg: EnvConfig,
    model: ModelProfile,
    space: SearchSpace,
) -> List[BenchResult]:
    """Coarse grid to eliminate bad regions and seed Optuna."""

    # Build probe grid
    if model.is_moe and space.ncmoe_range:
        r = space.ncmoe_range
        vals = list(r)
        ncmoe_vals = [vals[0], vals[len(vals) // 2], vals[-1]] if len(vals) >= 3 else vals
        param_grid = [
            {"ncmoe": nc, "threads": t, "ctk": ctk, "flash_attn": space.flash_attn, "nkvo": False}
            for nc in ncmoe_vals
            for t in space.thread_candidates[:2]
            for ctk in space.kv_quant_options[:2]
        ]
    else:
        ngl_vals = [999]
        param_grid = [
            {"ngl": ngl, "threads": t, "ctk": ctk, "flash_attn": space.flash_attn, "nkvo": False}
            for ngl in ngl_vals
            for t in space.thread_candidates[:2]
            for ctk in space.kv_quant_options[:2]
        ]

    results = []
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Quick probe", total=len(param_grid))
        for params in param_grid:
            r = run_bench(env_cfg, model, params)
            results.append(r)
            best = max((x.tg_tokens_per_sec for x in results if x.valid), default=0)
            status = f"{r.tg_tokens_per_sec:.1f} t/s" if r.valid else "crash"
            progress.update(task, advance=1, description=f"Quick probe  best: {best:.1f} t/s  last: {status}")

    valid = [r for r in results if r.valid]
    if not valid:
        raise RuntimeError("All quick probe runs crashed. Check llama.cpp installation.")
    return results


# ── Phase 6b: Optuna TPE ──────────────────────────────────────────────────

def optuna_search(
    env_cfg: EnvConfig,
    model: ModelProfile,
    space: SearchSpace,
    seed_results: List[BenchResult],
) -> Tuple[dict, float]:
    """Bayesian search warm-started from quick probe results."""

    def objective(trial: optuna.Trial) -> float:
        params: dict = {"flash_attn": space.flash_attn}

        if model.is_moe and space.ncmoe_range:
            params["ncmoe"] = trial.suggest_int(
                "ncmoe", space.ncmoe_range.start, space.ncmoe_range.stop - 1
            )
        else:
            params["ngl"] = 999

        params["threads"] = trial.suggest_categorical("threads", space.thread_candidates)
        params["ctk"] = trial.suggest_categorical("ctk", space.kv_quant_options)
        params["nkvo"] = trial.suggest_categorical("nkvo", space.nkvo_options)

        result = run_bench(env_cfg, model, params)
        if result.crashed:
            return float("-inf")
        return result.tg_tokens_per_sec

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=8, seed=42),
    )

    # Warm start from quick probe
    for r in seed_results:
        if not r.valid:
            continue
        p = r.params.copy()
        dists: dict = {}
        vals: dict = {}
        if "ncmoe" in p and space.ncmoe_range:
            dists["ncmoe"] = optuna.distributions.IntDistribution(
                space.ncmoe_range.start, space.ncmoe_range.stop - 1
            )
            vals["ncmoe"] = p["ncmoe"]
        if "threads" in p:
            dists["threads"] = optuna.distributions.CategoricalDistribution(space.thread_candidates)
            vals["threads"] = p["threads"]
        if "ctk" in p:
            dists["ctk"] = optuna.distributions.CategoricalDistribution(space.kv_quant_options)
            vals["ctk"] = p["ctk"]
        if "nkvo" in p:
            dists["nkvo"] = optuna.distributions.CategoricalDistribution(space.nkvo_options)
            vals["nkvo"] = p["nkvo"]
        if dists and vals:
            try:
                study.add_trial(optuna.trial.create_trial(
                    params=vals,
                    distributions=dists,
                    value=r.tg_tokens_per_sec,
                ))
            except Exception:
                pass

    completed = [0]

    def callback(study, trial):
        completed[0] += 1

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Optuna TPE", total=OPTUNA_TRIALS)

        def cb(study, trial):
            best = study.best_value if study.best_value else 0
            progress.update(task, advance=1, description=f"Optuna TPE   best: {best:.1f} t/s")

        study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_SEC, callbacks=[cb])

    best = study.best_params.copy()
    best["flash_attn"] = space.flash_attn
    if "ngl" not in best:
        best["ngl"] = 999
    return best, study.best_value


# ── Phase 6c: Context Binary Search ───────────────────────────────────────

def context_search(
    env_cfg: EnvConfig,
    model: ModelProfile,
    best_params: dict,
    space: SearchSpace,
    peak_tg: float,
) -> Tuple[int, dict]:
    """Find largest context where TG >= 60% of peak."""
    candidates = space.context_candidates
    if not candidates:
        return 8192, best_params

    min_tg = peak_tg * MIN_ACCEPTABLE_TG_RATIO

    lo, hi = 0, len(candidates) - 1

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Context search", total=hi + 1)

        while lo < hi:
            mid = (lo + hi + 1) // 2
            ctx = candidates[mid]
            r = run_bench(env_cfg, model, best_params, ctx=ctx)
            progress.advance(task, description=f"Context search  ctx={ctx}")
            if r.crashed or r.tg_tokens_per_sec < min_tg:
                hi = mid - 1
            else:
                lo = mid

    optimal_ctx = candidates[lo]

    # Check if nkvo helps for the chosen context
    if optimal_ctx > 32768 and not best_params.get("nkvo", False):
        params_nkvo = {**best_params, "nkvo": True}
        r = run_bench(env_cfg, model, params_nkvo, ctx=optimal_ctx)
        if r.valid and r.tg_tokens_per_sec >= min_tg:
            best_params = params_nkvo

    return optimal_ctx, best_params


# ── Phase 6d: Stability Pass ───────────────────────────────────────────────

def stability_pass(
    env_cfg: EnvConfig,
    model: ModelProfile,
    best_params: dict,
    optimal_ctx: int,
) -> Tuple[float, float, float]:
    """Run best params N times, return (mean_tg, std_tg, cv)."""
    tg_values = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn()) as progress:
        task = progress.add_task("Stability check", total=STABILITY_RUNS)
        for _ in range(STABILITY_RUNS):
            r = run_bench(env_cfg, model, best_params, ctx=optimal_ctx)
            if r.valid:
                tg_values.append(r.tg_tokens_per_sec)
            progress.advance(task)

    if not tg_values:
        return 0.0, 0.0, 1.0

    mean_tg = statistics.mean(tg_values)
    std_tg = statistics.stdev(tg_values) if len(tg_values) > 1 else 0.0
    cv = std_tg / mean_tg if mean_tg > 0 else 0.0

    if cv > STABILITY_CV_WARN:
        warn(
            f"High result variance (CV={cv:.0%}). "
            "Background processes may be interfering."
        )

    return mean_tg, std_tg, cv


# ── Full pipeline ──────────────────────────────────────────────────────────

def run_full(
    env_cfg: EnvConfig,
    model: ModelProfile,
    space: SearchSpace,
) -> dict:
    """Run all 4 phases and return result dict."""
    # 6a
    info("Phase 1/4: Quick probe")
    probe_results = quick_probe(env_cfg, model, space)

    # 6b
    info("Phase 2/4: Optuna TPE search")
    best_params, peak_tg = optuna_search(env_cfg, model, space, probe_results)

    # 6c
    info("Phase 3/4: Context binary search")
    optimal_ctx, best_params = context_search(env_cfg, model, best_params, space, peak_tg)

    # 6d
    info("Phase 4/4: Stability check")
    mean_tg, std_tg, cv = stability_pass(env_cfg, model, best_params, optimal_ctx)

    return {
        "best_params": best_params,
        "tg_tokens_per_sec": mean_tg,
        "tg_std": std_tg,
        "stability_cv": cv,
        "optimal_ctx": optimal_ctx,
        "optuna_best": peak_tg,
    }
