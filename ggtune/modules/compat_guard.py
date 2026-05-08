"""Module 12: llama.cpp Compatibility Guard."""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import requests
from rich.console import Console
from rich.table import Table
from rich import box

from ggtune.config import LLAMA_CPP_PINNED_BUILD, WATCH_PATTERNS

console = Console()

DUMMY_MODEL_ARGS: list[str] = []  # no model needed for --help / --list-devices


@dataclass
class CompatTest:
    name: str
    description: str
    args: list[str]  # args after the binary
    binary: str      # "llama-bench" or "llama-cli"
    validator: Callable[[str], bool]
    critical: bool


@dataclass
class TestResult:
    test: CompatTest
    passed: bool
    output: str
    error: str


@dataclass
class CompatReport:
    results: List[TestResult]
    build: str

    @property
    def all_critical_passed(self) -> bool:
        return all(r.passed for r in self.results if r.test.critical)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


COMPAT_TESTS = [
    CompatTest(
        name="--list-devices",
        description="llama-cli must list available compute devices",
        args=["--list-devices"],
        binary="llama-cli",
        validator=lambda out: any(x in out for x in ["CPU", "CUDA", "Metal", "Available devices", "Device"]),
        critical=True,
    ),
    CompatTest(
        name="llama-bench --help",
        description="llama-bench must accept --help",
        args=["--help"],
        binary="llama-bench",
        validator=lambda out: "-m" in out or "model" in out.lower(),
        critical=True,
    ),
    CompatTest(
        name="-fa flag",
        description="flash attention flag must be documented",
        args=["--help"],
        binary="llama-bench",
        validator=lambda out: "-fa" in out,
        critical=False,
    ),
    CompatTest(
        name="-ncmoe flag",
        description="MoE experts flag must be documented",
        args=["--help"],
        binary="llama-cli",
        validator=lambda out: "ncmoe" in out.lower() or "moe" in out.lower(),
        critical=False,
    ),
    CompatTest(
        name="-nkvo flag",
        description="KV offload flag must be documented",
        args=["--help"],
        binary="llama-cli",
        validator=lambda out: "nkvo" in out.lower(),
        critical=False,
    ),
]


def run_tests(bin_dir: str) -> CompatReport:
    from ggtune.utils.shell import make_env_with_lib
    bin_path = Path(bin_dir)
    build = LLAMA_CPP_PINNED_BUILD
    env = make_env_with_lib(bin_dir)
    results = []

    for test in COMPAT_TESTS:
        binary = bin_path / test.binary
        if not binary.exists():
            results.append(TestResult(test=test, passed=False, output="", error="binary not found"))
            if test.critical:
                raise RuntimeError(f"Critical binary not found: {binary}")
            continue

        try:
            proc = subprocess.run(
                [str(binary)] + test.args,
                capture_output=True, text=True, timeout=15, env=env,
            )
            combined = proc.stdout + proc.stderr
            passed = test.validator(combined)
            results.append(TestResult(test=test, passed=passed, output=combined[:500], error=""))
            if test.critical and not passed:
                raise RuntimeError(
                    f"Critical compat test '{test.name}' failed. "
                    f"llama.cpp may have changed its API."
                )
        except subprocess.TimeoutExpired:
            results.append(TestResult(test=test, passed=False, output="", error="timeout"))
        except RuntimeError:
            raise
        except Exception as e:
            results.append(TestResult(test=test, passed=False, output="", error=str(e)))

    return CompatReport(results=results, build=build)


def print_report(report: CompatReport) -> None:
    table = Table(title=f"Compat report — llama.cpp {report.build}", box=box.ROUNDED)
    table.add_column("Test")
    table.add_column("Status", justify="center")
    table.add_column("Critical", justify="center")
    table.add_column("Description")

    for r in report.results:
        status = "[green]✓ pass[/]" if r.passed else "[red]✗ fail[/]"
        crit = "[red]YES[/]" if r.test.critical else "no"
        table.add_row(r.test.name, status, crit, r.test.description)

    console.print(table)

    if not report.all_passed:
        fails = [r for r in report.results if not r.passed]
        console.print(f"\n[yellow]{len(fails)} test(s) failed.[/]")
        for r in fails:
            if r.error:
                console.print(f"  [dim]{r.test.name}:[/] {r.error}")


@dataclass
class Change:
    build: int
    title: str
    url: str
    affected_areas: list[str]
    is_breaking: bool


def check_for_changes(current_build: str) -> List[Change]:
    try:
        current = int(current_build.lstrip("b"))
    except ValueError:
        return []

    try:
        resp = requests.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases",
            params={"per_page": 20},
            timeout=10,
        )
        releases = resp.json()
    except Exception:
        return []

    relevant = []
    for rel in releases:
        tag = rel.get("tag_name", "")
        try:
            build_num = int(tag.lstrip("b"))
        except ValueError:
            continue
        if build_num <= current:
            break

        body = rel.get("body", "").lower()
        matches = [p for p in WATCH_PATTERNS if p in body]
        if matches:
            relevant.append(Change(
                build=build_num,
                title=rel.get("name", tag),
                url=rel.get("html_url", ""),
                affected_areas=matches,
                is_breaking="breaking" in body,
            ))

    return relevant


def print_changes(changes: List[Change], current_build: str) -> None:
    if not changes:
        console.print(f"[green]No relevant changes since {current_build}.[/]")
        return

    console.print(f"[yellow]Changes since {current_build} that may affect GGTune:[/]")
    for c in changes:
        flag = " [red][BREAKING][/]" if c.is_breaking else ""
        areas = ", ".join(c.affected_areas[:3])
        console.print(f"  b{c.build}  {c.title}{flag}  [dim]({areas})[/]")
        console.print(f"         {c.url}")
