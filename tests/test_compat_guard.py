"""Tests for compat guard — mocked subprocess."""
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
from ggtune.modules.compat_guard import run_tests, COMPAT_TESTS

_SFX = ".exe" if platform.system() == "Windows" else ""


def _make_proc(stdout="", returncode=0):
    p = MagicMock()
    p.stdout = stdout
    p.stderr = ""
    p.returncode = returncode
    return p


def test_all_pass(tmp_path):
    # Create fake binaries
    bench = tmp_path / f"llama-bench{_SFX}"
    cli = tmp_path / f"llama-cli{_SFX}"
    bench.touch()
    cli.touch()

    outputs = {
        "llama-cli": "Available devices:\n  CPU (default)\n  CUDA0: RTX 3060\n -m model -fa\n-ncmoe\n-nkvo",
        "llama-bench": "Usage: llama-bench -m model\n  -fa INT\n  -pg prompt,gen",
    }

    def fake_run(cmd, **kwargs):
        name = Path(cmd[0]).stem
        return _make_proc(stdout=outputs.get(name, ""))

    with patch("subprocess.run", side_effect=fake_run):
        report = run_tests(str(tmp_path))

    assert report.all_critical_passed


def test_missing_binary_non_critical(tmp_path):
    bench = tmp_path / f"llama-bench{_SFX}"
    cli = tmp_path / f"llama-cli{_SFX}"
    bench.touch()
    cli.touch()

    # Critical tests (--list-devices, llama-bench --help) pass; non-critical
    # ones (-fa, -ncmoe, -nkvo flags) are absent from the --help text.
    outputs = {
        "llama-cli": "Available devices:\n  CPU (default)\n  CUDA0: RTX 3060",
        "llama-bench": "Usage: llama-bench -m model",
    }

    def fake_run(cmd, **kwargs):
        name = Path(cmd[0]).stem
        return _make_proc(stdout=outputs.get(name, ""))

    with patch("subprocess.run", side_effect=fake_run):
        report = run_tests(str(tmp_path))

    # Should not raise even though non-critical tests fail
    assert report.all_critical_passed
    assert not report.all_passed
    # -fa is not in bench --help output, so that non-critical test should fail
    fa_result = next(r for r in report.results if r.test.name == "-fa flag")
    assert not fa_result.passed
