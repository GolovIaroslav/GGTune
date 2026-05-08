"""Tests for compat guard — mocked subprocess."""
from unittest.mock import patch, MagicMock
import subprocess
from ggtune.modules.compat_guard import run_tests, COMPAT_TESTS


def _make_proc(stdout="", returncode=0):
    p = MagicMock()
    p.stdout = stdout
    p.stderr = ""
    p.returncode = returncode
    return p


def test_all_pass(tmp_path):
    # Create fake binaries
    bench = tmp_path / "llama-bench"
    cli = tmp_path / "llama-cli"
    bench.touch()
    cli.touch()

    outputs = {
        "llama-cli": "Available devices:\n  CPU (default)\n  CUDA0: RTX 3060\n -m model -fa\n-ncmoe\n-nkvo",
        "llama-bench": "Usage: llama-bench -m model\n  -fa INT\n  -pg prompt,gen",
    }

    def fake_run(cmd, **kwargs):
        name = cmd[0].split("/")[-1]
        return _make_proc(stdout=outputs.get(name, ""))

    with patch("subprocess.run", side_effect=fake_run):
        report = run_tests(str(tmp_path))

    assert report.all_critical_passed


def test_missing_binary_non_critical(tmp_path):
    bench = tmp_path / "llama-bench"
    cli = tmp_path / "llama-cli"
    bench.touch()
    cli.touch()

    def fake_run(cmd, **kwargs):
        return _make_proc(stdout="Usage: llama-bench -m model -fa 1")

    with patch("subprocess.run", side_effect=fake_run):
        report = run_tests(str(tmp_path))

    # Should not raise even if non-critical tests fail
    assert isinstance(report.all_critical_passed, bool)
