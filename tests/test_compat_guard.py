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

    # Critical tests must pass; non-critical (fa/nkvo/ncmoe) may fail
    cli_outputs = {"--list-devices": "Available devices:\n  CPU (default)"}

    def fake_run(cmd, **kwargs):
        binary = cmd[0].split("/")[-1]
        arg = cmd[1] if len(cmd) > 1 else ""
        if binary == "llama-cli":
            return _make_proc(stdout=cli_outputs.get(arg, ""))
        # llama-bench: return minimal valid output (critical test passes, -fa not present)
        return _make_proc(stdout="Usage: llama-bench -m model")

    with patch("subprocess.run", side_effect=fake_run):
        report = run_tests(str(tmp_path))

    assert report.all_critical_passed
    # -fa is not in bench --help output, so that non-critical test should fail
    fa_result = next(r for r in report.results if r.test.name == "-fa flag")
    assert not fa_result.passed
