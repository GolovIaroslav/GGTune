"""Tests for llama-bench output parsing — the most fragile part."""
from pathlib import Path
from ggtune.modules.benchmark_engine import parse_bench_output

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_csv_format():
    text = (FIXTURES / "bench_output_b5262.txt").read_text()
    tg, pp = parse_bench_output(text)
    assert tg == pytest.approx(45.72, rel=0.01)
    assert pp == pytest.approx(1234.56, rel=0.01)


def test_parse_markdown_format():
    text = (FIXTURES / "bench_output_b5100.txt").read_text()
    tg, pp = parse_bench_output(text)
    assert tg > 0
    assert pp > 0


def test_parse_empty():
    tg, pp = parse_bench_output("")
    assert tg == 0.0
    assert pp == 0.0


def test_parse_crash_output():
    tg, pp = parse_bench_output("CUDA error: out of memory\nAborted (core dumped)")
    assert tg == 0.0
    assert pp == 0.0


import pytest
