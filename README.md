# GGTune

Stop guessing llama.cpp parameters. GGTune benchmarks your actual hardware and finds the fastest settings for any GGUF model automatically.

```
$ ggtune run Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

Scanning hardware...  RTX 3060 Laptop 6GB · 32GB RAM · 8 cores
Reading model...      MoE · 128 experts · 35B params · Q2_K_XL

Running benchmark (38 trials, ~22 min)...
  Quick probe    ████████████████  12/12
  Optuna TPE     ████████████████  30/30  best so far: 41.2 t/s
  Context search ████████████████  6/6
  Stability      ████████████████  3/3

Results
  Speed:    45.7 t/s  (was 12 t/s with default settings)
  Context:  40 000 tokens
  Params:   -ncmoe 30 -t 8 -ctk q8_0 -ctv q8_0 -fa on

Alias written to ~/.zshrc → type 'qwen' to start
```

---

## Why

Running a model with wrong parameters is painful. Too many threads and it's slower than half. Context too large and it crashes. MoE experts in the wrong place and VRAM sits idle. The official docs tell you what the flags do, not what values to use for your specific GPU and model combination.

GGTune runs the actual benchmark with your hardware and your model, searches the parameter space intelligently, and hands you a working configuration.

---

## Features

- **Automatic llama.cpp setup** — detects if CUDA is missing, explains why, builds from source with the right flags. No manual cmake.
- **GGUF-aware search** — reads model metadata before benchmarking. MoE models get MoE-specific parameter search, dense models get layer offload search.
- **Smart optimizer** — quick grid probe to rule out bad regions, then Bayesian search (Optuna TPE) to converge fast. Usually 30–50 benchmark runs total.
- **Context binary search** — finds the largest context that fits in VRAM without killing generation speed.
- **Model browser** — searches HuggingFace for models that fit your VRAM, filtered to quantizations that actually make sense for your hardware.
- **Compatibility guard** — pins a tested llama.cpp version. Before any update, runs smoke tests to catch broken flags or changed output formats.
- **Cached profiles** — benchmark results are saved per model + hardware fingerprint. Second run on the same model takes seconds.
- **Ready to use** — writes an alias to your shell config. `qwen` and it opens in the browser.

---

## Install

**Requirements:** Python 3.10+, git, cmake, CUDA toolkit (for GPU acceleration)

```bash
pip install ggtune
```

Or from source:

```bash
git clone https://github.com/yourname/ggtune
cd ggtune
pip install -e .
```

llama.cpp is managed automatically. GGTune will build it on first run if it's not found, or if the existing binary doesn't have CUDA support.

---

## Usage

**Tune a model you already have:**
```bash
ggtune run /path/to/model.gguf
ggtune run /path/to/model.gguf --alias mymodel
```

**Find and download a model first:**
```bash
ggtune browse              # shows models that fit your VRAM
ggtune browse --vram 6     # filter for 6GB VRAM
```

**Quick tune (5 minutes instead of 20):**
```bash
ggtune quick /path/to/model.gguf
```

**Check what's installed:**
```bash
ggtune hw                  # hardware report
ggtune info model.gguf     # model metadata
ggtune show                # saved benchmark profiles
```

**Keep llama.cpp working:**
```bash
ggtune compat              # run compatibility tests
ggtune update --check      # check for relevant llama.cpp changes
ggtune update              # safe update with automatic rollback
```

---

## Model Browser

GGTune can search HuggingFace for models that fit your hardware. It focuses on [Unsloth](https://huggingface.co/unsloth) quantizations because they consistently perform better on consumer GPUs — dynamic quantization keeps quality higher at smaller sizes compared to standard GGUF quants.

```bash
$ ggtune browse

Models for RTX 3060 6GB + 32GB RAM

  #  Model                              Size   VRAM   Est. speed
  1  unsloth/Qwen3.6-35B-A3B Q2_K_XL   12 GB  part   ~45 t/s
  2  unsloth/Qwen3-14B UD-Q4_K_XL       9 GB  ✓      ~55 t/s
  3  unsloth/Llama-3.1-8B Q8_0          8 GB  ✓      ~70 t/s
  4  unsloth/gemma-3-12b UD-Q4_K_M      7 GB  ✓      ~60 t/s

[1-4] download   [q] quit
> 3

Downloading... ████████████████ 8.5 GB
Run benchmark now? [Y/n]
```

---

## How it works

**Phase 1 — Quick probe.** Runs a coarse grid of 12 parameter combinations to eliminate regions that crash or are clearly slow.

**Phase 2 — Bayesian search.** Uses Optuna with TPE sampler, warm-started with results from phase 1. Converges to near-optimal in ~30 trials.

**Phase 3 — Context search.** Binary search over context sizes [4k, 8k, 16k, 32k, 40k, 64k, 128k]. Finds the largest context where generation speed stays above 60% of peak.

**Phase 4 — Stability check.** Runs the winner 3 times, reports coefficient of variation. Warns if results are noisy (background processes, thermal throttling).

For MoE models the key parameter is `-ncmoe` — how many experts to keep in RAM. Too few and the model thrashes disk. Too many and you OOM. The search space is built from the model's actual expert counts read from GGUF metadata.

---

## llama.cpp compatibility

llama.cpp updates daily and occasionally changes flag names, output formats, or behavior. GGTune handles this by:

- **Pinning a tested build** — you get a specific version that's known to work, not whatever is latest
- **Smoke tests on install/update** — checks that `llama-bench` outputs parseable CSV, that `-ncmoe` flag exists, that `--list-devices` works
- **Safe updates** — builds the new version alongside the old one, runs tests, swaps only if everything passes
- **Rollback** — previous version is kept until you explicitly clean up

To see what changed in llama.cpp since your installed version:
```bash
ggtune update --check
```

This shows only changes that affect GGTune's functionality — benchmark output format, CUDA detection, MoE flags, KV cache flags — not every commit.

---

## Supported hardware

| GPU | Status |
|-----|--------|
| NVIDIA (CUDA) | ✅ Full support |
| AMD (ROCm) | 🚧 Planned |
| Apple Silicon (Metal) | 🚧 Planned |
| CPU only | ✅ Works, slow |

| OS | Status |
|----|--------|
| Linux | ✅ Full support |
| macOS | ✅ Full support |
| Windows | 🚧 Planned |

---

## Configuration

GGTune stores everything in `~/.llamatune/`:

```
~/.llamatune/
├── env.json          # llama.cpp install info, pinned build
├── profiles/         # cached benchmark results per model+hardware
└── llamatune.log     # full log of every benchmark run
```

To force a fresh benchmark ignoring the cache:
```bash
ggtune run model.gguf --force
```

---

## Contributing

The codebase is split into independent modules — hardware scanner, GGUF reader, benchmark engine, HuggingFace browser, compatibility guard. Each module has a clear input/output contract and can be developed and tested in isolation.

The most fragile part is `benchmark_engine.py` — specifically the function that parses `llama-bench` output. If something breaks after a llama.cpp update, that's usually where to look. There are fixture files in `tests/fixtures/` with captured benchmark outputs from different builds for regression testing.

```bash
# Run tests
pytest tests/

# Run just the benchmark parser tests (fastest feedback loop)
pytest tests/test_bench_parser.py -v

# Check compatibility with installed llama.cpp
ggtune compat --report
```

When llama.cpp changes something that breaks GGTune, the fix is usually:
1. `ggtune compat --report` to see which test failed
2. Run the failing command manually to see the new output format
3. Fix the parser in `benchmark_engine.py`
4. Update `LLAMA_CPP_PINNED_BUILD` in `config.py`
5. Add the new output format as a fixture in `tests/fixtures/`

---

## License

MIT
