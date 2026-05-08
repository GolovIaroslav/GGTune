# GGTune

Finds the fastest llama.cpp parameters for your GPU and model by actually running benchmarks on your hardware.

```
$ ggtune
```

Interactive TUI guides you through scanning for models, running benchmarks, and saving the result as a shell alias.

---

## What it does

You give it a GGUF file. It runs `llama-bench` with different combinations of `-ncmoe`, `-ctk`, `-ctv`, `-t`, `-fa` and finds what's fastest on your machine. For MoE models it searches the expert offload range. For dense models it searches GPU layer counts. At the end you get a ready-to-use command with the best settings.

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (AMD/Apple planned)
- Linux or macOS
- `git`, `cmake`, CUDA toolkit (for building llama.cpp)

---

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/GolovIaroslav/GGTune
cd GGTune
uv sync
uv run ggtune
```

If you already have llama.cpp built somewhere, GGTune will find it automatically. If not, it can build it for you from the TUI (option 6).

---

## Usage

Just run `ggtune` (or `uv run ggtune` if not activated). The TUI has everything:

- Scan for models on your machine
- Enter a model path manually
- Browse HuggingFace for models that fit your VRAM
- Run full or quick benchmark
- View saved results
- Update llama.cpp
- Check compatibility

CLI commands also work if you prefer:

```bash
ggtune run /path/to/model.gguf        # full benchmark
ggtune quick /path/to/model.gguf      # ~5 min quick version
ggtune hw                              # hardware info
ggtune show                            # saved profiles
ggtune compat                          # test llama.cpp binaries
```

---

## Supported models

Any GGUF format. The main focus is quantized models from [unsloth](https://huggingface.co/unsloth) and [bartowski](https://huggingface.co/bartowski) — the search space is tuned for them. Dense models and MoE models both work.

---

## How the benchmark works

1. Quick grid probe — tries a coarse set of parameter combinations to eliminate bad regions
2. Bayesian search — Optuna TPE, warm-started from probe results, ~30 trials
3. Context search — binary search to find the largest context where speed stays above 60% of peak
4. Stability check — runs the winner 3× to confirm results are consistent

Results are cached per model + hardware fingerprint. Running it again on the same model is instant.

---

## Configuration

Everything lives in `~/.llamatune/`:

```
~/.llamatune/
├── env.json      # llama.cpp location and build info
└── profiles/     # cached benchmark results
```

---

## License

[MIT](LICENSE)
