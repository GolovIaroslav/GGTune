# GGTune

Finds the fastest llama.cpp parameters for your GPU and model by actually running benchmarks on your hardware.

```
$ ggtune
```

No config files. Just point it at a model and it figures out the best settings for your machine.

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

If you already have llama.cpp built somewhere, GGTune will find it automatically. If not, choose **"llama.cpp — version / update"** from the main menu to build it.

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
ggtune browse                          # find models on HuggingFace
ggtune scan                            # find .gguf files on disk
ggtune info /path/to/model.gguf       # model metadata
ggtune hw                              # hardware info
ggtune show                            # saved profiles
ggtune clear --all                     # clear profile cache
ggtune compat                          # test llama.cpp binaries
ggtune update                          # update llama.cpp
ggtune help                            # show all commands
```

---

## How the benchmark works

1. Grid probe — coarse sweep to eliminate bad parameter regions
2. Bayesian search — Optuna TPE, ~30 trials, warm-started from probe results
3. Context search — finds the largest context where speed stays above 60% of peak
4. Stability pass — reruns the winner to confirm results are consistent

Results are cached per model + hardware fingerprint. Running it again on the same model is instant.

Built with: Python, [Optuna](https://optuna.org), [Rich](https://github.com/Textualize/rich), [llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## Configuration

Everything lives in `~/.llamatune/`:

```
~/.llamatune/env.json       # llama.cpp location and build info
~/.llamatune/profiles/      # cached benchmark results
```

---

## License

[MIT](LICENSE)
