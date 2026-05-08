from pathlib import Path

LLAMA_CPP_PINNED_BUILD = "b9014"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"

CONFIG_DIR = Path.home() / ".llamatune"
PROFILES_DIR = CONFIG_DIR / "profiles"
ENV_FILE = CONFIG_DIR / "env.json"
LOG_FILE = CONFIG_DIR / "llamatune.log"

LLAMA_INSTALL_DIR = Path.home() / ".local" / "llamatune" / "llama.cpp"

BENCH_WARMUP_RUNS = 1
BENCH_MEASUREMENT_RUNS = 2
OPTUNA_TRIALS = 30
OPTUNA_TIMEOUT_SEC = 900
QUICK_PROBE_WARMUP = 1

MIN_ACCEPTABLE_TG_RATIO = 0.6  # context search: drop to 60% of peak → stop
STABILITY_RUNS = 3
STABILITY_CV_WARN = 0.15  # warn if coefficient of variation > 15%

CONTEXT_CANDIDATES = [4096, 8192, 16384, 32768, 40000, 65536, 131072]

HF_API = "https://huggingface.co/api"
RECOMMENDED_AUTHORS = ["unsloth", "bartowski"]

WATCH_PATTERNS = [
    "llama-bench", "gguf", "moe", "ncmoe",
    "flash.att", "fa", "kv.cache", "nkvo",
    "cuda", "cublas", "list-devices", "breaking",
]
