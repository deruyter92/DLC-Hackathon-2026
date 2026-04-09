"""Project path constants used by training and benchmarking scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmarking"
