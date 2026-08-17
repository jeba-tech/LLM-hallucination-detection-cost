from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# .env lives in the project root (next to .env.example), not in src/
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATASET_PATH = DATA_DIR / "truthfulqa_subset.csv"
RAW_RUNS_PATH = RESULTS_DIR / "raw_runs.csv"
PARETO_PLOT_PATH = RESULTS_DIR / "pareto_plot.png"

# Two Groq models of different scale. Gemini was the originally planned second
# provider but its free tier allows only 20 requests/day, which cannot support
# this experiment; Groq publishes its limits in response headers and allows
# 1,000-14,400/day. The comparison is therefore across model scale (8B vs 70B)
# rather than across providers.
SMALL_MODEL = "llama-3.1-8b-instant"  # 14,400 req/day, 6,000 tok/min
LARGE_MODEL = "llama-3.3-70b-versatile"  # 1,000 req/day, 12,000 tok/min

# Only the 8B model has the daily budget for the scaled n=500 run. The 70B
# results from the n=50 pilot remain in raw_runs.csv and are still reported —
# just at the smaller sample size. Add LARGE_MODEL back to extend it, ~76
# questions per day as its quota resets.
MODELS = [SMALL_MODEL]

# ponytail: hardcoded per-1M-token pricing snapshot using published pay-as-you-go
# rates, not what the free tier actually bills — this is only used to compute the
# cost axis for the Pareto plot.
# UNVERIFIED: these are carried over from the gemini-2.0-flash / older-Llama era.
# Look up current published rates for both models and correct them before
# reporting any $ figure in the writeup — the cost axis is meaningless otherwise.
PRICING_PER_1M_TOKENS = {
    SMALL_MODEL: {"input": 0.05, "output": 0.08},
    LARGE_MODEL: {"input": 0.59, "output": 0.79},
}

SAMPLE_NS = [1, 3, 5, 10]

# The n=50 pilot was underpowered: every detector's bootstrap AUROC interval
# straddled chance. Scaled to 500, which fits the 8B model's 14,400 req/day
# budget and tightens those intervals roughly 3x.
DATASET_SUBSET_SIZE = 500
# The pilot sample is kept as a prefix of the larger one so previously logged
# results (including the 70B runs) stay valid. Do not change this.
PILOT_SIZE = 50
RANDOM_SEED = 42

# Pacing. The binding constraint on Groq is tokens/min, not requests/min:
# measured mean is ~100 tokens/call, so 6,000 tok/min permits ~60 calls/min.
# 45 leaves margin for longer-than-average answers without relying on retries.
RPM = {SMALL_MODEL: 45.0, LARGE_MODEL: 40.0}
MAX_RETRIES = 6
