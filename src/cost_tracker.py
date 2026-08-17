"""Append one row per detector call to results/raw_runs.csv."""
import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass

import config
from llm_clients import Usage

FIELDS = [
    "question_id",
    "method",
    "model",
    "n_samples",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_s",
    "prediction",
    "score",
    "label",
]


def _cost(model: str, usage: Usage) -> float:
    price = config.PRICING_PER_1M_TOKENS[model]
    return (usage.tokens_in * price["input"] + usage.tokens_out * price["output"]) / 1_000_000


@dataclass
class CallAccumulator:
    """Collects usage across the (possibly many) API calls one detection makes."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def add(self, model: str, usage: Usage) -> None:
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        self.cost_usd += _cost(model, usage)


@contextmanager
def timed_call():
    acc = CallAccumulator()
    start = time.perf_counter()
    yield acc
    acc.latency_s = time.perf_counter() - start  # type: ignore[attr-defined]


def ensure_header() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not config.RAW_RUNS_PATH.exists():
        with open(config.RAW_RUNS_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def log_row(
    question_id: int,
    method: str,
    model: str,
    n_samples: int,
    acc: CallAccumulator,
    latency_s: float,
    prediction: bool,
    label: bool,
    score: float | None = None,
) -> None:
    """`score` is the detector's raw continuous signal (confidence, agreement
    ratio, entropy) before thresholding — logged so thresholds can be swept in
    analysis without re-running the experiment. None for inherently binary
    detectors such as llm_judge."""
    ensure_header()
    if _completed is not None:
        _completed.add((question_id, method, model, n_samples))
    with open(config.RAW_RUNS_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(
            {
                "question_id": question_id,
                "method": method,
                "model": model,
                "n_samples": n_samples,
                "tokens_in": acc.tokens_in,
                "tokens_out": acc.tokens_out,
                "cost_usd": acc.cost_usd,
                "latency_s": latency_s,
                "prediction": prediction,
                "score": "" if score is None else score,
                "label": label,
            }
        )


_completed: set[tuple[int, str, str, int]] | None = None


def _load_completed() -> set[tuple[int, str, str, int]]:
    """Read the existing run log once into a set, so resume checks are O(1)."""
    done: set[tuple[int, str, str, int]] = set()
    if not config.RAW_RUNS_PATH.exists():
        return done
    with open(config.RAW_RUNS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add(
                    (int(row["question_id"]), row["method"], row["model"], int(row["n_samples"]))
                )
            except (KeyError, ValueError):
                continue  # skip malformed/partial trailing row from an interrupted run
    return done


def already_done(question_id: int, method: str, model: str, n_samples: int) -> bool:
    global _completed
    if _completed is None:
        _completed = _load_completed()
    return (question_id, method, model, n_samples) in _completed
