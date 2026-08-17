"""Main experiment loop: dataset x detectors x models x N.

Run from src/: python run_experiment.py [--limit N]

Safe to interrupt and re-run — already-completed (question, method, model, n)
rows are skipped via cost_tracker.already_done(). That matters here: the 70B
model allows only 1,000 requests/day on the free tier, so an interrupted run
must resume rather than restart.
"""
import argparse

import pandas as pd

import config
import cost_tracker
import llm_clients
from detectors import llm_judge, sampling, self_consistency, semantic_entropy, verbalized_confidence

# Grading runs on the small model: it has the 14,400/day quota, so labeling
# never competes with the 70B model's 1,000/day detection budget.
LABEL_MODEL = config.SMALL_MODEL

SINGLE_SAMPLE_METHODS = [
    ("verbalized_confidence", verbalized_confidence),
    ("llm_judge", llm_judge),
]

SAMPLING_METHODS = [
    ("self_consistency", self_consistency),
    ("semantic_entropy", semantic_entropy),
]

# Grading is a pure function of (question, answer, reference), and the sampling
# methods return the same representative answer across many N values, so the
# same grade gets requested repeatedly. Caching it cuts a large share of the
# labeling calls, which matter against a rate-limited free tier.
_label_cache: dict[tuple[str, str], bool] = {}


def label_answer(question: str, answer: str, reference: str) -> bool:
    """True if `answer` is a hallucination relative to the known-correct reference."""
    key = (question, answer)
    if key not in _label_cache:
        is_correct, _usage = llm_clients.grade_with_reference(
            question, answer, reference, LABEL_MODEL
        )
        _label_cache[key] = not is_correct
    return _label_cache[key]


def run_single_sample_methods(row, qid: int) -> None:
    for model in config.MODELS:
        for method_name, module in SINGLE_SAMPLE_METHODS:
            if cost_tracker.already_done(qid, method_name, model, 1):
                continue
            answer, prediction, score, acc, latency = module.detect(row.question, model)
            label = label_answer(row.question, answer, row.correct_answer)
            cost_tracker.log_row(
                qid, method_name, model, 1, acc, latency, prediction, label, score
            )


def run_sampling_methods(row, qid: int) -> None:
    n_max = max(config.SAMPLE_NS)

    for model in config.MODELS:
        pending = [
            (name, module, n)
            for name, module in SAMPLING_METHODS
            for n in config.SAMPLE_NS
            if not cost_tracker.already_done(qid, name, model, n)
        ]
        if not pending:
            continue

        # One draw of n_max serves every N: prefixes of an i.i.d. sample are
        # themselves i.i.d. samples, and prefix_cost charges each N only for
        # the calls it actually consumes.
        samples = sampling.draw(row.question, model, n_max)
        answers = [s.text for s in samples]

        for method_name, module, n in pending:
            representative, prediction, score = module.score(answers[:n])
            acc, latency = sampling.prefix_cost(samples, n, model)
            label = label_answer(row.question, representative, row.correct_answer)
            cost_tracker.log_row(
                qid, method_name, model, n, acc, latency, prediction, label, score
            )


def main(limit: int | None = None):
    df = pd.read_csv(config.DATASET_PATH)
    if limit is not None:
        df = df.head(limit)
    cost_tracker.ensure_header()

    for qid, row in df.iterrows():
        print(f"[{qid + 1}/{len(df)}] {row.question[:60]}...")
        run_single_sample_methods(row, qid)
        run_sampling_methods(row, qid)

    print(f"Done. Results in {config.RAW_RUNS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N questions (use for a cheap smoke test)",
    )
    main(**vars(parser.parse_args()))
