"""Shared sampling for the two sampling-based detectors.

Both self-consistency and semantic entropy score the same set of sampled
generations — they differ only in how they measure disagreement. So the
samples are drawn once per (question, model) and both detectors score them.

Samples are also drawn once at max(SAMPLE_NS) and the N-sweep reads prefixes
of that list. Because samples are i.i.d. at fixed temperature, the first n of
10 draws is distributed identically to an independent draw of n — but costs
10 calls total instead of 1+3+5+10=19. Per-call usage and latency are kept
separately so each prefix is still charged only for the calls it uses.
"""
import time
from dataclasses import dataclass

import llm_clients
from cost_tracker import CallAccumulator


@dataclass
class Sample:
    text: str
    usage: llm_clients.Usage
    latency_s: float


def draw(question: str, model: str, n_max: int) -> list[Sample]:
    samples = []
    for _ in range(n_max):
        start = time.perf_counter()
        text, usage = llm_clients.answer(question, model, temperature=0.7)
        samples.append(Sample(text.strip(), usage, time.perf_counter() - start))
    return samples


def prefix_cost(samples: list[Sample], n: int, model: str) -> tuple[CallAccumulator, float]:
    """Cost and latency attributable to just the first n samples."""
    acc = CallAccumulator()
    for sample in samples[:n]:
        acc.add(model, sample.usage)
    latency_s = sum(sample.latency_s for sample in samples[:n])
    return acc, latency_s
