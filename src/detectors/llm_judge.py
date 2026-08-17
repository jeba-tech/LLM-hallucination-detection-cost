"""Method 2: LLM-as-judge.

Generate an answer, then ask the same model (no reference given) whether
that answer is true. FALSE verdict => flagged as hallucinated.
"""
import time

import llm_clients
from cost_tracker import CallAccumulator


def detect(
    question: str, model: str
) -> tuple[str, bool, float | None, CallAccumulator, float]:
    start = time.perf_counter()
    acc = CallAccumulator()

    answer, gen_usage = llm_clients.answer(question, model, temperature=0.0)
    acc.add(model, gen_usage)

    is_true, judge_usage = llm_clients.judge(question, answer, model)
    acc.add(model, judge_usage)

    prediction_hallucinated = not is_true
    latency_s = time.perf_counter() - start
    # no continuous score: the judge returns a bare TRUE/FALSE verdict
    return answer, prediction_hallucinated, None, acc, latency_s
