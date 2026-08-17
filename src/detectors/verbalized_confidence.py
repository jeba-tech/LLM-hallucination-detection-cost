"""Method 1: single-pass verbalized confidence.

Ask the model to answer AND self-report a 0-100 confidence in one call.
Low self-reported confidence => flagged as hallucinated.

This replaces the originally planned token-log-prob signal: no free-tier API
(Groq, Gemini) exposes token log-probs, but verbalized confidence keeps the
same role in the experiment — a genuine 1-call, cheapest-possible detector —
and is an established baseline (Lin et al. 2022; Tian et al. 2023).

Unlike the log-prob version this runs on BOTH models, so method 1 now has a
cross-model comparison the original plan couldn't provide.
"""
import time

import llm_clients
from cost_tracker import CallAccumulator

# ponytail: LLMs are systematically overconfident when verbalizing confidence
# (most answers land 80-100), so this threshold is high by design. Worth a
# sensitivity check against the observed score distribution before trusting F1.
THRESHOLD = 80.0  # confidence below this => flagged hallucinated

# Used when the model ignores the output format and no score can be parsed.
# Treated as low confidence: an unparseable response is itself a bad sign.
UNPARSEABLE_CONFIDENCE = 0.0


def detect(question: str, model: str) -> tuple[str, bool, float, CallAccumulator, float]:
    start = time.perf_counter()
    acc = CallAccumulator()

    answer, confidence, usage = llm_clients.generate_with_confidence(question, model)
    acc.add(model, usage)

    score = UNPARSEABLE_CONFIDENCE if confidence is None else confidence
    prediction_hallucinated = score < THRESHOLD

    latency_s = time.perf_counter() - start
    return answer, prediction_hallucinated, score, acc, latency_s
