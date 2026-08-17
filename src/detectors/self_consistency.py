"""Method 3: self-consistency.

Sample the same question N times at temperature>0. Low agreement across
samples => flagged as hallucinated.

Samples come from detectors.sampling, shared with semantic_entropy.
"""
import re
import string
from collections import Counter

AGREEMENT_THRESHOLD = 0.5  # majority-answer share below this => flagged hallucinated


def _normalize(text: str) -> str:
    """Standard short-answer QA normalization (SQuAD-style): casefold, drop
    articles and punctuation, collapse whitespace. Without this, trivial
    surface differences ("Paris" vs "paris." vs "The answer is Paris") count
    as disagreement and the agreement signal is mostly noise.
    """
    text = text.casefold()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score(answers: list[str]) -> tuple[str, bool, float]:
    """Returns (representative answer, flagged as hallucinated, agreement ratio)."""
    normalized = [_normalize(a) for a in answers]
    top_answer, top_count = Counter(normalized).most_common(1)[0]
    agreement = top_count / len(answers)

    prediction_hallucinated = agreement < AGREEMENT_THRESHOLD
    # report the majority-vote answer as the "representative" generation
    representative = next(a for a, norm in zip(answers, normalized) if norm == top_answer)

    return representative, prediction_hallucinated, agreement
