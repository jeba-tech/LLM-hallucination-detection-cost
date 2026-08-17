"""Method 4: semantic entropy.

Same samples as self-consistency, but cluster by embedding similarity instead
of exact string match, then compute entropy over cluster sizes.
High entropy (many distinct meanings) => flagged as hallucinated.
"""
from functools import lru_cache
from math import log

import numpy as np
from sklearn.cluster import AgglomerativeClustering

ENTROPY_THRESHOLD = 0.6  # normalized entropy above this => flagged hallucinated
CLUSTER_DISTANCE_THRESHOLD = 0.3  # cosine distance for merging clusters


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _cluster_entropy(answers: list[str]) -> float:
    if len(answers) <= 1:
        return 0.0

    embeddings = _embedder().encode(answers, normalize_embeddings=True)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    ).fit(embeddings)

    counts = np.bincount(clustering.labels_)
    probs = counts / len(answers)
    entropy = -sum(p * log(p) for p in probs if p > 0)
    max_entropy = log(len(answers))  # normalize to [0, 1]
    return entropy / max_entropy if max_entropy > 0 else 0.0


def score(answers: list[str]) -> tuple[str, bool, float]:
    """Returns (representative answer, flagged as hallucinated, normalized entropy)."""
    normalized_entropy = _cluster_entropy(answers)
    prediction_hallucinated = normalized_entropy > ENTROPY_THRESHOLD
    return answers[0], prediction_hallucinated, normalized_entropy
