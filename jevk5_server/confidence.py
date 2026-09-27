from __future__ import annotations

from typing import Iterable


def calculate_confidence(probabilities: Iterable[float]) -> float:
    """Calculate TypeSafe confidence from a probability distribution.

    Formula from TypeSafe documentation:
        confidence = (count * max(p) - 1) / (count - 1)
    clamped to [0.0, 1.0].
    """
    probs = list(probabilities)
    count = len(probs)
    if count <= 1:
        return 1.0

    peak = max(probs)
    conf = (count * peak - 1.0) / (count - 1.0)
    return max(0.0, min(1.0, conf))
