"""Shannon entropy of source token distribution."""

from __future__ import annotations

import math
from collections import Counter


def shannon_entropy_from_frequencies(freq: Counter) -> float:
    """Compute Shannon entropy (bits per token) from a token frequency distribution.

    Higher entropy = flatter distribution = more "surprising" tokens = higher opacity.

    Args:
        freq: Counter mapping tokens to counts.

    Returns:
        Entropy in bits. Normalised to [0, 1] by dividing by log2(vocabulary_size).
    """
    total = sum(freq.values())
    if total == 0 or len(freq) <= 1:
        return 0.0

    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalise by maximum possible entropy (uniform distribution)
    max_entropy = math.log2(len(freq))
    if max_entropy == 0:
        return 0.0

    return entropy / max_entropy
