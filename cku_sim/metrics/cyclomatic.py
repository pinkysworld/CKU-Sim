"""Cyclomatic complexity density — CC/LOC as an opacity proxy.

For C/C++ files, we use a simple heuristic counter rather than requiring
a full parser.  This counts decision points (if, for, while, case, &&, ||, ?)
which correlates well with McCabe's CC for most practical purposes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex matching decision-point keywords in C/C++ source
DECISION_KEYWORDS = re.compile(
    r"\b(if|else\s+if|for|while|case|catch|switch)\b"
    r"|(\?\s)"         # ternary operator
    r"|(&&|\|\|)"      # logical operators
)


def cyclomatic_complexity_file(filepath: Path) -> int:
    """Estimate cyclomatic complexity of a single C/C++ file.

    Uses keyword/operator counting as a proxy for full CFG analysis.
    Starts at 1 (base complexity for a straight-line program).

    Returns:
        Estimated cyclomatic complexity.
    """
    try:
        text = filepath.read_text(errors="replace")
    except Exception:
        return 1

    # Strip single-line comments
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    # Strip multi-line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Strip string literals (avoid counting keywords in strings)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)

    decisions = len(DECISION_KEYWORDS.findall(text))
    return 1 + decisions


def cyclomatic_density(files: list[Path], total_loc: int) -> float:
    """Compute cyclomatic complexity density: total CC / LOC.

    Higher density = more control-flow irregularity = higher opacity.

    Args:
        files: Source file paths.
        total_loc: Total lines of code (from count_loc).

    Returns:
        CC density. Typically in range [0.01, 0.5].
    """
    if total_loc == 0:
        return 0.0

    total_cc = sum(cyclomatic_complexity_file(f) for f in files)
    density = total_cc / total_loc

    logger.debug(f"Cyclomatic density: CC={total_cc}, LOC={total_loc}, density={density:.6f}")
    return density
