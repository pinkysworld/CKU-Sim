"""Halstead volume metric — lexical diversity as an opacity proxy.

V = N * log2(η)
where N = total operators + operands, η = distinct operators + operands.

We normalise by LOC to make it scale-independent.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Simple operator/operand classification for C-family languages
OPERATORS = re.compile(
    r"(==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|->|::|"
    r"[+\-*/%=<>&|^~!?:;,.\[\]{}()])"
)
OPERANDS = re.compile(r"\b[A-Za-z_]\w*\b|\b\d+\.?\d*\b")


def halstead_metrics_file(filepath: Path) -> tuple[int, int, int, int]:
    """Extract Halstead primitives from a single file.

    Returns:
        (n1, n2, N1, N2) = (distinct operators, distinct operands,
                            total operators, total operands)
    """
    try:
        text = filepath.read_text(errors="replace")
    except Exception:
        return (0, 0, 0, 0)

    # Strip comments and strings
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
    text = re.sub(r"'(?:[^'\\]|\\.)*'", "''", text)

    ops = OPERATORS.findall(text)
    opnds = OPERANDS.findall(text)

    # Remove C keywords from operands (they're more like operators)
    c_keywords = {
        "if", "else", "for", "while", "do", "switch", "case", "break",
        "continue", "return", "goto", "default", "typedef", "struct",
        "union", "enum", "sizeof", "static", "extern", "const", "volatile",
        "register", "auto", "void", "int", "char", "short", "long",
        "float", "double", "signed", "unsigned", "inline",
    }
    opnds = [o for o in opnds if o not in c_keywords]

    op_counter = Counter(ops)
    opnd_counter = Counter(opnds)

    n1 = len(op_counter)
    n2 = len(opnd_counter)
    N1 = sum(op_counter.values())
    N2 = sum(opnd_counter.values())

    return (n1, n2, N1, N2)


def halstead_volume_normalised(files: list[Path]) -> float:
    """Compute normalised Halstead volume across all files.

    Volume V = N * log2(η), where N = N1+N2, η = n1+n2.
    Normalised by dividing by N to get log2(η), i.e. bits per token.
    This makes it scale-independent and comparable across projects.

    Returns:
        Normalised volume (bits per token). Higher = more diverse vocabulary = more opaque.
    """
    total_n1 = total_n2 = total_N1 = total_N2 = 0

    for fpath in files:
        n1, n2, N1, N2 = halstead_metrics_file(fpath)
        total_n1 += n1
        total_n2 += n2
        total_N1 += N1
        total_N2 += N2

    eta = total_n1 + total_n2  # vocabulary size
    N = total_N1 + total_N2    # program length

    if eta <= 1 or N == 0:
        return 0.0

    # Normalised volume = log2(η)
    # This is V/N = bits per token — scale-independent
    normalised = math.log2(eta)

    # Further normalise to [0, 1] range using empirical bounds
    # log2(η) for real codebases typically ranges from ~8 to ~18
    # We map [8, 18] -> [0, 1]
    normalised = max(0.0, min(1.0, (normalised - 8.0) / 10.0))

    logger.debug(
        f"Halstead: η={eta}, N={N}, log2(η)={math.log2(eta):.2f}, "
        f"normalised={normalised:.4f}"
    )

    return normalised
