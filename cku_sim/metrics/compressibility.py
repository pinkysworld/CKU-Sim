"""Compressibility index — the primary KC proxy from Nguyen (2026).

CI = compressed_size / raw_size

Higher CI = less compressible = more structurally opaque = higher CKU.
This is the *opacity-oriented* version: 0 = perfectly compressible, 1 = incompressible.
"""

from __future__ import annotations

import gzip
import lzma
import logging

logger = logging.getLogger(__name__)


def compressibility_index(data: bytes, method: str = "gzip") -> float:
    """Compute compressibility index for a byte stream.

    Returns the ratio compressed_size / raw_size.
    Higher values indicate less compressible (more opaque) data.

    Args:
        data: Raw bytes (typically a tar archive of source files).
        method: Compression algorithm — "gzip", "lzma", or "zstd".

    Returns:
        Float in [0, 1]. Higher = more opaque.
    """
    if not data:
        return 0.0

    raw_size = len(data)

    if method == "gzip":
        compressed = gzip.compress(data, compresslevel=9)
    elif method == "lzma":
        compressed = lzma.compress(data, preset=9)
    elif method == "zstd":
        try:
            import zstandard as zstd
            compressor = zstd.ZstdCompressor(level=22)
            compressed = compressor.compress(data)
        except ImportError:
            logger.warning("zstandard not installed, falling back to gzip")
            compressed = gzip.compress(data, compresslevel=9)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    compressed_size = len(compressed)
    ci = compressed_size / raw_size

    logger.debug(
        f"CI({method}): {raw_size} -> {compressed_size} bytes, "
        f"ratio={ci:.4f}"
    )

    return ci
