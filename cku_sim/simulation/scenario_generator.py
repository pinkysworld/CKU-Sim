"""Generate synthetic codebases with controlled structural opacity.

Used to replicate Simulation 1 from Nguyen (2026): demonstrating that
compressibility index cleanly separates regular from irregular code.
"""

from __future__ import annotations

import random
import string


def generate_regular_source(size_bytes: int = 50_000, seed: int = 42) -> bytes:
    """Generate highly regular (low-opacity) synthetic C-like source.

    Produces repetitive function definitions with predictable structure.
    Should compress very well (high CI in original paper's convention,
    low CI in our opacity-oriented convention).
    """
    rng = random.Random(seed)
    lines = [
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "",
    ]

    func_idx = 0
    while len("\n".join(lines).encode()) < size_bytes:
        lines.append(f"int process_item_{func_idx}(int value) {{")
        lines.append(f"    int result = value * {func_idx + 1};")
        lines.append(f"    if (result > 100) {{")
        lines.append(f"        result = result - 100;")
        lines.append(f"    }}")
        lines.append(f"    return result;")
        lines.append(f"}}")
        lines.append("")
        func_idx += 1

    return "\n".join(lines).encode()[:size_bytes]


def generate_irregular_source(size_bytes: int = 50_000, seed: int = 42) -> bytes:
    """Generate highly irregular (high-opacity) synthetic C-like source.

    Produces diverse, non-repeating code with varied control flow,
    random identifiers, and heterogeneous structure.
    Should compress poorly (high opacity).
    """
    rng = random.Random(seed)
    lines = [
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <math.h>",
        "",
    ]

    types = ["int", "float", "double", "char*", "unsigned long", "short"]
    ops = ["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"]
    comparisons = ["<", ">", "<=", ">=", "==", "!="]

    def rand_id(length=None):
        length = length or rng.randint(3, 12)
        return rng.choice(string.ascii_lowercase) + "".join(
            rng.choices(string.ascii_lowercase + string.digits + "_", k=length - 1)
        )

    func_idx = 0
    while len("\n".join(lines).encode()) < size_bytes:
        ret_type = rng.choice(types)
        fname = rand_id(rng.randint(5, 15))
        nparams = rng.randint(0, 5)
        params = ", ".join(
            f"{rng.choice(types)} {rand_id()}" for _ in range(nparams)
        )
        lines.append(f"{ret_type} {fname}({params}) {{")

        # Random local variables
        nvars = rng.randint(1, 6)
        varnames = []
        for _ in range(nvars):
            vname = rand_id()
            vtype = rng.choice(types)
            init = rng.randint(-1000, 1000)
            lines.append(f"    {vtype} {vname} = {init};")
            varnames.append(vname)

        # Random control flow
        nblocks = rng.randint(1, 4)
        for _ in range(nblocks):
            block_type = rng.choice(["if", "for", "while", "switch"])
            if block_type == "if":
                v = rng.choice(varnames) if varnames else "0"
                cmp = rng.choice(comparisons)
                val = rng.randint(-500, 500)
                lines.append(f"    if ({v} {cmp} {val}) {{")
                # Random operations inside
                for _ in range(rng.randint(1, 3)):
                    a = rng.choice(varnames) if varnames else "x"
                    b = rng.choice(varnames) if varnames else "y"
                    op = rng.choice(ops)
                    lines.append(f"        {a} = {a} {op} {b};")
                if rng.random() > 0.5:
                    lines.append(f"    }} else {{")
                    a = rng.choice(varnames) if varnames else "x"
                    lines.append(f"        {a} = {rng.randint(0, 999)};")
                lines.append(f"    }}")
            elif block_type == "for":
                ivar = rand_id(3)
                limit = rng.randint(1, 100)
                lines.append(f"    for (int {ivar} = 0; {ivar} < {limit}; {ivar}++) {{")
                a = rng.choice(varnames) if varnames else "x"
                op = rng.choice(ops[:5])
                lines.append(f"        {a} = {a} {op} {ivar};")
                lines.append(f"    }}")
            elif block_type == "while":
                v = rng.choice(varnames) if varnames else "x"
                lines.append(f"    while ({v} > 0) {{")
                lines.append(f"        {v} = {v} - {rng.randint(1, 10)};")
                lines.append(f"    }}")
            elif block_type == "switch":
                v = rng.choice(varnames) if varnames else "x"
                lines.append(f"    switch ({v} % {rng.randint(2,8)}) {{")
                for case_val in range(rng.randint(2, 5)):
                    lines.append(f"        case {case_val}:")
                    a = rng.choice(varnames) if varnames else "x"
                    lines.append(f"            {a} = {rng.randint(-999, 999)};")
                    lines.append(f"            break;")
                lines.append(f"        default:")
                lines.append(f"            break;")
                lines.append(f"    }}")

        # Return
        if ret_type == "char*":
            lines.append(f'    return "{rand_id(8)}";')
        elif "float" in ret_type or "double" in ret_type:
            lines.append(f"    return {rng.uniform(-100, 100):.4f};")
        else:
            rv = rng.choice(varnames) if varnames else "0"
            lines.append(f"    return {rv};")

        lines.append(f"}}")
        lines.append("")
        func_idx += 1

    return "\n".join(lines).encode()[:size_bytes]


def generate_spectrum(
    n_samples: int = 50,
    size_bytes: int = 50_000,
    base_seed: int = 42,
) -> list[tuple[str, bytes, float]]:
    """Generate a spectrum of synthetic codebases from regular to irregular.

    Args:
        n_samples: Number of samples to generate.
        size_bytes: Target size per sample.
        base_seed: Random seed base.

    Returns:
        List of (label, source_bytes, regularity_score) tuples.
        regularity_score: 0.0 = fully irregular, 1.0 = fully regular.
    """
    samples = []
    for i in range(n_samples):
        regularity = i / (n_samples - 1)  # 0.0 to 1.0
        seed = base_seed + i

        # Interpolate by mixing regular and irregular chunks
        regular = generate_regular_source(size_bytes, seed)
        irregular = generate_irregular_source(size_bytes, seed + 1000)

        split = int(len(regular) * regularity)
        mixed = regular[:split] + irregular[: size_bytes - split]

        label = f"synth_{i:03d}_reg{regularity:.2f}"
        samples.append((label, mixed, regularity))

    return samples
