"""
Apply `if type(x) is not torch.Tensor:` guards around simple
`x = convert_to_tensor(x)` patterns in torch backend files.

Only transforms cases where:
- LHS and RHS variable names match exactly (e.g. `x = convert_to_tensor(x)`)
- No extra arguments (dtype=, sparse=, etc.)

Usage:
    python benchmarks/apply_type_guards.py [--dry-run]
"""

import re
import sys


def transform(source: str) -> tuple[str, int]:
    """Return (transformed_source, n_changes)."""
    # Match: <indent><varname> = convert_to_tensor(<same-varname>)
    # The closing paren must be immediately after the varname (no extra args)
    pattern = re.compile(
        r"^( +)(\w+) = convert_to_tensor\(\2\)\s*$", re.MULTILINE
    )

    count = 0

    def replace(m: re.Match) -> str:
        nonlocal count
        indent = m.group(1)
        var = m.group(2)
        count += 1
        return (
            f"{indent}if type({var}) is not torch.Tensor:\n"
            f"{indent}    {var} = convert_to_tensor({var})"
        )

    result = pattern.sub(replace, source)
    return result, count


FILES = [
    "keras/src/backend/torch/nn.py",
    "keras/src/backend/torch/numpy.py",
    "keras/src/backend/torch/math.py",
]

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    root = "/Users/hellorahul/Projects/keras"
    total = 0
    for rel in FILES:
        path = f"{root}/{rel}"
        with open(path) as f:
            src = f.read()
        new_src, n = transform(src)
        total += n
        print(f"{rel}: {n} transforms")
        if not dry_run and n:
            with open(path, "w") as f:
                f.write(new_src)
    print(f"\nTotal: {total} transforms {'(dry run)' if dry_run else 'applied'}")
