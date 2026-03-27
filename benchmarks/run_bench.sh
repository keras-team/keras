#!/bin/bash
# run_bench.sh — Full benchmark across all frameworks and backends
#
# Pass 1: pure JAX + pure PyTorch + Keras[torch]   -> bench_torch.json
# Pass 2: Keras[jax] only                          -> bench_jax.json
# Then prints a combined grouped summary.
#
# Usage:
#   ./benchmarks/run_bench.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="$SCRIPT_DIR/bench.py"

echo "════════════════════════════════════════════════════════════════════"
echo "  Pass 1 of 2: Pure JAX + Pure PyTorch + Keras[torch]"
echo "════════════════════════════════════════════════════════════════════"
KERAS_BACKEND=torch python "$BENCH" --tag torch

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Pass 2 of 2: Keras[jax]  (skipping pure JAX/torch — already run)"
echo "════════════════════════════════════════════════════════════════════"
KERAS_BACKEND=jax python "$BENCH" --tag jax --skip jax,torch

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Combined Summary"
echo "════════════════════════════════════════════════════════════════════"
python3 << 'PYEOF'
import json, os, sys

files = {"torch": "bench_torch.json", "jax": "bench_jax.json"}
combined = {}
for tag, fname in files.items():
    if os.path.exists(fname):
        combined.update(json.load(open(fname)))
    else:
        print(f"  WARNING: {fname} not found", file=sys.stderr)

groups = {
    "Pure JAX":     [k for k in combined if k.startswith("jax ")],
    "Pure PyTorch": [k for k in combined if k.startswith("torch ")],
    "Keras[torch]": [k for k in combined if "keras[torch]" in k],
    "Keras[jax]":   [k for k in combined if "keras[jax]" in k],
}

for name, keys in groups.items():
    if not keys:
        continue
    print(f"\n  {name}")
    print(f"  {'-'*64}")
    for k in sorted(keys):
        print(f"    {k:<52s}  {combined[k]:8.2f} ms")

# Overhead ratios (Keras vs native PyTorch for matching ops)
pt = {k: v for k, v in combined.items() if k.startswith("torch ")}
kt = {k: v for k, v in combined.items() if "keras[torch]" in k}
kj = {k: v for k, v in combined.items() if "keras[jax]" in k}

pairs = [
    ("torch  CNN  eager",                    "keras[torch]  CNN  eager",              "CNN eager"),
    ("torch  CNN  compile",                  "keras[torch]  CNN  compile",            "CNN compile"),
    ("torch  LLM  forward  eager",           "keras[torch]  LLM  forward  eager",     "LLM fwd eager"),
    ("torch  LLM  forward  compile",         "keras[torch]  LLM  forward  compile",   "LLM fwd compile"),
    (f"torch  LLM  generate 32tok  eager",   "keras[torch]  LLM  generate 32tok  eager",  "LLM gen eager"),
    (f"torch  LLM  generate 32tok  compile", "keras[torch]  LLM  generate 32tok  compile","LLM gen compile"),
]

print("\n\n  Keras[torch] overhead vs raw PyTorch")
print(f"  {'-'*64}")
print(f"  {'Op':<22s}  {'PyTorch':>10s}  {'Keras':>10s}  {'Overhead':>10s}")
print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
for pt_k, kt_k, label in pairs:
    if pt_k in combined and kt_k in combined:
        pt_v, kt_v = combined[pt_k], combined[kt_k]
        ratio = kt_v / pt_v if pt_v > 0 else float("inf")
        print(f"  {label:<22s}  {pt_v:8.2f}ms  {kt_v:8.2f}ms  {ratio:8.2f}x")
PYEOF
