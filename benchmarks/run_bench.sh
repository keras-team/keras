#!/bin/bash
# run_bench.sh — Full benchmark across all frameworks and backends
#
# Pass 1: pure JAX + pure PyTorch                  -> bench_pure.json
# Pass 2: Keras[torch] (skipping pure frameworks)  -> bench_keras_torch.json
# Pass 3: Keras[jax]   (skipping pure frameworks)  -> bench_keras_jax.json
# Then prints a combined grouped summary.
#
# Usage:
#   ./benchmarks/run_bench.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="$SCRIPT_DIR/bench.py"

echo "════════════════════════════════════════════════════════════════════"
echo "  Pass 1 of 3: Pure JAX + Pure PyTorch"
echo "════════════════════════════════════════════════════════════════════"
KERAS_BACKEND=torch python3 "$BENCH" --tag pure --skip keras

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Pass 2 of 3: Keras with torch backend"
echo "════════════════════════════════════════════════════════════════════"
KERAS_BACKEND=torch python3 "$BENCH" --tag keras_torch --skip jax,torch

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Pass 3 of 3: Keras with jax backend"
echo "════════════════════════════════════════════════════════════════════"
KERAS_BACKEND=jax python3 "$BENCH" --tag keras_jax --skip jax,torch

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Combined Summary — All Frameworks & Backends"
echo "════════════════════════════════════════════════════════════════════"
python3 << 'PYEOF'
import json, os, sys

# Load all benchmark files
files_to_load = {
    "Pure Frameworks": "bench_pure.json",
    "Keras[torch]": "bench_keras_torch.json", 
    "Keras[jax]": "bench_keras_jax.json",
}

combined = {}
for name, fname in files_to_load.items():
    if os.path.exists(fname):
        data = json.load(open(fname))
        combined.update(data)
        print(f"  ✓ Loaded {fname}")
    else:
        print(f"  ✗ Missing {fname}", file=sys.stderr)

print("\n" + "="*70)

# Group results by category
groups = {
    "Pure JAX":     sorted(k for k in combined if k.startswith("jax ")),
    "Pure PyTorch": sorted(k for k in combined if k.startswith("torch ")),
    "Keras[torch]": sorted(k for k in combined if "keras[torch]" in k),
    "Keras[jax]":   sorted(k for k in combined if "keras[jax]" in k),
}

for name, keys in groups.items():
    if not keys:
        continue
    print(f"\n  {name}")
    print(f"  {'-'*66}")
    for k in keys:
        print(f"    {k:<52s}  {combined[k]:8.2f} ms")

# Keras[torch] vs PyTorch overhead
print("\n\n  Keras[torch] Overhead vs PyTorch")
print(f"  {'-'*70}")
print(f"  {'Operation':<24s}  {'PyTorch':>12s}  {'Keras[torch]':>12s}  {'Overhead':>12s}")
print(f"  {'-'*24}  {'-'*12}  {'-'*12}  {'-'*12}")

pairs_torch = [
    ("torch  CNN  eager", "keras[torch]  CNN  eager", "CNN eager"),
    ("torch  CNN  compile", "keras[torch]  CNN  compile", "CNN compile"),
    ("torch  LLM  forward  eager", "keras[torch]  LLM  forward  eager", "LLM forward eager"),
    ("torch  LLM  forward  compile", "keras[torch]  LLM  forward  compile", "LLM forward compile"),
    ("torch  LLM  generate 32tok  eager", "keras[torch]  LLM  generate 32tok  eager", "LLM generate eager"),
    ("torch  LLM  generate 32tok  compile", "keras[torch]  LLM  generate 32tok  compile", "LLM generate compile"),
]

for torch_k, keras_k, label in pairs_torch:
    if torch_k in combined and keras_k in combined:
        t_val, k_val = combined[torch_k], combined[keras_k]
        ratio = k_val / t_val
        print(f"  {label:<24s}  {t_val:10.2f} ms  {k_val:10.2f} ms  {ratio:10.2f}x")

# Keras[jax] vs JAX overhead
print("\n\n  Keras[jax] Overhead vs JAX")
print(f"  {'-'*70}")
print(f"  {'Operation':<24s}  {'Pure JAX':>12s}  {'Keras[jax]':>12s}  {'Overhead':>12s}")
print(f"  {'-'*24}  {'-'*12}  {'-'*12}  {'-'*12}")

pairs_jax = [
    ("jax  CNN  eager", "keras[jax]  CNN  eager", "CNN eager"),
    ("jax  CNN  jit", "keras[jax]  CNN  jit", "CNN jit"),
    ("jax  LLM  forward  eager", "keras[jax]  LLM  forward  eager", "LLM forward eager"),
    ("jax  LLM  forward  jit", "keras[jax]  LLM  forward  jit", "LLM forward jit"),
    ("jax  LLM  generate 32tok  eager", "keras[jax]  LLM  generate 32tok  eager", "LLM generate eager"),
    ("jax  LLM  generate 32tok  jit+scan", "keras[jax]  LLM  generate 32tok  jit", "LLM generate jit"),
]

for jax_k, keras_k, label in pairs_jax:
    if jax_k in combined and keras_k in combined:
        j_val, k_val = combined[jax_k], combined[keras_k]
        ratio = k_val / j_val
        print(f"  {label:<24s}  {j_val:10.2f} ms  {k_val:10.2f} ms  {ratio:10.2f}x")

# Cross-backend comparison for Keras
print("\n\n  Keras Backend Comparison (torch vs jax)")
print(f"  {'-'*70}")
print(f"  {'Operation':<24s}  {'Keras[torch]':>12s}  {'Keras[jax]':>12s}  {'Ratio':>12s}")
print(f"  {'-'*24}  {'-'*12}  {'-'*12}  {'-'*12}")

cross_pairs = [
    ("keras[torch]  CNN  eager", "keras[jax]  CNN  eager", "CNN eager"),
    ("keras[torch]  LLM  forward  eager", "keras[jax]  LLM  forward  eager", "LLM forward eager"),
    ("keras[torch]  LLM  generate 32tok  eager", "keras[jax]  LLM  generate 32tok  eager", "LLM generate eager"),
]

for torch_k, jax_k, label in cross_pairs:
    if torch_k in combined and jax_k in combined:
        t_val, j_val = combined[torch_k], combined[jax_k]
        ratio = t_val / j_val
        marker = "(torch faster)" if ratio > 1 else "(jax faster)"
        print(f"  {label:<24s}  {t_val:10.2f} ms  {j_val:10.2f} ms  {ratio:10.2f}x {marker}")

print("\n" + "="*70 + "\n")
PYEOF
