#!/bin/bash
# run_bench.sh — Compare pip-installed Keras (baseline) vs PR branch (optimized)
#
# Usage:
#   ./benchmarks/run_bench.sh                      # uses parent dir as PR source
#   ./benchmarks/run_bench.sh /path/to/keras-pr     # explicit PR source
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="$SCRIPT_DIR/bench.py"
KERAS_SRC="${1:-$SCRIPT_DIR/..}"

export KERAS_BACKEND=torch

echo "════════════════════════════════════════════════════════════"
echo "  Step 1: Install pip Keras (baseline)"
echo "════════════════════════════════════════════════════════════"
pip install keras -q 2>&1 | tail -1 || true
echo "  $(python -c 'import keras; print(f"keras {keras.__version__} @ {keras.__file__}")')"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 2: Benchmark baseline"
echo "════════════════════════════════════════════════════════════"
python "$BENCH" --tag baseline

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 3: Install PR branch from $KERAS_SRC"
echo "════════════════════════════════════════════════════════════"
pip install --no-deps --no-build-isolation -e "$KERAS_SRC" -q 2>&1 | tail -1 || true
echo "  $(python -c 'import keras; print(f"keras {keras.__version__} @ {keras.__file__}")')"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Step 4: Benchmark optimized"
echo "════════════════════════════════════════════════════════════"
python "$BENCH" --tag optimized

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Comparison (Keras-specific tests)"
echo "════════════════════════════════════════════════════════════"
python3 << 'PYEOF'
import json, sys
try:
    a = json.load(open("bench_baseline.json"))
    b = json.load(open("bench_optimized.json"))
except FileNotFoundError:
    print("  Missing results files"); sys.exit(1)

print(f"  {'Test':<52s} {'Baseline':>10s} {'PR':>10s} {'Speedup':>8s}")
print(f"  {'-'*52}  {'-'*9}  {'-'*9}  {'-'*7}")
for k in sorted(a):
    if k in b and 'keras' in k:
        sp = a[k] / b[k] if b[k] > 0 else float('inf')
        marker = " <<<" if sp > 1.05 else ""
        print(f"  {k:<52s} {a[k]:8.2f}ms {b[k]:8.2f}ms  {sp:6.2f}x{marker}")
PYEOF
