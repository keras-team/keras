#!/usr/bin/env bash
# Compare Keras inference overhead: pip-installed (baseline) vs local repo.
#
# Usage:
#   cd keras/
#   bash benchmarks/compare.sh
#
# Prerequisites:
#   - venv at ./venv with torch, jax, flax, numpy installed
#   - keras pip-installed in that venv as baseline
#   - local repo at current directory with modifications
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_DIR/venv"
BENCH="$SCRIPT_DIR/bench.py"
RESULTS_DIR="$REPO_DIR/bench_results"
mkdir -p "$RESULTS_DIR"

if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi

source "$VENV/bin/activate"

echo "============================================================"
echo "  Step 1: Pure framework baselines (no Keras overhead)"
echo "============================================================"
python "$BENCH" --mode pure --out "$RESULTS_DIR/pure.json"

echo ""
echo "============================================================"
echo "  Step 2: Baseline — pip-installed Keras"
echo "============================================================"
# Temporarily uninstall local editable keras and install from PyPI
pip install --quiet keras --no-deps 2>/dev/null
KERAS_BACKEND=torch python "$BENCH" --mode keras --out "$RESULTS_DIR/baseline_torch.json"

echo ""
echo "============================================================"
echo "  Step 3: Optimized — local repo Keras"
echo "============================================================"
# Reinstall local editable version
pip install --quiet -e "$REPO_DIR" --no-deps 2>/dev/null
KERAS_BACKEND=torch python "$BENCH" --mode keras --out "$RESULTS_DIR/optimized_torch.json"

echo ""
echo "============================================================"
echo "  Step 4: Comparison"
echo "============================================================"
python3 << 'PYEOF'
import json, os

rd = os.environ.get("RESULTS_DIR", "bench_results")
pure = json.load(open(f"{rd}/pure.json"))
base = json.load(open(f"{rd}/baseline_torch.json"))
opt = json.load(open(f"{rd}/optimized_torch.json"))

print(f"\n{'Op':<45s}  {'Pure':>8s}  {'Baseline':>9s}  {'Optimized':>10s}  {'BL/Pure':>8s}  {'Opt/Pure':>9s}  {'Speedup':>8s}")
print("-" * 100)

comparisons = [
    ("CNN eager",
     "torch  CNN  eager",
     "keras[torch]  CNN  eager",
     "keras[torch]  CNN  eager"),
    ("CNN compile",
     "torch  CNN  compile",
     "keras[torch]  CNN  compile",
     "keras[torch]  CNN  compile"),
    ("LLM forward eager",
     "torch  LLM  forward  eager",
     "keras[torch]  LLM  forward  eager",
     "keras[torch]  LLM  forward  eager"),
    ("LLM forward compile",
     "torch  LLM  forward  compile",
     "keras[torch]  LLM  forward  compile",
     "keras[torch]  LLM  forward  compile"),
    ("LLM generate 8tok eager",
     "torch  LLM  generate 8tok  eager",
     "keras[torch]  LLM  generate 8tok  eager",
     "keras[torch]  LLM  generate 8tok  eager"),
]

for label, pure_key, base_key, opt_key in comparisons:
    p = pure.get(pure_key, 0)
    b = base.get(base_key, 0)
    o = opt.get(opt_key, 0)
    if p and b and o:
        bl_ratio = b / p
        opt_ratio = o / p
        speedup = b / o
        flag = "✓" if opt_ratio <= 1.5 else ("~" if opt_ratio <= 2.0 else "✗")
        print(f"  {label:<43s}  {p:>7.3f}  {b:>8.3f}ms  {o:>9.3f}ms  {bl_ratio:>7.2f}x  {opt_ratio:>8.2f}x  {speedup:>7.2f}x {flag}")

print()
PYEOF

echo ""
echo "Results saved to $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
