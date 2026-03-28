#!/usr/bin/env bash
# =============================================================================
# bench_gemma3.sh — Run Gemma3 benchmark across backends + keras variants.
#
# Usage:
#   ./bench_gemma3.sh [--profile] [--detail] [--gen-tokens N] [--batch N] [--seq N]
#
# What it does:
#   1. Activates the local venv
#   2. Runs benchmarks with KERAS_BACKEND=torch  using LOCAL   keras
#   3. Runs benchmarks with KERAS_BACKEND=jax    using LOCAL   keras
#   4. Runs benchmarks with KERAS_BACKEND=torch  using PIP     keras
#   5. Runs benchmarks with KERAS_BACKEND=jax    using PIP     keras
#   6. Prints a final comparison table
#
# The "local" keras is the editable install at /mnt/c/projects/Google/keras.
# The "pip" keras is the version in the system site-packages of the venv.
#
# Requires:
#   - WSL with the keras venv at /mnt/c/projects/Google/keras/venv_keras
#   - keras-hub at /mnt/c/projects/Google/keras-hub (local editable or on path)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERAS_ROOT="$(dirname "$SCRIPT_DIR")"                    # .../keras
HUB_ROOT="/mnt/c/projects/Google/keras-hub"
VENV="$KERAS_ROOT/venv_keras"
BENCH="$SCRIPT_DIR/gemma3_bench.py"
RESULTS_DIR="$SCRIPT_DIR/gemma3_results"
mkdir -p "$RESULTS_DIR"

# ── CLI args ─────────────────────────────────────────────────────────────────
PROFILE_FLAG=""
DETAIL_FLAG=""
GEN_T=10
BATCH=2
SEQ=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)    PROFILE_FLAG="GEMMA3_PROFILE=1"; shift ;;
        --detail)     DETAIL_FLAG="GEMMA3_DETAIL=1";  shift ;;
        --gen-tokens) GEN_T="$2";  shift 2 ;;
        --batch)      BATCH="$2";  shift 2 ;;
        --seq)        SEQ="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export GEMMA3_GENERATE_T="$GEN_T"
export GEMMA3_BATCH="$BATCH"
export GEMMA3_SEQ="$SEQ"
[[ -n "$PROFILE_FLAG" ]] && export GEMMA3_PROFILE=1
[[ -n "$DETAIL_FLAG"  ]] && export GEMMA3_DETAIL=1

# ── Helpers ──────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

section() { echo -e "\n${BOLD}${CYAN}━━━  $1  ━━━${NC}\n"; }
ok()      { echo -e "${GREEN}✓ $1${NC}"; }

HR="────────────────────────────────────────────────────────────────────"

# ── Activate the venv ────────────────────────────────────────────────────────
section "Activating venv"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
ok "venv activated: $VENV"

# Add the local keras-hub so `import keras_hub` finds it
export PYTHONPATH="$HUB_ROOT:${PYTHONPATH:-}"
ok "PYTHONPATH includes $HUB_ROOT"

# ── Function: run one benchmark variant ──────────────────────────────────────
run_variant() {
    local backend="$1"
    local keras_mode="$2"   # "local" or "pip"
    local label="$backend/$keras_mode"
    local out="$RESULTS_DIR/${backend}_${keras_mode}.txt"

    section "Running: $label"

    # Switch keras: local uses the editable install (already on PYTHONPATH
    # because of pip install -e); pip uses the installed wheel.
    if [[ "$keras_mode" == "local" ]]; then
        # Ensure the local editable copy is first on the path
        local PYTHONPATH_BACKUP="$PYTHONPATH"
        export PYTHONPATH="$KERAS_ROOT:$PYTHONPATH"
        echo "Using LOCAL keras from: $KERAS_ROOT"
    else
        # Temporarily shadow the editable install by removing it from
        # PYTHONPATH; pip-installed keras will be found in site-packages.
        local PYTHONPATH_BACKUP="$PYTHONPATH"
        # Remove $KERAS_ROOT from PYTHONPATH
        export PYTHONPATH="${PYTHONPATH//$KERAS_ROOT:/}"
        echo "Using PIP keras from site-packages"
    fi

    # Let GPU be used if available (no CUDA_VISIBLE_DEVICES override).
    KERAS_BACKEND="$backend" \
    TF_CPP_MIN_LOG_LEVEL=3 \
        python "$BENCH" 2>&1 | tee "$out"

    export PYTHONPATH="$PYTHONPATH_BACKUP"
    ok "Results saved → $out"
}

# ── Run all four combinations ─────────────────────────────────────────────────
run_variant torch local
run_variant jax   local
run_variant torch pip
run_variant jax   pip

# ── Comparison table ──────────────────────────────────────────────────────────
section "COMPARISON TABLE"
echo "$HR"
printf "%-58s %12s %12s %12s %12s\n" "Metric" "torch/local" "jax/local" "torch/pip" "jax/pip"
echo "$HR"

# Parse ms/call values out of the result files using grep + awk
extract() {
    local file="$1"
    local pattern="$2"
    # Get the first match, extract the number before "ms/call"
    grep -m1 "$pattern" "$file" 2>/dev/null \
        | grep -oP '\d+\.\d+(?= ms)' \
        | head -1
}

print_row() {
    local label="$1"
    local pattern="$2"
    local v1 v2 v3 v4
    v1=$(extract "$RESULTS_DIR/torch_local.txt" "$pattern")
    v2=$(extract "$RESULTS_DIR/jax_local.txt"   "$pattern")
    v3=$(extract "$RESULTS_DIR/torch_pip.txt"   "$pattern")
    v4=$(extract "$RESULTS_DIR/jax_pip.txt"     "$pattern")
    printf "%-58s %12s %12s %12s %12s\n" \
        "$label" \
        "${v1:--}" "${v2:--}" "${v3:--}" "${v4:--}"
}

print_row "backbone(inputs) [no grad]"    "backbone(inputs)"
print_row "backbone.predict_on_batch"     "predict_on_batch"
print_row "backbone.predict"              "backbone.predict"
print_row "causal_lm(inputs)"             "causal_lm\(inputs\)"
print_row "call_with_cache (1 token)"     "call_with_cache"
print_row "generate() [cached]"           "generate() \[cached"

echo "$HR"
echo
echo -e "${YELLOW}Results saved to: $RESULTS_DIR/${NC}"
echo -e "${YELLOW}Goal: torch/local should approach jax/local performance.${NC}"
