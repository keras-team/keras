#!/usr/bin/env bash
# Run a Keras layer benchmark across multiple backends and compare results.
#
# Usage:
#   ./benchmarks/run_multi_backend.sh MODULE [FLAGS...]
#
# Examples:
#   # Run all conv benchmarks on jax, torch, and tensorflow:
#   ./benchmarks/run_multi_backend.sh benchmarks.layer_benchmark.conv_benchmark
#
#   # Run a single benchmark with custom settings:
#   ./benchmarks/run_multi_backend.sh benchmarks.layer_benchmark.conv_benchmark \
#       --benchmark_name=benchmark_conv2D --num_samples=2048 --batch_size=256
#
#   # Compare only jax and torch:
#   BACKENDS="jax torch" ./benchmarks/run_multi_backend.sh \
#       benchmarks.layer_benchmark.conv_benchmark
#
# Set the BACKENDS environment variable to override the default list.

set -euo pipefail

MODULE="${1:?Usage: $0 MODULE [FLAGS...]}"
shift

BACKENDS="${BACKENDS:-jax torch tensorflow}"

declare -A results  # backend:layer:mode -> throughput

for backend in $BACKENDS; do
    echo "=== Backend: $backend ==="

    output=$(
        KERAS_BACKEND="$backend" _BENCHMARK_RESULT_JSON=1 \
        python3 -m "$MODULE" "$@" 2>&1
    ) || true

    # Print non-result output (warnings, progress, etc.).
    echo "$output" | grep -v "^BENCHMARK_RESULT:" || true

    # Parse JSON result lines in a single Python call per line.
    while IFS= read -r line; do
        json="${line#BENCHMARK_RESULT:}"
        read -r layer mode tp < <(
            echo "$json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['layer'], d['mode'], d['throughput'])
"
        )
        results["$backend:$layer:$mode"]="$tp"
    done < <(echo "$output" | grep "^BENCHMARK_RESULT:" || true)

    echo
done

# Collect unique layer:mode pairs preserving insertion order.
declare -A seen_keys
ordered_keys=()
for key in "${!results[@]}"; do
    lm="${key#*:}"  # strip backend prefix
    if [[ -z "${seen_keys[$lm]+x}" ]]; then
        seen_keys["$lm"]=1
        ordered_keys+=("$lm")
    fi
done

if [[ ${#ordered_keys[@]} -eq 0 ]]; then
    echo "No benchmark results collected."
    exit 0
fi

# Print comparison table.
echo "Comparison (samples/sec)"
echo

header=$(printf "%-30s" "Benchmark")
for backend in $BACKENDS; do
    header+=$(printf " %15s" "$backend")
done
echo "$header"
printf '%.0s-' {1..80}
echo

for lm in "${ordered_keys[@]}"; do
    layer="${lm%%:*}"
    mode="${lm#*:}"
    row=$(printf "%-30s" "$layer ($mode)")
    for backend in $BACKENDS; do
        val="${results[$backend:$lm]:-n/a}"
        row+=$(printf " %15s" "$val")
    done
    echo "$row"
done
