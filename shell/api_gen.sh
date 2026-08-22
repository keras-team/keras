#!/bin/bash
set -Eeuo pipefail

# Resolve the repo root as the parent of this script's parent directory
# (this script lives in `<repo>/shell/`).
base_dir=$(dirname "$(dirname "$0")")

echo "Generating api directory with public APIs..."
# Generate API Files
python3 "${base_dir}"/api_gen.py

# Format code because `api_gen.py` might order
# imports differently.
echo "Formatting api directory..."
# Run pre-commit only on the freshly generated API files, skipping the
# `api-gen` hook itself (to avoid recursively regenerating the API).
# `|| true` keeps this script from failing when pre-commit reformats
# files (which it reports as a non-zero exit); output is discarded since
# we only care about the formatting side effect, not the report.
(SKIP=api-gen pre-commit run --files $(find "${base_dir}"/keras/api -type f) --hook-stage pre-commit || true) > /dev/null
