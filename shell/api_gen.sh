#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Generating api directory with public APIs..."
# Generate API Files
python3 "${base_dir}"/api_gen.py

# Format code because `api_gen.py` might order
# imports differently.
echo "Formatting api directory..."
(SKIP=api-gen pre-commit run --files $(find "${base_dir}"/keras/api -type f) --hook-stage pre-commit || true) > /dev/null
