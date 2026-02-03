#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Generating api directory with public APIs..."
# Generate API Files (prefer venv or Python 3.11 to match CI)
if [ -x "${base_dir}/.venv311/bin/python3.11" ]; then
  "${base_dir}/.venv311/bin/python3.11" "${base_dir}"/api_gen.py
elif [ -x "${base_dir}/.venv/bin/python3" ]; then
  "${base_dir}/.venv/bin/python3" "${base_dir}"/api_gen.py
elif command -v python3.11 &>/dev/null; then
  python3.11 "${base_dir}"/api_gen.py
else
  python3 "${base_dir}"/api_gen.py
fi

# Format code because `api_gen.py` might order
# imports differently.
echo "Formatting api directory..."
(SKIP=api-gen pre-commit run --files $(find "${base_dir}"/keras/api -type f) --hook-stage pre-commit || true) > /dev/null
