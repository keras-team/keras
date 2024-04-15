#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Generating public APIs..."
python3 "${base_dir}"/api_gen.py

echo "Formatting code..."
isort --sp "${base_dir}/pyproject.toml" .

black --config "${base_dir}/pyproject.toml" .

flake8 --config "${base_dir}/setup.cfg" .

