#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

ruff check --exit-zero --config "${base_dir}/pyproject.toml" .

black --config "${base_dir}/pyproject.toml" --check .

