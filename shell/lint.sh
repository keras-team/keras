#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

isort --sp "${base_dir}/pyproject.toml" --check .

black --config "${base_dir}/pyproject.toml" --check .

flake8 --config "${base_dir}/setup.cfg" .

