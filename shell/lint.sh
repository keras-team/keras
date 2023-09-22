#!/bin/bash
set -Eeuo pipefail

# Define the base directory
base_dir=$(dirname "$(dirname "$0")")

# Run isort to check import statement sorting
isort --settings-path "${base_dir}/pyproject.toml" --check .

# Run black to check code formatting
black --config "${base_dir}/pyproject.toml" --check .

# Run flake8 for linting using the provided configuration
flake8 --config "${base_dir}/setup.cfg" .
