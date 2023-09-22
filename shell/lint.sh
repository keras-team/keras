#!/bin/bash
set -Eeuo pipefail

# Define the base directory
base_dir=$(dirname "$(dirname "$0")")

# Run isort to sort import statements
isort --settings-path "${base_dir}/pyproject.toml" .

# Run black to format the code
black --config "${base_dir}/pyproject.toml" .

# Run flake8 for linting using the provided configuration
flake8 --config "${base_dir}/setup.cfg" .
