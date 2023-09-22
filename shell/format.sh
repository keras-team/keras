#!/bin/bash
set -Eeuo pipefail

# Define the base directory
base_dir=$(dirname "$(dirname "$0")")

# Run isort with the specified configuration file
isort --settings-path "${base_dir}/pyproject.toml" .

# Run black with the specified configuration file
black --config "${base_dir}/pyproject.toml" .

# Run flake8 with the specified configuration file
flake8 --config "${base_dir}/setup.cfg" .
