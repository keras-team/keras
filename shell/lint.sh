#!/bin/bash
set -Euo pipefail

base_dir=$(dirname $(dirname $0))

ruff check --config "${base_dir}/pyproject.toml" .
exitcode=$?

ruff format --check --config "${base_dir}/pyproject.toml" .
exitcode=$(($exitcode + $?))

exit $exitcode
