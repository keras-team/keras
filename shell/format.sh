#!/bin/bash
set -Eeuo pipefail

if ! command -v pre-commit 2>&1 >/dev/null
then
    echo 'Please `pip install pre-commit` to run format.sh.'
    exit 1
fi

base_dir=$(dirname $(dirname $0))

echo "Formatting all files..."
SKIP=api-gen pre-commit run --all-files
