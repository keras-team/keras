#!/bin/bash
isort --sl keras
black --line-length 80 keras
flake8 keras || echo "flake8 reports issues. Run lint.sh to enforce strict checks."
