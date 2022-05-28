#!/bin/bash
isort --sl keras
black --line-length 80 keras
flake8 keras
