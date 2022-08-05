#!/bin/bash
isort --sl keras
flynt --line-length 80 keras
black --line-length 80 keras
flake8 keras
