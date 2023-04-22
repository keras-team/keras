#!/bin/bash -e

base_dir=$(dirname $(dirname $0))
targets="${base_dir}/*.py ${base_dir}/keras_core/"

isort --sp "${base_dir}/setup.cfg" --sl ${targets}
black --line-length 80 ${targets}

flake8 --config "${base_dir}/setup.cfg" --max-line-length=200 ${targets}
