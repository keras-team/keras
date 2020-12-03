#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function build_wheel() {
  TMPDIR="$1"
  DEST="$2"
  PROJECT_NAME="$3"

  mkdir -p "$TMPDIR"
  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/keras ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi
  cp -r "bazel-bin/keras/tools/pip_package/build_pip_package.runfiles/org_keras/keras" "$TMPDIR"
  cp keras/tools/pip_package/setup.py "$TMPDIR"

  # Make sure init files exist.
  touch "${TMPDIR}/keras/__init__.py"
  touch "${TMPDIR}/keras/applications/__init__.py"
  touch "${TMPDIR}/keras/benchmarks/__init__.py"
  touch "${TMPDIR}/keras/datasets/__init__.py"
  touch "${TMPDIR}/keras/distribute/__init__.py"
  touch "${TMPDIR}/keras/engine/__init__.py"
  touch "${TMPDIR}/keras/estimator/__init__.py"
  touch "${TMPDIR}/keras/feature_column/__init__.py"
  touch "${TMPDIR}/keras/google_utils/__init__.py"
  touch "${TMPDIR}/keras/initializers/__init__.py"
  touch "${TMPDIR}/keras/layers/__init__.py"
  touch "${TMPDIR}/keras/layers/ops/__init__.py"
  touch "${TMPDIR}/keras/layers/preprocessing/__init__.py"
  touch "${TMPDIR}/keras/legacy_tf_layers/__init__.py"
  touch "${TMPDIR}/keras/mixed_precision/__init__.py"
  touch "${TMPDIR}/keras/mixed_precision/experimental/__init__.py"
  touch "${TMPDIR}/keras/premade/__init__.py"
  touch "${TMPDIR}/keras/preprocessing/__init__.py"
  touch "${TMPDIR}/keras/saving/__init__.py"
  touch "${TMPDIR}/keras/utils/__init__.py"
  touch "${TMPDIR}/keras/wrappers/__init__.py"


  pushd ${TMPDIR} > /dev/null
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel --universal --project_name $PROJECT_NAME
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
  rm -rf "${TMPDIR}"
}

function main() {
  NIGHTLY_BUILD=0

  while true; do
    if [[ -z "$1" ]]; then
      break
    elif [[ "$1" == "--nightly" ]]; then
      NIGHTLY_BUILD=1
    else
      DSTDIR="$(real_path $1)"
    fi
    shift
  done

  PROJECT_NAME="keras"
  if [[ ${NIGHTLY_BUILD} == "1" ]]; then
    PROJECT_NAME="keras-nightly"
  fi

  SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"

  if [[ -z "$DSTDIR" ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PROJECT_NAME"
}

main "$@"
