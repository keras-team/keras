#!/bin/bash
# Copyright 2020 Google Inc. All Rights Reserved.
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

set -e
set -x

cd "${KOKORO_ROOT}/"

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

PYTHON_BINARY="/usr/bin/python3.9"

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate

# Check the python version
python --version
python3 --version

cd "src/github/keras"

# Keep pip version at 20.1.1 to avoid the slow resolver issue.
pip install -U pip==20.1.1 setuptools
pip install -r requirements.txt
# Uninstall the keras-nightly package so that we will only test the version of
# keras code from local workspace.
pip uninstall -y keras-nightly

# TODO(scottzhu): Using --define=use_fast_cpp_protos=false to suppress the
# protobuf build issue for now. We should have a proper solution for this.
bazel test --test_timeout 300,450,1200,3600 --test_output=errors --keep_going \
   --define=use_fast_cpp_protos=false \
   --build_tests_only \
   --build_tag_filters="-no_oss,-oss_excluded" \
   --test_tag_filters="-no_oss,-oss_excluded" \
   -- //keras/...
