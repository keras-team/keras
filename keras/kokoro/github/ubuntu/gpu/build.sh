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

# LD Library Path needs to be same as TensorFlow Ubuntu Docker build -
# https://github.com/tensorflow/build/blob/master/tf_sig_build_dockerfiles/Dockerfile
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.1/lib64"
export TF_CUDA_COMPUTE_CAPABILITIES=6.0
TF_CUDA_CONFIG_REPO="@ubuntu16.04-py3-gcc7_manylinux2010-cuda10.1-cudnn7-tensorrt6.0_config_cuda"

tag_filters="gpu,-no_gpu,-nogpu,-benchmark-test,-no_oss,-oss_excluded,-oss_serial,-no_gpu_presubmit"
# There are only 4 GPU available on the local test machine.
TF_GPU_COUNT=4
TF_TESTS_PER_GPU=8
LOCAL_TEST_JOBS=32  # TF_GPU_COUNT * TF_TESTS_PER_GPU

# TODO(scottzhu): Using --define=use_fast_cpp_protos=false to suppress the
# protobuf build issue for now. We should have a proper solution for this.
bazel test --test_timeout 300,600,1200,3600 --test_output=errors --keep_going \
   --define=use_fast_cpp_protos=false \
   --build_tests_only \
   --action_env=TF_CUDA_COMPUTE_CAPABILITIES="${TF_CUDA_COMPUTE_CAPABILITIES}" \
   --action_env=TF_CUDA_CONFIG_REPO="${TF_CUDA_CONFIG_REPO}" \
   --action_env=TF_CUDA_VERSION=10 \
   --action_env=TF_CUDNN_VERSION=7 \
   --test_env=TF_GPU_COUNT=${TF_GPU_COUNT} \
   --test_env=TF_TESTS_PER_GPU=${TF_TESTS_PER_GPU} \
   --build_tag_filters="${tag_filters}" \
   --test_tag_filters="${tag_filters}" \
   --run_under=@org_keras//keras/tools/gpu_build:parallel_gpu_execute \
   --local_test_jobs=${LOCAL_TEST_JOBS} \
   -- //keras/...
