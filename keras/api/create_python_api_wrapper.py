# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Thin wrapper to call TensorFlow's API generation script.

This file exists to provide a main function for the py_binary in the API
generation genrule. It just calls the main function for the actual API
generation script in TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras  # pylint: disable=unused-import
from tensorflow.python.tools.api.generator import create_python_api

if __name__ == '__main__':
  create_python_api.main()
