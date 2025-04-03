# Copyright 2024 The etils Authors.
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

"""Environment utils."""

import os
import sys


def is_notebook() -> bool:
  """Returns True if running in a notebook (Colab, Jupyter) environment."""
  # Use sys.module as we do not want to trigger an import (slow)
  # Check whether we're running in a IPython notebook (and not terminal)
  if IPython := sys.modules.get('IPython'):  # pylint: disable=invalid-name
    ipython = IPython.get_ipython()
    if ipython and 'IPKernelApp' in ipython.config:
      return True
  return False


def is_test() -> bool:
  """Returns True if running in a test environment."""
  return 'TEST_TMPDIR' in os.environ
