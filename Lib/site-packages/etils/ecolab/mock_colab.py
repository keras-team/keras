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

"""Utils for tests."""

from __future__ import annotations

import contextlib
import sys
import types
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def mock_colabtools():
  """colabtools only works in Colab, so mock it.."""
  module_mock = _ColabtoolsFrontEndMock('colabtools.frontend')
  sys.modules['colabtools'] = mock.MagicMock()
  sys.modules['colabtools.frontend'] = module_mock
  yield
  del sys.modules['colabtools.frontend']
  del sys.modules['colabtools']


class _ColabtoolsFrontEndMock(types.ModuleType):
  """Mock of `from colabtools import frontend`."""

  def GetUsersWhoHaveConnectedToThisSession(self) -> list[str]:  # pylint: disable=invalid-name
    return ['some_ldap']


@pytest.fixture(scope='module', autouse=True)
def mock_collapse():
  with mock.patch('etils.ecolab.colab_utils.collapse', contextlib.nullcontext):
    yield
