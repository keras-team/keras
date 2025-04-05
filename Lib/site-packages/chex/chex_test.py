# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for chex."""

from absl.testing import absltest
import chex


class ChexTest(absltest.TestCase):
  """Test chex can be imported correctly."""

  def test_import(self):
    self.assertTrue(hasattr(chex, 'assert_devices_available'))


if __name__ == '__main__':
  absltest.main()
