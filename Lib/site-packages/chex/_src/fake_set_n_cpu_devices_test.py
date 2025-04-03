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
"""Test for `set_n_cpu_devices` from `fake.py`.

This test is isolated to ensure hermeticity because its execution changes
XLA backend configuration.
"""

import unittest
from absl.testing import absltest
from chex._src import asserts
from chex._src import fake


class DevicesSetterTest(absltest.TestCase):

  def test_set_n_cpu_devices(self):
    try:
      # Should not initialize backends.
      fake.set_n_cpu_devices(4)
    except RuntimeError as set_cpu_exception:
      raise unittest.SkipTest(
          "set_n_cpu_devices: backend's already been initialized. "
          'Run this test in isolation from others.') from set_cpu_exception

    # Hence, this one does not fail.
    fake.set_n_cpu_devices(6)

    # This assert initializes backends.
    asserts.assert_devices_available(6, 'cpu', backend='cpu')

    # Which means that next call must fail.
    with self.assertRaisesRegex(RuntimeError,
                                'Attempted to set 8 devices, but 6 CPUs.+'):
      fake.set_n_cpu_devices(8)


if __name__ == '__main__':
  absltest.main()
