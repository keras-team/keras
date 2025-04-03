# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for methods in `join.py`."""

from absl.testing import absltest
import numpy as np
from optax.schedules import _join
from optax.schedules import _schedule


class JoinTest(absltest.TestCase):

  def test_join_schedules(self):
    my_schedule = _join.join_schedules(
        schedules=[
            _schedule.constant_schedule(1.0),
            _schedule.constant_schedule(2.0),
            _schedule.constant_schedule(1.0),
        ],
        boundaries=[3, 6],
    )
    np.testing.assert_allclose(1.0, my_schedule(0), atol=0.0)
    np.testing.assert_allclose(1.0, my_schedule(1), atol=0.0)
    np.testing.assert_allclose(1.0, my_schedule(2), atol=0.0)
    np.testing.assert_allclose(2.0, my_schedule(3), atol=0.0)
    np.testing.assert_allclose(2.0, my_schedule(4), atol=0.0)
    np.testing.assert_allclose(2.0, my_schedule(5), atol=0.0)
    np.testing.assert_allclose(1.0, my_schedule(6), atol=0.0)


if __name__ == "__main__":
  absltest.main()
