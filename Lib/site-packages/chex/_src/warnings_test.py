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
"""Tests for `warnings.py`."""

import functools

from absl.testing import absltest
from chex._src import warnings


@functools.partial(warnings.warn_only_n_pos_args_in_future, n=1)
def f(a, b, c):
  return a + b + c


@warnings.warn_deprecated_function
def g0(a, b, c):
  return a + b + c


@functools.partial(warnings.warn_deprecated_function, replacement='h')
def g1(a, b, c):
  return a + b + c


def h1(a, b, c):
  return a + b + c


h2 = warnings.create_deprecated_function_alias(h1, 'path.h2', 'path.h1')


class WarningsTest(absltest.TestCase):

  def test_warn_only_n_pos_args_in_future(self):
    with self.assertWarns(Warning):
      f(1, 2, 3)
    with self.assertWarns(Warning):
      f(1, 2, c=3)

  def test_warn_deprecated_function_no_replacement(self):
    with self.assertWarns(Warning) as cm:
      g0(1, 2, 3)

    warning_message = str(cm.warnings[0].message)

    self.assertIn('The function g0 is deprecated', warning_message)
    # the warning message doesn't have a replacement function
    self.assertNotIn('please use', warning_message)

  def test_warn_deprecated_function(self):
    with self.assertWarns(Warning) as cm:
      g1(1, 2, 3)

    warning_message = str(cm.warnings[0].message)
    self.assertIn('The function g1 is deprecated', warning_message)
    # check that the warning message points to replacement function
    self.assertIn('Please use h instead', warning_message)

  def test_create_deprecated_function_alias(self):
    with self.assertWarns(Warning):
      h2(1, 2, 3)


if __name__ == '__main__':
  absltest.main()
