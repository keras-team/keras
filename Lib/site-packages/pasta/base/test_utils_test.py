# coding=utf-8
"""Tests for google3.third_party.py.pasta.base.test_utils."""
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import unittest

from pasta.base import test_utils


class CheckAstEqualityTest(test_utils.TestCase):

  def test_empty(self):
    src = ""
    t = ast.parse(src)
    self.checkAstsEqual(t, t)

  def test_one_global(self):
    src = "X = 1\n"
    t = ast.parse(src)
    self.checkAstsEqual(t, t)

  def test_two_globals(self):
    src = "X = 1\nY = 2\n"
    t = ast.parse(src)
    self.checkAstsEqual(t, t)

  def test_different_number_of_nodes(self):
    src1 = "X = 1\ndef Foo():\n  return None\n"
    src2 = src1 + "Y = 2\n"
    t1 = ast.parse(src1)
    t2 = ast.parse(src2)
    with self.assertRaises(AssertionError):
      self.checkAstsEqual(t1, t2)

  def test_simple_function_def(self):
    code = ("def foo(x):\n"
            "  return x + 1\n")
    t = ast.parse(code)
    self.checkAstsEqual(t, t)


def suite():
  result = unittest.TestSuite()
  result.addTests(unittest.makeSuite(CheckAstEqualityTest))
  return result


if __name__ == '__main__':
  unittest.main()
