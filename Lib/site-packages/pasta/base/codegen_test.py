# coding=utf-8
"""Tests for generating code from a non-annotated ast."""
# Copyright 2017 Google LLC
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
import os.path
import unittest
from six import with_metaclass

import pasta
from pasta.base import codegen
from pasta.base import test_utils

TESTDATA_DIR = os.path.realpath(
    os.path.join(os.path.dirname(pasta.__file__), '../testdata'))


def _is_syntax_valid(filepath):
  with open(filepath, 'r') as f:
    try:
      ast.parse(f.read())
    except SyntaxError:
      return False
  return True


class AutoFormatTestMeta(type):

  def __new__(mcs, name, bases, inst_dict):
    # Helper function to generate a test method
    def auto_format_test_generator(input_file):
      def test(self):
        with open(input_file, 'r') as handle:
          src = handle.read()
        t = ast.parse(src)
        auto_formatted = codegen.to_str(t)
        self.assertMultiLineEqual(src, auto_formatted)
      return test

    # Add a test method for each input file
    test_method_prefix = 'test_auto_format_'
    data_dir = os.path.join(TESTDATA_DIR, 'codegen')
    for dirpath, _, files in os.walk(data_dir):
      for filename in files:
        if filename.endswith('.in'):
          full_path = os.path.join(dirpath, filename)
          inst_dict[test_method_prefix + filename[:-3]] = unittest.skipIf(
              not _is_syntax_valid(full_path),
              'Test contains syntax not supported by this version.',
              )(auto_format_test_generator(full_path))
    return type.__new__(mcs, name, bases, inst_dict)


class AutoFormatTest(with_metaclass(AutoFormatTestMeta, test_utils.TestCase)):
  """Tests that code without formatting info is printed neatly."""

  def test_imports(self):
    src = 'from a import b\nimport c, d\nfrom ..e import f, g\n'
    t = ast.parse(src)
    self.assertEqual(src, pasta.dump(t))

  @test_utils.requires_features('exec_node')
  def test_exec_node_default(self):
    src = 'exec foo in bar'
    t = ast.parse(src)
    self.assertEqual('exec(foo, bar)\n', pasta.dump(t))

  @test_utils.requires_features('bytes_node')
  def test_bytes(self):
    src = "b'foo'"
    t = ast.parse(src)
    self.assertEqual("b'foo'\n", pasta.dump(t))

  def test_default_indentation(self):
    for indent in ('  ', '    ', '\t'):
      src ='def a():\n' + indent + 'b\n'
      t = pasta.parse(src)
      t.body.extend(ast.parse('def c(): d').body)
      self.assertEqual(codegen.to_str(t),
                       src + 'def c():\n' + indent + 'd\n')


def suite():
  result = unittest.TestSuite()
  result.addTests(unittest.makeSuite(AutoFormatTest))
  return result


if __name__ == '__main__':
  unittest.main()
