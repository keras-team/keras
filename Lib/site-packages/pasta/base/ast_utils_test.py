# coding=utf-8
"""Tests for ast_utils."""
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

import pasta
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import test_utils


class UtilsTest(test_utils.TestCase):

  def test_sanitize_source(self):
    coding_lines = (
        '# -*- coding: latin-1 -*-',
        '# -*- coding: iso-8859-15 -*-',
        '# vim: set fileencoding=ascii :',
        '# This Python file uses the following encoding: utf-8',
    )
    src_template = '{coding}\na = 123\n'
    sanitized_src = '# (removed coding)\na = 123\n'
    for line in coding_lines:
      src = src_template.format(coding=line)

      # Replaced on lines 1 and 2
      self.assertEqual(sanitized_src, ast_utils.sanitize_source(src))
      src_prefix = '"""Docstring."""\n'
      self.assertEqual(src_prefix + sanitized_src,
                       ast_utils.sanitize_source(src_prefix + src))

      # Unchanged on line 3
      src_prefix = '"""Docstring."""\n# line 2\n'
      self.assertEqual(src_prefix + src,
                       ast_utils.sanitize_source(src_prefix + src))


class AlterChildTest(test_utils.TestCase):

  def testRemoveChildMethod(self):
    src = """\
class C():
  def f(x):
    return x + 2
  def g(x):
    return x + 3
"""
    tree = pasta.parse(src)
    class_node = tree.body[0]
    meth1_node = class_node.body[0]

    ast_utils.remove_child(class_node, meth1_node)

    result = pasta.dump(tree)
    expected = """\
class C():
  def g(x):
    return x + 3
"""
    self.assertEqual(result, expected)

  def testRemoveAlias(self):
    src = "from a import b, c"
    tree = pasta.parse(src)
    import_node = tree.body[0]
    alias1 = import_node.names[0]
    ast_utils.remove_child(import_node, alias1)

    self.assertEqual(pasta.dump(tree), "from a import c")

  def testRemoveFromBlock(self):
    src = """\
if a:
  print("foo!")
  x = 1
"""
    tree = pasta.parse(src)
    if_block = tree.body[0]
    print_stmt = if_block.body[0]
    ast_utils.remove_child(if_block, print_stmt)

    expected = """\
if a:
  x = 1
"""
    self.assertEqual(pasta.dump(tree), expected)

  def testReplaceChildInBody(self):
    src = 'def foo():\n  a = 0\n  a += 1 # replace this\n  return a\n'
    replace_with = pasta.parse('foo(a + 1)  # trailing comment\n').body[0]
    expected = 'def foo():\n  a = 0\n  foo(a + 1) # replace this\n  return a\n'
    t = pasta.parse(src)

    parent = t.body[0]
    node_to_replace = parent.body[1]
    ast_utils.replace_child(parent, node_to_replace, replace_with)

    self.assertEqual(expected, pasta.dump(t))

  def testReplaceChildInvalid(self):
    src = 'def foo():\n  return 1\nx = 1\n'
    replace_with = pasta.parse('bar()').body[0]
    t = pasta.parse(src)

    parent = t.body[0]
    node_to_replace = t.body[1]
    with self.assertRaises(errors.InvalidAstError):
      ast_utils.replace_child(parent, node_to_replace, replace_with)
