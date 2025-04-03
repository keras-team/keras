# coding=utf-8
"""Helpers for working with python ASTs."""
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
import re

from pasta.augment import errors
from pasta.base import formatting as fmt

# From PEP-0263 -- https://www.python.org/dev/peps/pep-0263/
_CODING_PATTERN = re.compile('^[ \t\v]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')


_AST_OP_NODES = (
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.FloorDiv, ast.Invert, ast.Not, ast.UAdd, ast.USub
) 


class _TreeNormalizer(ast.NodeTransformer):
  """Replaces all op nodes with unique instances."""

  def visit(self, node):
    if isinstance(node, _AST_OP_NODES):
      return node.__class__()
    return super(_TreeNormalizer, self).visit(node)


_tree_normalizer = _TreeNormalizer()


def parse(src):
  """Replaces ast.parse; ensures additional properties on the parsed tree.

  This enforces the assumption that each node in the ast is unique.
  """
  tree = ast.parse(sanitize_source(src))
  _tree_normalizer.visit(tree)
  return tree


def sanitize_source(src):
  """Strip the 'coding' directive from python source code, if present.

  This is a workaround for https://bugs.python.org/issue18960. Also see PEP-0263.
  """
  src_lines = src.splitlines(True)
  for i, line in enumerate(src_lines[:2]):
    if _CODING_PATTERN.match(line):
      src_lines[i] = re.sub('#.*$', '# (removed coding)', line)
  return ''.join(src_lines)


def find_nodes_by_type(node, accept_types):
  visitor = FindNodeVisitor(lambda n: isinstance(n, accept_types))
  visitor.visit(node)
  return visitor.results


class FindNodeVisitor(ast.NodeVisitor):

  def __init__(self, condition):
    self._condition = condition
    self.results = []

  def visit(self, node):
    if self._condition(node):
      self.results.append(node)
    super(FindNodeVisitor, self).visit(node)


def get_last_child(node):
  """Get the last child node of a block statement.

  The input must be a block statement (e.g. ast.For, ast.With, etc).

  Examples:
    1. with first():
         second()
         last()

    2. try:
         first()
       except:
         second()
       finally:
         last()

  In both cases, the last child is the node for `last`.
  """
  if isinstance(node, ast.Module):
    try:
      return node.body[-1]
    except IndexError:
      return None
  if isinstance(node, ast.If):
    if (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) and
        fmt.get(node.orelse[0], 'is_elif')):
      return get_last_child(node.orelse[0])
    if node.orelse:
      return node.orelse[-1]
  elif isinstance(node, ast.With):
    if (len(node.body) == 1 and isinstance(node.body[0], ast.With) and
        fmt.get(node.body[0], 'is_continued')):
      return get_last_child(node.body[0])
  elif hasattr(ast, 'Try') and isinstance(node, ast.Try):
    if node.finalbody:
      return node.finalbody[-1]
    if node.orelse:
      return node.orelse[-1]
  elif hasattr(ast, 'TryFinally') and isinstance(node, ast.TryFinally):
    if node.finalbody:
      return node.finalbody[-1]
  elif hasattr(ast, 'TryExcept') and isinstance(node, ast.TryExcept):
    if node.orelse:
      return node.orelse[-1]
    if node.handlers:
      return get_last_child(node.handlers[-1])
  return node.body[-1]


def remove_child(parent, child):
  for _, field_value in ast.iter_fields(parent):
    if isinstance(field_value, list) and child in field_value:
      field_value.remove(child)
      return
  raise errors.InvalidAstError('Unable to find list containing child %r on '
                               'parent node %r' % (child, parent))


def replace_child(parent, node, replace_with):
  """Replace a node's child with another node while preserving formatting.

  Arguments:
    parent: (ast.AST) Parent node to replace a child of.
    node: (ast.AST) Child node to replace.
    replace_with: (ast.AST) New child node.
  """
  # TODO(soupytwist): Don't refer to the formatting dict directly
  if hasattr(node, fmt.PASTA_DICT):
    fmt.set(replace_with, 'prefix', fmt.get(node, 'prefix'))
    fmt.set(replace_with, 'suffix', fmt.get(node, 'suffix'))
  for field in parent._fields:
    field_val = getattr(parent, field, None)
    if field_val == node:
      setattr(parent, field, replace_with)
      return
    elif isinstance(field_val, list):
      try:
        field_val[field_val.index(node)] = replace_with
        return
      except ValueError:
        pass
  raise errors.InvalidAstError('Node %r is not a child of %r' % (node, parent))


def has_docstring(node):
  return (hasattr(node, 'body') and node.body and
          isinstance(node.body[0], ast.Expr) and
          isinstance(node.body[0].value, ast.Str))
