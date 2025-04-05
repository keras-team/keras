# coding=utf-8
"""Annotate python syntax trees with formatting from the source file."""
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

import abc
import ast
import contextlib
import functools
import itertools
import six
from six.moves import zip
import sys

from pasta.base import ast_constants
from pasta.base import ast_utils
from pasta.base import formatting as fmt
from pasta.base import token_generator


# ==============================================================================
# == Helper functions for decorating nodes with prefix + suffix               ==
# ==============================================================================

def _gen_wrapper(f, scope=True, prefix=True, suffix=True, max_suffix_lines=None,
                 semicolon=False, comment=False, statement=False):
  @contextlib.wraps(f)
  def wrapped(self, node, *args, **kwargs):
    with (self.scope(node, trailing_comma=False) if scope else _noop_context()):
      if prefix:
        self.prefix(node, default=self._indent if statement else '')
      f(self, node, *args, **kwargs)
      if suffix:
        self.suffix(node, max_lines=max_suffix_lines, semicolon=semicolon,
                    comment=comment, default='\n' if statement else '')
  return wrapped


@contextlib.contextmanager
def _noop_context():
  yield


def expression(f):
  """Decorates a function where the node is an expression."""
  return _gen_wrapper(f, max_suffix_lines=0)


def fstring_expression(f):
  """Decorates a function where the node is a FormattedValue in an fstring."""
  return _gen_wrapper(f, scope=False)


def space_around(f):
  """Decorates a function where the node has whitespace prefix and suffix."""
  return _gen_wrapper(f, scope=False)


def space_left(f):
  """Decorates a function where the node has whitespace prefix."""
  return _gen_wrapper(f, scope=False, suffix=False)


def statement(f):
  """Decorates a function where the node is a statement."""
  return _gen_wrapper(f, scope=False, max_suffix_lines=1, semicolon=True,
                      comment=True, statement=True)


def module(f):
  """Special decorator for the module node."""
  return _gen_wrapper(f, scope=False, comment=True)


def block_statement(f):
  """Decorates a function where the node is a statement with children."""
  @contextlib.wraps(f)
  def wrapped(self, node, *args, **kwargs):
    self.prefix(node, default=self._indent)
    f(self, node, *args, **kwargs)
    if hasattr(self, 'block_suffix'):
      last_child = ast_utils.get_last_child(node)
      # Workaround for ast.Module which does not have a lineno
      if last_child and last_child.lineno != getattr(node, 'lineno', 0):
        indent = (fmt.get(last_child, 'prefix') or '\n').splitlines()[-1]
        self.block_suffix(node, indent)
    else:
      self.suffix(node, comment=True)
  return wrapped


# ==============================================================================
# == NodeVisitors for annotating an AST                                       ==
# ==============================================================================

class BaseVisitor(ast.NodeVisitor):
  """Walks a syntax tree in the order it appears in code.

  This class has a dual-purpose. It is implemented (in this file) for annotating
  an AST with formatting information needed to reconstruct the source code, but
  it also is implemented in pasta.base.codegen to reconstruct the source code.

  Each visit method in this class specifies the order in which both child nodes
  and syntax tokens appear, plus where to account for whitespace, commas,
  parentheses, etc.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self._stack = []
    self._indent = ''
    self._indent_diff = ''
    self._default_indent_diff = '  '

  def visit(self, node):
    self._stack.append(node)
    super(BaseVisitor, self).visit(node)
    assert node is self._stack.pop()

  def prefix(self, node, default=''):
    """Account for some amount of whitespace as the prefix to a node."""
    self.attr(node, 'prefix', [lambda: self.ws(comment=True)], default=default)

  def suffix(self, node, max_lines=None, semicolon=False, comment=False,
             default=''):
    """Account for some amount of whitespace as the suffix to a node."""
    def _ws():
      return self.ws(max_lines=max_lines, semicolon=semicolon, comment=comment)
    self.attr(node, 'suffix', [_ws], default=default)

  def indented(self, node, children_attr):
    children = getattr(node, children_attr)
    prev_indent = self._indent
    prev_indent_diff = self._indent_diff
    new_diff = fmt.get(children[0], 'indent_diff')
    if new_diff is None:
      new_diff = self._default_indent_diff
    self._indent_diff = new_diff
    self._indent = prev_indent + self._indent_diff
    for child in children:
      yield child
    self.attr(node, 'block_suffix_%s' % children_attr, [])
    self._indent = prev_indent
    self._indent_diff = prev_indent_diff

  def set_default_indent_diff(self, indent):
    self._default_indent_diff = indent

  @contextlib.contextmanager
  def scope(self, node, attr=None, trailing_comma=False, default_parens=False):
    """Context manager to handle a parenthesized scope.

    Arguments:
      node: (ast.AST) Node to store the scope prefix and suffix on.
      attr: (string, optional) Attribute of the node contained in the scope, if
        any. For example, as `None`, the scope would wrap the entire node, but
        as 'bases', the scope might wrap only the bases of a class.
      trailing_comma: (boolean) If True, allow a trailing comma at the end.
      default_parens: (boolean) If True and no formatting information is
        present, the scope would be assumed to be parenthesized.
    """
    if attr:
      self.attr(node, attr + '_prefix', [],
                default='(' if default_parens else '')
    yield
    if attr:
      self.attr(node, attr + '_suffix', [],
                default=')' if default_parens else '')

  def token(self, token_val):
    """Account for a specific token."""

  def attr(self, node, attr_name, attr_vals, deps=None, default=None):
    """Handles an attribute on the given node."""

  def ws(self, max_lines=None, semicolon=False, comment=True):
    """Account for some amount of whitespace.

    Arguments:
      max_lines: (int) Maximum number of newlines to consider.
      semicolon: (boolean) If True, parse up to the next semicolon (if present).
      comment: (boolean) If True, look for a trailing comment even when not in
        a parenthesized scope.
    """
    return ''

  def dots(self, num_dots):
    """Account for a number of dots."""
    return '.' * num_dots

  def ws_oneline(self):
    """Account for up to one line of whitespace."""
    return self.ws(max_lines=1)

  def optional_token(self, node, attr_name, token_val, default=False):
    """Account for a suffix that may or may not occur."""

  def one_of_symbols(self, *symbols):
    """Account for one of the given symbols."""
    return symbols[0]

  # ============================================================================
  # == BLOCK STATEMENTS: Statements that contain a list of statements         ==
  # ============================================================================

  # Keeps the entire suffix, so @block_statement is not useful here.
  @module
  def visit_Module(self, node):
    self.generic_visit(node)

  @block_statement
  def visit_If(self, node):
    tok = 'elif' if fmt.get(node, 'is_elif') else 'if'
    self.attr(node, 'open_if', [tok, self.ws], default=tok + ' ')
    self.visit(node.test)
    self.attr(node, 'open_block', [self.ws, ':', self.ws_oneline],
              default=':\n')

    for stmt in self.indented(node, 'body'):
      self.visit(stmt)

    if node.orelse:
      if (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) and
          self.check_is_elif(node.orelse[0])):
        fmt.set(node.orelse[0], 'is_elif', True)
        self.visit(node.orelse[0])
      else:
        self.attr(node, 'elseprefix', [self.ws])
        self.token('else')
        self.attr(node, 'open_else', [self.ws, ':', self.ws_oneline],
                  default=':\n')
        for stmt in self.indented(node, 'orelse'):
          self.visit(stmt)

  @abc.abstractmethod
  def check_is_elif(self, node):
    """Return True if the node continues a previous `if` statement as `elif`.

    In python 2.x, `elif` statments get parsed as If nodes. E.g, the following
    two syntax forms are indistinguishable in the ast in python 2.

    if a:
      do_something()
    elif b:
      do_something_else()

    if a:
      do_something()
    else:
      if b:
        do_something_else()

    This method should return True for the 'if b' node if it has the first form.
    """

  @block_statement
  def visit_While(self, node):
    self.attr(node, 'while_keyword', ['while', self.ws], default='while ')
    self.visit(node.test)
    self.attr(node, 'open_block', [self.ws, ':', self.ws_oneline],
              default=':\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)

    if node.orelse:
      self.attr(node, 'else', [self.ws, 'else', self.ws, ':', self.ws_oneline],
                default=self._indent + 'else:\n')
      for stmt in self.indented(node, 'orelse'):
        self.visit(stmt)

  @block_statement
  def visit_For(self, node):
    if hasattr(ast, 'AsyncFor') and isinstance(node, ast.AsyncFor):
      self.attr(node, 'for_keyword', ['async', self.ws, 'for', self.ws],
                default='async for ')
    else:
      self.attr(node, 'for_keyword', ['for', self.ws], default='for ')
    self.visit(node.target)
    self.attr(node, 'for_in', [self.ws, 'in', self.ws], default=' in ')
    self.visit(node.iter)
    self.attr(node, 'open_block', [self.ws, ':', self.ws_oneline],
              default=':\n')
    for stmt in self.indented(node, 'body'):
        self.visit(stmt)

    if node.orelse:
      self.attr(node, 'else', [self.ws, 'else', self.ws, ':', self.ws_oneline],
                default=self._indent + 'else:\n')

      for stmt in self.indented(node, 'orelse'):
        self.visit(stmt)

  def visit_AsyncFor(self, node):
    return self.visit_For(node)

  @block_statement
  def visit_With(self, node):
    if hasattr(node, 'items'):
      return self.visit_With_3(node)
    if not getattr(node, 'is_continued', False):
      self.attr(node, 'with', ['with', self.ws], default='with ')
    self.visit(node.context_expr)
    if node.optional_vars:
      self.attr(node, 'with_as', [self.ws, 'as', self.ws], default=' as ')
      self.visit(node.optional_vars)

    if len(node.body) == 1 and self.check_is_continued_with(node.body[0]):
      node.body[0].is_continued = True
      self.attr(node, 'with_comma', [self.ws, ',', self.ws], default=', ')
    else:
      self.attr(node, 'open_block', [self.ws, ':', self.ws_oneline],
                default=':\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)

  def visit_AsyncWith(self, node):
    return self.visit_With(node)

  @abc.abstractmethod
  def check_is_continued_try(self, node):
    pass

  @abc.abstractmethod
  def check_is_continued_with(self, node):
    """Return True if the node continues a previous `with` statement.

    In python 2.x, `with` statments with many context expressions get parsed as
    a tree of With nodes. E.g, the following two syntax forms are
    indistinguishable in the ast in python 2.

    with a, b, c:
      do_something()

    with a:
      with b:
        with c:
          do_something()

    This method should return True for the `with b` and `with c` nodes.
    """

  def visit_With_3(self, node):
    if hasattr(ast, 'AsyncWith') and isinstance(node, ast.AsyncWith):
      self.attr(node, 'with', ['async', self.ws, 'with', self.ws],
                default='async with ')
    else:
      self.attr(node, 'with', ['with', self.ws], default='with ')

    for i, withitem in enumerate(node.items):
      self.visit(withitem)
      if i != len(node.items) - 1:
        self.token(',')

    self.attr(node, 'with_body_open', [':', self.ws_oneline], default=':\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)

  @space_around
  def visit_withitem(self, node):
    self.visit(node.context_expr)
    if node.optional_vars:
      self.attr(node, 'as', [self.ws, 'as', self.ws], default=' as ')
      self.visit(node.optional_vars)

  @block_statement
  def visit_ClassDef(self, node):
    for i, decorator in enumerate(node.decorator_list):
      self.attr(node, 'decorator_prefix_%d' % i, [self.ws, '@'], default='@')
      self.visit(decorator)
      self.attr(node, 'decorator_suffix_%d' % i, [self.ws],
                default='\n' + self._indent)
    self.attr(node, 'class_def', ['class', self.ws, node.name, self.ws],
              default='class %s' % node.name, deps=('name',))
    class_args = getattr(node, 'bases', []) + getattr(node, 'keywords', [])
    with self.scope(node, 'bases', trailing_comma=bool(class_args),
                    default_parens=True):
      for i, base in enumerate(node.bases):
        self.visit(base)
        self.attr(node, 'base_suffix_%d' % i, [self.ws])
        if base != class_args[-1]:
          self.attr(node, 'base_sep_%d' % i, [',', self.ws], default=', ')
      if hasattr(node, 'keywords'):
        for i, keyword in enumerate(node.keywords):
          self.visit(keyword)
          self.attr(node, 'keyword_suffix_%d' % i, [self.ws])
          if keyword != node.keywords[-1]:
            self.attr(node, 'keyword_sep_%d' % i, [',', self.ws], default=', ')
    self.attr(node, 'open_block', [self.ws, ':', self.ws_oneline],
              default=':\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)

  @block_statement
  def visit_FunctionDef(self, node):
    for i, decorator in enumerate(node.decorator_list):
      self.attr(node, 'decorator_symbol_%d' % i, [self.ws, '@', self.ws],
                default='@')
      self.visit(decorator)
      self.attr(node, 'decorator_suffix_%d' % i, [self.ws_oneline],
                default='\n' + self._indent)
    if (hasattr(ast, 'AsyncFunctionDef') and
        isinstance(node, ast.AsyncFunctionDef)):
      self.attr(node, 'function_def',
                [self.ws, 'async', self.ws, 'def', self.ws, node.name, self.ws],
                deps=('name',), default='async def %s' % node.name)
    else:
      self.attr(node, 'function_def',
                [self.ws, 'def', self.ws, node.name, self.ws],
                deps=('name',), default='def %s' % node.name)
    # In Python 3, there can be extra args in kwonlyargs
    kwonlyargs = getattr(node.args, 'kwonlyargs', [])
    args_count = sum((len(node.args.args + kwonlyargs),
                      1 if node.args.vararg else 0,
                      1 if node.args.kwarg else 0))
    with self.scope(node, 'args', trailing_comma=args_count > 0,
                    default_parens=True):
      self.visit(node.args)

    if getattr(node, 'returns', None):
      self.attr(node, 'returns_prefix', [self.ws, '->', self.ws],
                deps=('returns',), default=' -> ')
      self.visit(node.returns)

    self.attr(node, 'open_block', [self.ws, ':', self.ws_oneline],
              default=':\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)

  def visit_AsyncFunctionDef(self, node):
    return self.visit_FunctionDef(node)

  @block_statement
  def visit_TryFinally(self, node):
    # Try with except and finally is a TryFinally with the first statement as a
    # TryExcept in Python2
    self.attr(node, 'open_try', ['try', self.ws, ':', self.ws_oneline],
              default='try:\n')
    # TODO(soupytwist): Find a cleaner solution for differentiating this.
    if len(node.body) == 1 and self.check_is_continued_try(node.body[0]):
      node.body[0].is_continued = True
      self.visit(node.body[0])
    else:
      for stmt in self.indented(node, 'body'):
        self.visit(stmt)
    self.attr(node, 'open_finally',
              [self.ws, 'finally', self.ws, ':', self.ws_oneline],
              default='finally:\n')
    for stmt in self.indented(node, 'finalbody'):
      self.visit(stmt)

  @block_statement
  def visit_TryExcept(self, node):
    if not getattr(node, 'is_continued', False):
      self.attr(node, 'open_try', ['try', self.ws, ':', self.ws_oneline],
                default='try:\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)
    for handler in node.handlers:
      self.visit(handler)
    if node.orelse:
      self.attr(node, 'open_else',
                [self.ws, 'else', self.ws, ':', self.ws_oneline],
                default='else:\n')
      for stmt in self.indented(node, 'orelse'):
        self.visit(stmt)

  @block_statement
  def visit_Try(self, node):
    # Python 3
    self.attr(node, 'open_try', [self.ws, 'try', self.ws, ':', self.ws_oneline],
              default='try:\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)
    for handler in node.handlers:
      self.visit(handler)
    if node.orelse:
      self.attr(node, 'open_else',
                [self.ws, 'else', self.ws, ':', self.ws_oneline],
                default='else:\n')
      for stmt in self.indented(node, 'orelse'):
        self.visit(stmt)
    if node.finalbody:
      self.attr(node, 'open_finally',
                [self.ws, 'finally', self.ws, ':', self.ws_oneline],
                default='finally:\n')
      for stmt in self.indented(node, 'finalbody'):
        self.visit(stmt)

  @block_statement
  def visit_ExceptHandler(self, node):
    self.token('except')
    if node.type:
      self.visit(node.type)
    if node.type and node.name:
      self.attr(node, 'as', [self.ws, self.one_of_symbols("as", ","), self.ws],
                default=' as ')
    if node.name:
      if isinstance(node.name, ast.AST):
        self.visit(node.name)
      else:
        self.token(node.name)
    self.attr(node, 'open_block', [self.ws, ':', self.ws_oneline],
              default=':\n')
    for stmt in self.indented(node, 'body'):
      self.visit(stmt)

  @statement
  def visit_Raise(self, node):
    if hasattr(node, 'cause'):
      return self.visit_Raise_3(node)

    self.token('raise')
    if node.type:
      self.attr(node, 'type_prefix', [self.ws], default=' ')
      self.visit(node.type)
    if node.inst:
      self.attr(node, 'inst_prefix', [self.ws, ',', self.ws], default=', ')
      self.visit(node.inst)
    if node.tback:
      self.attr(node, 'tback_prefix', [self.ws, ',', self.ws], default=', ')
      self.visit(node.tback)

  def visit_Raise_3(self, node):
    if node.exc:
      self.attr(node, 'open_raise', ['raise', self.ws], default='raise ')
      self.visit(node.exc)
      if node.cause:
        self.attr(node, 'cause_prefix', [self.ws, 'from', self.ws],
                  default=' from ')
        self.visit(node.cause)
    else:
      self.token('raise')

  # ============================================================================
  # == STATEMENTS: Instructions without a return value                        ==
  # ============================================================================

  @statement
  def visit_Assert(self, node):
    self.attr(node, 'assert_open', ['assert', self.ws], default='assert ')
    self.visit(node.test)
    if node.msg:
      self.attr(node, 'msg_prefix', [',', self.ws], default=', ')
      self.visit(node.msg)

  @statement
  def visit_Assign(self, node):
    for i, target in enumerate(node.targets):
      self.visit(target)
      self.attr(node, 'equal_%d' % i, [self.ws, '=', self.ws], default=' = ')
    self.visit(node.value)

  @statement
  def visit_AugAssign(self, node):
    self.visit(node.target)
    op_token = '%s=' % ast_constants.NODE_TYPE_TO_TOKENS[type(node.op)][0]
    self.attr(node, 'operator', [self.ws, op_token, self.ws],
              default=' %s ' % op_token)
    self.visit(node.value)

  @statement
  def visit_AnnAssign(self, node):
    # TODO: Check default formatting for different values of "simple"
    self.visit(node.target)
    self.attr(node, 'colon', [self.ws, ':', self.ws], default=': ')
    self.visit(node.annotation)
    if node.value:
      self.attr(node, 'equal', [self.ws, '=', self.ws], default=' = ')
      self.visit(node.value)

  @expression
  def visit_Await(self, node):
    self.attr(node, 'await', ['await', self.ws], default='await ')
    self.visit(node.value)

  @statement
  def visit_Break(self, node):
    self.token('break')

  @statement
  def visit_Continue(self, node):
    self.token('continue')

  @statement
  def visit_Delete(self, node):
    self.attr(node, 'del', ['del', self.ws], default='del ')
    for i, target in enumerate(node.targets):
      self.visit(target)
      if target is not node.targets[-1]:
        self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')

  @statement
  def visit_Exec(self, node):
    # If no formatting info is present, will use parenthesized style
    self.attr(node, 'exec', ['exec', self.ws], default='exec')
    with self.scope(node, 'body', trailing_comma=False, default_parens=True):
      self.visit(node.body)
      if node.globals:
        self.attr(node, 'in_globals',
                  [self.ws, self.one_of_symbols('in', ','), self.ws],
                  default=', ')
        self.visit(node.globals)
        if node.locals:
          self.attr(node, 'in_locals', [self.ws, ',', self.ws], default=', ')
          self.visit(node.locals)

  @statement
  def visit_Expr(self, node):
    self.visit(node.value)

  @statement
  def visit_Global(self, node):
    self.token('global')
    identifiers = []
    for ident in node.names:
      if ident != node.names[0]:
        identifiers.extend([self.ws, ','])
      identifiers.extend([self.ws, ident])
    self.attr(node, 'names', identifiers)

  @statement
  def visit_Import(self, node):
    self.token('import')
    for i, alias in enumerate(node.names):
      self.attr(node, 'alias_prefix_%d' % i, [self.ws], default=' ')
      self.visit(alias)
      if alias != node.names[-1]:
        self.attr(node, 'alias_sep_%d' % i, [self.ws, ','], default=',')

  @statement
  def visit_ImportFrom(self, node):
    self.token('from')
    self.attr(node, 'module_prefix', [self.ws], default=' ')

    module_pattern = []
    if node.level > 0:
      module_pattern.extend([self.dots(node.level), self.ws])
    if node.module:
      parts = node.module.split('.')
      for part in parts[:-1]:
        module_pattern += [self.ws, part, self.ws, '.']
      module_pattern += [self.ws, parts[-1]]

    self.attr(node, 'module', module_pattern,
              deps=('level', 'module'),
              default='.' * node.level + (node.module or ''))
    self.attr(node, 'module_suffix', [self.ws], default=' ')

    self.token('import')
    with self.scope(node, 'names', trailing_comma=True):
      for i, alias in enumerate(node.names):
        self.attr(node, 'alias_prefix_%d' % i, [self.ws], default=' ')
        self.visit(alias)
        if alias is not node.names[-1]:
          self.attr(node, 'alias_sep_%d' % i, [self.ws, ','], default=',')

  @expression
  def visit_NamedExpr(self, node):
    self.visit(target)
    self.attr(node, 'equal' % i, [self.ws, ':=', self.ws], default=' := ')
    self.visit(node.value)

  @statement
  def visit_Nonlocal(self, node):
    self.token('nonlocal')
    identifiers = []
    for ident in node.names:
      if ident != node.names[0]:
        identifiers.extend([self.ws, ','])
      identifiers.extend([self.ws, ident])
    self.attr(node, 'names', identifiers)

  @statement
  def visit_Pass(self, node):
    self.token('pass')

  @statement
  def visit_Print(self, node):
    self.attr(node, 'print_open', ['print', self.ws], default='print ')
    if node.dest:
      self.attr(node, 'redirection', ['>>', self.ws], default='>>')
      self.visit(node.dest)
      if node.values:
        self.attr(node, 'values_prefix', [self.ws, ',', self.ws], default=', ')
      elif not node.nl:
        self.attr(node, 'trailing_comma', [self.ws, ','], default=',')

    for i, value in enumerate(node.values):
      self.visit(value)
      if value is not node.values[-1]:
        self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
      elif not node.nl:
        self.attr(node, 'trailing_comma', [self.ws, ','], default=',')

  @statement
  def visit_Return(self, node):
    self.token('return')
    if node.value:
      self.attr(node, 'return_value_prefix', [self.ws], default=' ')
      self.visit(node.value)

  @expression
  def visit_Yield(self, node):
    self.token('yield')
    if node.value:
      self.attr(node, 'yield_value_prefix', [self.ws], default=' ')
      self.visit(node.value)

  @expression
  def visit_YieldFrom(self, node):
    self.attr(node, 'yield_from', ['yield', self.ws, 'from', self.ws],
              default='yield from ')
    self.visit(node.value)

  # ============================================================================
  # == EXPRESSIONS: Anything that evaluates and can be in parens              ==
  # ============================================================================

  @expression
  def visit_Attribute(self, node):
    self.visit(node.value)
    self.attr(node, 'dot', [self.ws, '.', self.ws], default='.')
    self.token(node.attr)

  @expression
  def visit_BinOp(self, node):
    op_symbol = ast_constants.NODE_TYPE_TO_TOKENS[type(node.op)][0]
    self.visit(node.left)
    self.attr(node, 'op', [self.ws, op_symbol, self.ws],
              default=' %s ' % op_symbol, deps=('op',))
    self.visit(node.right)

  @expression
  def visit_BoolOp(self, node):
    op_symbol = ast_constants.NODE_TYPE_TO_TOKENS[type(node.op)][0]
    for i, value in enumerate(node.values):
      self.visit(value)
      if value is not node.values[-1]:
        self.attr(node, 'op_%d' % i, [self.ws, op_symbol, self.ws],
                  default=' %s ' % op_symbol, deps=('op',))

  @expression
  def visit_Call(self, node):
    self.visit(node.func)

    with self.scope(node, 'arguments', default_parens=True):
      # python <3.5: starargs and kwargs are in separate fields
      # python 3.5+: starargs args included as a Starred nodes in the arguments
      #              and kwargs are included as keywords with no argument name.
      if sys.version_info[:2] >= (3, 5):
        any_args = self.visit_Call_arguments35(node)
      else:
        any_args = self.visit_Call_arguments(node)
      if any_args:
        self.optional_token(node, 'trailing_comma', ',')

  def visit_Call_arguments(self, node):
    def arg_location(tup):
      arg = tup[1]
      if isinstance(arg, ast.keyword):
        arg = arg.value
      return (getattr(arg, "lineno", 0), getattr(arg, "col_offset", 0))

    if node.starargs:
      sorted_keywords = sorted(
          [(None, kw) for kw in node.keywords] + [('*', node.starargs)],
          key=arg_location)
    else:
      sorted_keywords = [(None, kw) for kw in node.keywords]
    all_args = [(None, n) for n in node.args] + sorted_keywords
    if node.kwargs:
      all_args.append(('**', node.kwargs))

    for i, (prefix, arg) in enumerate(all_args):
      if prefix is not None:
        self.attr(node, '%s_prefix' % prefix, [self.ws, prefix], default=prefix)
      self.visit(arg)
      if arg is not all_args[-1][1]:
        self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
    return bool(all_args)

  def visit_Call_arguments35(self, node):
    def arg_compare(a1, a2):
      """Old-style comparator for sorting args."""
      def is_arg(a):
        return not isinstance(a, (ast.keyword, ast.Starred))

      # No kwarg can come before a regular arg (but Starred can be wherever)
      if is_arg(a1) and isinstance(a2, ast.keyword):
        return -1
      elif is_arg(a2) and isinstance(a1, ast.keyword):
        return 1

      # If no lineno or col_offset on one of the args, they compare as equal
      # (since sorting is stable, this should leave them mostly where they
      # were in the initial list).
      def get_pos(a):
        if isinstance(a, ast.keyword):
          a = a.value
        return (getattr(a, 'lineno', None), getattr(a, 'col_offset', None))

      pos1 = get_pos(a1)
      pos2 = get_pos(a2)

      if None in pos1 or None in pos2:
        return 0

      # If both have lineno/col_offset set, use that to sort them
      return -1 if pos1 < pos2 else 0 if pos1 == pos2 else 1

    # Note that this always sorts keywords identically to just sorting by
    # lineno/col_offset, except in cases where that ordering would have been
    # a syntax error (named arg before unnamed arg).
    all_args = sorted(node.args + node.keywords,
                      key=functools.cmp_to_key(arg_compare))

    for i, arg in enumerate(all_args):
      self.visit(arg)
      if arg is not all_args[-1]:
        self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
    return bool(all_args)

  def visit_Starred(self, node):
    self.attr(node, 'star', ['*', self.ws], default='*')
    self.visit(node.value)

  @expression
  def visit_Compare(self, node):
    self.visit(node.left)
    for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
      self.attr(node, 'op_prefix_%d' % i, [self.ws], default=' ')
      self.visit(op)
      self.attr(node, 'op_suffix_%d' % i, [self.ws], default=' ')
      self.visit(comparator)

  @expression
  def visit_Dict(self, node):
    self.token('{')
    for i, key, value in zip(range(len(node.keys)), node.keys, node.values):
      if key is None:
        # Handle Python 3.5+ dict unpacking syntax (PEP-448)
        self.attr(node, 'starstar_%d' % i, [self.ws, '**'], default='**')
      else:
        self.visit(key)
        self.attr(node, 'key_val_sep_%d' % i, [self.ws, ':', self.ws],
                  default=': ')
      self.visit(value)
      if value is not node.values[-1]:
        self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
    self.optional_token(node, 'extracomma', ',', allow_whitespace_prefix=True)
    self.attr(node, 'close_prefix', [self.ws, '}'], default='}')

  @expression
  def visit_DictComp(self, node):
    self.attr(node, 'open_dict', ['{', self.ws], default='{')
    self.visit(node.key)
    self.attr(node, 'key_val_sep', [self.ws, ':', self.ws], default=': ')
    self.visit(node.value)
    for comp in node.generators:
      self.visit(comp)
    self.attr(node, 'close_dict', [self.ws, '}'], default='}')

  @expression
  def visit_GeneratorExp(self, node):
    self._comp_exp(node)

  @expression
  def visit_IfExp(self, node):
    self.visit(node.body)
    self.attr(node, 'if', [self.ws, 'if', self.ws], default=' if ')
    self.visit(node.test)
    self.attr(node, 'else', [self.ws, 'else', self.ws], default=' else ')
    self.visit(node.orelse)

  @expression
  def visit_Lambda(self, node):
    self.attr(node, 'lambda_def', ['lambda', self.ws], default='lambda ')
    self.visit(node.args)
    self.attr(node, 'open_lambda', [self.ws, ':', self.ws], default=': ')
    self.visit(node.body)

  @expression
  def visit_List(self, node):
    self.attr(node, 'list_open', ['[', self.ws], default='[')

    for i, elt in enumerate(node.elts):
      self.visit(elt)
      if elt is not node.elts[-1]:
        self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
    if node.elts:
      self.optional_token(node, 'extracomma', ',', allow_whitespace_prefix=True)

    self.attr(node, 'list_close', [self.ws, ']'], default=']')

  @expression
  def visit_ListComp(self, node):
    self._comp_exp(node, open_brace='[', close_brace=']')

  def _comp_exp(self, node, open_brace=None, close_brace=None):
    if open_brace:
      self.attr(node, 'compexp_open', [open_brace, self.ws], default=open_brace)
    self.visit(node.elt)
    for i, comp in enumerate(node.generators):
      self.visit(comp)
    if close_brace:
      self.attr(node, 'compexp_close', [self.ws, close_brace],
                default=close_brace)

  @expression
  def visit_Name(self, node):
    self.token(node.id)

  @expression
  def visit_NameConstant(self, node):
    self.token(str(node.value))

  @expression
  def visit_Repr(self, node):
    self.attr(node, 'repr_open', ['`', self.ws], default='`')
    self.visit(node.value)
    self.attr(node, 'repr_close', [self.ws, '`'], default='`')

  @expression
  def visit_Set(self, node):
    self.attr(node, 'set_open', ['{', self.ws], default='{')

    for i, elt in enumerate(node.elts):
      self.visit(elt)
      if elt is not node.elts[-1]:
        self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
      else:
        self.optional_token(node, 'extracomma', ',',
                            allow_whitespace_prefix=True)

    self.attr(node, 'set_close', [self.ws, '}'], default='}')

  @expression
  def visit_SetComp(self, node):
    self._comp_exp(node, open_brace='{', close_brace='}')

  @expression
  def visit_Subscript(self, node):
    self.visit(node.value)
    self.attr(node, 'slice_open', [self.ws, '[', self.ws], default='[')
    self.visit(node.slice)
    self.attr(node, 'slice_close', [self.ws, ']'], default=']')

  @expression
  def visit_Tuple(self, node):
    with self.scope(node, 'elts', default_parens=True):
      for i, elt in enumerate(node.elts):
        self.visit(elt)
        if elt is not node.elts[-1]:
          self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws],
                    default=', ')
        else:
          self.optional_token(node, 'extracomma', ',',
                              allow_whitespace_prefix=True,
                              default=len(node.elts) == 1)

  @expression
  def visit_UnaryOp(self, node):
    op_symbol = ast_constants.NODE_TYPE_TO_TOKENS[type(node.op)][0]
    self.attr(node, 'op', [op_symbol, self.ws], default=op_symbol, deps=('op',))
    self.visit(node.operand)

  # ============================================================================
  # == OPERATORS AND TOKENS: Anything that's just whitespace and tokens       ==
  # ============================================================================

  @space_around
  def visit_Ellipsis(self, node):
    self.token('...')

  def visit_Add(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Sub(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Mult(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Div(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Mod(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Pow(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_LShift(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_RShift(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_BitAnd(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_BitOr(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_BitXor(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_FloorDiv(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Invert(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Not(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_UAdd(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_USub(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Eq(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_NotEq(self, node):
    self.attr(node, 'operator', [self.one_of_symbols('!=', '<>')])

  def visit_Lt(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_LtE(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Gt(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_GtE(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_Is(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_IsNot(self, node):
    self.attr(node, 'content', ['is', self.ws, 'not'], default='is not')

  def visit_In(self, node):
    self.token(ast_constants.NODE_TYPE_TO_TOKENS[type(node)][0])

  def visit_NotIn(self, node):
    self.attr(node, 'content', ['not', self.ws, 'in'], default='not in')

  # ============================================================================
  # == MISC NODES: Nodes which are neither statements nor expressions         ==
  # ============================================================================

  def visit_alias(self, node):
    name_pattern = []
    parts = node.name.split('.')
    for part in parts[:-1]:
      name_pattern += [self.ws, part, self.ws, '.']
    name_pattern += [self.ws, parts[-1]]
    self.attr(node, 'name', name_pattern,
              deps=('name',),
              default=node.name)
    if node.asname is not None:
      self.attr(node, 'asname', [self.ws, 'as', self.ws], default=' as ')
      self.token(node.asname)

  @space_around
  def visit_arg(self, node):
    self.token(node.arg)
    if node.annotation is not None:
      self.attr(node, 'annotation_prefix', [self.ws, ':', self.ws],
                default=': ')
      self.visit(node.annotation)

  @space_around
  def visit_arguments(self, node):
    # In Python 3, args appearing after *args must be kwargs
    kwonlyargs = getattr(node, 'kwonlyargs', [])
    kw_defaults = getattr(node, 'kw_defaults', [])
    assert len(kwonlyargs) == len(kw_defaults)

    total_args = sum((len(node.args + kwonlyargs),
                      len(getattr(node, 'posonlyargs', [])),
                      1 if node.vararg else 0,
                      1 if node.kwarg else 0))
    arg_i = 0

    pos_args = getattr(node, 'posonlyargs', []) + node.args
    positional = pos_args[:-len(node.defaults)] if node.defaults else pos_args
    keyword = node.args[-len(node.defaults):] if node.defaults else node.args

    for arg in positional:
      self.visit(arg)
      arg_i += 1
      if arg_i < total_args:
        self.attr(node, 'comma_%d' % arg_i, [self.ws, ',', self.ws],
                  default=', ')
      if arg_i == len(getattr(node, 'posonlyargs', [])):
        self.attr(node, 'posonly_sep', [self.ws, '/', self.ws, ',', self.ws],
                  default='/, ')

    for i, (arg, default) in enumerate(zip(keyword, node.defaults)):
      self.visit(arg)
      self.attr(node, 'default_%d' % i, [self.ws, '=', self.ws],
                default='=')
      self.visit(default)
      arg_i += 1
      if arg_i < total_args:
        self.attr(node, 'comma_%d' % arg_i, [self.ws, ',', self.ws],
                  default=', ')

    if node.vararg:
      self.attr(node, 'vararg_prefix', [self.ws, '*', self.ws], default='*')
      if isinstance(node.vararg, ast.AST):
        self.visit(node.vararg)
      else:
        self.token(node.vararg)
        self.attr(node, 'vararg_suffix', [self.ws])
      arg_i += 1
      if arg_i < total_args:
        self.token(',')
    elif kwonlyargs:
      # If no vararg, but we have kwonlyargs, insert a naked *, which will
      # definitely not be the last arg.
      self.attr(node, 'kwonly_sep', [self.ws, '*', self.ws, ',', self.ws]);

    for i, (arg, default) in enumerate(zip(kwonlyargs, kw_defaults)):
      self.visit(arg)
      if default is not None:
        self.attr(node, 'kw_default_%d' % i, [self.ws, '=', self.ws],
                  default='=')
        self.visit(default)
      arg_i += 1
      if arg_i < total_args:
        self.attr(node, 'comma_%d' % arg_i, [self.ws, ',', self.ws],
                  default=', ')

    if node.kwarg:
      self.attr(node, 'kwarg_prefix', [self.ws, '**', self.ws], default='**')
      if isinstance(node.kwarg, ast.AST):
        self.visit(node.kwarg)
      else:
        self.token(node.kwarg)
        self.attr(node, 'kwarg_suffix', [self.ws])

  @space_around
  def visit_comprehension(self, node):
    if getattr(node, 'is_async', False):
      self.attr(node, 'for', [self.ws, 'async', self.ws, 'for', self.ws],
                default=' async for ')
    else:
      self.attr(node, 'for', [self.ws, 'for', self.ws], default=' for ')
    self.visit(node.target)
    self.attr(node, 'in', [self.ws, 'in', self.ws], default=' in ')
    self.visit(node.iter)
    for i, if_expr in enumerate(node.ifs):
      self.attr(node, 'if_%d' % i, [self.ws, 'if', self.ws], default=' if ')
      self.visit(if_expr)

  @space_around
  def visit_keyword(self, node):
    if node.arg is None:
      self.attr(node, 'stars', ['**', self.ws], default='**')
    else:
      self.token(node.arg)
      self.attr(node, 'eq', [self.ws, '='], default='=')
    self.visit(node.value)

  @space_left
  def visit_Index(self, node):
    self.visit(node.value)

  @space_left
  def visit_ExtSlice(self, node):
    for i, dim in enumerate(node.dims):
      self.visit(dim)
      if dim is not node.dims[-1]:
        self.attr(node, 'dim_sep_%d' % i, [self.ws, ',', self.ws], default=', ')
    self.optional_token(node, 'trailing_comma', ',', default=False)

  @space_left
  def visit_Slice(self, node):
    if node.lower:
      self.visit(node.lower)
    self.attr(node, 'lowerspace', [self.ws, ':', self.ws], default=':')
    if node.upper:
      self.visit(node.upper)

    self.attr(node, 'stepspace1', [self.ws])
    self.optional_token(node, 'step_colon', ':')
    self.attr(node, 'stepspace2', [self.ws])
    if node.step and self.check_slice_includes_step(node):
      self.optional_token(node, 'step_colon_2', ':', default=True)
      node.step.is_explicit_step = True
      self.visit(node.step)

  def check_slice_includes_step(self, node):
    """Helper function for Slice node to determine whether to visit its step."""
    # This is needed because of a bug in the 2.7 parser which treats
    # a[::] as Slice(lower=None, upper=None, step=Name(id='None'))
    # but also treats a[::None] exactly the same.
    if not node.step:
      return False
    if getattr(node.step, 'is_explicit_step', False):
      return True
    return not (isinstance(node.step, ast.Name) and node.step.id == 'None')

  @fstring_expression
  def visit_FormattedValue(self, node):
    self.visit(node.value)
    if node.conversion != -1:
      self.attr(node, 'conversion',
                [self.ws, '!', chr(node.conversion)], deps=('conversion',),
                default='!%c' % node.conversion)
    if node.format_spec:
      self.attr(node, 'format_spec_prefix', [self.ws, ':', self.ws],
                default=':')
      self.visit(node.format_spec)


class AnnotationError(Exception):
  """An exception for when we failed to annotate the tree."""


class AstAnnotator(BaseVisitor):

  def __init__(self, source):
    super(AstAnnotator, self).__init__()
    self.tokens = token_generator.TokenGenerator(source)

  def visit(self, node):
    try:
      fmt.set(node, 'indent', self._indent)
      fmt.set(node, 'indent_diff', self._indent_diff)
      super(AstAnnotator, self).visit(node)
    except (TypeError, ValueError, IndexError, KeyError) as e:
      raise AnnotationError(e)

  def indented(self, node, children_attr):
    """Generator which annotates child nodes with their indentation level."""
    children = getattr(node, children_attr)
    cur_loc = self.tokens._loc
    next_loc = self.tokens.peek_non_whitespace().start
    # Special case: if the children are on the same line, then there is no
    # indentation level to track.
    if cur_loc[0] == next_loc[0]:
      indent_diff = self._indent_diff
      self._indent_diff = None
      for child in children:
        yield child
      self._indent_diff = indent_diff
      return

    prev_indent = self._indent
    prev_indent_diff = self._indent_diff

    # Find the indent level of the first child
    indent_token = self.tokens.peek_conditional(
        lambda t: t.type == token_generator.TOKENS.INDENT)
    new_indent = indent_token.src
    new_diff = _get_indent_diff(prev_indent, new_indent)
    if not new_diff:
      new_diff = ' ' * 4  # Sensible default
      print('Indent detection failed (line %d); inner indentation level is not '
            'more than the outer indentation.' % cur_loc[0], file=sys.stderr)

    # Set the indent level to the child's indent and iterate over the children
    self._indent = new_indent
    self._indent_diff = new_diff
    for child in children:
      yield child
    # Store the suffix at this indentation level, which could be many lines
    fmt.set(node, 'block_suffix_%s' % children_attr,
            self.tokens.block_whitespace(self._indent))

    # Dedent back to the previous level
    self._indent = prev_indent
    self._indent_diff = prev_indent_diff

  @expression
  def visit_Num(self, node):
    """Annotate a Num node with the exact number format."""
    token_number_type = token_generator.TOKENS.NUMBER
    contentargs = [lambda: self.tokens.next_of_type(token_number_type).src]
    if self.tokens.peek().src == '-':
      contentargs.insert(0, '-')
    self.attr(node, 'content', contentargs, deps=('n',), default=str(node.n))

  @expression
  def visit_Str(self, node):
    """Annotate a Str node with the exact string format."""
    self.attr(node, 'content', [self.tokens.str], deps=('s',), default=node.s)

  @expression
  def visit_JoinedStr(self, node):
    """Annotate a JoinedStr node with the fstr formatting metadata."""
    fstr_iter = self.tokens.fstr()()
    res = ''
    values = (v for v in node.values if isinstance(v, ast.FormattedValue))
    while True:
      res_part, tg = next(fstr_iter)
      res += res_part
      if tg is None:
        break
      prev_tokens = self.tokens
      self.tokens = tg
      self.visit(next(values))
      self.tokens = prev_tokens

    self.attr(node, 'content', [lambda: res], default=res)

  @expression
  def visit_Bytes(self, node):
    """Annotate a Bytes node with the exact string format."""
    self.attr(node, 'content', [self.tokens.str], deps=('s',), default=node.s)
    
  @space_around
  def visit_Ellipsis(self, node):
    # Ellipsis is sometimes split into 3 tokens and other times a single token
    # Account for both forms when parsing the input.
    if self.tokens.peek().src == '...':
      self.token('...')
    else:
      for i in range(3):
        self.token('.')

  def check_is_elif(self, node):
    """Return True iff the If node is an `elif` in the source."""
    next_tok = self.tokens.next_name()
    return isinstance(node, ast.If) and next_tok.src == 'elif'

  def check_is_continued_try(self, node):
    """Return True iff the TryExcept node is a continued `try` in the source."""
    return (isinstance(node, ast.TryExcept) and
            self.tokens.peek_non_whitespace().src != 'try')

  def check_is_continued_with(self, node):
    """Return True iff the With node is a continued `with` in the source."""
    return isinstance(node, ast.With) and self.tokens.peek().src == ','

  def check_slice_includes_step(self, node):
    """Helper function for Slice node to determine whether to visit its step."""
    # This is needed because of a bug in the 2.7 parser which treats
    # a[::] as Slice(lower=None, upper=None, step=Name(id='None'))
    # but also treats a[::None] exactly the same.
    return self.tokens.peek_non_whitespace().src not in '],'

  def ws(self, max_lines=None, semicolon=False, comment=True):
    """Parse some whitespace from the source tokens and return it."""
    next_token = self.tokens.peek()
    if semicolon and next_token and next_token.src == ';':
      result = self.tokens.whitespace() + self.token(';')
      next_token = self.tokens.peek()
      if next_token.type in (token_generator.TOKENS.NL,
                             token_generator.TOKENS.NEWLINE):
        result += self.tokens.whitespace(max_lines=1)
      return result
    return self.tokens.whitespace(max_lines=max_lines, comment=comment)

  def dots(self, num_dots):
    """Parse a number of dots."""
    def _parse_dots():
      return self.tokens.dots(num_dots)
    return _parse_dots

  def block_suffix(self, node, indent_level):
    fmt.set(node, 'suffix', self.tokens.block_whitespace(indent_level))

  def token(self, token_val):
    """Parse a single token with exactly the given value."""
    token = self.tokens.next()
    if token.src != token_val:
      raise AnnotationError("Expected %r but found %r\nline %d: %s" % (
          token_val, token.src, token.start[0], token.line))

    # If the token opens or closes a parentheses scope, keep track of it
    if token.src in '({[':
      self.tokens.hint_open()
    elif token.src in ')}]':
      self.tokens.hint_closed()

    return token.src

  def optional_token(self, node, attr_name, token_val,
                     allow_whitespace_prefix=False, default=False):
    """Try to parse a token and attach it to the node."""
    del default
    fmt.append(node, attr_name, '')
    token = (self.tokens.peek_non_whitespace()
             if allow_whitespace_prefix else self.tokens.peek())
    if token and token.src == token_val:
      parsed = ''
      if allow_whitespace_prefix:
        parsed += self.ws()
      fmt.append(node, attr_name,
                           parsed + self.tokens.next().src + self.ws())

  def one_of_symbols(self, *symbols):
    """Account for one of the given symbols."""
    def _one_of_symbols():
      next_token = self.tokens.next()
      found = next((s for s in symbols if s == next_token.src), None)
      if found is None:
        raise AnnotationError(
            'Expected one of: %r, but found: %r' % (symbols, next_token.src))
      return found
    return _one_of_symbols

  def attr(self, node, attr_name, attr_vals, deps=None, default=None):
    """Parses some source and sets an attribute on the given node.

    Stores some arbitrary formatting information on the node. This takes a list
    attr_vals which tell what parts of the source to parse. The result of each
    function is concatenated onto the formatting data, and strings in this list
    are a shorthand to look for an exactly matching token.

    For example:
      self.attr(node, 'foo', ['(', self.ws, 'Hello, world!', self.ws, ')'],
                deps=('s',), default=node.s)

    is a rudimentary way to parse a parenthesized string. After running this,
    the matching source code for this node will be stored in its formatting
    dict under the key 'foo'. The result might be `(\n  'Hello, world!'\n)`.

    This also keeps track of the current value of each of the dependencies.
    In the above example, we would have looked for the string 'Hello, world!'
    because that's the value of node.s, however, when we print this back, we
    want to know if the value of node.s has changed since this time. If any of
    the dependent values has changed, the default would be used instead.

    Arguments:
      node: (ast.AST) An AST node to attach formatting information to.
      attr_name: (string) Name to store the formatting information under.
      attr_vals: (list of functions/strings) Each item is either a function
        that parses some source and return a string OR a string to match
        exactly (as a token).
      deps: (optional, set of strings) Attributes of the node which attr_vals
        depends on.
      default: (string) Unused here.
    """
    del default  # unused
    if deps:
      for dep in deps:
        fmt.set(node, dep + '__src', getattr(node, dep, None))
    attr_parts = []
    for attr_val in attr_vals:
      if isinstance(attr_val, six.string_types):
        attr_parts.append(self.token(attr_val))
      else:
        attr_parts.append(attr_val())
    fmt.set(node, attr_name, ''.join(attr_parts))

  def scope(self, node, attr=None, trailing_comma=False, default_parens=False):
    """Return a context manager to handle a parenthesized scope.

    Arguments:
      node: (ast.AST) Node to store the scope prefix and suffix on.
      attr: (string, optional) Attribute of the node contained in the scope, if
        any. For example, as `None`, the scope would wrap the entire node, but
        as 'bases', the scope might wrap only the bases of a class.
      trailing_comma: (boolean) If True, allow a trailing comma at the end.
      default_parens: (boolean) If True and no formatting information is
        present, the scope would be assumed to be parenthesized.
    """
    del default_parens
    return self.tokens.scope(node, attr=attr, trailing_comma=trailing_comma)

  def _optional_token(self, token_type, token_val):
    token = self.tokens.peek()
    if not token or token.type != token_type or token.src != token_val:
      return ''
    else:
      self.tokens.next()
      return token.src + self.ws()


def _get_indent_width(indent):
  width = 0
  for c in indent:
    if c == ' ':
      width += 1
    elif c == '\t':
      width += 8 - (width % 8)
  return width


def _ltrim_indent(indent, remove_width):
  width = 0
  for i, c in enumerate(indent):
    if width == remove_width:
      break
    if c == ' ':
      width += 1
    elif c == '\t':
      if width + 8 - (width % 8) <= remove_width:
        width += 8 - (width % 8)
      else:
        return ' ' * (width + 8 - remove_width) + indent[i + 1:]
  return indent[i:]


def _get_indent_diff(outer, inner):
  """Computes the whitespace added to an indented block.

  Finds the portion of an indent prefix that is added onto the outer indent. In
  most cases, the inner indent starts with the outer indent, but this is not
  necessarily true. For example, the outer block could be indented to four
  spaces and its body indented with one tab (effectively 8 spaces).

  Arguments:
    outer: (string) Indentation of the outer block.
    inner: (string) Indentation of the inner block.
  Returns:
    The string whitespace which is added to the indentation level when moving
    from outer to inner.
  """
  outer_w = _get_indent_width(outer)
  inner_w = _get_indent_width(inner)
  diff_w = inner_w - outer_w

  if diff_w <= 0:
    return None

  return _ltrim_indent(inner, inner_w - diff_w)
