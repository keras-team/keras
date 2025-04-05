# coding=utf-8
"""Perform static analysis on python syntax trees."""
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
import collections
import six

# TODO: Support relative imports

# Represents a reference to something external to the module.
# Fields:
#   name: (string) The full dotted name being referenced.
#   node: (ast.AST) The AST node where the reference is defined.
#   name_ref: (Name) The name object that refers to the imported name, if
#     applicable. This may not be the same id if the import is aliased.
ExternalReference = collections.namedtuple('ExternalReference',
                                           ('name', 'node', 'name_ref'))


class ScopeVisitor(ast.NodeVisitor):

  def __init__(self):
    super(ScopeVisitor, self).__init__()
    self._parent = None
    self.root_scope = self.scope = RootScope(None)

  def visit(self, node):
    if node is None:
      return
    if self.root_scope.node is None:
      self.root_scope.node = node
    self.root_scope.set_parent(node, self._parent)
    tmp = self._parent
    self._parent = node
    super(ScopeVisitor, self).visit(node)
    self._parent = tmp

  def visit_in_order(self, node, *attrs):
    for attr in attrs:
      val = getattr(node, attr, None)
      if val is None:
        continue
      if isinstance(val, list):
        for item in val:
          self.visit(item)
      elif isinstance(val, ast.AST):
        self.visit(val)

  def visit_Import(self, node):
    for alias in node.names:
      name_parts = alias.name.split('.')

      if not alias.asname:
        # If not aliased, define the top-level module of the import
        cur_name = self.scope.define_name(name_parts[0], alias)
        self.root_scope.add_external_reference(name_parts[0], alias,
                                               name_ref=cur_name)

        # Define names of sub-modules imported
        partial_name = name_parts[0]
        for part in name_parts[1:]:
          partial_name += '.' + part
          cur_name = cur_name.lookup_name(part)
          cur_name.define(alias)
          self.root_scope.add_external_reference(partial_name, alias,
                                                 name_ref=cur_name)

      else:
        # If the imported name is aliased, define that name only
        name = self.scope.define_name(alias.asname, alias)

        # Define names of sub-modules imported
        for i in range(1, len(name_parts)):
          self.root_scope.add_external_reference('.'.join(name_parts[:i]),
                                                 alias)
        self.root_scope.add_external_reference(alias.name, alias, name_ref=name)

    self.generic_visit(node)

  def visit_ImportFrom(self, node):
    if node.module:
      name_parts = node.module.split('.')
      for i in range(1, len(name_parts) + 1):
        self.root_scope.add_external_reference('.'.join(name_parts[:i]), node)
    for alias in node.names:
      name = self.scope.define_name(alias.asname or alias.name, alias)
      if node.module:
        self.root_scope.add_external_reference(
            '.'.join((node.module, alias.name)), alias, name_ref=name)
      # TODO: else? relative imports
    self.generic_visit(node)

  def visit_Name(self, node):
    if isinstance(node.ctx, (ast.Store, ast.Param)):
      self.scope.define_name(node.id, node)
    elif isinstance(node.ctx, ast.Load):
      self.scope.lookup_name(node.id).add_reference(node)
      self.root_scope.set_name_for_node(node, self.scope.lookup_name(node.id))
    self.generic_visit(node)

  def visit_FunctionDef(self, node):
    # Visit decorator list first to avoid declarations in args
    self.visit_in_order(node, 'decorator_list')
    if isinstance(self.root_scope.parent(node), ast.ClassDef):
      pass # TODO: Support referencing methods by "self" where possible
    else:
      self.scope.define_name(node.name, node)
    try:
      self.scope = self.scope.create_scope(node)
      self.visit_in_order(node, 'args', 'returns', 'body')
    finally:
      self.scope = self.scope.parent_scope

  def visit_arguments(self, node):
    self.visit_in_order(node, 'defaults', 'args')
    if six.PY2:
      # In python 2.x, these names are not Name nodes. Define them explicitly
      # to be able to find references in the function body.
      for arg_attr_name in ('vararg', 'kwarg'):
        arg_name = getattr(node, arg_attr_name, None)
        if arg_name is not None:
          self.scope.define_name(arg_name, node)
    else:
      # Visit defaults first to avoid declarations in args
      self.visit_in_order(node, 'vararg', 'kwarg')

  def visit_arg(self, node):
    self.scope.define_name(node.arg, node)
    self.generic_visit(node)

  def visit_ClassDef(self, node):
    self.visit_in_order(node, 'decorator_list', 'bases')
    self.scope.define_name(node.name, node)
    try:
      self.scope = self.scope.create_scope(node)
      self.visit_in_order(node, 'body')
    finally:
      self.scope = self.scope.parent_scope

  def visit_Attribute(self, node):
    self.generic_visit(node)
    node_value_name = self.root_scope.get_name_for_node(node.value)
    if node_value_name:
      node_name = node_value_name.lookup_name(node.attr)
      self.root_scope.set_name_for_node(node, node_name)
      node_name.add_reference(node)


class Scope(object):

  def __init__(self, parent_scope, node):
    self.parent_scope = parent_scope
    self.names = {}
    self.node = node

  def define_name(self, name, node):
    try:
      name_obj = self.names[name]
    except KeyError:
      name_obj = self.names[name] = Name(name)
    name_obj.define(node)
    return name_obj

  def lookup_name(self, name):
    try:
      return self.names[name]
    except KeyError:
      pass
    if self.parent_scope is None:
      name_obj = self.names[name] = Name(name)
      return name_obj
    return self.parent_scope.lookup_name(name)

  def get_root_scope(self):
    return self.parent_scope.get_root_scope()

  def lookup_scope(self, node):
    return self.get_root_scope().lookup_scope(node)

  def create_scope(self, node):
    subscope = Scope(self, node)
    self.get_root_scope()._set_scope_for_node(node, subscope)
    return subscope


class RootScope(Scope):

  def __init__(self, node):
    super(RootScope, self).__init__(None, node)
    self.external_references = {}
    self._parents = {}
    self._nodes_to_names = {}
    self._node_scopes = {}

  def add_external_reference(self, name, node, name_ref=None):
    ref = ExternalReference(name=name, node=node, name_ref=name_ref)
    if name in self.external_references:
      self.external_references[name].append(ref)
    else:
      self.external_references[name] = [ref]

  def get_root_scope(self):
    return self

  def parent(self, node):
    return self._parents.get(node, None)

  def set_parent(self, node, parent):
    self._parents[node] = parent
    if parent is None:
      self._node_scopes[node] = self

  def get_name_for_node(self, node):
    return self._nodes_to_names.get(node, None)

  def set_name_for_node(self, node, name):
    self._nodes_to_names[node] = name

  def lookup_scope(self, node):
    while node:
      try:
        return self._node_scopes[node]
      except KeyError:
        node = self.parent(node)
    return None

  def _set_scope_for_node(self, node, node_scope):
    self._node_scopes[node] = node_scope


# Should probably also have a scope?
class Name(object):

  def __init__(self, id):
    self.id = id
    self.definition = None
    self.reads = []
    self.attrs = {}

  def add_reference(self, node):
    self.reads.append(node)

  def define(self, node):
    if self.definition:
      self.reads.append(node)
    else:
      self.definition = node

  def lookup_name(self, name):
    try:
      return self.attrs[name]
    except KeyError:
      name_obj = self.attrs[name] = Name('.'.join((self.id, name)))
      return name_obj


def analyze(tree):
  v = ScopeVisitor()
  v.visit(tree)
  return v.scope
