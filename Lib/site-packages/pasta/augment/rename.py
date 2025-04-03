# coding=utf-8
"""Rename names in a python module."""
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
import six

from pasta.augment import import_utils
from pasta.base import ast_utils
from pasta.base import scope


def rename_external(t, old_name, new_name):
  """Rename an imported name in a module.

  This will rewrite all import statements in `tree` that reference the old
  module as well as any names in `tree` which reference the imported name. This
  may introduce new import statements, but only if necessary.

  For example, to move and rename the module `foo.bar.utils` to `foo.bar_utils`:
  > rename_external(tree, 'foo.bar.utils', 'foo.bar_utils')

  - import foo.bar.utils
  + import foo.bar_utils

  - from foo.bar import utils
  + from foo import bar_utils

  - from foo.bar import logic, utils
  + from foo.bar import logic
  + from foo import bar_utils

  Arguments:
    t: (ast.Module) Module syntax tree to perform the rename in. This will be
      updated as a result of this function call with all affected nodes changed
      and potentially new Import/ImportFrom nodes added.
    old_name: (string) Fully-qualified path of the name to replace.
    new_name: (string) Fully-qualified path of the name to update to.

  Returns:
    True if any changes were made, False otherwise.
  """
  sc = scope.analyze(t)

  if old_name not in sc.external_references:
    return False

  has_changed = False
  renames = {}
  already_changed = []
  for ref in sc.external_references[old_name]:
    if isinstance(ref.node, ast.alias):
      parent = sc.parent(ref.node)
      # An alias may be the most specific reference to an imported name, but it
      # could if it is a child of an ImportFrom, the ImportFrom node's module
      # may also need to be updated.
      if isinstance(parent, ast.ImportFrom) and parent not in already_changed:
        assert _rename_name_in_importfrom(sc, parent, old_name, new_name)
        renames[old_name.rsplit('.', 1)[-1]] = new_name.rsplit('.', 1)[-1]
        already_changed.append(parent)
      else:
        ref.node.name = new_name + ref.node.name[len(old_name):]
        if not ref.node.asname:
          renames[old_name] = new_name
      has_changed = True
    elif isinstance(ref.node, ast.ImportFrom):
      if ref.node not in already_changed:
        assert _rename_name_in_importfrom(sc, ref.node, old_name, new_name)
        renames[old_name.rsplit('.', 1)[-1]] = new_name.rsplit('.', 1)[-1]
        already_changed.append(ref.node)
        has_changed = True

  for rename_old, rename_new in six.iteritems(renames):
    _rename_reads(sc, t, rename_old, rename_new)
  return has_changed


def _rename_name_in_importfrom(sc, node, old_name, new_name):
  if old_name == new_name:
    return False

  module_parts = node.module.split('.')
  old_parts = old_name.split('.')
  new_parts = new_name.split('.')

  # If just the module is changing, rename it
  if module_parts[:len(old_parts)] == old_parts:
    node.module = '.'.join(new_parts + module_parts[len(old_parts):])
    return True
    
  # Find the alias node to be changed
  for alias_to_change in node.names:
    if alias_to_change.name == old_parts[-1]:
      break
  else:
    return False

  alias_to_change.name = new_parts[-1]

  # Split the import if the package has changed
  if module_parts != new_parts[:-1]:
    if len(node.names) > 1:
      new_import = import_utils.split_import(sc, node, alias_to_change)
      new_import.module = '.'.join(new_parts[:-1])
    else:
      node.module = '.'.join(new_parts[:-1])

  return True


def _rename_reads(sc, t, old_name, new_name):
  """Updates all locations in the module where the given name is read.

  Arguments:
    sc: (scope.Scope) Scope to work in. This should be the scope of `t`.
    t: (ast.AST) The AST to perform updates in.
    old_name: (string) Dotted name to update.
    new_name: (string) Dotted name to replace it with.

  Returns:
    True if any changes were made, False otherwise.
  """
  name_parts = old_name.split('.')
  try:
    name = sc.names[name_parts[0]]
    for part in name_parts[1:]:
      name = name.attrs[part]
  except KeyError:
    return False

  has_changed = False
  for ref_node in name.reads:
    if isinstance(ref_node, (ast.Name, ast.Attribute)):
      ast_utils.replace_child(sc.parent(ref_node), ref_node,
                              ast.parse(new_name).body[0].value)
      has_changed = True

  return has_changed
