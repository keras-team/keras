# Copyright 2024 The Treescope Authors.
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

"""Algorithms for expanding and collapsing foldable nodes."""

from __future__ import annotations

import collections
from typing import Any, Collection

from treescope._internal.parts import part_interface


def expand_to_layout_marks(
    tree: part_interface.RenderableTreePart,
    marks: Collection[Any],
    collapse_weak_others: bool,
) -> None:
  """Expands a tree so that any part with a marker in the given set is visible.

  This function ignores the existing expand state, and instead just ensures that
  every foldable node that contains a layout marker in `marks` is expanded.
  Optionally it will also ensure that all sibling foldable nodes are collapsed.

  Expand states for those nodes are set to EXPANDED or COLLAPSED, and expand
  states for other nodes are not modified. This means you can call
  `expand_for_balanced_layout` after this function to reformat the subtrees
  of `tree` that haven't yet been assigned strong expand/collapse states. This
  is useful for producing balanced layouts once the user expands a collapsed
  node.

  Args:
    tree: The tree to update.
    marks: The marks that should be made visible.
    collapse_weak_others: Whether to collapse foldables that do NOT have the
      given marks and have a weak expansion state.
  """
  for foldable in tree.foldables_in_this_part():
    found = _process_foldable_by_marks(foldable, marks, collapse_weak_others)
    if collapse_weak_others and not found:
      foldable.set_expand_state(part_interface.ExpandState.COLLAPSED)


def _process_foldable_by_marks(
    foldable: part_interface.FoldableTreeNode,
    marks: Collection[Any],
    collapse_weak_others: bool,
) -> bool:
  """Expands a foldable based on marks, and may also collapse its children.

  Args:
    foldable: The foldable to process.
    marks: The marks that should be made visible.
    collapse_weak_others: Whether to collapse foldables that do NOT have the
      given marks and have a weak expansion state.

  Returns:
    True if this node should be made visible (e.g. all parents expanded) based
    on the marks.
  """
  # Check children.
  might_need_collapsing = []
  for child_foldable in foldable.as_expanded_part().foldables_in_this_part():
    if not _process_foldable_by_marks(
        child_foldable, marks, collapse_weak_others
    ):
      might_need_collapsing.append(child_foldable)

  if any(mark in foldable.layout_marks_in_this_part for mark in marks):
    # We need to expand this node.
    foldable.set_expand_state(part_interface.ExpandState.EXPANDED)
    if collapse_weak_others:
      # We should also collapse any child that wasn't marked as needing to be
      # expanded.
      for child_foldable in might_need_collapsing:
        if child_foldable.get_expand_state().is_weak():
          child_foldable.set_expand_state(part_interface.ExpandState.COLLAPSED)
    # Inform caller that we found something.
    return True
  else:
    # This node doesn't need to be expanded. Don't immediately mark it as
    # collapsed, because we may want to mark a parent as collapsed instead.
    # The parent will collapse this node if necessary.
    return False


def expand_for_balanced_layout(
    tree: part_interface.RenderableTreePart,
    max_height: int | None = 20,
    target_width: int = 60,
    relax_height_constraint_for_only_child: bool = True,
    recursive_expand_height_for_collapsed_nodes: int = 7,
) -> None:
  """Updates a tree's (weak) expand states to achieve a "balanced" layout.

  A balanced layout tries to evenly allocate space along vertical and horizontal
  axes. Similar to many pretty-printers, nodes are usually expanded to avoid
  long lines and make them fit comfortably within `target_width` characters.
  However, there is also a `max_height` parameter that takes precedence, which
  can be used to prevent outputs from being too large for very large objects,
  and makes it easier for users to interactively expand only the subtrees that
  they want to inspect.

  Unlike other pretty-printers, for simplicity we don't track the current
  indent level or remaining line width, but instead make local decisions for
  each node. This means the overall rendering may end up longer than
  `target_width` because the leaves (each of width `target_width`) may be
  indented. (This simplifies the system because we do not have to track the
  exact indent levels and starting/ending positions of each node, and don't
  have to propagate that information up through the rendered parts.)

  This function only modifies WEAKLY_EXPANDED or WEAKLY_COLLAPSED nodes.

  Args:
    tree: The tree to update.
    max_height: Maximum number of lines for the rendering, which we should try
      not to exceed when expanding nodes. Can sometimes be exceeded for "only
      child" subtrees if `relax_height_constraint_for_only_child` is set. If
      None, then an arbitrary height is allowed, and only `target_width` will be
      used to determine expansion size.
    target_width: Target number of characters for the rendering. If a node is
      shorter than this width when collapsed, we will leave it on a single line
      instead of expanding it.
    relax_height_constraint_for_only_child: If True, we allow the height
      constraint to be violated once if there is only a single node that could
      possibly be expanded. In particular, we repeatedly expand nodes until we
      reach a node shorter than `target_width` or a node with two or more
      children. If we match or exceed the height constraint while doing this, we
      stop after expanding that node (but leave it expanded). If we haven't
      reached the constraint, we continue expanding under the stricter rules.
    recursive_expand_height_for_collapsed_nodes: Whenever we mark a node as
      collapsed, we first recursively expand its children as if it was expanded
      with this as the max height, then collapse only the outermost node. This
      means it will render as a single line, but if expanded by the user, it
      will expand not just the clicked node but also some of its children. This
      is intended to support faster interactive exploration of large objects.
  """
  # Tracks the number of lines we have so far committed to outputting via our
  # expansion decisions.
  committed_height = 1 + tree.newlines_in_expanded_parent
  # Tracks the collection of foldable nodes we can consider expanding.
  candidate_foldables = collections.deque(tree.foldables_in_this_part())

  # Step 1: Handle only children.
  if relax_height_constraint_for_only_child:
    while (
        # We haven't yet reached the height constraint.
        (max_height is None or committed_height < max_height)
        # There's only one node that could be expanded.
        and len(candidate_foldables) == 1
        # This node doesn't fit in the target width.
        and candidate_foldables[0].collapsed_width > target_width
        # This node isn't already forced to be collapsed.
        and candidate_foldables[0].get_expand_state()
        != part_interface.ExpandState.COLLAPSED
    ):
      # We should expand this node.
      foldable = candidate_foldables.popleft()
      expanded_foldable_contents = foldable.as_expanded_part()
      foldable.set_expand_state(part_interface.ExpandState.EXPANDED)
      committed_height += expanded_foldable_contents.newlines_in_expanded_parent
      candidate_foldables.extend(
          expanded_foldable_contents.foldables_in_this_part()
      )

  # Step 2: Expand nodes in breadth-first order while we have space.
  while candidate_foldables:
    foldable = candidate_foldables.popleft()
    expanded_foldable_contents = foldable.as_expanded_part()

    if foldable.get_expand_state().is_weak():
      # Infer a state for this node.
      height_if_expanded = (
          committed_height
          + expanded_foldable_contents.newlines_in_expanded_parent
      )
      if max_height is not None and height_if_expanded > max_height:
        # This node takes up too much space if expanded. Collapse it.
        foldable.set_expand_state(part_interface.ExpandState.COLLAPSED)
      elif foldable.collapsed_width > target_width:
        # This node is too long to render on one line. Expand it.
        foldable.set_expand_state(part_interface.ExpandState.EXPANDED)
      else:
        # This node could render either collapsed or expanded. Commit to its
        # current weak preference.
        if (
            foldable.get_expand_state()
            == part_interface.ExpandState.WEAKLY_EXPANDED
        ):
          foldable.set_expand_state(part_interface.ExpandState.EXPANDED)
        else:
          foldable.set_expand_state(part_interface.ExpandState.COLLAPSED)

    if foldable.get_expand_state() == part_interface.ExpandState.EXPANDED:
      # Record it as expanded and add children to our queue.
      committed_height += expanded_foldable_contents.newlines_in_expanded_parent
      candidate_foldables.extend(
          expanded_foldable_contents.foldables_in_this_part()
      )
    else:
      # Recursively process the children with this function, and don't add them
      # to the current queue (since their parent isn't expanded)
      expand_for_balanced_layout(
          expanded_foldable_contents,
          max_height=recursive_expand_height_for_collapsed_nodes,
          target_width=target_width,
          relax_height_constraint_for_only_child=(
              relax_height_constraint_for_only_child
          ),
          recursive_expand_height_for_collapsed_nodes=(
              recursive_expand_height_for_collapsed_nodes
          ),
      )


def expand_to_depth(
    tree: part_interface.RenderableTreePart, depth: int
) -> None:
  """Expands a tree up to a given depth.

  This function ignores the existing expand state, and instead rewrites it to
  expand up to the given depth.

  Args:
    tree: Tree to expand.
    depth: Depth to expand to. At depth 0, all foldables will be collapsed.
  """
  for foldable in tree.foldables_in_this_part():
    if depth > 0:
      foldable.set_expand_state(part_interface.ExpandState.EXPANDED)
      expand_to_depth(foldable.as_expanded_part(), depth - 1)
    else:
      foldable.set_expand_state(part_interface.ExpandState.COLLAPSED)
