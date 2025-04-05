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

"""Parts and builders for Treescope's intermediate output format.

The functions exposed here can be used to construct a tree of parts that can be
rendered to text or to interactive HTML. Node handlers and the
__treescope_repr__ method can use them to render custom types.

Note that the internal definition of `RenderableTreePart` is considered an
implementation detail, and is subject to change. To build renderable tree parts,
you should instead use the functions exposed here (or higher-level wrappers in
`treescope.repr_lib`).
"""

# pylint: disable=g-importing-member,g-multiple-import,unused-import


from treescope._internal.parts.basic_parts import (
    build_full_line_with_annotations,
    empty_part,
    floating_annotation_with_separate_focus,
    fold_condition,
    in_outlined_box,
    indented_children,
    on_separate_lines,
    roundtrip_condition,
    siblings_with_annotations,
    siblings,
    summarizable_condition,
    text,
    vertical_space,
    with_layout_mark,
)
from treescope._internal.parts.common_structures import (
    build_copy_button,
    build_custom_foldable_tree_node,
    build_foldable_tree_node_from_children,
    build_one_line_tree_node,
    maybe_qualified_type_name,
)
from treescope._internal.parts.common_styles import (
    abbreviation_color,
    comment_color_when_expanded,
    comment_color,
    custom_text_color,
    deferred_placeholder_style,
    error_color,
    qualified_type_name_style,
    custom_style,
)
from treescope._internal.parts.custom_dataclass_util import (
    build_field_children,
    render_dataclass_constructor,
)
from treescope._internal.parts.embedded_iframe import (
    embedded_iframe,
)
from treescope._internal.parts.foldable_impl import (
    abbreviatable,
    abbreviation_level,
    abbreviatable_with_annotations,
)
from treescope._internal.parts.part_interface import (
    RenderableTreePart,
    RenderableAndLineAnnotations,
    Rendering,
    ExpandState,
    NodePath,
)


__all__ = [
    "abbreviatable",
    "abbreviation_level",
    "abbreviatable_with_annotations",
    "build_full_line_with_annotations",
    "empty_part",
    "floating_annotation_with_separate_focus",
    "fold_condition",
    "in_outlined_box",
    "indented_children",
    "on_separate_lines",
    "roundtrip_condition",
    "siblings_with_annotations",
    "siblings",
    "summarizable_condition",
    "text",
    "vertical_space",
    "with_layout_mark",
    "build_copy_button",
    "build_custom_foldable_tree_node",
    "build_foldable_tree_node_from_children",
    "build_one_line_tree_node",
    "maybe_qualified_type_name",
    "abbreviation_color",
    "comment_color_when_expanded",
    "comment_color",
    "custom_text_color",
    "deferred_placeholder_style",
    "error_color",
    "qualified_type_name_style",
    "custom_style",
    "build_field_children",
    "render_dataclass_constructor",
    "embedded_iframe",
    "RenderableTreePart",
    "RenderableAndLineAnnotations",
    "Rendering",
    "ExpandState",
    "NodePath",
]
