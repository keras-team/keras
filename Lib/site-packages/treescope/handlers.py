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

"""Treescope rendering handlers, for use in renderer configurations.

These handlers are responsible for implementing each of the steps of the
default Treescope renderer, and can be used individually to build custom
renderer configurations.
"""

# pylint: disable=g-importing-member,g-multiple-import,unused-import

from treescope._internal.handlers.autovisualizer_hook import (
    use_autovisualizer_if_present,
)
from treescope._internal.handlers.basic_types_handler import (
    handle_basic_types,
)
from treescope._internal.handlers.canonical_alias_postprocessor import (
    replace_with_canonical_aliases,
)
from treescope._internal.handlers.custom_type_handlers import (
    handle_via_global_registry,
    handle_via_treescope_repr_method,
)
from treescope._internal.handlers.function_reflection_handlers import (
    handle_code_objects_with_reflection,
)
from treescope._internal.handlers.generic_pytree_handler import (
    handle_arbitrary_pytrees,
)
from treescope._internal.handlers.generic_repr_handler import (
    handle_anything_with_repr,
)
from treescope._internal.handlers.repr_html_postprocessor import (
    append_repr_html_when_present,
)
from treescope._internal.handlers.shared_value_postprocessor import (
    check_for_shared_values,
    setup_shared_value_context,
)

__all__ = [
    "use_autovisualizer_if_present",
    "handle_basic_types",
    "replace_with_canonical_aliases",
    "handle_via_global_registry",
    "handle_via_treescope_repr_method",
    "handle_code_objects_with_reflection",
    "handle_arbitrary_pytrees",
    "handle_anything_with_repr",
    "append_repr_html_when_present",
    "check_for_shared_values",
    "setup_shared_value_context",
]
