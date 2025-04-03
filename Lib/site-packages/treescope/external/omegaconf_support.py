# Copyright 2025 The Treescope Authors.
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

"""Lazy setup logic for adding OmegaConf support to treescope."""
from __future__ import annotations

import functools

from treescope import type_registries
from treescope import repr_lib
from treescope import canonical_aliases

# pylint: disable=import-outside-toplevel
try:
  import omegaconf
except ImportError:
  omegaconf = None
# pylint: enable=import-outside-toplevel


def set_up_omegaconf() -> None:
  """Registers handlers for OmegaConf types."""
  if omegaconf is None:
    raise RuntimeError(
        "Cannot set up OmegaConf support in treescope: omegaconf cannot be"
        " imported."
    )
  type_registries.TREESCOPE_HANDLER_REGISTRY[omegaconf.DictConfig] = (
      functools.partial(repr_lib.handle_custom_mapping, roundtrippable=True)
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[omegaconf.ListConfig] = (
      functools.partial(repr_lib.handle_custom_listlike, roundtrippable=True)
  )
  # Register canonical aliases for all types and functions omegaconf exports
  # in omegaconf.__all__.
  canonical_aliases.populate_from_public_api(
      omegaconf, canonical_aliases.prefix_filter("omegaconf")
  )
