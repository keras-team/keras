# Copyright 2024 The etils Authors.
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

"""Numpy API.

When possible, utils are meant to work with
both numpy and jax.numpy.

"""

import sys

# pylint: disable=g-bad-import-order,g-importing-member

# Lazy, array types and xnp utils
from etils.enp import typing
from etils.enp import compat
from etils.enp.array_spec import ArraySpec
from etils.enp.array_types import dtypes
from etils.enp.checking import check_and_normalize_arrays
from etils.enp.numpy_utils import get_np_module  # DEPRECATED: Use `lazy`
from etils.enp.numpy_utils import is_array  # DEPRECATED: Use `lazy` instead
from etils.enp.numpy_utils import lazy
from etils.enp.numpy_utils import NpModule

# Str compatibility
from etils.enp.numpy_utils import is_array_str
from etils.enp.numpy_utils import is_dtype_str
from etils.enp.numpy_utils import normalize_bytes2str

# Additional numpy ops
from etils.enp import linalg
from etils.enp.einops_utils import flatten
from etils.enp.einops_utils import unflatten
from etils.enp.interp_utils import interp
from etils.enp.numpy_utils import tau
from etils.enp.geo_utils import angle_between
from etils.enp.geo_utils import batch_dot
from etils.enp.geo_utils import project_onto_plane
from etils.enp.geo_utils import project_onto_vector

# Inside tests, can use `enp.testing`
if 'pytest' in sys.modules:  # < Ensure open source does not trigger import
  try:
    from etils.enp import testing  # pylint: disable=g-import-not-at-top
  except ImportError:
    pass

del sys
