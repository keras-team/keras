# Copyright 2024 The JAX Authors.
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

"""Python bindings for the MLIR Mosaic GPU dialect.

Note: this file *must* be called `mosaic_gpu.py`, in order to match the dialect
name. Otherwise, MLIR is unable to find the module during dialect search.
"""

# ruff: noqa: F401
# ruff: noqa: F403


# pylint: disable=g-bad-import-order
from jaxlib.mosaic.dialect.gpu._mosaic_gpu_gen_ops import *  # pylint: disable=wildcard-import  # type: ignore[import-not-found]
from jaxlib.mosaic.dialect.gpu._mosaic_gpu_gen_enums import *  # pylint: disable=wildcard-import  # type: ignore[import-not-found]
from jaxlib.mlir._mlir_libs._mosaic_gpu_ext import *  # pylint: disable=wildcard-import  # type: ignore[import-not-found]

try:
  from jaxlib.mlir.dialects._ods_common import _cext
except ImportError:
  from mlir.dialects._ods_common import _cext  # type: ignore[import-not-found]


# Add the parent module to the search prefix
_cext.globals.append_dialect_search_prefix(__name__[:__name__.rfind(".")])
