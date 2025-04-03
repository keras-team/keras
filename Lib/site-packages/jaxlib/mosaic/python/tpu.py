# Copyright 2023 The JAX Authors.
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

"""Python bindings for the MLIR TPU dialect."""

# ruff: noqa: F401
# ruff: noqa: F403


# pylint: disable=g-bad-import-order
from ._tpu_gen import *  # pylint: disable=wildcard-import
from ._tpu_gen import _Dialect
from jaxlib.mlir._mlir_libs._tpu_ext import *  # pylint: disable=wildcard-import
try:
  from jaxlib.mlir.dialects._ods_common import _cext
except ImportError:
  from mlir.dialects._ods_common import _cext


_cext.globals.append_dialect_search_prefix("jax.jaxlib.mosaic.python")


@_cext.register_operation(_Dialect, replace=True)
class TraceOp(TraceOp):  # noqa: F405
  """An extension to the automatically generated TraceOp bindings."""

  def __init__(self, results, message, level, *, loc=None, ip=None):
    super().__init__(results, message, level, loc=loc, ip=ip)
    self.regions[0].blocks.append(*[])  # Append the block.

  @property
  def body(self):
    return self.regions[0].blocks[0]


@_cext.register_operation(_Dialect, replace=True)
class RegionOp(RegionOp):  # noqa: F405
  """An extension to the automatically generated RegionOp bindings."""

  def __init__(self, results, *, loc=None, ip=None):
    super().__init__(results, loc=loc, ip=ip)
    self.regions[0].blocks.append()  # Append the block.

  @property
  def body(self):
    return self.regions[0].blocks[0]
