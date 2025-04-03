#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Optional

from ._builtin_ops_gen import *
from ._builtin_ops_gen import _Dialect
from ..extras.meta import region_op

try:
    from ..ir import *
    from ._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


@_ods_cext.register_operation(_Dialect, replace=True)
class ModuleOp(ModuleOp):
    """Specialization for the module op class."""

    def __init__(self, *, loc=None, ip=None):
        super().__init__(loc=loc, ip=ip)
        body = self.regions[0].blocks.append()

    @property
    def body(self):
        return self.regions[0].blocks[0]


@region_op
def module(
    *,
    sym_name=None,
    sym_visibility=None,
    attrs: Optional[Dict[str, Attribute]] = None,
    loc=None,
    ip=None,
):
    mod = ModuleOp.__base__(
        sym_name=sym_name, sym_visibility=sym_visibility, loc=loc, ip=ip
    )
    if attrs is None:
        attrs = {}
    for attr_name, attr in attrs.items():
        mod.operation.attributes[attr_name] = attr

    return mod
