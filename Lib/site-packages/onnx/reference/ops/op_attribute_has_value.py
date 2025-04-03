# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class AttributeHasValue(OpRun):
    def _run(  # type: ignore
        self,
        value_float=None,  # noqa: ARG002
        value_floats=None,  # noqa: ARG002
        value_graph=None,  # noqa: ARG002
        value_graphs=None,  # noqa: ARG002
        value_int=None,  # noqa: ARG002
        value_ints=None,  # noqa: ARG002
        value_sparse_tensor=None,  # noqa: ARG002
        value_sparse_tensors=None,  # noqa: ARG002
        value_string=None,  # noqa: ARG002
        value_strings=None,  # noqa: ARG002
        value_tensor=None,  # noqa: ARG002
        value_tensors=None,  # noqa: ARG002
        value_type_proto=None,  # noqa: ARG002
        value_type_protos=None,  # noqa: ARG002
    ):
        # TODO: support overridden attributes.
        for att in self.onnx_node.attribute:
            if att.name.startswith("value_"):
                return (np.array([True]),)
        return (np.array([False]),)
