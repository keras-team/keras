"""openvino.opset16 has a compatibility issue for ops missing due to an
OpenVINO 2026.0.0 bug (https://github.com/openvinotoolkit/openvino/issues/34780).

All ops listed below were introduced in opset15 but were accidentally removed
from the opset16 re-export list. The bug has been fixed upstream; this temporary workaround 
ensures that the missing ops are available in opset16 by re-importing them from opset15 if 
they are not present. This workaround can be removed once a fixed release is available.
"""

import openvino.opset16 as ov_opset

_MISSING_FROM_OPSET16 = [
    "bitwise_left_shift",
    "bitwise_right_shift",
    "col2im",
    "embedding_bag_offsets",
    "embedding_bag_packed",
    "roi_align_rotated",
    "scatter_nd_update",
    "slice_scatter",
    "string_tensor_pack",
    "string_tensor_unpack",
]

if not hasattr(ov_opset, "scatter_nd_update"):
    import openvino.opset15 as _opset15

    for _op in _MISSING_FROM_OPSET16:
        if not hasattr(ov_opset, _op):
            setattr(ov_opset, _op, getattr(_opset15, _op))

    del _opset15, _op
