# SPDX-License-Identifier: Apache-2.0

"""
tf2onnx.keras2onnx_api - Ease migration from keras2onnx to tf2onnx.
Use tf2onnx.keras2onnx_api.convert_keras instead of deprecated keras2onnx.convert_keras
"""

# pylint: disable=unused-argument,missing-docstring

from onnx import mapping, defs
import tensorflow as tf
import tf2onnx
from tf2onnx.constants import OPSET_TO_IR_VERSION

def to_tf_tensor_spec(onnx_type, name=None, unknown_dim=1):
    shp = [unknown_dim if isinstance(n_, str) else n_ for n_ in onnx_type.shape]
    return tf.TensorSpec(shp, mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.to_onnx_type().tensor_type.elem_type],
                         name=name)

def _process_initial_types(initial_types, unknown_dim=1):
    if initial_types is None:
        return None

    input_specs = []
    c_ = 0
    while c_ < len(initial_types):
        name = None
        type_idx = c_
        if isinstance(initial_types[c_], str):
            name = initial_types[c_]
            type_idx = c_ + 1
        ts_spec = to_tf_tensor_spec(initial_types[type_idx], name, unknown_dim)
        input_specs.append(ts_spec)
        c_ += 1 if name is None else 2

    return input_specs

def get_maximum_opset_supported():
    return min(max(OPSET_TO_IR_VERSION.keys()), defs.onnx_opset_version())

def convert_keras(model, name=None, doc_string='', target_opset=None, initial_types=None,
                  channel_first_inputs=None, debug_mode=False, custom_op_conversions=None):
    """
    :param model: keras model
    :param name: the converted onnx model internal name
    :param doc_string: doc string
    :param target_opset: the targeted onnx model opset
    :param initial_types: the overridden input type for the target ONNX model.
    :param channel_first_inputs: A list of channel first input
    :param debug_mode: ignored
    :param custom_op_conversions: ignored
    :return an ONNX ModelProto
    """
    if target_opset is None:
        target_opset = get_maximum_opset_supported()
    input_signature = _process_initial_types(initial_types, unknown_dim=None)
    name = name or model.name

    model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=target_opset,
                                          inputs_as_nchw=channel_first_inputs)
    model.graph.name = name
    model.graph.doc_string = doc_string

    return model
