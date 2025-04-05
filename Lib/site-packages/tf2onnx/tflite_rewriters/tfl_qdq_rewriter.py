# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.tflite_rewriters.tfl_qdq_rewriter - Remove qdq sequences to dequantize model
"""
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring

def rewrite_tfl_qdq(g, ops):
    pattern0 = \
        OpTypePattern('TFL_DEQUANTIZE', name='dequant', inputs=[
            OpTypePattern('TFL_QUANTIZE', name='quant'),
        ])

    matcher = GraphMatcher(pattern0, allow_reorder=False)
    match_results = list(matcher.match_ops(ops))
    if match_results:
        for match in match_results:
            dequant = match.get_op("dequant")
            quant = match.get_op("quant")
            inp_node = quant.inputs[0]
            for k in ["scale", "quantized_dimension", "zero_point"]:
                if dequant.get_attr_value(k) != quant.get_attr_value(k):
                    continue
            needed_relu = None
            if all(k in quant.attr and len(quant.get_attr_value(k)) == 1 for k in ["min", "max"]):
                min_val = quant.get_attr_value("min")[0]
                max_val = quant.get_attr_value("max")[0]
                if min_val == 0.0 and 5.999 <= max_val <= 6.0:
                    needed_relu = "TFL_RELU6"
                elif min_val == 0.0:
                    # This may introduce unneeded relu ops but will be correct.
                    # If the --dequantize feature is used a lot in the future we can optimize this.
                    needed_relu = "TFL_RELU"
                if inp_node.type == needed_relu:
                    # If it's really obviously unneeded, we skip it.
                    needed_relu = None
                elif "TFL_" + inp_node.get_attr_value("fused_activation_function", b'').decode() == needed_relu:
                    needed_relu = None

            if needed_relu is not None:
                relu_name = inp_node.name + "_relu"

                relu6 = g.make_node(needed_relu, [quant.input[0]], op_name_scope=relu_name,
                                    skip_conversion=False, shapes=quant.output_shapes, dtypes=quant.output_dtypes)
                g.replace_all_inputs(dequant.output[0], relu6.output[0])
            else:
                g.replace_all_inputs(dequant.output[0], quant.input[0])

            g.remove_node(dequant.name)
            if len(g.find_output_consumers(quant.output[0])) == 0:
                g.remove_node(quant.name)

    return ops
