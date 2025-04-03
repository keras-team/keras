# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.eye_rewriter - supports tf.eye
"""

from onnx import onnx_pb
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

# pylint: disable=invalid-name,unused-argument,missing-docstring, unused-variable


def rewrite_eye(g, ops):
    # schema of eye is eye(num_rows, num_columns=None), if num_columns not specified then it's equal to num_rows
    # tf.eye is implemented by a sub_graph which contains op "MatrixDiag" or "MatrixSetDiag" while
    # these two ops are un-supported directly in onnx
    # but onnx op EyeLike can be used to map the sub_graph
    # "rewrite_eye" supports tf.eye(non_const) and tf.eye(non_const1, non_const2).
    # tf.eye(const) and tf.eye(const1, const2) are not supported in this rewriter

    # ConstantOfShape in opset 9 is used, so if opset less than 9 then do nothing
    if g.opset < 9:
        return g.get_nodes()

    pattern1 = \
        OpTypePattern("MatrixDiag", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill", inputs=[
                OpTypePattern("Const", name="fill_value"),
                OpTypePattern("ConcatV2", inputs=[
                    "*",
                    "*",
                    OpTypePattern("Pack", inputs=[
                        OpTypePattern("Minimum|Cast", name="min_or_cast")
                    ])
                ])
            ])
        ])
    pattern2 = \
        OpTypePattern("MatrixSetDiag", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill"),
            OpTypePattern("Fill", inputs=[
                OpTypePattern("Const", name="fill_value"),
                OpTypePattern("ConcatV2", inputs=[
                    "*",
                    "*",
                    OpTypePattern("Pack", inputs=[
                        OpTypePattern("Minimum|Cast", name="min_or_cast")
                    ])
                ])
            ])
        ])
    pattern3 = \
        OpTypePattern("MatrixDiag", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill", inputs=[
                OpTypePattern("ConcatV2", inputs=[
                    "*",
                    OpTypePattern("ExpandDims", inputs=[
                        OpTypePattern("Minimum|Cast", name="min_or_cast"),
                        "*"
                    ]),
                    "*",
                ]),
                OpTypePattern("Const", name="fill_value"),
            ])
        ])
    pattern4 = \
        OpTypePattern("MatrixSetDiag", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill"),
            OpTypePattern("Fill", inputs=[
                OpTypePattern("ConcatV2", inputs=[
                    "*",
                    OpTypePattern("ExpandDims", inputs=[
                        OpTypePattern("Minimum|Cast", name="min_or_cast"),
                        "*"
                    ]),
                    "*",
                ]),
                OpTypePattern("Const", name="fill_value"),
            ]),
        ])
    pattern5 = \
        OpTypePattern("MatrixDiagV3", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill", inputs=[
                OpTypePattern("ConcatV2", inputs=[
                    "*",
                    OpTypePattern("ExpandDims", inputs=[
                        OpTypePattern("Minimum|Cast", name="min_or_cast"),
                        "*"
                    ]),
                    "*",
                ]),
                OpTypePattern("Const", name="fill_value"),
            ]),
            "*", "*", "*", "*",
        ])
    pattern6 = \
        OpTypePattern("MatrixSetDiagV3", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill"),
            OpTypePattern("Fill", inputs=[
                OpTypePattern("ConcatV2", inputs=[
                    "*",
                    OpTypePattern("ExpandDims", inputs=[
                        OpTypePattern("Minimum|Cast", name="min_or_cast"),
                        "*"
                    ]),
                    "*",
                ]),
                OpTypePattern("Const", name="fill_value"),
            ]), "*"
        ])
    pattern7 = \
        OpTypePattern("MatrixDiag", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill", inputs=[
                OpTypePattern("Reshape", inputs=[
                    OpTypePattern("Minimum|Cast", name="min_or_cast"),
                    "*",
                ]),
                OpTypePattern("Const", name="fill_value"),
            ])
        ])
    pattern8 = \
        OpTypePattern("MatrixSetDiag", name="output_eye_matrix", inputs=[
            OpTypePattern("Fill"),
            OpTypePattern("Fill", inputs=[
                OpTypePattern("Reshape", inputs=[
                    OpTypePattern("Minimum|Cast", name="min_or_cast"),
                    "*",
                ]),
                OpTypePattern("Const", name="fill_value"),
            ])
        ])

    for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8]:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        for match_result in match_results:
            if match_result.get_op("fill_value").get_tensor_value() != 1:
                continue

            min_or_cast = match_result.get_op("min_or_cast")
            if min_or_cast.type == "Minimum":
                min_node = min_or_cast
            elif min_or_cast.type == "Cast" and min_or_cast.inputs[0].type == "Minimum":
                min_node = min_or_cast.inputs[0]
            else:
                continue

            num_rows = min_node.inputs[0]
            num_columns = min_node.inputs[1]

            old_output = match_result.get_op("output_eye_matrix")
            output_dtypes = [g.get_dtype(old_output.output[0])]
            output_shapes = [g.get_shape(old_output.output[0])]
            g.remove_node(old_output.name)

            # onnx op "EyeLike" need a 2D tensor, so generate it

            num_rows = GraphBuilder(g).make_unsqueeze(
                {"axes": [0], "data": num_rows.output[0]}, return_node=True)
            num_columns = GraphBuilder(g).make_unsqueeze(
                {"axes": [0], "data": num_columns.output[0]}, return_node=True)
            matrix_shape = g.make_node("Concat", [num_rows.output[0], num_columns.output[0]], attr={"axis": 0})
            # cast nodes added for "ConstantOfShape" in ONNX only accepts int64 data.
            matrix_shape_int64 = g.make_node("Cast", matrix_shape.output, attr={"to": onnx_pb.TensorProto.INT64})
            zero_matrix = g.make_node("ConstantOfShape", matrix_shape_int64.output)

            g.make_node("EyeLike", zero_matrix.output, attr={"dtype": output_dtypes[0]},
                        name=old_output.name, shapes=output_shapes, dtypes=output_dtypes, outputs=old_output.output)

    return g.get_nodes()
