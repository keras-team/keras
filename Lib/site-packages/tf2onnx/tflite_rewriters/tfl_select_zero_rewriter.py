# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.tflite_rewriters.tfl_select_zero_rewriter - TFLite has a pattern to remove NaN when multiplying/dividing by 0
"""
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring,unused-argument

def rewrite_tfl_select_zero(g, ops):
    pattern0 = \
        OpTypePattern('TFL_SELECT_V2', name='select', inputs=[
            OpTypePattern('TFL_EQUAL', name='equal', inputs=[
                OpTypePattern('Const|ConstV2', name='const_eq'),
                OpTypePattern('*', name='term_eq'),
            ], allow_reorder=True),
            OpTypePattern('Const|ConstV2', name='const_select'),
            OpTypePattern('TFL_MUL|TFL_DIV', name='mul', inputs=[
                OpTypePattern('*', name='term_mul1'),
                OpTypePattern('*', name='term_mul2'),
            ]),
        ])

    matcher = GraphMatcher(pattern0, allow_reorder=False)
    match_results = list(matcher.match_ops(ops))
    if match_results:
        for match in match_results:
            select = match.get_op("select")
            term_eq = match.get_op("term_eq")
            const_select = match.get_op("const_select")
            const_eq = match.get_op("const_eq")
            term_mul1 = match.get_op("term_mul1")
            term_mul2 = match.get_op("term_mul2")
            if const_select.get_tensor_value(as_list=True) != 0:
                continue
            if const_eq.get_tensor_value(as_list=True) != 0:
                continue
            if term_mul1.name != term_eq.name:
                term_mul1, term_mul2 = term_mul2, term_mul1
            if term_mul1.name != term_eq.name:
                continue
            # Tell downstream conversion to avoid Mul/Add optimization
            select.set_attr("handles_nan", True)

    return ops
