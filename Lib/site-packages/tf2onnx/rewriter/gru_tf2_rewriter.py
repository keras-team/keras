# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.gru_tf2_rewriter - Rewrites GRU pattern used by tf2.
"""

from tf2onnx.graph_matcher import GraphMatcher
from tf2onnx.rewriter.rnn_utils import make_grucell_pattern, keras_gru_pattern
from tf2onnx.tf_loader import find_function
from tf2onnx.rewriter.unit_rnn_rewriter_base import UnitRnnContext
from tf2onnx.rewriter.gru_rewriter import GRUUnitRewriter
from tf2onnx.graph_builder import GraphBuilder

# pylint: disable=invalid-name,unused-argument,missing-docstring, unused-variable


def rewrite_gru_tf2(g, ops):
    pattern1 = make_grucell_pattern("Identity")
    pattern2 = keras_gru_pattern

    for pattern in [pattern1, pattern2]:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        for match_result in match_results:
            activation_op = match_result.get_op("optional_activation")
            activations = ["Sigmoid", activation_op.type]
            if activation_op.type not in ["Relu", "Tanh", "Sigmoid"]:
                continue

            if pattern is pattern1:
                concat = match_result.get_op("cell_inputs")
                if len(concat.inputs) != 3:
                    continue
                get_item = concat.inputs[0]
                init_state = concat.inputs[1]
            else:
                get_item = match_result.get_op("gru_input")
                init_state = match_result.get_op("state")
            if not get_item.type == "TensorListGetItem":
                continue
            x_e = get_item.inputs[0]
            if not x_e.is_graph_input():
                continue
            x_idx = g.input_names.index(x_e.output[0])
            if not init_state.is_graph_input():
                continue
            init_state_idx = g.input_names.index(init_state.output[0])

            cell_output = match_result.get_op("cell_output")
            final_consumers = g.find_output_consumers(cell_output.output[0])
            select_ops = [n for n in final_consumers if n.type == "Select"]
            def has_tensor_list_consumer(n):
                return any(c.type == "TensorListSetItem" for c in g.find_output_consumers(n.output[0]))
            select_ops = [n for n in select_ops if has_tensor_list_consumer(n)]
            if len(select_ops) == 1:
                greater_eq = select_ops[0].inputs[0]
                if greater_eq.type != "GreaterEqual":
                    continue
                seq_len = greater_eq.inputs[1]
                if not seq_len.is_graph_input():
                    continue
                seq_len_idx = g.input_names.index(seq_len.output[0])
                final_consumers = g.find_output_consumers(select_ops[0].output[0])
            else:
                seq_len_idx = None

            tensor_set_items = [n for n in final_consumers if n.type == "TensorListSetItem"]
            if len(tensor_set_items) != 1:
                continue

            if not tensor_set_items[0].inputs[0].is_graph_input():
                continue
            out_idx = g.input_names.index(tensor_set_items[0].input[0])

            hk = match_result.get_op("hidden_kernel")
            while hk.type == "Identity":
                hk = hk.inputs[0]
            if not hk.is_graph_input():
                continue
            hk_idx = g.input_names.index(hk.output[0])

            hb = match_result.get_op("hidden_bias")
            if not hb.is_graph_input():
                continue
            hb_idx = g.input_names.index(hb.output[0])

            gk = match_result.get_op("gate_kernel")
            while gk.type == "Identity":
                gk = gk.inputs[0]
            if not gk.is_graph_input():
                continue
            gk_idx = g.input_names.index(gk.output[0])

            gb = match_result.get_op("gate_bias")
            if not gb.is_graph_input():
                continue
            gb_idx = g.input_names.index(gb.output[0])

            bias_add = match_result.get_op("bias_add")
            if bias_add is not None and bias_add.data_format != "NHWC":
                continue

            g.gru_rewriter_context = {
                "x_idx": x_idx,
                "out_idx": out_idx,
                "initial_state_idx": init_state_idx,
                "hidden_kernel_idx": hk_idx,
                "hidden_bias_idx": hb_idx,
                "gate_kernel_idx": gk_idx,
                "gate_bias_idx": gb_idx,
                "seq_len_idx": seq_len_idx,
                "activations": activations,
                "from_keras": pattern is pattern2,
                "linear_before_reset": 1 if pattern is pattern2 else 0,
            }

    for op in ops:
        if op.is_while():
            body_graph = find_function(op.get_attr_str("body"))
            if body_graph.gru_rewriter_context is None:
                continue
            body_context = body_graph.gru_rewriter_context
            hk = op.input[body_context["hidden_kernel_idx"]]
            hb = op.input[body_context["hidden_bias_idx"]]
            gk = op.input[body_context["gate_kernel_idx"]]
            gb = op.input[body_context["gate_bias_idx"]]
            if not all(g.is_const(w) for w in [hk, hb, gk, gb]):
                continue
            hk_const = g.get_tensor_value(hk, as_list=False)
            hb_const = g.get_tensor_value(hb, as_list=False)
            gk_const = g.get_tensor_value(gk, as_list=False)
            gb_const = g.get_tensor_value(gb, as_list=False)

            initial_state_sq = op.input[body_context["initial_state_idx"]]
            initial_state = GraphBuilder(g).make_unsqueeze({"data": initial_state_sq, "axes": [0]})

            context = UnitRnnContext()
            context.from_keras = body_context["from_keras"]
            context.weights.update({
                "hidden_kernel": hk_const,
                "hidden_bias": hb_const,
                "gate_kernel": gk_const,
                "gate_bias": gb_const
            })
            context.attributes["activations"] = body_context["activations"]
            context.attributes["linear_before_reset"] = body_context["linear_before_reset"]
            tensor_array_inp = op.inputs[body_context["x_idx"]]
            if not tensor_array_inp.type == "TensorListFromTensor":
                continue

            final_consumers = g.find_output_consumers(op.output[body_context["out_idx"]])
            output_ys = [n.output[0] for n in final_consumers if n.type == "TensorListStack"]

            context.onnx_input_ids["X"] = tensor_array_inp.input[0]
            if body_context["seq_len_idx"] is None:
                context.onnx_input_ids["sequence_lens"] = ""
            else:
                context.onnx_input_ids["sequence_lens"] = op.input[body_context["seq_len_idx"]]
            context.onnx_input_ids["initial_state"] = initial_state

            gru_rewriter = GRUUnitRewriter(g)
            gru_rewriter.process_weights_and_bias(context)
            gru_node = gru_rewriter.create_rnn_node(context)
            squeeze_output = GraphBuilder(g).make_squeeze({"data": gru_node.output[0], "axes": [1]})
            for output in output_ys:
                g.replace_all_inputs(output, squeeze_output)

            f_state_squeeze = GraphBuilder(g).make_squeeze({"data": gru_node.output[1], "axes": [0]})
            g.replace_all_inputs(op.output[body_context["initial_state_idx"]], f_state_squeeze)

    return g.get_nodes()
