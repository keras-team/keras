# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.lstm_tf2_rewriter - Rewrites LSTM pattern used by tf2.
"""

import numpy as np
from tf2onnx.graph_matcher import GraphMatcher
from tf2onnx.rewriter.rnn_utils import make_lstm_pattern
from tf2onnx.tf_loader import find_function
from tf2onnx.rewriter.lstm_rewriter_base import LSTMContext
from tf2onnx.rewriter.lstm_rewriter import LSTMRewriter
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx import utils

# pylint: disable=invalid-name,unused-argument,missing-docstring, unused-variable

def _make_lstm_pattern_from_params(params):
    return make_lstm_pattern(enter_or_id="Identity") if not params.get("from_keras", False) \
        else make_lstm_pattern(
            from_keras=True,
            use_bias=params.get("use_bias", False),
            activation=params.get("activation", ""),
            recurrent_activation=params.get("recurrent_activation", "")
        )

def rewriter_lstm_tf2(g, ops):
    lstm_params_variations = [
        # default activations
        {"enter_or_id": "Identity"},                                                  # TF LSTM
        {"from_keras": True, "use_bias": False},                                        # keras LSTM
        {"from_keras": True, "use_bias": True},                                         # keras LSTM with bias
        # hard sigmoid as recurrent activation
        {"from_keras": True, "use_bias": False, "recurrent_activation": "hard_sigmoid"}, # keras LSTM
        {"from_keras": True, "use_bias": True, "recurrent_activation": "hard_sigmoid"}   # keras LSTM with bias
        # Note: add other LSTM variations as needed
    ]
    for params in lstm_params_variations:
        pattern = _make_lstm_pattern_from_params(params)
        matcher = GraphMatcher(pattern, allow_reorder=False)
        match_results = list(matcher.match_ops(ops))

        for match_result in match_results:
            is_ft_hard_sigmoid = params.get("recurrent_activation", "") == "hard_sigmoid"
            recurrent_activation_f = "HardSigmoid" if is_ft_hard_sigmoid else  \
                match_result.get_op("ft").type
            activation_g = match_result.get_op("gt").type
            activation_h = match_result.get_op("ct'").type

            default_activations = ["Relu", "Sigmoid", "Tanh"]
            if ((activation_g not in default_activations) or
                    (activation_h not in default_activations) or
                    (not is_ft_hard_sigmoid and recurrent_activation_f not in default_activations)):
                continue

            activations_fgh = [
                recurrent_activation_f,
                activation_g,
                activation_h
            ]

            # extract input x_t
            from_keras = params.get("from_keras", False)
            if from_keras:
                get_item = match_result.get_op("xt")
            else:
                concat = match_result.get_op("xh")
                if len(concat.inputs) != 3:
                    continue
                get_item = concat.inputs[0]
            if not get_item.type == "TensorListGetItem":
                continue
            x_e = get_item.inputs[0]
            if not x_e.is_graph_input():
                continue
            x_idx = g.input_names.index(x_e.output[0])

            # extract output h_t
            ht_mul = match_result.get_op("ht")
            final_consumers = g.find_output_consumers(ht_mul.output[0])
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

            # extract input h_(t-1) and c_(t-1)
            init_state = match_result.get_op("ht-1") if from_keras else concat.inputs[1]
            if init_state.is_graph_input():
                # c and h are separate
                h_idx = g.input_names.index(init_state.output[0])
                c_e = match_result.get_op("c")
                if not c_e.is_graph_input():
                    continue
                c_idx = g.input_names.index(c_e.output[0])
                ch_info = {
                    "state_is_tuple": True,
                    "c_idx": c_idx,
                    "h_idx": h_idx,
                }
            else:
                # c and h are concatenated
                if not init_state.type == "Slice":
                    continue
                ch_e = init_state.inputs[0]
                if not ch_e.is_graph_input():
                    continue
                ch_idx = g.input_names.index(ch_e.output[0])

                c_e = match_result.get_op("c")
                if not c_e.type == "Slice" or c_e.input[0] != ch_e.output[0]:
                    continue
                ch_info = {
                    "state_is_tuple": False,
                    "ch_idx": ch_idx,
                }

            # extract weights and bias
            w_idx = hk_idx = gk_idx = 0
            ft_bias = None

            if from_keras:
                # hidden kernel
                hk = match_result.get_op("R")
                while hk.type == "Identity":
                    hk = hk.inputs[0]
                if not hk.is_graph_input():
                    continue
                hk_idx = g.input_names.index(hk.output[0])

                # gate kernel
                gk = match_result.get_op("W")
                while gk.type == "Identity":
                    gk = gk.inputs[0]
                if not gk.is_graph_input():
                    continue
                gk_idx = g.input_names.index(gk.output[0])

                # Wb and Rb are concatenated
                b_idx = None
                if from_keras and params.get("use_bias", False):
                    bias_add = match_result.get_op("bias_add")
                    if bias_add is not None and bias_add.data_format != "NHWC":
                        continue

                    b_e = match_result.get_op("cell_bias")
                    while b_e.type == "Identity":
                        b_e = b_e.inputs[0]
                    if not b_e.is_graph_input():
                        continue
                    b_idx = g.input_names.index(b_e.output[0])

            else:
                # W and R are concatenated
                w_e = match_result.get_op("cell_kernel")
                if not w_e.is_graph_input():
                    continue
                w_idx = g.input_names.index(w_e.output[0])

                bias_add = match_result.get_op("bias_add")
                if bias_add is not None and bias_add.data_format != "NHWC":
                    continue

                b_e = match_result.get_op("cell_bias")
                if not b_e.is_graph_input():
                    continue
                b_idx = g.input_names.index(b_e.output[0])

                ft_bias_node = match_result.get_op("ft_bias")
                if not ft_bias_node.is_const():
                    continue
                if g.get_dtype(ft_bias_node.output[0]) != g.get_dtype(b_e.output[0]):
                    continue
                ft_bias = ft_bias_node.get_tensor_value(as_list=False)

            g.lstm_rewriter_context = {
                # common
                "x_idx": x_idx,
                "out_idx": out_idx,
                "seq_len_idx": seq_len_idx,
                "bias_idx": b_idx,
                "from_keras": from_keras,
                "activations_fgh": activations_fgh,
                **ch_info, # {state_is_tuple, h_idx, c_idx} or {state_is_tuple, ch_idx}

                # TF
                "weight_idx": w_idx,
                "ft_bias": ft_bias,

                # Keras
                "w_idx": gk_idx,
                "r_idx": hk_idx,
            }

    for op in ops:
        if op.is_while():
            body_graph = find_function(op.get_attr_str("body"))
            if body_graph.lstm_rewriter_context is None:
                continue
            body_context = body_graph.lstm_rewriter_context

            # parse weights
            consts = []
            if body_context["from_keras"]:
                wx = op.input[body_context["w_idx"]]
                wh = op.input[body_context["r_idx"]]
                wx_const = g.get_tensor_value(wx, as_list=False)
                wh_const = g.get_tensor_value(wh, as_list=False)
                consts.extend([wx, wh])
            else:
                w = op.input[body_context["weight_idx"]]
                w_const = g.get_tensor_value(w, as_list=False)
                consts.append(w)

            # parse bias
            if body_context["bias_idx"] is not None:
                b = op.input[body_context["bias_idx"]]
                b_const = g.get_tensor_value(b, as_list=False)
                consts.append(b)
            else:
                b_const = None

            if not all(g.is_const(c) for c in consts):
                continue

            # parse states
            if body_context["state_is_tuple"]:
                initial_c_sq = op.input[body_context["c_idx"]]
                initial_h_sq = op.input[body_context["h_idx"]]
                initial_c = GraphBuilder(g).make_unsqueeze({"data": initial_c_sq, "axes": [0]})
                initial_h = GraphBuilder(g).make_unsqueeze({"data": initial_h_sq, "axes": [0]})
            else:
                initial_ch = op.input[body_context["ch_idx"]]
                if not g.is_const(initial_ch):
                    continue
                initial_ch_const = g.get_tensor_value(initial_ch, as_list=False)
                if not len(initial_ch_const.shape) == 2:
                    continue
                initial_ch_const = np.expand_dims(initial_ch_const, axis=0)
                initial_c_const, initial_h_const = np.split(initial_ch_const, 2, axis=2)
                initial_c = g.make_const(utils.make_name("initial_c"), initial_c_const).output[0]
                initial_h = g.make_const(utils.make_name("initial_h"), initial_h_const).output[0]

            # build LSTMContext
            context = LSTMContext()
            context.from_keras = body_context["from_keras"]

            if context.from_keras:
                context.weights.append({"w": wx_const, "r": wh_const, "bias": b_const})
            else:
                context.weights.append({"weight": w_const, "bias": b_const, "ft_bias": body_context["ft_bias"]})

            context.onnx_input_ids.append({})
            context.input_size.append(None)
            context.hidden_size.append(None)
            context.attributes.append({"activations": body_context['activations_fgh']})
            tensor_array_inp = op.inputs[body_context["x_idx"]]
            if not tensor_array_inp.type == "TensorListFromTensor":
                continue

            final_consumers = g.find_output_consumers(op.output[body_context["out_idx"]])
            output_ys = [n.output[0] for n in final_consumers if n.type == "TensorListStack"]

            context.onnx_input_ids[0]["X"] = tensor_array_inp.input[0]
            if body_context["seq_len_idx"] is None:
                context.onnx_input_ids[0]["sequence_lens"] = ""
            else:
                context.onnx_input_ids[0]["sequence_lens"] = op.input[body_context["seq_len_idx"]]
            context.onnx_input_ids[0]["initial_c"] = initial_c
            context.onnx_input_ids[0]["initial_h"] = initial_h

            lstm_rewriter = LSTMRewriter(g)
            lstm_rewriter.num_lstm_layers = 1

            lstm_rewriter.process_weights_and_bias(context)
            lstm_node = lstm_rewriter.create_rnn_node(context)[0]

            squeeze_output = GraphBuilder(g).make_squeeze({"data": lstm_node.output[0], "axes": [1]})
            for output in output_ys:
                g.replace_all_inputs(output, squeeze_output)

            if body_context["state_is_tuple"]:
                c_squeeze = GraphBuilder(g).make_squeeze({"data": lstm_node.output[2], "axes": [0]})
                h_squeeze = GraphBuilder(g).make_squeeze({"data": lstm_node.output[1], "axes": [0]})
                g.replace_all_inputs(op.output[body_context["c_idx"]], c_squeeze)
                g.replace_all_inputs(op.output[body_context["h_idx"]], h_squeeze)
            else:
                concat_ch = g.make_node("Concat", [lstm_node.output[2], lstm_node.output[1]],
                                        attr={"axis": 2}).output[0]
                ch_squeeze = GraphBuilder(g).make_squeeze({"data": concat_ch, "axes": [0]})
                ch_output = op.output[body_context["ch_idx"]]
                g.replace_all_inputs(ch_output, ch_squeeze)

    return g.get_nodes()
