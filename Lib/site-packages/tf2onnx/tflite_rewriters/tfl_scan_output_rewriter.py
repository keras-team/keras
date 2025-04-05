# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.tflite_rewriters.tfl_scan_output_rewriter - Identify a common slice/concat pattern in tflite subgraphs
Effectively replace A = A[:i] + [B] + A[i+1:] with A[i] = B
"""
import numpy as np

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring

def rewrite_tfl_scan_outputs(g, ops):
    pattern0 = \
        OpTypePattern('TFL_CONCATENATION', name='concat', inputs=[
            OpTypePattern('TFL_SLICE', name='begin_slice'),
            OpTypePattern('*', name='middle'),
            OpTypePattern('TFL_SLICE', name='end_slice')
        ])

    matcher = GraphMatcher(pattern0, allow_reorder=False)
    match_results = list(matcher.match_ops(ops))
    if match_results:
        for match in match_results:
            concat = match.get_op("concat")
            begin_slice = match.get_op("begin_slice")
            middle = match.get_op("middle")
            end_slice = match.get_op("end_slice")
            middle_shape = g.get_shape(middle.output[0])

            # Both slices must be slicing the same tensor
            if begin_slice.input[0] != end_slice.input[0]:
                continue
            original_tensor = begin_slice.input[0]
            if concat.get_attr_int("axis") != 0:
                continue
            # The inserted slice must have length 1 (to be a single index)
            if middle_shape is None or len(middle_shape) == 0 or middle_shape[0] != 1:
                continue
            rank = len(middle_shape)
            scan_output = middle.output[0]
            if not begin_slice.inputs[1].is_const() or not end_slice.inputs[2].is_const():
                continue
            # The first slice must start from the beginning (0) for all dims
            if not all(v == 0 for v in begin_slice.inputs[1].get_tensor_value()):
                continue
            # The second slice must slice to the end (-1) for all dims
            if not all(v == -1 for v in end_slice.inputs[2].get_tensor_value()):
                continue
            # The other slice dims are assembled by concatenation if rank > 1
            if rank > 1:
                begin_concat = begin_slice.inputs[2]
                end_concat = end_slice.inputs[1]
                if not begin_concat.type == "TFL_CONCATENATION":
                    continue
                if not end_concat.type == "TFL_CONCATENATION":
                    continue
                # Except for dim 0, slice from beginning to end
                if not all(get_uniform_const_val(inp) == -1 for inp in begin_concat.inputs[1:]):
                    continue
                if not all(get_uniform_const_val(inp) == 0 for inp in end_concat.inputs[1:]):
                    continue
                begin_idx = begin_concat.inputs[0]
                end_idx = end_concat.inputs[0]
            else:
                begin_idx = begin_slice.inputs[2]
                end_idx = end_slice.inputs[1]
            # For dim 0, slice to i for first part and from i+1 for second
            if not node_is_one_plus_node(begin_idx, end_idx):
                continue
            out1, _ = get_out_and_offset(begin_idx)
            graph_inps = [n.output[0] for n in g.inputs]
            # To be a scan output, i must be a graph input
            if out1 not in graph_inps:
                continue
            # The array being sliced must be a graph input
            if original_tensor not in graph_inps:
                continue
            # The input/output index of i
            idx = graph_inps.index(out1)
            # The input/output index of the array
            scan_output_idx = graph_inps.index(original_tensor)
            # For a scan output, i must be assigned to i+1 with each iteration
            if not node_is_one_plus_node(g.get_node_by_output(out1), g.get_node_by_output(g.outputs[idx])):
                continue
            if len(g.find_output_consumers(concat.output[0])) > 1:
                continue

            if g.opset < 10 and len(g.find_output_consumers(concat.output[0])) <= 1:
                # If opset is < 10, conversion of the subgraph will fail unless we remove the slice nodes
                # We add a tmp node to replace them.
                shape = g.get_shape(concat.output[0])
                dtype = g.get_dtype(concat.output[0])
                tmp_node = g.make_node("TMP_SCAN_OUTPUT", [original_tensor, scan_output],
                                       shapes=[shape], dtypes=[dtype])
                g.replace_all_inputs(concat.output[0], tmp_node.output[0])

            to_remove = []
            out = g.outputs[scan_output_idx]
            node = g.get_node_by_output(out)
            to_remove.append(node)

            while len(node.input) > 0 and node != concat:
                out = node.input[0]
                node = g.get_node_by_output(out)
                to_remove.append(node)

            to_remove += [begin_slice, end_slice, concat]

            out = original_tensor
            node = g.get_node_by_output(out)
            to_remove.append(node)

            while len(node.input) > 0:
                out = node.input[0]
                node = g.get_node_by_output(out)
                to_remove.append(node)

            if not g.is_safe_to_remove_nodes(to_remove):
                continue

            g.scan_outputs.append((scan_output_idx, scan_output))
    return ops

def get_uniform_const_val(n):
    if not n.is_const():
        return None
    v = n.get_tensor_value(as_list=False).flatten()
    if len(v) == 0:
        return None
    if np.all(v == v[0]):
        return v[0]
    return None

def get_out_and_offset(n):
    if n.type in ['TFL_RESHAPE', 'TFL_IDENTITY', 'Identity']:
        return get_out_and_offset(n.inputs[0])
    if n.type == 'TFL_ADD':
        v1 = get_uniform_const_val(n.inputs[0])
        v2 = get_uniform_const_val(n.inputs[1])
        if v1 is not None and v2 is not None:
            return '', v1 + v2
        if v1 is not None:
            inp2, o2 = get_out_and_offset(n.inputs[1])
            return inp2, v1 + o2
        if v2 is not None:
            inp1, o1 = get_out_and_offset(n.inputs[0])
            return inp1, v2 + o1
    return n.output[0], 0

def node_is_one_plus_node(node, one_plus_node):
    n1, o1 = get_out_and_offset(node)
    n2, o2 = get_out_and_offset(one_plus_node)
    return n1 == n2 and o1 + 1 == o2
