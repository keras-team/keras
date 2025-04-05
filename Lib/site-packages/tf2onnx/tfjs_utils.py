# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.tfjs_utils - utilities for parsing tfjs files into onnx graphs

Main functions of interest are graphs_from_tfjs and read_tfjs_graph
"""

import json
import os
import base64
import gzip
import struct
import logging

from onnx import numpy_helper, helper
import numpy as np
from google.protobuf.json_format import ParseDict
import tensorflow as tf
from tensorflow.python.framework import c_api_util
from tensorflow.core.framework import types_pb2, node_def_pb2

from tf2onnx import utils
from tf2onnx.graph import Graph
from tf2onnx import tf_utils

logger = logging.getLogger(__name__)


tf_api_def_map = c_api_util.ApiDefMap()


def read_tfjs_attr(attr, tf_dtypes=False):
    """
    Reads the value of a single tfjs node attribute. If tf_dtypes is True, tensorflow dtypes are returned instead of
    onnx dtypes
    """
    k = list(attr.keys())[0]
    return read_tfjs_attr_helper(k, attr[k], tf_dtypes)


def fix_string_attr(tfjs_node):
    """
    Older tfjs models store strings as lists of ints (representing byte values). This function finds and replaces
    those strings, so protobuf can correctly decode the json.
    """
    def fix(v):
        if isinstance(v, list):
            return base64.encodebytes(bytes(v)).decode()
        return v
    if 'attr' not in tfjs_node:
        return
    for v in tfjs_node['attr'].values():
        if 's' in v:
            v['s'] = fix(v['s'])
        if 'list' in v and 's' in v['list']:
            for i, x in enumerate(v['list']['s']):
                v['list']['s'][i] = fix(x)


def read_tfjs_attr_helper(k, v, tf_dtypes=False):
    """
    A tfjs attribute value is itself a dict with a single key specifying the type and a value with the actual data
    like { axis: { i: -1 }} or { pads: { list: { i: [1, 2, 3, 4] } } }. This helper takes the key specifying the
    type (like 'i' or 'list') and the value and decodes the attribute value.
    """
    supported_types = ['func', 'shape', 'type', 'list', 's', 'i', 'f', 'b']
    utils.make_sure(k in supported_types, "Unrecognized tfjs attribute type %s", k)
    if k == 'list':
        non_empty_keys = [k2 for k2, v2 in v.items() if len(v2) > 0]
        if len(non_empty_keys) == 0:
            return []
        k2 = non_empty_keys[0]
        return [read_tfjs_attr_helper(k2, v2, tf_dtypes) for v2 in v[k2]]
    if k == 'type':
        dtype = v
        if not isinstance(dtype, int):
            dtype = getattr(types_pb2, dtype)
        if not tf_dtypes:
            dtype = tf_utils.map_tf_dtype(dtype)
        return dtype
    if k == 'func':
        return v['name']
    if k == 'shape':
        return [int(d['size']) for d in v.get('dim', [])]
    if k == 's':
        return base64.decodebytes(v.encode())
    if k == 'i':
        # ints are stored in the tfjs json as strings
        return int(v)
    return v


def tfjs_node_to_tf_node_def(node):
    """Converts a tfjs node to a tf node_def for use in tf shape inferencing"""
    node_def = node_def_pb2.NodeDef()
    ParseDict(node, node_def)
    return node_def


def resolve_output(output, op_info, func_name=None):
    """
    Given an output name from a tfjs model and an op_info dict containing info about the nodes available, (and the
    function name if this is a subgraph), returns the canonical name to use as the output name in the onnx model.
    The resulting string is always "node_name:port_number"
    """
    cnt = output.count(':')
    # outputs in the tfjs model can use one of 3 different formats interchangably.
    if cnt == 0:
        # If no port is specified, it is referring to port 0
        if output in op_info:
            return output + ':0'
        # Output isn't from an op and may be an input (no port number)
        return output
    if cnt == 1:
        # Already in our standard format
        return output
    # Format is node_name:output_name:subindex
    node, output_arg_name, index = output.split(':')
    if node not in op_info and func_name is not None:
        # In very rare cases, tfjs prepends the func_name to a node but forgets to fix the outputs
        long_node_name = func_name + "/" + node
        if long_node_name in op_info:
            node = long_node_name
    op_type, tf_attr, inp_dtypes = op_info[node]
    names, _ = get_output_names_and_dtypes(op_type, tf_attr, inp_dtypes)
    idx = names.index(output_arg_name) + int(index)
    return node + ':' + str(idx)


def get_output_names_and_dtypes(op_type, tf_attr, inp_dtypes):
    """Parses the tf documentation to determine the names and dtypes of the outputs of the op"""
    # TODO: ['Prelu', 'Conv1D', 'DepthwiseConv2d', 'FusedDepthwiseConv2dNative', 'Ones', 'Zeros']
    if op_type == 'Prelu':
        return ['activations'], [inp_dtypes[0]]
    try:
        tf_op_def = tf_api_def_map.get_op_def(op_type)
    except ValueError:
        raise ValueError("Failed to determine dtypes for op type %s. May be an unsupported op type." % op_type)
    dtypes = []
    names = []
    for arg in tf_op_def.output_arg:
        num_copies = 1
        if arg.type_list_attr:
            dtypes += tf_attr[arg.type_list_attr]
            num_copies = len(tf_attr[arg.type_list_attr])
        else:
            if arg.type_attr:
                dtype = tf_attr[arg.type_attr]
            else:
                dtype = arg.type
            if arg.number_attr:
                dtypes += [dtype] * tf_attr[arg.number_attr]
                num_copies = tf_attr[arg.number_attr]
            else:
                dtypes.append(dtype)
        names += [arg.name] * num_copies
    return names, dtypes


def get_output_dtypes(op_type, tf_attr, inp_dtypes):
    """Returns a list of the tf dtypes for the op's outputs"""
    _, out_dtypes = get_output_names_and_dtypes(op_type, tf_attr, inp_dtypes)
    return out_dtypes


def get_output_shapes(node_def, input_dtypes, input_shapes, inp_consts):
    """Returns a list of the output shapes of an op. input_dtypes should be tf dtypes."""
    from tf2onnx.tf_loader import tf_session, tf_placeholder  # pylint: disable=import-outside-toplevel

    if node_def.op in ["Prelu", "Enter"]:
        return [input_shapes[0]]

    if node_def.op == "Merge":
        # Find the first non-None shape (if it exists) and return it
        non_none = ([t for t in input_shapes if t is not None] + [None])[0]
        # The second output of merge is a scalar int indicating which input was selected
        return [non_none, []]

    if node_def.op == "Placeholder":
        shape = None
        if 'shape' in node_def.attr:
            shape = [d.size for d in node_def.attr['shape'].shape.dim]
            shape = [None if d == -1 else d for d in shape]
            if len(shape) == 0:
                # According to TF docs, "If the shape has 0 dimensions, the shape is unconstrained."
                shape = None
        return [shape]

    del node_def.input[:]
    node_def.name = "node"
    if "_class" in node_def.attr:
        # Remove colocation information (list of nodes tf wants computed on same device)
        del node_def.attr["_class"]

    g = tf.Graph()
    with g.as_default():
        for i, (dtype, shape, const) in enumerate(zip(input_dtypes, input_shapes, inp_consts)):
            inp = "input" + str(i)
            if const is None:
                if shape is not None and -1 in shape:
                    shape = [d if d != -1 else None for d in shape]
                tf_placeholder(dtype, name=inp, shape=shape)
            else:
                tf.constant(const, dtype, name=inp)
            node_def.input.append(inp)
        mini_graph_def = g.as_graph_def()
        mini_graph_def.node.append(node_def)
    g2 = tf.Graph()
    with g2.as_default():
        with tf_session() as sess:
            tf.import_graph_def(mini_graph_def, name='')
            node = sess.graph.get_operation_by_name("node")
            outputs_shapes = [tf_utils.get_tf_tensor_shape(out) for out in node.outputs]
            return outputs_shapes


def sort_tfjs_functions(funcs):
    """Topologically sorts a list of tfjs functions"""
    dependencies = {}
    name_to_func = {}
    for f in funcs:
        name = f['signature']['name']
        dependencies[name] = get_tfjs_func_dependencies(f)
        name_to_func[name] = f
    ordered = utils.topological_sort(dependencies)
    return [name_to_func[n] for n in ordered]


def get_tfjs_func_dependencies(func):
    """Returns a list of names of functions the provided tfjs func depends on"""
    dependencies = set()
    for node in func.get('nodeDef', []):
        for v in node.get('attr', {}).values():
            if list(v.keys())[0] == 'func':
                dependencies.add(read_tfjs_attr(v))
    return list(dependencies)


def read_model_json(model_path):
    """Given the path to a model.json file, parses the json and returns a dict (and flag indicating if the weights are
    compressed)"""
    zip_compressed = False
    with open(model_path, "rb") as f:
        magic_number = f.read(2)
        f.seek(0)
        if magic_number == b'\x1f\x8b':
            # Sometimes models from tfhub look normal but are gzip compressed without warning
            unziped_bytes = gzip.decompress(f.read())
            model = json.loads(unziped_bytes)
            zip_compressed = True
        else:
            model = json.load(f)
    return model, zip_compressed


def graphs_from_tfjs(model_path, input_names=None, output_names=None, shape_override=None,
                     ignore_default=None, use_default=None):
    """Given the path to a model.json file, parses the model into onnx graphs and returns the main graph and a
    topologically sorted list of subgraphs."""
    model, zip_compressed = read_model_json(model_path)

    model_format = model['modelTopology'].get('format')
    if model_format is None:
        if 'keras_version' in model['modelTopology']:
            model_format = 'layers-model'
        else:
            model_format = 'graph-model'
    utils.make_sure(model_format == 'graph-model', "tf2onnx only supports conversion from tfjs graph models, "
                    "not format %s. Use Google's tfjs converter to convert to a graph model, then try again.",
                    model_format)

    weights_manifest = model['weightsManifest'][0]

    sharded_data = []
    for path in weights_manifest["paths"]:
        with open(os.path.join(os.path.dirname(model_path), path), "rb") as f:
            shard_bytes = f.read()
            if zip_compressed:
                shard_bytes = gzip.decompress(shard_bytes)
            sharded_data.append(shard_bytes)

    weights_data = b''.join(sharded_data)
    weights = {}

    i = 0
    for weight in weights_manifest['weights']:
        weight_name, np_arr, num_bytes = read_tfjs_weight(weight, weights_data, offset=i)
        weights[weight_name] = np_arr
        i += num_bytes

    utils.make_sure(len(weights_data) == i, "Total weight bytes %d doesn't match read bytes %d", len(weights_data), i)
    topology = model['modelTopology']

    tensors_to_rename = {}
    if output_names is None and 'signature' in model:
        outputs = model['signature'].get('outputs')
        inputs = model['signature'].get('inputs')
        if outputs is not None:
            output_names = [v['name'] for v in outputs.values()]
            tensors_to_rename.update({v['name']: k for k, v in outputs.items()})
        if inputs is not None:
            tensors_to_rename.update({v['name']: k for k, v in inputs.items()})

    main_g = read_tfjs_graph(topology['node'], weights, None, input_names, output_names, shape_override,
                             ignore_default, use_default)
    main_g.rename_tensors(tensors_to_rename)
    subgraphs = []
    funcs = sort_tfjs_functions(topology.get('library', {}).get('function', []))
    for func in funcs:
        sub_g = read_tfjs_graph(func.get('nodeDef', []), weights, func, None, None, shape_override,
                                ignore_default, use_default)
        subgraphs.append(sub_g)

    return main_g, subgraphs


def read_tfjs_weight(weight, weights_data, offset):
    """Returns the name, numpy array, and number of bytes for a tfjs weight"""
    name = weight['name']
    count = np.product(weight['shape'], dtype=np.int64)
    if weight['dtype'] == 'string':
        num_strings = np.prod(weight['shape'], dtype=np.int64)
        string_list, num_bytes = read_string_weight(weights_data, offset, num_strings)
        np_arr = np.array(string_list).reshape(weight['shape'])
        return name, np_arr, num_bytes
    np_dtype = np.dtype(weight['dtype'])
    if 'quantization' in weight:
        q_info = weight['quantization']
        q_dtype = np.dtype(q_info['dtype'])
        np_arr = np.frombuffer(weights_data, dtype=q_dtype, count=count, offset=offset)
        num_bytes = np_arr.nbytes
        if 'scale' in q_info:
            np_arr = np_arr.astype(np_dtype) * q_info['scale'] + q_info['min']
        else:
            np_arr = np_arr.astype(np_dtype)
    else:
        np_arr = np.frombuffer(weights_data, dtype=np_dtype, count=count, offset=offset)
        num_bytes = np_arr.nbytes
    np_arr = np_arr.reshape(weight['shape'])
    return name, np_arr, num_bytes


def read_string_weight(weights_data, offset, num_strings):
    """Decodes binary weight data for a tfjs string"""
    string_list = []
    j = offset
    for _ in range(num_strings):
        # TFJS strings start with a 4 byte unsigned int indicating their length, followed by the bytes of the string
        length = struct.unpack('<I', weights_data[j:j + 4])[0]
        j += 4
        string_list.append(weights_data[j:j + length])
        j += length
    return string_list, j - offset


def read_tfjs_function(func):
    """Parses properties of a tfjs function."""
    tf_dtypes = {}
    output_shapes = {}
    signature = func['signature']
    inputs = []
    for i, inp in enumerate(signature['inputArg']):
        inp_name = inp['name']
        inputs.append(inp_name)
        tf_dtypes[inp_name] = getattr(types_pb2, inp['type'])
        out_shapes_attr = func.get('argAttr', {}).get(str(i), {}).get('attr', {}).get('_output_shapes')
        if out_shapes_attr is not None:
            output_shapes[inp_name] = read_tfjs_attr(out_shapes_attr)[0]
        else:
            output_shapes[inp_name] = None
    ret_map = func['ret']
    outputs = [ret_map[out['name']] for out in signature['outputArg']]
    name = signature['name']
    return tf_dtypes, output_shapes, inputs, outputs, name


def read_tfjs_graph(nodes, weights, func=None, graph_inputs=None, graph_outputs=None, shape_override=None,
                    ignore_default=None, use_default=None):
    """Creates an onnx graph from the provided tfjs nodes"""
    if shape_override is None:
        shape_override = {}
    onnx_nodes = []
    output_shapes = {}
    tf_dtypes = {}
    op_info = {}
    graph_name = 'tfjs_model'
    func_name = None

    def update_shapes(new_shapes):
        if isinstance(new_shapes, dict):
            new_shapes = new_shapes.items()
        for k, v in new_shapes:
            output_shapes[k] = shape_override.get(k, v)

    if func is not None:
        tf_dtypes, fn_input_shapes, graph_inputs, graph_outputs, func_name = read_tfjs_function(func)
        update_shapes(fn_input_shapes)
        graph_name = func_name
        for inp in graph_inputs:
            onnx_nodes.append(helper.make_node("Placeholder", [], outputs=[inp], name=inp))

    if graph_inputs is None:
        placeholder_ops = ["Placeholder", "PlaceholderWithDefault", "PlaceholderV2"]
        graph_inputs = [n['name'] + ':0' for n in nodes if n['op'] in placeholder_ops]

    for node in nodes:
        if node['op'] == "NextIteration":
            # NextIteration nodes can violate the topological sort with cyclic dependencies, so we do them first.
            node_name = node['name']
            output_name = node_name + ':0'
            output_shapes[output_name] = None
            tf_dtypes[output_name] = read_tfjs_attr(node['attr']['T'], tf_dtypes=True)
            op_info[node_name] = (node['op'], {'dtype': tf_dtypes[output_name]}, [tf_dtypes[output_name]])

    for node in nodes:
        op_type = node['op']
        node_name = node['name']
        if op_type == "Const":
            np_arr = weights[node_name]
            out_name = node_name + ':0'
            tf_dtype = read_tfjs_attr(node['attr']['dtype'], tf_dtypes=True)
            onnx_dtype = tf_utils.map_tf_dtype(tf_dtype)
            # The dtype of a Const in tfjs can differ from that of the weight used to get its value
            np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
            onnx_tensor = numpy_helper.from_array(np_arr.astype(np_dtype), out_name)
            onnx_node = helper.make_node("Const", [], outputs=[out_name], name=node_name, value=onnx_tensor)
            onnx_nodes.append(onnx_node)
            output_shapes[out_name] = shape_override.get(out_name, list(np_arr.shape))
            tf_dtypes[out_name] = tf_dtype
            op_info[node_name] = (op_type, {'dtype': tf_dtypes[out_name]}, [])
            continue
        tf_attr = {}
        onnx_attr = {}
        fix_string_attr(node)
        node_def = tfjs_node_to_tf_node_def(node)
        for k, v in node.get('attr', {}).items():
            tf_attr[k] = read_tfjs_attr(v, tf_dtypes=True)
            if k in tf_utils.TF_IGNORED_NODE_ATTRS:
                continue
            if k == 'DstT':
                k = 'to'
            onnx_attr[k] = read_tfjs_attr(v)
        if op_type == "FusedDepthwiseConv2dNative":
            # This op isn't in tensorflow but can be converted to a TF op
            op_type = "_FusedDepthwiseConv2dNative"
            err_msg = "explicit_paddings for supported for _FusedDepthwiseConv2dNative"
            if "explicit_paddings" in tf_attr:
                utils.make_sure(len(tf_attr['explicit_paddings']) == 0, err_msg)
                del tf_attr['explicit_paddings']
                del onnx_attr['explicit_paddings']
                del node_def.attr['explicit_paddings']
            node_def.op = op_type

        input_names = [inp for inp in node.get('input', []) if not inp.startswith('^')]
        input_names = [resolve_output(inp, op_info, func_name) for inp in input_names]
        inp_dtypes = [tf_dtypes[inp] for inp in input_names]
        inp_shapes = [output_shapes[inp] for inp in input_names]
        inp_consts = [weights.get(inp.split(':')[0]) for inp in input_names]
        out_dtypes = get_output_dtypes(op_type, tf_attr, inp_dtypes)
        out_shapes = get_output_shapes(node_def, inp_dtypes, inp_shapes, inp_consts)
        op_info[node_name] = (op_type, tf_attr, inp_dtypes)

        output_names = [node_name + ":" + str(i) for i in range(len(out_dtypes))]
        tf_dtypes.update(zip(output_names, out_dtypes))
        update_shapes(zip(output_names, out_shapes))

        if op_type == "PlaceholderWithDefault":
            remove = False
            if ignore_default and node_name in ignore_default:
                op_type = 'Placeholder'
                input_names = []
            elif use_default and node_name in use_default:
                remove = True
            elif node_name.endswith('keras_learning_phase'):
                logger.warning("Removing optional input %s that appears to be a keras learning phase parameter. "
                               "Use --ignore_default to force this into an input.", node_name)
                remove = True
            if remove:
                op_type = 'Identity'
                graph_inputs = [inp for inp in graph_inputs if inp != node_name + ":0"]

        onnx_node = utils.make_onnx_node_with_attr(op_type, input_names, output_names, name=node_name, **onnx_attr)
        onnx_nodes.append(onnx_node)

    for inp in graph_inputs:
        if output_shapes[inp] is None:
            logger.warning("Input %s has unknown shape. Specify shape with --inputs flag.", inp)

    dtypes = {k: tf_utils.map_tf_dtype(v) for k, v in tf_dtypes.items()}
    if graph_outputs is None:
        output_to_node = {out: node.name for node in onnx_nodes for out in node.output}
        node_to_outputs = {node.name: list(node.output) for node in onnx_nodes}
        used_nodes = set(output_to_node[out] for node in onnx_nodes for out in node.input)
        unused_nodes = [node for node in onnx_nodes if node.name not in used_nodes]
        graph_outputs = [out for node in unused_nodes for out in node_to_outputs[node.name]]
    graph_outputs_mapped = [resolve_output(out, op_info, func_name) for out in graph_outputs]

    g = Graph(onnx_nodes, output_shapes, dtypes, input_names=graph_inputs, output_names=graph_outputs_mapped,
              is_subgraph=func is not None, graph_name=graph_name)
    g.rename_tensors(dict(zip(graph_outputs_mapped, graph_outputs)))
    return g
