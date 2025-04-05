# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.tf_utils - misc utilities for tf2onnx that interface with tensorflow
"""

import collections
from packaging.version import Version

import numpy as np
import tensorflow as tf

from tensorflow.core.framework import types_pb2, tensor_pb2, graph_pb2
from tensorflow.python.framework import tensor_util

from onnx import onnx_pb, numpy_helper

from tf2onnx import utils
from tf2onnx.utils import make_sure, is_tf_const_op, port_name, map_onnx_to_numpy_type
from . import logging

logger = logging.getLogger(__name__)

#
#  mapping dtypes from tensorflow to onnx
#
TF_TO_ONNX_DTYPE = {
    types_pb2.DT_FLOAT: onnx_pb.TensorProto.FLOAT,
    types_pb2.DT_HALF: onnx_pb.TensorProto.FLOAT16,
    types_pb2.DT_BFLOAT16: onnx_pb.TensorProto.FLOAT16,
    types_pb2.DT_DOUBLE: onnx_pb.TensorProto.DOUBLE,
    types_pb2.DT_INT32: onnx_pb.TensorProto.INT32,
    types_pb2.DT_INT16: onnx_pb.TensorProto.INT16,
    types_pb2.DT_INT8: onnx_pb.TensorProto.INT8,
    types_pb2.DT_UINT8: onnx_pb.TensorProto.UINT8,
    types_pb2.DT_UINT16: onnx_pb.TensorProto.UINT16,
    types_pb2.DT_UINT32: onnx_pb.TensorProto.UINT32,
    types_pb2.DT_UINT64: onnx_pb.TensorProto.UINT64,
    types_pb2.DT_INT64: onnx_pb.TensorProto.INT64,
    types_pb2.DT_STRING: onnx_pb.TensorProto.STRING,
    types_pb2.DT_COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    types_pb2.DT_COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
    types_pb2.DT_BOOL: onnx_pb.TensorProto.BOOL,
    types_pb2.DT_RESOURCE: onnx_pb.TensorProto.INT64,  # TODO: hack to allow processing on control flow
    types_pb2.DT_VARIANT: onnx_pb.TensorProto.UNDEFINED,
    types_pb2.DT_QUINT8: onnx_pb.TensorProto.UINT8,
}


def tf_to_onnx_tensor(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    np_data = get_tf_tensor_data(tensor)
    if np_data.dtype == object:
        # assume np_data is string, numpy_helper.from_array accepts ndarray,
        # in which each item is of str while the whole dtype is of object.
        try:
            # Faster but fails on Unicode
            np_data = np_data.astype(str).astype(object)
        except UnicodeDecodeError:
            decode = np.vectorize(lambda x: x.decode('UTF-8'))
            np_data = decode(np_data).astype(object)
        except:  # pylint: disable=bare-except
            raise RuntimeError("Not support type: {}".format(type(np_data.flat[0])))
    return numpy_helper.from_array(np_data, name=name)


def get_tf_tensor_data(tensor):
    """Get data from tensor."""
    make_sure(isinstance(tensor, tensor_pb2.TensorProto), "Require TensorProto")
    np_data = tensor_util.MakeNdarray(tensor)
    make_sure(isinstance(np_data, np.ndarray), "%r isn't ndarray", np_data)
    return np_data


def get_tf_const_value(op, as_list=True):
    """
    If as_list=True, return the array as a (possibly nested) list.
    Otherwise, return data of type np.ndarray.

    If a tensor is a scalar having value 1,
        when as_list=False, return np.array(1), type is <class 'numpy.ndarray'>
        when as_list=True, return 1, type is <class 'int'>.
    """
    make_sure(is_tf_const_op(op), "%r isn't a const op", op.name)
    value = get_tf_tensor_data(op.get_attr("value"))
    if as_list:
        value = value.tolist()
    return value


def get_tf_shape_attr(node):
    """Get shape from tensorflow attr "shape"."""
    dims = None
    try:
        shape = get_tf_node_attr(node, "shape")
        if not shape.unknown_rank:
            dims = [int(d.size) for d in shape.dim]
    except:  # pylint: disable=bare-except
        pass
    return dims


def get_tf_tensor_shape(tensor):
    shape = []
    try:
        shape = tensor.get_shape().as_list()
    except Exception:  # pylint: disable=broad-except
        shape = None
    return shape


def map_tf_dtype(dtype):
    if dtype:
        dtype = TF_TO_ONNX_DTYPE[dtype]
    return dtype


def get_tf_node_attr(node, name):
    """Parser TF node attribute."""
    return node.get_attr(name)


def get_tf_version():
    return Version(tf.__version__)

def compress_graph_def(graph_def):
    """
    Remove large const values from graph. This lets us import the graph and run shape inference without TF crashing.
    """
    node_defs = list(graph_def.node)
    const_node_values = {}
    for node_def in node_defs:
        if node_def.op == 'Const':
            tensor = node_def.attr["value"].tensor
            # Small constants are sometimes used to store shape information and must be maintained
            if len(tensor.tensor_content) > 1000:
                make_sure(node_def.name not in const_node_values, "Two nodes in graph have same name %s", node_def.name)
                const_node_values[node_def.name] = tensor.tensor_content
                tensor.tensor_content = b''
    return const_node_values

def get_index_from_strided_slice_of_shape(node, outputs_to_values):
    """Returns the  index of the dimension that the strided slice is reading from the shape node or None"""
    attr_vals = {
        'shrink_axis_mask': 1,
        'ellipsis_mask': 0,
        'begin_mask': 0,
        'new_axis_mask': 0,
        'end_mask': 0
    }
    for a in node.node_def.attr:
        if a in attr_vals:
            i = get_tf_node_attr(node, a)
            if i != attr_vals[a]:
                return None
    i1 = outputs_to_values.get(node.inputs[1].name)
    i2 = outputs_to_values.get(node.inputs[2].name)
    i3 = outputs_to_values.get(node.inputs[3].name)
    if i1 is None or i2 is None or i3 is None:
        return None
    if i1.shape != (1,) or i2.shape != (1,) or i3.shape != (1,):
        return None
    i1, i2, i3 = i1[0], i2[0], i3[0]
    if i1 + 1 != i2 or i3 != 1:
        return None
    return i1

def compute_const_folding_using_tf(g, const_node_values, graph_outputs):
    """Find nodes with constant inputs and compute their values using TF"""
    if const_node_values is None:
        const_node_values = {}
    graph_outputs = set(graph_outputs)
    from tf2onnx.tf_loader import tf_session, tf_placeholder  # pylint: disable=import-outside-toplevel

    ops = g.get_operations()
    outputs_to_values = {}
    outputs_to_dtypes = {}
    outputs_to_shapes = {}
    shape_node_outputs = {}

    def is_small_shape(x):
        return np.product(x) <= 1000

    def is_huge_shape(x):
        return np.product(x) >= 1000000

    for node in ops:
        # Load values of constants. Use const_node_values if possible
        if node.type in ["Const", "ConstV2"]:
            tensor = node.node_def.attr["value"].tensor
            if node.name in const_node_values:
                tensor.tensor_content = const_node_values[node.name]
            outputs_to_values[node.outputs[0].name] = get_tf_tensor_data(tensor)
            outputs_to_dtypes[node.outputs[0].name] = node.outputs[0].dtype
        for out in node.outputs:
            outputs_to_shapes[out.name] = get_tf_tensor_shape(out)

    for node in ops:
        if node.type == "Shape":
            shape = outputs_to_shapes.get(node.inputs[0].name)
            if shape is not None:
                shape_node_outputs[node.outputs[0].name] = shape

    unneeded_outputs = set()
    progress = True
    while progress:
        progress = False
        for node in ops:
            # Find ops with constant inputs and compute their values
            input_names = [i.name for i in node.inputs]
            output_names = [i.name for i in node.outputs]
            if node.type == 'StridedSlice' and input_names[0] in shape_node_outputs \
                                           and output_names[0] not in outputs_to_values \
                                           and output_names[0] not in unneeded_outputs:
                shape = shape_node_outputs[input_names[0]]
                i = get_index_from_strided_slice_of_shape(node, outputs_to_values)
                if i is not None and 0 <= i < len(shape) and shape[i] is not None:
                    np_dtype = map_onnx_to_numpy_type(map_tf_dtype(node.outputs[0].dtype))
                    outputs_to_values[output_names[0]] = np.array(shape[i], dtype=np_dtype)
                    outputs_to_dtypes[node.outputs[0].name] = node.outputs[0].dtype
                    progress = True
            can_fold = node.type not in ['Enter', 'Placeholder', 'PlaceholderWithDefault', 'Switch', 'Merge',
                                         'NextIteration', 'Exit', 'QuantizeAndDequantizeV2', 'QuantizeAndDequantizeV3',
                                         'QuantizeAndDequantizeV4']
            can_fold = can_fold and not node.type.startswith('Random')
            can_fold = can_fold and len(input_names) > 0 and all(inp in outputs_to_values for inp in input_names)
            # We can only fold nodes with a single output
            can_fold = can_fold and len(output_names) == 1 and output_names[0] not in outputs_to_values
            # Skip if value already computed, used, and discarded
            can_fold = can_fold and output_names[0] not in unneeded_outputs and output_names[0] not in graph_outputs
            if can_fold:
                # Make a mini graph containing just the node to fold
                g2 = tf.Graph()
                with g2.as_default():
                    for inp in input_names:
                        tf_placeholder(outputs_to_dtypes[inp], name=inp.split(':')[0])
                    mini_graph_def = g2.as_graph_def()
                    mini_graph_def.node.append(node.node_def)
                g3 = tf.Graph()
                with g3.as_default():
                    feed_dict = {}
                    inp_shapes = []
                    for inp in input_names:
                        inp_np = outputs_to_values[inp]
                        feed_dict[inp] = inp_np
                        inp_shapes.append(inp_np.shape)
                    try:
                        with tf_session() as sess:
                            tf.import_graph_def(mini_graph_def, name='')
                            results = sess.run(output_names, feed_dict=feed_dict)
                        if is_huge_shape(results[0].shape) and all(is_small_shape(inp) for inp in inp_shapes):
                            logger.debug("Skipping folding of node %s since result shape %s is much larger "
                                         "than input shapes %s", node.name, results[0].shape, inp_shapes)
                        else:
                            outputs_to_values[output_names[0]] = results[0]
                            outputs_to_dtypes[output_names[0]] = node.outputs[0].dtype
                            progress = True
                    except Exception:  # pylint: disable=broad-except
                        logger.debug("Could not fold node %s", node.name)
        unneeded_outputs.update(outputs_to_values.keys())
        for node in ops:
            # Mark values we need to keep
            input_names = [i.name for i in node.inputs]
            output_names = [i.name for i in node.outputs]
            if len(output_names) == 1 and output_names[0] in outputs_to_values:
                continue
            for i in input_names:
                if i in unneeded_outputs:
                    unneeded_outputs.remove(i)
        for node in unneeded_outputs:
            # Remove unneeded values to prevent memory usage explosion
            if node in outputs_to_values:
                del outputs_to_values[node]
                del outputs_to_dtypes[node]

    for node in ops:
        # We don't need the constants any more
        if node.type in ["Const", "ConstV2"] and node.outputs[0].name in outputs_to_values:
            del outputs_to_values[node.outputs[0].name]
            del outputs_to_dtypes[node.outputs[0].name]

    logger.info("Computed %d values for constant folding", len(outputs_to_values))
    return outputs_to_values, outputs_to_dtypes

class HashTableInfo:
    def __init__(self, shared_name, key_dtype, val_dtype, resource_input=None):
        self.shared_name = shared_name
        self.key_dtype = key_dtype
        self.val_dtype = val_dtype
        self.resource_input = resource_input

def get_hash_table_info(nodes_or_graph_def):
    """
    Return lists of the shared_names, key_dtypes, and value_dtypes of all hash tables declared in the graph_def
    or list of nodes
    """
    if isinstance(nodes_or_graph_def, graph_pb2.GraphDef):
        nodes = nodes_or_graph_def.node
    else:
        nodes = nodes_or_graph_def
    info = []
    for n in nodes:
        if n.op == "LookupTableFindV2":
            info.append(HashTableInfo(None, n.attr['Tin'].type, n.attr['Tout'].type, n.input[0]))
        if n.op in ["HashTableV2", "MutableHashTableV2"]:
            if all(k in n.attr for k in ['shared_name', 'key_dtype', 'value_dtype']):
                name = n.attr['shared_name'].s
                if name != b'':
                    info.append(HashTableInfo(name, n.attr['key_dtype'].type, n.attr['value_dtype'].type))
    return info

def replace_placeholders_with_tables(graph_def, placeholder_to_table_info):
    """
    Given a graph_def and a map from placeholder names to a tuple of table names, key dtypes, and value dtypes,
    Replaces placeholder ops in the graph_def with HashTableV2 ops
    """
    for n in graph_def.node:
        if n.op == "Placeholder" and n.name in placeholder_to_table_info:
            info = placeholder_to_table_info[n.name]
            for a in list(n.attr):
                del n.attr[a]
            n.op = "HashTableV2"
            n.attr['shared_name'].s = info.shared_name
            n.attr['key_dtype'].type = info.key_dtype
            n.attr['value_dtype'].type = info.val_dtype

def read_tf_node_def_attrs(node_def, input_dtypes, input_shapes):
    """Given a tf node def, returns a dict of attribute names to values"""
    from tf2onnx.tf_loader import tf_session, tf_placeholder  # pylint: disable=import-outside-toplevel
    del node_def.input[:]
    node_def.name = "node"

    # read_tf_node_attrs uses some tf methods that require the node to be loaded into a valid TF graph
    g = tf.Graph()
    with g.as_default():
        for i, (dtype, shape) in enumerate(zip(input_dtypes, input_shapes)):
            inp = "input" + str(i)
            tf_placeholder(dtype, name=inp, shape=shape)
            node_def.input.append(inp)
        mini_graph_def = g.as_graph_def()
        mini_graph_def.node.append(node_def)
    g2 = tf.Graph()
    with g2.as_default():
        with tf_session() as sess:
            tf.import_graph_def(mini_graph_def, name='')
            node = sess.graph.get_operation_by_name("node")
            return read_tf_node_attrs(node)


# ignore the following attributes
TF_IGNORED_NODE_ATTRS = {
    "T", "unknown_rank", "_class", "Tshape", "use_cudnn_on_gpu", "Index", "Tpaddings",
    "TI", "Tparams", "Tindices", "Tlen", "Tdim", "Tin", "dynamic_size", "Tmultiples",
    "Tblock_shape", "Tcrops", "index_type", "Taxis", "U", "maxval",
    "Tout", "Tlabels", "Tindex", "element_shape", "Targmax", "Tperm", "Tcond",
    "T_threshold", "shape_type", "_lower_using_switch_merge",
    "parallel_iterations", "_num_original_outputs", "output_types", "output_shapes",
    "key_dtype", "value_dtype", "Tin", "Tout", "capacity", "component_types", "shapes",
    "Toutput_types", "dense_shapes", "Tdense", "Tsegmentids", "Tshift", "Tnumsegments", "SrcT",
    "Tcomplex", "Treal",    # For RFFT, Tcomplex is ignored because
                            # onnx.helper.make_node fails,
                            # TODO: it should be added back.
}

TF_SUBGRAPH_ATTRS = {
    "body", "cond", "then_branch", "else_branch", "f"
}


def read_tf_node_attrs(node):
    """Given a tf Node, returns a dict of attribute names to values"""
    attr = {}
    attr_cnt = collections.Counter()

    for a in node.node_def.attr:
        attr_cnt[a] += 1
        value = get_tf_node_attr(node, a)
        if a in TF_IGNORED_NODE_ATTRS or a in TF_SUBGRAPH_ATTRS or isinstance(value, tensor_pb2.TensorProto):
            pass
        elif a == "shape":
            shape = get_tf_shape_attr(node)
            if shape is not None:
                attr[a] = shape
        elif a == "DstT":
            attr["to"] = map_tf_dtype(value)
        elif isinstance(value, tf.DType):
            attr[a] = map_tf_dtype(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tf.DType):
            attr[a] = [map_tf_dtype(v) for v in value]
        else:
            attr[a] = get_tf_node_attr(node, a)

    return attr, attr_cnt

def tflist_to_onnx(g, shape_override, const_node_values=None, ignore_default=None, use_default=None):
    """
    Convert the tf-node list into an onnx graph with minimal rewrites so
    we can use the onnx graph as intermediate graph.
    """

    node_list = g.get_operations()
    functions = {}

    # some stats
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}

    # find outputs
    ops = node_list

    # create dict with output to shape mappings
    for node in ops:
        for out in node.outputs:
            shape = shape_override.get(out.name)
            if shape is None:
                shape = get_tf_tensor_shape(out)
            dtypes[out.name] = map_tf_dtype(out.dtype)
            output_shapes[out.name] = shape

    for node in ops:
        attr, new_attr_cnt = read_tf_node_attrs(node)
        attr_cnt += new_attr_cnt
        takeit = True
        op_cnt[node.type] += 1
        for a in node.node_def.attr:
            attr_cnt[a] += 1
            value = get_tf_node_attr(node, a)
            if a == "T":
                if value and not isinstance(value, list):
                    dtypes[node.name] = map_tf_dtype(value)
            elif a in TF_SUBGRAPH_ATTRS:
                input_shapes = [inp.get_shape() for inp in node.inputs]
                nattr = get_tf_node_attr(node, a)
                attr[a] = nattr.name
                functions[nattr.name] = input_shapes
            elif isinstance(value, tensor_pb2.TensorProto):
                if const_node_values and node.name in const_node_values:
                    value.tensor_content = const_node_values[node.name]
                onnx_tensor = tf_to_onnx_tensor(value, name=port_name(node.name))
                attr[a] = onnx_tensor

        node_type = node.type
        input_names = [i.name for i in node.inputs]
        output_names = [i.name for i in node.outputs]

        if node_type == 'PlaceholderWithDefault':
            if ignore_default and node.name in ignore_default:
                node_type = 'Placeholder'
                input_names = []
            elif use_default and node.name in use_default:
                node_type = 'Identity'
            elif node.name.endswith('keras_learning_phase'):
                logger.warning("Removing optional input %s that appears to be a keras learning phase parameter. "
                               "Use --ignore_default to force this into an input.", node.name)
                node_type = 'Identity'

        if takeit:
            try:
                onnx_node = utils.make_onnx_node_with_attr(node_type, input_names, output_names, name=node.name, **attr)
                onnx_nodes.append(onnx_node)
            except Exception as ex:
                logger.error("pass1 convert failed for %s, ex=%s", node, ex)
                raise

    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, functions


def tensorflow_to_onnx(graph, shape_override, const_node_values=None, ignore_default=None, use_default=None):
    """
    Load tensorflow graph and do a conversion.
    """
    return tflist_to_onnx(graph, shape_override, const_node_values, ignore_default, use_default)
