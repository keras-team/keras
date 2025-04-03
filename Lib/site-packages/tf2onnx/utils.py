# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.utils - misc utilities for tf2onnx
"""

import os
import collections
import inspect
import re
import shutil
import tempfile
import types
import zipfile
import logging

from typing import Any, Optional, Sequence
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from google.protobuf import text_format
from onnx import helper, onnx_pb, defs, numpy_helper, AttributeProto, ModelProto, NodeProto, __version__
from . import constants


logger = logging.getLogger(__file__)


# pylint: disable=unexpected-keyword-arg


#
# mapping dtypes from onnx to numpy
#
ONNX_TO_NUMPY_DTYPE = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.FLOAT16: np.float16,
    onnx_pb.TensorProto.DOUBLE: np.float64,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.INT16: np.int16,
    onnx_pb.TensorProto.INT8: np.int8,
    onnx_pb.TensorProto.UINT8: np.uint8,
    onnx_pb.TensorProto.UINT16: np.uint16,
    onnx_pb.TensorProto.UINT32: np.uint32,
    onnx_pb.TensorProto.UINT64: np.uint64,
    onnx_pb.TensorProto.INT64: np.int64,
    onnx_pb.TensorProto.UINT64: np.uint64,
    onnx_pb.TensorProto.BOOL: bool,
    onnx_pb.TensorProto.COMPLEX64: np.complex64,
    onnx_pb.TensorProto.COMPLEX128: np.complex128,
    onnx_pb.TensorProto.STRING: object,
}

#
#  onnx dtype names
#
ONNX_DTYPE_NAMES = {
    onnx_pb.TensorProto.FLOAT: "float",
    onnx_pb.TensorProto.FLOAT16: "float16",
    onnx_pb.TensorProto.DOUBLE: "double",
    onnx_pb.TensorProto.INT32: "int32",
    onnx_pb.TensorProto.INT16: "int16",
    onnx_pb.TensorProto.INT8: "int8",
    onnx_pb.TensorProto.UINT8: "uint8",
    onnx_pb.TensorProto.UINT16: "uint16",
    onnx_pb.TensorProto.UINT32: "uint32",
    onnx_pb.TensorProto.UINT64: "uint64",
    onnx_pb.TensorProto.INT64: "int64",
    onnx_pb.TensorProto.STRING: "string",
    onnx_pb.TensorProto.BOOL: "bool",
    onnx_pb.TensorProto.COMPLEX64: "complex64",
    onnx_pb.TensorProto.COMPLEX128: "complex128"
}


class TensorValueInfo(object):
    def __init__(self, tensor_id, g):
        self.id = tensor_id
        if self.id:
            self.dtype = g.get_dtype(tensor_id)
            self.shape = g.get_shape(tensor_id)


ONNX_UNKNOWN_DIMENSION = -1
ONNX_EMPTY_INPUT = ""

# index for internally generated names
INTERNAL_NAME = 1

# Fake onnx op type which is used for Graph input.
GRAPH_INPUT_TYPE = "NON_EXISTENT_ONNX_TYPE"


def make_name(name):
    """Make op name for inserted ops."""
    global INTERNAL_NAME
    INTERNAL_NAME += 1
    return "{}__{}".format(name, INTERNAL_NAME)


def split_nodename_and_shape(name):
    """input name with shape into name and shape."""
    # pattern for a node name
    inputs = []
    shapes = {}
    # input takes in most cases the format name:0, where 0 is the output number
    # in some cases placeholders don't have a rank which onnx can't handle so we let uses override the shape
    # by appending the same, ie : [1,28,28,3]
    name_pattern = r"(?:([\w\d/\-\._:]+)(\[[\-\d,]+\])?),?"
    splits = re.split(name_pattern, name)
    for i in range(1, len(splits), 3):
        inputs.append(splits[i])
        if splits[i + 1] is not None:
            shape = [int(n) for n in splits[i + 1][1:-1].split(",")]
            shape = [n if n >= 0 else None for n in shape]
            shapes[splits[i]] = shape
    if not shapes:
        shapes = None
    return inputs, shapes


def map_numpy_to_onnx_dtype(np_dtype):
    for onnx_dtype, numpy_dtype in ONNX_TO_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported numpy dtype '%s' for mapping to onnx" % np_dtype)


def map_onnx_to_numpy_type(onnx_type):
    return ONNX_TO_NUMPY_DTYPE[onnx_type]


def node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def make_onnx_shape(shape):
    """shape with -1 is not valid in onnx ... make it a name."""
    if shape:
        # don't do this if input is a scalar
        return [make_name("unk") if i == -1 else i for i in shape]
    return shape


def port_name(name, nr=0):
    """Map node output number to name."""
    return name + ":" + str(nr)

class SeqType:
    """Wrap around TensorProto.* to signify a tensor sequence of a given type"""
    def __init__(self, tensor_dtype):
        self.dtype = tensor_dtype

    def __eq__(self, other):
        if isinstance(other, SeqType):
            return self.dtype == other.dtype
        return NotImplemented

    def __repr__(self):
        return "SeqType(%r)" % self.dtype

def make_onnx_inputs_outputs(name, elem_type, shape, **kwargs):
    """Wrapper for creating onnx graph inputs or outputs
       name,  # type: Text
       elem_type,  # type: TensorProto.DataType
       shape,  # type: Optional[Sequence[int]]
    """
    if elem_type is None:
        elem_type = onnx_pb.TensorProto.UNDEFINED
    elif isinstance(elem_type, SeqType):
        return helper.make_tensor_sequence_value_info(name, elem_type.dtype, make_onnx_shape(shape), **kwargs)
    return helper.make_tensor_value_info(
        name,
        elem_type,
        make_onnx_shape(shape),
        **kwargs
    )

_attr_type_in_signature = inspect.signature(helper.make_attribute).parameters.get("attr_type", None) is not None


def make_onnx_node_with_attr(op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: Optional[str] = None,
                             domain: Optional[str] = None, **kwargs: Any) -> NodeProto:
    """
    Since ONNX 1.15.0, helper.make_attribute() does not support empty iterators.
    But tf2onnx will leverage ONNX attributes to transfer some extra data along with the ONNX node
    across different conversion stages.
    This function removes empty lists from kwargs and adds them back with attr_type=INTS attributes by default.
    """
    if _attr_type_in_signature:
        attr_empty_lists = {}
        valid_attrs = {}
        if kwargs:
            for key, value in sorted(kwargs.items()):
                if not isinstance(value, bytes) and \
                    isinstance(value, collections.abc.Sequence) and len(list(value)) == 0:
                    attr_empty_lists[key] = value
                else:
                    valid_attrs[key] = value

        onnx_node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **valid_attrs)

        if attr_empty_lists:
            for key, value in attr_empty_lists.items():
                onnx_node.attribute.extend([helper.make_attribute(key, value, attr_type=AttributeProto.INTS)])
    else:
        onnx_node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **kwargs)

    return onnx_node


def find_opset(opset):
    """Find opset."""
    if opset is None or opset == 0:
        opset = defs.onnx_opset_version()
        if opset > constants.PREFERRED_OPSET:
            # if we use a newer onnx opset than most runtimes support, default to the one most supported
            opset = constants.PREFERRED_OPSET
    return opset


def get_subgraphs_from_onnx(model_proto):
    """Returns an iterator over the graphs/subgraphs of a model (using dfs)"""
    stack = [model_proto.graph]
    while stack:
        g = stack.pop()
        yield g
        for node in g.node:
            for attr in node.attribute:
                if hasattr(attr, "g"):
                    stack.append(attr.g)
                if hasattr(attr, "graphs"):
                    stack.extend(attr.graphs)


def initialize_name_counter(model_proto):
    """Avoid name conflicts by initializing the counter used by make_name based on the provided model"""
    suffix_regex = re.compile(r"__(\d+)(:\d+)?$")
    def avoid_name(name):
        global INTERNAL_NAME
        suffix = suffix_regex.search(name)
        if suffix:
            INTERNAL_NAME = max(INTERNAL_NAME, int(suffix.group(1)) + 1)
    for g in get_subgraphs_from_onnx(model_proto):
        for n in g.node:
            avoid_name(n.name)
            for out in n.output:
                avoid_name(out)


def save_onnx_model(save_path_root, onnx_file_name, feed_dict, model_proto, include_test_data=False, as_text=False,
                    external_tensor_storage=None):
    """Save onnx model as file. Save a pbtxt file as well if as_text is True"""
    save_path = save_path_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if include_test_data:
        data_path = os.path.join(save_path, "test_data_set_0")
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        i = 0
        for data_key in feed_dict:
            data = feed_dict[data_key]
            t = numpy_helper.from_array(data)
            t.name = data_key
            data_full_path = os.path.join(data_path, "input_" + str(i) + ".pb")
            save_protobuf(data_full_path, t)
            i += 1

    if external_tensor_storage is None:
        target_path = os.path.join(save_path, onnx_file_name + ".onnx")
        save_protobuf(target_path, model_proto)
    else:
        zip_path = os.path.join(save_path, onnx_file_name + ".zip")
        save_onnx_zip(zip_path, model_proto, external_tensor_storage)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(save_path)
        target_path = os.path.join(save_path, "__MODEL_PROTO.onnx")

    if as_text:
        save_protobuf(target_path + ".pbtxt", model_proto, as_text=True)

    return target_path


def save_onnx_zip(target_path, model_proto, external_tensor_storage):
    with zipfile.ZipFile(target_path, 'w') as z:
        z.writestr("__MODEL_PROTO.onnx", model_proto.SerializeToString())
        for k, v in external_tensor_storage.name_to_tensor_data.items():
            z.writestr(k, v)


def make_sure(bool_val, error_msg, *args):
    if not bool_val:
        raise ValueError("make_sure failure: " + error_msg % args)


def combine_seeds(seed, seed2):
    """Produces an onnx float seed from two tf int seeds. Returns None if both seeds are 0."""
    if seed != 0 or seed2 != 0:
        # Produce a unique value depending on both seeds. (diagonal grid traversal)
        combined_seed = (seed + seed2 + 1) * (seed + seed2 + 2) // 2 - seed
        return float(combined_seed)
    return None


def topological_sort(dependencies):
    """
    Given a dictionary mapping items to lists of dependencies, returns a topological ordering of the items.
    Raises a ValueError for cyclic dependencies.
    """
    stack = list(dependencies.keys())
    visiting = set()
    visited = set()
    ordered = []
    while stack:
        x = stack.pop()
        if x in visited:
            continue
        if x in visiting:
            visiting.remove(x)
            visited.add(x)
            ordered.append(x)
            continue
        stack.append(x)
        visiting.add(x)
        for y in dependencies[x]:
            if y in visiting:
                raise ValueError("Cyclic dependencies present: %r" % dependencies)
            if y not in visited:
                stack.append(y)
    return ordered


def check_io(input_names, output_names, valid_outputs):
    """Asserts that input_names and output_names are contained within valid_outputs else raises an error"""
    io_to_check = []
    if input_names:
        io_to_check.extend(input_names)
    if output_names:
        io_to_check.extend(output_names)
    if io_to_check:
        # check output existence in case user passed in wrong output ids
        non_exists = set(io_to_check) - set(valid_outputs)
        if non_exists:
            logger.error("\nFailed to convert: inputs/outputs specified do not exist, make sure your passed"
                         "in format: input/output_node_name:port_id. Problematic inputs/outputs are: %s \n",
                         non_exists)
            raise ValueError("Inputs/Outputs Not Found")


def is_cpp_protobuf():
    return isinstance(ModelProto().ParseFromString, types.BuiltinFunctionType)


def construct_graph_from_nodes(parent_g, nodes, outputs, shapes, dtypes):
    """Construct Graph from nodes and outputs with specified shapes and dtypes."""
    # pylint: disable=protected-access
    g = parent_g.create_new_graph_with_same_config()
    g.parent_graph = parent_g
    nodes = set(nodes)
    all_outputs = set()
    for op in nodes:
        all_outputs |= set(op.output)

        branches = {}
        body_graphs = op.graph.contained_graphs.pop(op.name, None)
        if body_graphs:
            for attr_name, body_graph in body_graphs.items():
                body_graph.parent_graph = g
                branches[attr_name] = body_graph

        _ = g.make_node(op.type, op.input, outputs=op.output, attr=op.attr, name=op.name,
                        skip_conversion=op.skip_conversion, infer_shape_dtype=False, branches=branches)

    for i in all_outputs:
        if i not in g._output_shapes:
            g._output_shapes[i] = parent_g._output_shapes[i]
        if i not in g._dtypes:
            g._dtypes[i] = parent_g._dtypes[i]

    # handle cell graph: insert identity node, since sometimes we need output same output_id
    # as state_output and scan_out, but ONNX don't allow the same output_id to appear more
    # than once as output node.
    new_output_names = []
    for output, shape, dtype in zip(outputs, shapes, dtypes):
        node = g.make_node("Identity", inputs=[output], op_name_scope="sub_graph_ending_node",
                           shapes=[shape], dtypes=[dtype], infer_shape_dtype=False)
        new_output_names.append(node.output[0])
    g.outputs = new_output_names
    return g


def tf_name_scope(name):
    return '/'.join(name.split('/')[:-1])


def get_temp_directory():
    return os.environ.get("TF2ONNX_TEMP_DIRECTORY", tempfile.mkdtemp())


def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_protobuf(path, message, as_text=False):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())

def model_proto_from_file(model_path):
    model_proto = ModelProto()
    with open(model_path, "rb") as f:
        model_proto.ParseFromString(f.read())
    return model_proto

def model_proto_from_zip(zip_path, external_tensor_storage):
    model_proto = ModelProto()
    with zipfile.ZipFile(zip_path, 'r') as z:
        for n in z.namelist():
            f = z.open(n)
            if n.endswith(".onnx"):
                model_proto.ParseFromString(f.read())
            else:
                external_tensor_storage.name_to_tensor_data[n] = f.read()
    return model_proto

def is_list_or_tuple(obj):
    return isinstance(obj, (list, tuple))


def is_unknown_dimension(dim):
    """  Return true if dim is not a positive integer value. """
    if dim is None or not isinstance(dim, int):
        return True
    return dim <= 0


def merge_shapes(shape1, shape2):
    """
    Merge 2 shapes, return merged shape, choose more specific dimension value from either side.
    Raise exception for mismatch.
    """
    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1

    make_sure(is_list_or_tuple(shape1), "invalid type for shape1")
    make_sure(is_list_or_tuple(shape2), "invalid type for shape2")
    make_sure(len(shape1) == len(shape2), "shapes rank mismatch: shape1=%s, shape2=%s", shape1, shape2)

    merged = []
    for d1, d2 in zip(shape1, shape2):
        d = d1
        if is_unknown_dimension(d1):
            d = d2
        elif not is_unknown_dimension(d2):
            make_sure(d1 == d2, "shapes dimension mismatch: shape1=%s, shape2=%s", shape1, shape2)
        merged.append(d)
    return merged


def are_shapes_compatible(src, dest):
    """
    Returns True iff src is compatible with dest.
    None is compatible with all shapes, different ranks are not considered as compatible
    """
    try:
        merge_shapes(src, dest)
        return True
    except:  # pylint: disable=bare-except
        return False


def are_shapes_equal(src, dest):
    """ Check whether 2 shapes are equal. """
    if src is None:
        return dest is None
    if dest is None:
        return src is None

    make_sure(is_list_or_tuple(src), "invalid type for src")
    make_sure(is_list_or_tuple(dest), "invalid type for dest")

    if len(src) != len(dest):
        return False
    return all(i == j for i, j in zip(src, dest))


def create_vague_shape_like(shape):
    make_sure(len(shape) >= 0, "rank should be >= 0")
    return [-1 for i in enumerate(shape)]


def get_onnx_version():
    return __version__


def make_opsetid(domain, version):
    make_sure(isinstance(version, int), "version must be an integer")
    return helper.make_opsetid(domain, version)


def is_onnx_domain(domain):
    if domain is None or domain == "":
        return True
    return False


def parse_bool(val):
    if val is None:
        return False
    return val.lower() in ("yes", "true", "t", "y", "1")


_is_debug_mode = parse_bool(os.environ.get(constants.ENV_TF2ONNX_DEBUG_MODE))


def is_debug_mode():
    return _is_debug_mode


def set_debug_mode(enabled):
    global _is_debug_mode
    _is_debug_mode = enabled


def get_max_value(np_dtype):
    return np.iinfo(np_dtype).max


def get_min_value(np_dtype):
    return np.iinfo(np_dtype).min


def get_url(url, path, max_retries=5):
    """ Download url and save to path. """
    retries = Retry(total=max_retries, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    response = session.get(url, allow_redirects=True)
    if response.status_code not in [200]:
        response.raise_for_status()

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "wb") as f:
        f.write(response.content)


def have_same_inference_value(g, output_1, output_2):
    """
    If two outputs have the same value in inference.
    Check whether they come from the same subgraph and the same subgraphs
    contain nodes with the same attributes and share the same ancestors.
    """

    def is_same(node_1, node_2):
        # go further util two instance isn't the same
        if node_1 == node_2:
            return True
        # check body graph
        if node_1.get_body_graphs() or node_2.get_body_graphs():
            logger.warning("Comparing two nodes containing body graph isn't supported.")
            return False
        # check domain
        if node_1.domain != node_2.domain:
            return False
        # check type
        if node_1.type != node_2.type:
            return False
        # check onnx attributes
        if node_1.get_onnx_attrs().keys() != node_2.get_onnx_attrs().keys():
            return False
        for name in node_1.get_onnx_attrs().keys(): # pylint: disable=consider-iterating-dictionary
            if node_1.get_attr_value(name) != node_2.get_attr_value(name):
                return False
        return True

    if output_1 == output_2:
        return True
    node_1 = g.get_node_by_output(output_1)
    node_2 = g.get_node_by_output(output_2)
    # compare their domain, attr, etc. see __eq__ in Node class
    if not is_same(node_1, node_2):
        return False

    for inp_1, inp_2 in zip(node_1.input, node_2.input):
        if not have_same_inference_value(g, inp_1, inp_2):
            return False
    return True


def is_tf_reverse_op(op):
    return op.type in ("ReverseV2", "ReverseSequence")


def is_tf_concat_op(op):
    return op.type in ("Concat", "ConcatV2", "ConcatV3")


def is_tf_tensor_array_gather_op(op):
    return op.type in ("TensorArrayGatherV2", "TensorArrayGatherV3")


def is_tf_tensor_array_write_op(op):
    return op.type in ("TensorArrayWriteV2", "TensorArrayWriteV3")

def is_tf_tensor_array_read_op(op):
    return op.type in ("TensorArrayReadV2", "TensorArrayReadV3")


def is_tf_tensor_array_op(op):
    return op.type in ("TensorArrayV2", "TensorArrayV3")


def is_tf_loopcond_op(op):
    return op.type == "LoopCond"


def is_tf_select_op(op):
    return op.type in ("Select", "SelectV2")


def is_tf_slice_op(op):
    return op.type == "Slice"


def is_tf_const_op(op):
    return op.type in ["Const", "ConstV2"]
