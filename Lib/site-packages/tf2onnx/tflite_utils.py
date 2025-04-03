# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.tflite_utils - utilities for parsing tflite files into onnx graph
"""

import collections
import importlib
import logging
import struct

from onnx import helper, onnx_pb, numpy_helper
from tensorflow.core.framework import types_pb2, tensor_pb2, node_def_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import numpy as np
from tf2onnx.tflite.TensorType import TensorType as TFLiteTensorType
from tf2onnx.tflite.Model import Model
from tf2onnx.flexbuffers import read_flexbuffer
from tf2onnx.tf_utils import read_tf_node_def_attrs
from tf2onnx.graph import Graph
from tf2onnx import utils

logger = logging.getLogger(__name__)

TFLITE_TO_ONNX_DTYPE = {
    TFLiteTensorType.FLOAT32: onnx_pb.TensorProto.FLOAT,
    TFLiteTensorType.FLOAT16: onnx_pb.TensorProto.FLOAT16,
    TFLiteTensorType.INT32: onnx_pb.TensorProto.INT32,
    TFLiteTensorType.UINT8: onnx_pb.TensorProto.UINT8,
    TFLiteTensorType.INT64: onnx_pb.TensorProto.INT64,
    TFLiteTensorType.STRING: onnx_pb.TensorProto.STRING,
    TFLiteTensorType.BOOL: onnx_pb.TensorProto.BOOL,
    TFLiteTensorType.INT16: onnx_pb.TensorProto.INT16,
    TFLiteTensorType.COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    TFLiteTensorType.INT8: onnx_pb.TensorProto.INT8,
    TFLiteTensorType.FLOAT64: onnx_pb.TensorProto.DOUBLE,
    TFLiteTensorType.COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
    TFLiteTensorType.UINT64: onnx_pb.TensorProto.UINT64,
    TFLiteTensorType.UINT32: onnx_pb.TensorProto.UINT32,
    TFLiteTensorType.UINT16: onnx_pb.TensorProto.UINT16,
    TFLiteTensorType.RESOURCE: onnx_pb.TensorProto.UNDEFINED,
    TFLiteTensorType.VARIANT: onnx_pb.TensorProto.UNDEFINED,
}


TFLITE_TO_TF_DTYPE = {
    TFLiteTensorType.FLOAT32: types_pb2.DT_FLOAT,
    TFLiteTensorType.FLOAT16: types_pb2.DT_HALF,
    TFLiteTensorType.INT32: types_pb2.DT_INT32,
    TFLiteTensorType.UINT8: types_pb2.DT_UINT8,
    TFLiteTensorType.INT64: types_pb2.DT_INT64,
    TFLiteTensorType.STRING: types_pb2.DT_STRING,
    TFLiteTensorType.BOOL: types_pb2.DT_BOOL,
    TFLiteTensorType.INT16: types_pb2.DT_INT16,
    TFLiteTensorType.COMPLEX64: types_pb2.DT_COMPLEX64,
    TFLiteTensorType.INT8: types_pb2.DT_INT8,
    TFLiteTensorType.FLOAT64: types_pb2.DT_DOUBLE,
    TFLiteTensorType.COMPLEX128: types_pb2.DT_COMPLEX128,
    TFLiteTensorType.UINT64: types_pb2.DT_UINT64,
    TFLiteTensorType.UINT32: types_pb2.DT_UINT32,
    TFLiteTensorType.UINT16: types_pb2.DT_UINT16,
    TFLiteTensorType.RESOURCE: types_pb2.DT_RESOURCE,
    TFLiteTensorType.VARIANT: types_pb2.DT_VARIANT,
}


def map_tflite_dtype_to_onnx(dtype):
    return TFLITE_TO_ONNX_DTYPE[dtype]


def map_tflite_dtype_to_tf(dtype):
    return TFLITE_TO_TF_DTYPE[dtype]


# The tflite schema uses snake case, but the python bindings use proper case
def snake_to_proper_case(name):
    return ''.join(n.capitalize() for n in name.split('_'))


def proper_to_snake_case(name):
    res = ''
    for c in name:
        if c.isupper() and res:
            res += '_'
        res += c.lower()
    return res

# Pulled from the tflite schema.fbs file. Needed to decode enum numbers into strings.
NODE_ATTR_NAME_TO_ENUM_TYPE = {
    'fused_activation_function': 'ActivationFunctionType',
    'padding': 'Padding',
    'type': 'LSHProjectionType',
    'weights_format': 'FullyConnectedOptionsWeightsFormat',
    'kernel_type': 'LSTMKernelType',
    'combiner': 'CombinerType',
    'in_data_type': 'TensorType',
    'out_data_type': 'TensorType',
    'output_type': 'TensorType',
    'out_type': 'TensorType',
    'mode': 'MirrorPadMode',
    'idx_out_type': 'TensorType',
}
NODE_ATTR_NAME_TO_ENUM_TYPE = {snake_to_proper_case(key): value for key, value in NODE_ATTR_NAME_TO_ENUM_TYPE.items()}

# Pulled from the tflite schema.fbs file.
FUNCTION_ATTRS = ['then_subgraph_index', 'else_subgraph_index', 'cond_subgraph_index',
                  'body_subgraph_index', 'subgraph']
FUNCTION_ATTRS = [snake_to_proper_case(attr) for attr in FUNCTION_ATTRS]


enum_cache = {}
def lookup_enum(idx, enum_name):
    """Given the name of a tflite enum class and an index, return a string with the name of the enum value"""
    if enum_name == 'TensorType':
        return map_tflite_dtype_to_onnx(idx)
    if enum_name in enum_cache:
        idx_to_name = enum_cache[enum_name]
    else:
        module = importlib.import_module('tf2onnx.tflite.' + enum_name)
        enum_class = getattr(module, enum_name)
        idx_to_name = {value: key for key, value in enum_class.__dict__.items() if not key.startswith('_')}
        enum_cache[enum_name] = idx_to_name
    utils.make_sure(idx in idx_to_name, "Can't lookup value %s for tflite enum %s. Please update tf2onnx or "
                    "submit an issue on GitHub.", idx, enum_name)
    return idx_to_name[idx]


def get_options_class(name):
    """Each tflite optype has a flatbuffer Options class (ex: AddOptions). Returns the options class given its name."""
    if name == "NONE":
        return None
    module = importlib.import_module('tf2onnx.tflite.' + name)
    return getattr(module, name)


def graphs_from_tflite(tflite_path, input_names=None, output_names=None):
    """
    Given the path to a tflite model, returns a tuple (main_graph, subgraphs) of graph.py Graph objects
    inputs/outputs will be taken from main graph in model if not overridden
    """
    tflite_graphs, opcodes, model, tensor_shapes = read_tflite_model(tflite_path)
    main_g = None
    subgraphs = []
    for i, tfl_graph in enumerate(tflite_graphs):
        is_main_g = i == len(tflite_graphs) - 1
        prefix = '' if is_main_g else tfl_graph.Name().decode() + '_'
        tensor_shapes_from_interpreter = None
        if is_main_g:
            tensor_shapes_from_interpreter = tensor_shapes
        onnx_nodes, _, _, output_shapes, dtypes, f_inputs, f_outputs, graph_name = \
            parse_tflite_graph(tfl_graph, opcodes, model, prefix, tensor_shapes_from_interpreter)
        g_inputs = f_inputs
        g_outputs = f_outputs
        if is_main_g:
            # Override IO in main graph
            utils.check_io(input_names, output_names, output_shapes.keys())
            if input_names:
                g_inputs = input_names
            if output_names:
                g_outputs = output_names
        g = Graph(onnx_nodes, output_shapes, dtypes, input_names=g_inputs, output_names=g_outputs,
                  is_subgraph=not is_main_g, graph_name=graph_name)
        if is_main_g:
            main_g = g
        else:
            subgraphs.append(g)
    return main_g, subgraphs


def read_tflite_model(tflite_path):
    """
    Given the path to a tflite model, returns tuple (tflite_graphs, opcodes_map, model)
    Graphs are topologically sorted and the main graph is last
    Pass these to parse_tflite_graph
    """
    with open(tflite_path, 'rb') as f:
        buf = f.read()
    buf = bytearray(buf)
    model = Model.GetRootAsModel(buf, 0)
    # To save space, each op in the model indicates its opcode as an index into the model's opcode map.
    opcodes_map = {}
    for i in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(i)
        # TFlite ran out of opcodes since they only used a byte. Old models store opcodes in DeprecatedBuiltinCode.
        # New models put PLACEHOLDER_FOR_GREATER_OP_CODES in this field to signify that BuiltinCode should be used.
        code = lookup_enum(op_code.DeprecatedBuiltinCode(), 'BuiltinOperator')
        if code == 'PLACEHOLDER_FOR_GREATER_OP_CODES':
            code = lookup_enum(op_code.BuiltinCode(), 'BuiltinOperator')
        if code == 'CUSTOM':
            code = op_code.CustomCode().decode()
        opcodes_map[i] = code
    # Shapes stored in tflite models are not always reliable so we get them from the interpreter if possible.
    tensor_shapes = {}
    try:
        interpreter = tf.lite.Interpreter(tflite_path)
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()

        for tensor_detail in tensor_details:
            name = tensor_detail.get('name')
            if "shape_signature" in tensor_detail:
                tensor_shapes[name] = tensor_detail["shape_signature"].tolist()
            elif "shape" in tensor_detail:
                tensor_shapes[name] = tensor_detail["shape"].tolist()
    except Exception as e:    # pylint: disable=broad-except
        logger.warning("Error loading model into tflite interpreter: %s", e)
    tflite_graphs = get_model_subgraphs(model)
    return tflite_graphs, opcodes_map, model, tensor_shapes


def get_subgraph_dependencies(model, graph_idx):
    """Returns a list of subgraph indices referenced by the indicated graph"""
    dependencies = []
    g = model.Subgraphs(graph_idx)
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        options_type_name = lookup_enum(op.BuiltinOptionsType(), 'BuiltinOptions')
        option_class = get_options_class(options_type_name)
        if option_class is not None:
            options = option_class()
            options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            for attr in FUNCTION_ATTRS:
                if hasattr(options, attr):
                    value = getattr(options, attr)()
                    dependencies.append(value)
    return dependencies


def get_model_subgraphs(model):
    """Returns topologically sorted subgraphs of a model. Guarantees main graph is placed at the end."""
    main_g = 0
    dependencies = {}
    idx_to_graph = {}
    for i in range(model.SubgraphsLength()):
        idx_to_graph[i] = model.Subgraphs(i)
        ds = get_subgraph_dependencies(model, i)
        utils.make_sure(main_g not in ds, "Main graph %s is a dependency of subgraph %s", main_g, i)
        dependencies[i] = ds

    ordered = utils.topological_sort(dependencies)

    return [idx_to_graph[i] for i in ordered]


def get_quantization_attr(quant_params):
    attr = {}
    attr['scale'] = quant_params.ScaleAsNumpy().tolist()
    attr['zero_point'] = quant_params.ZeroPointAsNumpy().tolist()
    attr['quantized_dimension'] = quant_params.QuantizedDimension()
    if not quant_params.MaxIsNone():
        attr['max'] = quant_params.MaxAsNumpy().tolist()
    if not quant_params.MinIsNone():
        attr['min'] = quant_params.MinAsNumpy().tolist()
    return attr


def parse_tflite_string_tensor(buffer_bytes, shape):
    """Returns an onnx tensor with the string data encoded in the tflite tensor data buffer"""
    def read_int(offset):
        return struct.unpack('<i', buffer_bytes[offset:offset+4])[0]
    offset = 0
    count = read_int(offset)
    offset += 4
    offset_list = []
    for i in range(count):
        offset_list.append(read_int(offset))
        offset += 4
    offset_list.append(len(buffer_bytes))
    string_list = []
    for i in range(count):
        string_list.append(buffer_bytes[offset_list[i]:offset_list[i+1]].decode("utf-8"))
    return numpy_helper.from_array(np.array(string_list, dtype=object).reshape(shape))


def op_has_scalar_output(input_shapes, optype, attr):
    """
    TFLite uses [] to denote both scalars and unknown output shapes. Return True if an op can have scalar outputs
    despite having non-scalar inputs. Otherwise, we will replace [] with None
    """
    if optype in ["TFL_STRIDED_SLICE", "StridedSlice"]:
        inp_rank = len(input_shapes[0])
        return attr['shrink_axis_mask'] == 2 ** inp_rank - 1
    if (optype.startswith("TFL_REDUCE") or optype in ['All']) and len(input_shapes) == 2:
        inp_rank = len(input_shapes[0])
        keep_dims = attr.get('keep_dims', True)
        # axes input can be a scalar for a single axis
        num_axes = 1 if input_shapes[1] == [] else input_shapes[1][0]
        return not keep_dims and inp_rank == num_axes
    if optype == "TFL_RESHAPE":
        return input_shapes[1] == [0]
    if optype == "Size":
        # Op from TF
        return True
    return False


def parse_tflite_graph(tflite_g, opcodes_map, model, input_prefix='', tensor_shapes_override=None):
    """
    Returns a Graph object along with some op count stats. All tflite op types are prefixed with "TFL_".
    Names of graph inputs are optionally prefixed with a string to prevent name conflicts in subgraphs.
    Quantizatized tensors are surrounded with quantize/dequantize ops
    """
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}
    tensor_names = {}
    if tensor_shapes_override is None:
        tensor_shapes_override = {}
    # Map tensor name to tflite Tensor object so we can fetch quantization info as needed
    name_to_tensor = {}
    # If a node takes a quantized tensor as input, we must add a dequantize op after it.
    # Store a mapping so we only need to make at most one dequantize op per tensor.
    tensor_name_to_dequant_output = {}

    # tflite uses generic names (arg0, arg1, etc.) for inputs but full names for other tensors, so
    # prefixing just the inputs should be fine. Other tensors are prefixed when we do inlining.
    input_indices = {tflite_g.Inputs(i) for i in range(tflite_g.InputsLength())}

    for i in range(tflite_g.TensorsLength()):
        tensor = tflite_g.Tensors(i)
        name = tensor.Name().decode()
        if i in input_indices:
            name = input_prefix + name
        tensor_names[i] = name
        name_to_tensor[name] = tensor

        if name in tensor_shapes_override:
            output_shapes[name] = tensor_shapes_override[name]
        elif tensor.ShapeIsNone():
            output_shapes[name] = None
        elif tensor.ShapeSignatureIsNone():
            # The shape signature uses -1 to signify unknown dims. Old models don't have this and use Shape instead.
            output_shapes[name] = tensor.ShapeAsNumpy().tolist()
        else:
            output_shapes[name] = tensor.ShapeSignatureAsNumpy().tolist()
        buf = model.Buffers(tensor.Buffer())
        dtypes[name] = map_tflite_dtype_to_onnx(tensor.Type())
        if not buf.DataIsNone() and tensor.Buffer() > 0:
            # For const values we use TF to decode the binary data from the buffer
            t = tensor_pb2.TensorProto()
            t.tensor_content = buf.DataAsNumpy().tobytes()
            if output_shapes[name] is None:
                output_shapes[name] = []
            for d in output_shapes[name]:
                t.tensor_shape.dim.add().size = d
            t.dtype = map_tflite_dtype_to_tf(tensor.Type())
            if t.dtype == tf.string:
                onnx_tensor = parse_tflite_string_tensor(t.tensor_content, output_shapes[name])
            else:
                np_data = tensor_util.MakeNdarray(t)
                onnx_tensor = numpy_helper.from_array(np_data, name=name)
            onnx_node = helper.make_node("Const", [], outputs=[name], name=name, value=onnx_tensor)
            onnx_nodes.append(onnx_node)
            op_cnt["Const"] += 1

    def get_dequant(tensor_name):
        """Creates a dequantize op for the provided tensor if needed and returns the output of the op, or
        the original tensor name if no dequantization is needed"""
        quant = name_to_tensor[tensor_name].Quantization()
        if quant is None or quant.ScaleIsNone() or quant.ZeroPointIsNone():
            return tensor_name
        if tensor_name in tensor_name_to_dequant_output:
            return tensor_name_to_dequant_output[tensor_name]
        dequant_name = tensor_name + "_dequant"
        attr = get_quantization_attr(quant)
        onnx_node = helper.make_node("TFL_DEQUANTIZE", [tensor_name], [dequant_name], name=dequant_name, **attr)
        onnx_nodes.append(onnx_node)
        tensor_name_to_dequant_output[tensor_name] = dequant_name
        output_shapes[dequant_name] = output_shapes[tensor_name].copy()
        dtypes[dequant_name] = onnx_pb.TensorProto.FLOAT
        return dequant_name

    def get_prequant(tensor_name):
        """Called by nodes with the name of the tensor they must output.
        If the output is supposed to be quantized, creates a Quantize op outputting the tensor.
        Returns the name that should be used for the "prequantized" tensor, or the original tensor if no quantization
        is needed"""
        quant = name_to_tensor[tensor_name].Quantization()
        if quant is None or quant.ScaleIsNone() or quant.ZeroPointIsNone():
            return tensor_name
        prequant_name = tensor_name + "_prequant"
        quantize_name = tensor_name + "_quantize"
        attr = get_quantization_attr(quant)
        onnx_node = helper.make_node("TFL_QUANTIZE", [prequant_name], [tensor_name], name=quantize_name, **attr)
        onnx_nodes.append(onnx_node)
        output_shapes[prequant_name] = output_shapes[tensor_name].copy()
        dtypes[prequant_name] = onnx_pb.TensorProto.FLOAT
        return prequant_name

    for i in range(tflite_g.OperatorsLength()):
        op = tflite_g.Operators(i)
        optype = 'TFL_' + opcodes_map[op.OpcodeIndex()]
        op_cnt[optype] += 1
        attr = {}
        options_type_name = lookup_enum(op.BuiltinOptionsType(), 'BuiltinOptions')
        option_class = get_options_class(options_type_name)
        wants_dequantized_input = True
        has_prequantized_output = True
        if optype == 'TFL_QUANTIZE':
            out_tensor = tflite_g.Tensors(op.Outputs(0))
            quant = out_tensor.Quantization()
            has_prequantized_output = False
            if quant is not None and not quant.ScaleIsNone() and not quant.ZeroPointIsNone():
                attr.update(get_quantization_attr(quant))
        elif optype == 'TFL_DEQUANTIZE':
            in_tensor = tflite_g.Tensors(op.Inputs(0))
            quant = in_tensor.Quantization()
            wants_dequantized_input = False
            if quant is not None and not quant.ScaleIsNone() and not quant.ZeroPointIsNone():
                attr.update(get_quantization_attr(quant))
        input_names = [tensor_names[op.Inputs(i)] for i in range(op.InputsLength()) if op.Inputs(i) != -1]
        output_names = [tensor_names[op.Outputs(i)] for i in range(op.OutputsLength()) if op.Outputs(i) != -1]
        if optype.startswith("TFL_Flex"):
            data = read_flexbuffer(op.CustomOptionsAsNumpy().tobytes(), decode_strings=False)
            utils.make_sure(isinstance(data, list), "Flex ops are expected to store data as a flexbuffer list")
            tf_op = data[0].decode("utf-8")
            tf_node_def = node_def_pb2.NodeDef()
            tf_node_def.ParseFromString(data[1])
            input_tf_dtypes = [map_tflite_dtype_to_tf(name_to_tensor[inp].Type()) for inp in input_names]
            def shape_to_tf_shape(dims):
                return [None if d < 0 else d for d in dims] if dims is not None else None
            input_shapes = [shape_to_tf_shape(output_shapes[inp]) for inp in input_names]
            tf_attrs, _ = read_tf_node_def_attrs(tf_node_def, input_tf_dtypes, input_shapes)
            attr.update(tf_attrs)
            optype = tf_op
        elif not op.CustomOptionsIsNone():
            custom_ops_format = lookup_enum(op.CustomOptionsFormat(), 'CustomOptionsFormat')
            if custom_ops_format == 'FLEXBUFFERS':
                data = None
                try:
                    data = read_flexbuffer(op.CustomOptionsAsNumpy().tobytes())
                except Exception as e:    # pylint: disable=broad-except
                    logger.warning("Could not parse attributes for custom op '%s': %s", optype, e)
                if isinstance(data, dict):
                    attr.update(data)
        if option_class is not None:
            options = option_class()
            options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            # All flatbuffer objects have these properties.
            block_list = [options_type_name + 'BufferHasIdentifier', 'Init',
                          'GetRootAs' + options_type_name, 'GetRootAs']
            # The rest of the properties of the options class provide its attribute names
            attr_names = {opt for opt in dir(options) if not opt.startswith('_') and opt not in block_list}
            for a in list(attr_names):
                # Flatbufffer list properties have 3 functions: *Length, *IsNone, and *AsNumpy
                if a + 'Length' in attr_names:
                    attr_names.remove(a + 'Length')
                    attr_names.remove(a + 'IsNone')
                    attr_names.remove(a)
            for a in attr_names:
                if a.endswith('AsNumpy'):
                    value = getattr(options, a)().tolist()
                    a = a[:-len('AsNumpy')]
                else:
                    # For enums we use a string with the value name, not enum index
                    value = getattr(options, a)()
                    if a in NODE_ATTR_NAME_TO_ENUM_TYPE:
                        value = lookup_enum(value, NODE_ATTR_NAME_TO_ENUM_TYPE[a])
                    elif a in FUNCTION_ATTRS:
                        value = model.Subgraphs(value).Name().decode()
                attr_cnt[a] += 1
                attr[proper_to_snake_case(a)] = value
        if wants_dequantized_input:
            input_names = [get_dequant(inp) for inp in input_names]
        if optype == "TFL_TFLite_Detection_PostProcess":
            # There's a bug in tflite for the output shapes of this op
            for out, shape in zip(output_names, [[-1, -1, 4], [-1, -1], [-1, -1], [-1]]):
                if len(output_shapes[out]) != len(shape):
                    output_shapes[out] = shape
        if all(output_shapes[out] == [] for out in output_names):
            # tflite uses [] to represent both scalars and completely unknown shapes
            # If an op has non-scalar inputs and all scalar outputs, it is very likely the shapes are actually unknown.
            inp_shapes = [output_shapes[inp] for inp in input_names]
            if not all(s == [] for s in inp_shapes):
                if any(s is None for s in inp_shapes) or not op_has_scalar_output(inp_shapes, optype, attr):
                    for out in output_names:
                        logger.warning("Replacing scalar output shape of %s with unknown shape", out)
                        output_shapes[out] = None
        if has_prequantized_output:
            output_names = [get_prequant(out) for out in output_names]
        onnx_node = utils.make_onnx_node_with_attr(optype, input_names, output_names, name=output_names[0], **attr)
        node_name = output_names[0] if output_names else utils.make_name(f"{optype}_Output")
        onnx_node = utils.make_onnx_node_with_attr(optype, input_names, output_names, name=node_name, **attr)
        onnx_nodes.append(onnx_node)

    inputs = [tensor_names[tflite_g.Inputs(i)] for i in range(tflite_g.InputsLength())]
    outputs = [tensor_names[tflite_g.Outputs(i)] for i in range(tflite_g.OutputsLength())]
    # TODO: Allow input/outputs to be overridden

    for inp in inputs:
        onnx_node = helper.make_node("Placeholder", [], outputs=[inp], name=inp)
        onnx_nodes.append(onnx_node)

    graph_name = (tflite_g.Name() or b'tflite graph').decode()
    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, inputs, outputs, graph_name
