# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors


import logging as log
import numpy as np
import sys
from openvino import PartialShape, Dimension, Type
from packaging.version import parse, Version
from typing import List, Dict, Union


# TODO: reuse this method in ovc and remove duplication
def get_static_shape(shape: [PartialShape, list, tuple], dynamic_value=None):
    # Current function returns list with static dimensions with following logic.
    # For dynamic dimensions return lower boundaries if they are set, otherwise
    # return upper boundaries if they are set. If dimension is fully dynamic then raise error.
    shape_list = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int):
            if dim == -1:
                shape_list.append(dynamic_value)
                continue
            shape_list.append(dim)
        elif isinstance(dim, np.int64):
            if dim == np.int64(-1):
                shape_list.append(dynamic_value)
                continue
            shape_list.append(dim)
        elif isinstance(dim, tuple):
            # tuple where (min_length, max_length), the format which uses MO cli parser
            assert len(dim) == 2, "Unknown dimension type {}".format(dim)
            if dim[0] > 0:
                shape_list.append(dim[0])
            elif dim[1] < np.iinfo(np.int64).max:
                shape_list.append(dim[1])
            else:
                shape_list.append(dynamic_value)
                continue
        elif isinstance(dim, Dimension):
            if dim.is_static or dim.get_min_length() > 0:
                shape_list.append(dim.get_min_length())
            elif dim.get_max_length() != -1:
                shape_list.append(dim.get_max_length())
            else:
                shape_list.append(dynamic_value)
                continue
        else:
            raise Exception("Unknown dimension type {}".format(dim))

    return tuple(shape_list)


def get_imported_module_version(imported_module):
    """
    Get imported module version
    :return: version(str) or raise AttributeError exception
    """
    version_attrs = ("__version__", "VERSION", "version")
    installed_version = None
    for attr in version_attrs:
        installed_version = getattr(imported_module, attr, None)
        if isinstance(installed_version, str):
            return installed_version
        else:
            installed_version = None

    if installed_version is None:
        raise AttributeError("{} module doesn't have version attribute".format(imported_module))
    else:
        return installed_version


# TODO: reuse this method in ovc and remove duplication
def get_environment_setup(framework):
    """
    Get environment setup such as Python version, TensorFlow version
    :param framework: framework name
    :return: a dictionary of environment variables
    """
    env_setup = dict()
    python_version = "{}.{}.{}".format(sys.version_info.major,
                                       sys.version_info.minor,
                                       sys.version_info.micro)
    env_setup['python_version'] = python_version
    try:
        if framework == 'tf':
            exec("import tensorflow")
            env_setup['tensorflow'] = get_imported_module_version(sys.modules["tensorflow"])
            exec("del tensorflow")
    except (AttributeError, ImportError):
        pass
    env_setup['sys_platform'] = sys.platform
    return env_setup


def trace_tf_model_if_needed(input_model, placeholder_shapes, placeholder_data_types, example_input):
    import tensorflow as tf
    if not isinstance(input_model,
                      (tf.keras.layers.Layer, tf.Module, tf.keras.Model, tf.types.experimental.GenericFunction)):
        return input_model
    return trace_tf_model(input_model, placeholder_shapes, placeholder_data_types, example_input)


def partial_shape_to_list(partial_shape: PartialShape):
    if partial_shape.rank.is_dynamic:
        return None
    res_list = []
    for dim in partial_shape:
        if dim.is_static:
            res_list.append(dim.get_length())
        else:
            res_list.append(None)
    return res_list


def get_input_spec_from_model(model, input_shapes=None):
    import tensorflow as tf
    if hasattr(model, "_build_input_shape") and model._build_input_shape is not None:
        if isinstance(model._build_input_shape, list):
            input_spec = [[tf.TensorSpec(shape) for shape in model._build_input_shape]]
        else:
            input_spec = [tf.TensorSpec(model._build_input_shape)]
    elif input_shapes and isinstance(input_shapes, list) and len(input_shapes) > 0:
        input_spec = []
        for input_shape in input_shapes:
            if isinstance(input_shape, PartialShape):
                input_spec.append(tf.TensorSpec(partial_shape_to_list(input_shape)))
            else:
                input_spec.append(tf.TensorSpec(None))
    else:
        input_spec = [tf.TensorSpec(None)]
    return input_spec


def get_concrete_func(tf_function, example_input, input_needs_packing, error_message, use_example_input=True):
    """
    Runs tracing of TF function and returns a concrete function.

    :param tf_function: TF function that needs to be traced.
    :param example_input: Example of function input.
    :param input_needs_packing: determines if input needs to be packed in a list before passing to TF function.
    It is used when original function was wrapped in outer TF function, which changes function signature.
    In this case wrapper TF function always expects list of inputs which are unpacked inside subfunction.
    So list/tuple are treated as multiple inputs of original model.
    Non list/tuple are treated as single input, and it needs packing to a list,
    as wrapper function always expect list of inputs.
    :param error_message: Error message which should be shown in case of tracing error.
    :param use_example_input: Determines if example_input should be used.

    :returns: Object of type tf.types.experimental.ConcreteFunction.
    """
    if input_needs_packing and not isinstance(example_input, (list, tuple)):
        example_input = [example_input]
    try:
        if use_example_input:
            if not input_needs_packing and isinstance(example_input, (list, tuple)):
                concrete_func = tf_function.get_concrete_function(*example_input)
            else:
                concrete_func = tf_function.get_concrete_function(example_input)

        else:
            concrete_func = tf_function.get_concrete_function()
    except Exception as e:
        raise Exception(error_message.format(e))
    return concrete_func


def get_signature_from_input(keras_model):
    if not hasattr(keras_model, 'input') or getattr(keras_model, 'input') is None:
        return None
    return getattr(keras_model, 'input')


def get_signature_from_input_signature(keras_model):
    if not hasattr(keras_model, 'input_signature') or getattr(keras_model, 'input_signature') is None:
        return None
    return getattr(keras_model, 'input_signature')


def create_generic_function_from_keras_model(keras_model):
    import tensorflow as tf
    assert isinstance(keras_model, (tf.keras.Model, tf.Module)), \
        "[TensorFlow Frontend] internal error: the input model must be of tf.keras.Model or tf.Module model type"

    keras_input_signature = get_signature_from_input(keras_model)
    if keras_input_signature is None:
        keras_input_signature = get_signature_from_input_signature(keras_model)
    if keras_input_signature is None:
        return None
    tf_input_signature = None
    wrapper_function = None
    if isinstance(keras_input_signature, dict):
        tf_input_signature = []
        for tensor_name, tensor_spec in keras_input_signature.items():
            tf_input_signature.append(tf.TensorSpec(shape=tensor_spec.shape,
                                                    dtype=tensor_spec.dtype,
                                                    name=tensor_name))
    elif isinstance(keras_input_signature, list):
        tf_input_signature = []
        for tensor_spec in keras_input_signature:
            tf_input_signature.append(tf.TensorSpec(shape=tensor_spec.shape,
                                                    dtype=tensor_spec.dtype,
                                                    name=tensor_spec.name))
    else:
        try:
            # single KerasTensor case
            tf_input_signature = []
            tf_input_signature.append(tf.TensorSpec(shape=keras_input_signature.shape,
                                                    dtype=keras_input_signature.dtype,
                                                    name=keras_input_signature.name))
        except:
            tf_input_signature = None
    if tf_input_signature is not None:
        @tf.function(input_signature=tf_input_signature)
        def wrapper_function_dict(*args):
            if isinstance(keras_input_signature, list):
                outputs = keras_model(args)
            else:
                input_dict = {}
                for ind, tensor_spec in enumerate(tf_input_signature):
                    input_dict[tensor_spec.name] = args[ind]
                outputs = keras_model(input_dict)
            # need to wrap the output into dictionary
            # it helps to preserve original keras tensor names
            post_outputs = {}
            if isinstance(outputs, dict):
                for output_name, output_value in outputs.items():
                    post_outputs[output_name] = output_value
            else:
                try:
                    if isinstance(outputs, list) and isinstance(keras_model.outputs, list) and \
                            len(outputs) == len(keras_model.outputs):
                        for output_value, output_tensor in zip(outputs, keras_model.outputs):
                            post_outputs[output_tensor.name] = output_value
                    else:
                        post_outputs[keras_model.output.name] = outputs
                except:
                    post_outputs = outputs
            return post_outputs

        wrapper_function = wrapper_function_dict
    return wrapper_function


def trace_tf_model(model, input_shapes, input_types, example_input):
    import tensorflow as tf
    if isinstance(model.__call__, tf.types.experimental.GenericFunction):
        tf_function = model.__call__
        input_needs_packing = False
    elif isinstance(model, tf.types.experimental.GenericFunction):
        tf_function = model
        input_needs_packing = False
    elif isinstance(model, (tf.keras.Model, tf.Module)):
        tf_function = create_generic_function_from_keras_model(model)
        if tf_function is not None:
            input_needs_packing = False
        else:
            # Wrap model to tf.Function.
            # In this case we loose input/output tensor names.
            @tf.function
            def tf_function(args):
                return model(*args)

            input_needs_packing = True
    else:
        # Wrap model to tf.Function.
        # In this case we loose input/output tensor names.
        @tf.function
        def tf_function(args):
            return model(*args)

        input_needs_packing = True

    def are_shapes_defined(shape: Union[List, Dict]):
        if shape is None:
            return False
        assert hasattr(shape, '__len__')
        if len(shape) == 0:
            return False

        if isinstance(shape, list):
            return np.all([shape is not None for shape in input_shapes])
        elif isinstance(shape, dict):
            return np.all([shape is not None for name, shape in input_shapes.items()])

    if example_input is not None:
        concrete_func = get_concrete_func(tf_function, example_input, input_needs_packing,
                                          "Could not trace the TF model with the following error: {}")
    else:
        if isinstance(tf_function, tf.types.experimental.GenericFunction) and \
                tf_function.input_signature is not None:
            concrete_func = get_concrete_func(tf_function, None, input_needs_packing,
                                              "Could not trace the TF model with the following error: {}",
                                              use_example_input=False)
        else:
            input_spec = get_input_spec_from_model(model, input_shapes)
            concrete_func = get_concrete_func(tf_function, input_spec, input_needs_packing,
                                              "Could not trace the TF model with the following error: {}.\n"
                                              "Please provide 'example_input'.")

    return concrete_func


def type_supported_by_tf_fe(input_model):
    import tensorflow as tf
    # Types that require tracing
    if isinstance(input_model,
                  (tf.keras.layers.Layer, tf.Module, tf.keras.Model, tf.types.experimental.GenericFunction)):
        return True
    # Types that do not require tracing
    if isinstance(input_model, (tf.Graph, tf.types.experimental.ConcreteFunction)):
        return True
    # GraphIterator
    elif model_is_graph_iterator(input_model):
        return True
    return False


def is_variable(func_input, captures):
    import tensorflow as tf
    if func_input.dtype == tf.resource:
        return True
    for capture in captures:
        if id(func_input) == id(capture[1]):
            return True
    return False


def create_tf_graph_iterator(input_model, placeholder_shapes, placeholder_data_types, example_input, share_weights):
    input_model = trace_tf_model_if_needed(input_model, placeholder_shapes, placeholder_data_types, example_input)

    import tensorflow as tf
    from openvino.frontend.tensorflow.graph_iterator import GraphIteratorTFGraph
    if model_is_graph_iterator(input_model):
        return input_model
    if isinstance(input_model, tf.Graph):
        return GraphIteratorTFGraph(input_model, share_weights)
    elif isinstance(input_model, tf.types.experimental.ConcreteFunction):
        # create a map for inputs to map internal tensor name to external one
        # collect all internal tensor names in a given order
        input_names_map = None
        if hasattr(input_model, 'inputs') and hasattr(input_model, 'structured_input_signature'):
            internal_tensor_names = []
            for func_input in input_model.inputs:
                if is_variable(func_input, input_model.graph.captures):
                    continue
                internal_tensor_names.append(func_input.name)
            if len(input_model.structured_input_signature) > 0 and \
                    len(internal_tensor_names) == len(input_model.structured_input_signature[0]):
                for internal_name, tensor_spec in zip(internal_tensor_names, input_model.structured_input_signature[0]):
                    input_names_map = input_names_map or {}
                    if not isinstance(tensor_spec, tf.TensorSpec):
                        input_names_map = None
                        break
                    input_names_map[internal_name] = tensor_spec.name
            elif len(input_model.structured_input_signature) > 1 and \
                    len(internal_tensor_names) == len(input_model.structured_input_signature[1]):
                external_tensor_names = sorted(input_model.structured_input_signature[1].keys())
                for internal_name, external_name in zip(internal_tensor_names, external_tensor_names):
                    input_names_map = input_names_map or {}
                    input_names_map[internal_name] = external_name

        output_names_map = None
        if hasattr(input_model, 'outputs') and hasattr(input_model, 'structured_outputs') and \
                isinstance(input_model.structured_outputs, dict):
            external_names = sorted(list(input_model.structured_outputs.keys()))
            internal_names = [tensor.name for tensor in input_model.outputs]
            if len(external_names) == len(internal_names):
                for external_name, internal_name in zip(external_names, internal_names):
                    output_names_map = output_names_map or {}
                    output_names_map[internal_name] = external_name
            else:
                for external_name, internal_tensor in input_model.structured_outputs.items():
                    internal_tf_tensor = None
                    if isinstance(internal_tensor, tf.Tensor):
                        internal_tf_tensor = internal_tensor
                    if isinstance(internal_tensor, list) and len(internal_tensor) > 0 and \
                            isinstance(internal_tensor[0], tf.Tensor):
                        internal_tf_tensor = internal_tensor[0]
                    if internal_tf_tensor is None:
                        output_names_map = None
                        break
                    output_names_map = output_names_map or {}
                    output_names_map[internal_tf_tensor.name] = external_name
        return GraphIteratorTFGraph(input_model.graph, share_weights, False, input_names_map, output_names_map)
    raise Exception("Could not wrap model of type {} to GraphIteratorTFGraph.".format(type(input_model)))


def extract_model_graph(argv):
    model = argv["input_model"]
    import tensorflow as tf
    trackable_is_imported = False
    try:
        from tensorflow.python.training.tracking.base import Trackable  # pylint: disable=no-name-in-module,import-error
        trackable_is_imported = True
    except:
        try:
            from tensorflow.python.trackable.base import Trackable
            trackable_is_imported = True
        except:
            log.warning("Could not import tensorflow.python.training.tracking.base.Trackable type.")
    env_setup = get_environment_setup("tf")
    if isinstance(model, tf.Graph):
        return True
    if isinstance(model, tf.compat.v1.GraphDef):
        graph = tf.Graph()
        with graph.as_default():
            tf.graph_util.import_graph_def(model, name='')
        argv["input_model"] = graph
        return True
    if isinstance(model, tf.compat.v1.Session):
        argv["input_model"] = model.graph
        return True
    if Version(env_setup["tensorflow"]) >= parse("2.6.0") and isinstance(model, (tf.types.experimental.GenericFunction,
                                                                                 tf.types.experimental.ConcreteFunction)):
        return True
    if isinstance(model, tf.train.Checkpoint):
        if isinstance(model.root, tf.keras.Model):
            argv["input_model"] = model.root
            return True
        else:
            raise Exception("Unknown checkpoint format.")

    if isinstance(model, (tf.keras.layers.Layer, tf.Module, tf.keras.Model)):
        return True
    if trackable_is_imported and isinstance(model, Trackable):
        if hasattr(model, "signatures") and len(model.signatures.items()):
            if "serving_default" in model.signatures:
                argv["input_model"] = model.signatures["serving_default"]
            elif "default" in model.signatures:
                argv["input_model"] = model.signatures["default"]
            else:
                for signature_name, signature in model.signatures.items():
                    argv["input_model"] = model.signatures[signature_name]
                    log.warning("Could not find the default signature. "
                                "The following signature was used for conversion: {}".format(signature_name))
                    break

        elif hasattr(model, "graph"):
            argv["input_model"] = model.graph
        else:
            raise Exception("Could not find signature of graph in a Trackable object.")
        return True
    if model_is_graph_iterator(model):
        return True
    return False


def model_is_graph_iterator(model):
    try:
        from openvino.frontend.tensorflow.graph_iterator import GraphIteratorTFGraph
    except:
        return False
    return isinstance(model, GraphIteratorTFGraph)


def tf_type_to_ov_type(val):
    import tensorflow as tf  # pylint: disable=import-error
    if not isinstance(val, tf.dtypes.DType):
        raise Exception("The provided type is not a TF type {}.".format(val))

    tf_to_ov_type = {
        tf.float32: Type.f32,
        tf.float16: Type.f16,
        tf.float64: Type.f64,
        tf.bfloat16: Type.bf16,
        tf.uint8: Type.u8,
        tf.int8: Type.i8,
        tf.int16: Type.i16,
        tf.int32: Type.i32,
        tf.int64: Type.i64,
        tf.bool: Type.boolean,
        tf.string: Type.string
    }
    if val not in tf_to_ov_type:
        raise Exception("The provided data type is not supported by OpenVino {}.".format(val))
    return tf_to_ov_type[val]
