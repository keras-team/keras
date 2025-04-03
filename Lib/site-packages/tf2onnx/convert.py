# SPDX-License-Identifier: Apache-2.0


"""
python -m tf2onnx.convert : api and commandline tool to convert a tensorflow model to onnx
"""

# pylint: disable=unused-argument,unused-import,ungrouped-imports,wrong-import-position

import argparse
import os
import sys
from packaging.version import Version

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf

from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import constants, logging, utils, optimizer
from tf2onnx import tf_loader
from tf2onnx.graph import ExternalTensorStorage
from tf2onnx.tf_utils import compress_graph_def, get_tf_version



# pylint: disable=unused-argument

_HELP_TEXT = """
Usage Examples:

python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx
python -m tf2onnx.convert --input frozen_graph.pb  --inputs X:0 --outputs output:0 --output model.onnx
python -m tf2onnx.convert --checkpoint checkpoint.meta  --inputs X:0 --outputs output:0 --output model.onnx

For help and additional information see:
    https://github.com/onnx/tensorflow-onnx

If you run into issues, open an issue here:
    https://github.com/onnx/tensorflow-onnx/issues
"""


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Convert tensorflow graphs to ONNX.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_HELP_TEXT)
    parser.add_argument("--input", help="input from graphdef")
    parser.add_argument("--graphdef", help="input from graphdef")
    parser.add_argument("--saved-model", help="input from saved model")
    parser.add_argument("--tag", help="tag to use for saved_model")
    parser.add_argument("--signature_def", help="signature_def from saved_model to use")
    parser.add_argument("--concrete_function", type=int, default=None,
                        help="For TF2.x saved_model, index of func signature in __call__ (--signature_def is ignored)")
    parser.add_argument("--checkpoint", help="input from checkpoint")
    parser.add_argument("--keras", help="input from keras model")
    parser.add_argument("--tflite", help="input from tflite model")
    parser.add_argument("--tfjs", help="input from tfjs model")
    parser.add_argument("--large_model", help="use the large model format (for models > 2GB)", action="store_true")
    parser.add_argument("--output", help="output model file")
    parser.add_argument("--inputs", help="model input_names (optional for saved_model, keras, and tflite)")
    parser.add_argument("--outputs", help="model output_names (optional for saved_model, keras, and tflite)")
    parser.add_argument("--ignore_default", help="comma-separated list of names of PlaceholderWithDefault "
                                                 "ops to change into Placeholder ops")
    parser.add_argument("--use_default", help="comma-separated list of names of PlaceholderWithDefault ops to "
                                              "change into Identity ops using their default value")
    parser.add_argument("--rename-inputs", help="input names to use in final model (optional)")
    parser.add_argument("--rename-outputs", help="output names to use in final model (optional)")
    parser.add_argument("--use-graph-names", help="(saved model only) skip renaming io using signature names",
                        action="store_true")
    parser.add_argument("--opset", type=int, default=None, help="opset version to use for onnx domain")
    parser.add_argument("--dequantize", help="remove quantization from model. Only supported for tflite currently.",
                        action="store_true")
    parser.add_argument("--custom-ops", help="comma-separated map of custom ops to domains in format OpName:domain. "
                                             "Domain 'ai.onnx.converters.tensorflow' is used by default.")
    parser.add_argument("--extra_opset", default=None,
                        help="extra opset with format like domain:version, e.g. com.microsoft:1")
    parser.add_argument("--load_op_libraries",
                        help="comma-separated list of tf op library paths to register before loading model")
    parser.add_argument("--target", default=",".join(constants.DEFAULT_TARGET), choices=constants.POSSIBLE_TARGETS,
                        help="target platform")
    parser.add_argument("--continue_on_error", help="continue_on_error", action="store_true")
    parser.add_argument("--verbose", "-v", help="verbose output, option is additive", action="count")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument("--output_frozen_graph", help="output frozen tf graph to file")

    # experimental
    parser.add_argument("--inputs-as-nchw", help="transpose inputs as from nhwc to nchw")
    parser.add_argument("--outputs-as-nchw", help="transpose outputs as from nhwc to nchw")
    args = parser.parse_args()

    args.shape_override = None
    if args.input:
        # for backward compatibility
        args.graphdef = args.input
    if args.graphdef or args.checkpoint:
        if not args.inputs or not args.outputs:
            parser.error("graphdef and checkpoint models need to provide inputs and outputs")
    if not any([args.graphdef, args.checkpoint, args.saved_model, args.keras, args.tflite, args.tfjs]):
        parser.print_help()
        sys.exit(1)
    if args.inputs:
        args.inputs, args.shape_override = utils.split_nodename_and_shape(args.inputs)
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.ignore_default:
        args.ignore_default = args.ignore_default.split(",")
    if args.use_default:
        args.use_default = args.use_default.split(",")
    if args.rename_outputs:
        args.rename_outputs = args.rename_outputs.split(",")
    if args.rename_inputs:
        args.rename_inputs = args.rename_inputs.split(",")
    if args.inputs_as_nchw:
        args.inputs_as_nchw = args.inputs_as_nchw.split(",")
    if args.outputs_as_nchw:
        args.outputs_as_nchw = args.outputs_as_nchw.split(",")
    if args.target:
        args.target = args.target.split(",")
    if args.signature_def:
        args.signature_def = [args.signature_def]
    if args.dequantize:
        if not args.tflite:
            parser.error("dequantize flag is currently only supported for tflite")
    if args.extra_opset:
        all_extra_opsets = args.extra_opset.split(',')
        extra_opset_list = []
        for extra_opset in all_extra_opsets:
            tokens = extra_opset.split(':')
            if len(tokens) != 2:
                parser.error("invalid extra_opset argument")
            extra_opset_list.append(utils.make_opsetid(tokens[0], int(tokens[1])))
        args.extra_opset = extra_opset_list
    if args.load_op_libraries:
        args.load_op_libraries = args.load_op_libraries.split(",")
    return args


def make_default_custom_op_handler(domain):
    def default_custom_op_handler(ctx, node, name, args):
        node.domain = domain
        return node
    return default_custom_op_handler


def _convert_common(frozen_graph, name="unknown", large_model=False, output_path=None,
                    output_frozen_graph=None, custom_ops=None, custom_op_handlers=None, optimizers=None, **kwargs):
    """Common processing for conversion."""

    model_proto = None
    external_tensor_storage = None
    const_node_values = None

    if custom_ops is not None:
        if custom_op_handlers is None:
            custom_op_handlers = {}
        custom_op_handlers.update(
            {op: (make_default_custom_op_handler(domain), []) for op, domain in custom_ops.items()})

    with tf.Graph().as_default() as tf_graph:
        if large_model:
            const_node_values = compress_graph_def(frozen_graph)
            external_tensor_storage = ExternalTensorStorage()
        if output_frozen_graph:
            utils.save_protobuf(output_frozen_graph, frozen_graph)
        if not kwargs.get("tflite_path") and not kwargs.get("tfjs_path"):
            tf.import_graph_def(frozen_graph, name='')
        g = process_tf_graph(tf_graph, const_node_values=const_node_values,
                             custom_op_handlers=custom_op_handlers, **kwargs)
        if constants.ENV_TF2ONNX_CATCH_ERRORS in os.environ:
            catch_errors = constants.ENV_TF2ONNX_CATCH_ERRORS.upper() == "TRUE"
        else:
            catch_errors = not large_model
        onnx_graph = optimizer.optimize_graph(g, catch_errors, optimizers=optimizers)
        model_proto = onnx_graph.make_model("converted from {}".format(name),
                                            external_tensor_storage=external_tensor_storage)
    if output_path:
        if large_model:
            utils.save_onnx_zip(output_path, model_proto, external_tensor_storage)
        else:
            utils.save_protobuf(output_path, model_proto)

    return model_proto, external_tensor_storage


def main():
    args = get_args()
    logging.basicConfig(level=logging.get_verbosity_level(args.verbose))
    if args.debug:
        utils.set_debug_mode(True)

    logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)

    extra_opset = args.extra_opset or []
    tflite_path = None
    tfjs_path = None
    custom_op_handlers = {}
    initialized_tables = None
    tensors_to_rename = {}
    if args.custom_ops:
        using_tf_opset = False
        for op in args.custom_ops.split(","):
            if ":" in op:
                op, domain = op.split(":")
            else:
                # default custom ops for tensorflow-onnx are in the "tf" namespace
                using_tf_opset = True
                domain = constants.TENSORFLOW_OPSET.domain
            custom_op_handlers[op] = (make_default_custom_op_handler(domain), [])
        if using_tf_opset:
            extra_opset.append(constants.TENSORFLOW_OPSET)

    if any(opset.domain == constants.CONTRIB_OPS_DOMAIN for opset in extra_opset):
        try:
            import tensorflow_text   # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError:
            logger.warning("tensorflow_text not installed. Model will fail to load if tensorflow_text ops are used.")

    # get the frozen tensorflow model from graphdef, checkpoint or saved_model.
    graph_def = None
    inputs = None
    outputs = None
    model_path = None

    if not utils.is_cpp_protobuf():
        logger.warning("***IMPORTANT*** Installed protobuf is not cpp accelerated. Conversion will be extremely slow. "
                       "See https://github.com/onnx/tensorflow-onnx/issues/1557")

    if args.load_op_libraries:
        for op_filepath in args.load_op_libraries:
            # change relative path to absolute path to satisfy the tf.load_op_library().
            if not os.path.isabs(op_filepath):
                op_filepath = os.getcwd() + "/" + op_filepath
            tf.load_op_library(op_filepath)
    if args.graphdef:
        graph_def, inputs, outputs = tf_loader.from_graphdef(args.graphdef, args.inputs, args.outputs)
        model_path = args.graphdef
    if args.checkpoint:
        graph_def, inputs, outputs = tf_loader.from_checkpoint(args.checkpoint, args.inputs, args.outputs)
        model_path = args.checkpoint
    if args.saved_model:
        graph_def, inputs, outputs, initialized_tables, tensors_to_rename = tf_loader.from_saved_model(
            args.saved_model, args.inputs, args.outputs, args.tag, args.signature_def, args.concrete_function,
            args.large_model, return_initialized_tables=True, return_tensors_to_rename=True,
            use_graph_names=args.use_graph_names)
        model_path = args.saved_model
    if args.keras:
        graph_def, inputs, outputs = tf_loader.from_keras(
            args.keras, args.inputs, args.outputs)
        model_path = args.keras
    if args.tflite:
        # Optional, but used to cut graph if provided.
        inputs = args.inputs
        outputs = args.outputs
        tflite_path = args.tflite
        model_path = tflite_path
    if args.tfjs:
        inputs = args.inputs
        outputs = args.outputs
        tfjs_path = args.tfjs
        model_path = tfjs_path

    if args.verbose:
        logger.info("inputs: %s", inputs)
        logger.info("outputs: %s", outputs)

    if args.rename_inputs:
        tensors_to_rename.update(zip(inputs, args.rename_inputs))
    if args.rename_outputs:
        tensors_to_rename.update(zip(outputs, args.rename_outputs))

    with tf.device("/cpu:0"):
        model_proto, _ = _convert_common(
            graph_def,
            name=model_path,
            continue_on_error=args.continue_on_error,
            target=args.target,
            opset=args.opset,
            custom_op_handlers=custom_op_handlers,
            extra_opset=extra_opset,
            shape_override=args.shape_override,
            input_names=inputs,
            output_names=outputs,
            inputs_as_nchw=args.inputs_as_nchw,
            outputs_as_nchw=args.outputs_as_nchw,
            large_model=args.large_model,
            tensors_to_rename=tensors_to_rename,
            ignore_default=args.ignore_default,
            use_default=args.use_default,
            tflite_path=tflite_path,
            dequantize=args.dequantize,
            tfjs_path=tfjs_path,
            initialized_tables=initialized_tables,
            output_frozen_graph=args.output_frozen_graph,
            output_path=args.output)


    # write onnx graph
    logger.info("")
    logger.info("Successfully converted TensorFlow model %s to ONNX", model_path)

    logger.info("Model inputs: %s", [n.name for n in model_proto.graph.input])
    logger.info("Model outputs: %s", [n.name for n in model_proto.graph.output])
    if args.output:
        if args.large_model:
            logger.info("Zipped ONNX model is saved at %s. Unzip before opening in onnxruntime.", args.output)
        else:
            logger.info("ONNX model is saved at %s", args.output)
    else:
        logger.info("To export ONNX model to file, please run with `--output` option")


def tensor_names_from_structed(concrete_func, input_names, output_names):
    tensors_to_rename = {}
    flat_structured_inp = tf.nest.flatten(concrete_func.structured_input_signature)
    structured_inputs = [t.name for t in flat_structured_inp if isinstance(t, tf.TensorSpec)]
    tensors_to_rename.update(zip(input_names, structured_inputs))
    if isinstance(concrete_func.structured_outputs, dict):
        for k, v in concrete_func.structured_outputs.items():
            tensors_to_rename[v.name] = k
    return tensors_to_rename


def _rename_duplicate_keras_model_names(model):
    """
    In very rare cases, keras has a bug where it will give multiple outputs the same name.
    We must edit the model or the TF trace will fail. Returns old_out_names (or None if no edit was made).
    IMPORTANT: model may be edited. Assign model.output_names to old_out_names to restore.
    """
    old_out_names = None
    if model.output_names and len(set(model.output_names)) != len(model.output_names):
        # In very rare cases, keras has a bug where it will give multiple outputs the same name
        # We must edit the model or the TF trace will fail
        old_out_names = model.output_names
        used_names = set()
        new_out_names = []
        for name in model.output_names:
            new_name = name
            i = 0
            while new_name in used_names:
                i += 1
                new_name = name + "_" + str(i)
            used_names.add(new_name)
            new_out_names.append(new_name)
        model.output_names = new_out_names
    return old_out_names


def _is_legacy_keras_model(model):
    """Inspects model class to determine if it is from tf or legacy keras"""

    logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)
    unknown_type_err = "model is not instance of tf.keras.Model or keras.Model"
    if isinstance(model, tf.keras.Model):
        return False
    try:
        import keras  # pylint: disable=import-outside-toplevel
        if isinstance(model, keras.Model):
            return True
        logger.warning(unknown_type_err)
    except ImportError:
        logger.warning(unknown_type_err)
    return False


def _from_keras_tf1(model, opset=None, custom_ops=None, custom_op_handlers=None, custom_rewriter=None,
                    inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None, shape_override=None,
                    target=None, large_model=False, output_path=None):
    """from_keras for tf 1.15"""
    input_names = [t.name for t in model.inputs]
    output_names = [t.name for t in model.outputs]
    old_out_names = _rename_duplicate_keras_model_names(model)
    tensors_to_rename = dict(zip(input_names, model.input_names))
    tensors_to_rename.update(zip(output_names, model.output_names))
    if old_out_names is not None:
        model.output_names = old_out_names

    if _is_legacy_keras_model(model):
        import keras  # pylint: disable=import-outside-toplevel
        sess = keras.backend.get_session()
    else:
        sess = tf.keras.backend.get_session(model.outputs)

    with tf.device("/cpu:0"):
        frozen_graph, initialized_tables = tf_loader.freeze_session(sess, input_names, output_names, get_tables=True)
        with tf.Graph().as_default():
            tf.import_graph_def(frozen_graph, name="")
            frozen_graph = tf_loader.tf_optimize(input_names, output_names, frozen_graph)
        model_proto, external_tensor_storage = _convert_common(
            frozen_graph,
            name=model.name,
            continue_on_error=True,
            target=target,
            opset=opset,
            custom_ops=custom_ops,
            custom_op_handlers=custom_op_handlers,
            custom_rewriter=custom_rewriter,
            extra_opset=extra_opset,
            shape_override=shape_override,
            input_names=input_names,
            output_names=output_names,
            inputs_as_nchw=inputs_as_nchw,
            outputs_as_nchw=outputs_as_nchw,
            large_model=large_model,
            tensors_to_rename=tensors_to_rename,
            initialized_tables=initialized_tables,
            output_path=output_path)

        return model_proto, external_tensor_storage


def from_keras(model, input_signature=None, opset=None, custom_ops=None, custom_op_handlers=None,
               custom_rewriter=None, inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None, shape_override=None,
               target=None, large_model=False, output_path=None, optimizers=None):
    """Returns a ONNX model_proto for a tf.keras model.

    Args:
        model: the tf.keras model we want to convert
        input_signature: a tf.TensorSpec or a numpy array defining the shape/dtype of the input
        opset: the opset to be used for the ONNX model, default is the latest
        custom_ops: if a model contains ops not recognized by onnx runtime,
            you can tag these ops with a custom op domain so that the
            runtime can still open the model. Type is a dictionary `{op name: domain}`.
        target: list of workarounds applied to help certain platforms
        custom_op_handlers: dictionary of custom ops handlers
        custom_rewriter: list of custom graph rewriters
        extra_opset: list of extra opset's, for example the opset's used by custom ops
        shape_override: dict with inputs that override the shapes given by tensorflow
        inputs_as_nchw: transpose inputs in list from nhwc to nchw
        outputs_as_nchw: transpose outputs in list from nhwc to nchw
        large_model: use the ONNX external tensor storage format
        output_path: save model to output_path
        optimizers: list (subset) of tf2onnx optimizers if applying all optimizers is not desired.

    Returns:
        An ONNX model_proto and an external_tensor_storage dict.
    """
    if get_tf_version() < Version("2.0"):
        return _from_keras_tf1(model, opset, custom_ops, custom_op_handlers, custom_rewriter, inputs_as_nchw,
                               outputs_as_nchw, extra_opset, shape_override, target, large_model, output_path)

    old_out_names = _rename_duplicate_keras_model_names(model)
    from tensorflow.python.keras.saving import saving_utils as _saving_utils # pylint: disable=import-outside-toplevel

    # let tensorflow do the checking if model is a valid model
    function = _saving_utils.trace_model_call(model, input_signature)
    try:
        concrete_func = function.get_concrete_function()
    except TypeError as e:
        # Legacy keras models don't accept the training arg tf provides so we hack around it
        if "got an unexpected keyword argument 'training'" not in str(e):
            raise e
        model_call = model.call
        def wrap_call(*args, training=False, **kwargs):
            return model_call(*args, **kwargs)
        model.call = wrap_call
        function = _saving_utils.trace_model_call(model, input_signature)
        try:
            # Legacy keras get make TF erroneously enter eager mode when it should be making symbolic tensors
            import tensorflow_core  # pylint: disable=import-outside-toplevel
            old_get_learning_phase = tensorflow_core.python.keras.backend.learning_phase
            tensorflow_core.python.keras.backend.learning_phase = \
                tensorflow_core.python.keras.backend.symbolic_learning_phase
        except ImportError:
            old_get_learning_phase = None
        try:
            concrete_func = function.get_concrete_function()
        finally:
            # Put everything back
            model.call = model_call
            if old_get_learning_phase is not None:
                tensorflow_core.python.keras.backend.learning_phase = old_get_learning_phase

    # These inputs will be removed during freezing (includes resources, etc.)
    if hasattr(concrete_func.graph, '_captures'):
        graph_captures = concrete_func.graph._captures  # pylint: disable=protected-access
        captured_inputs = [t_name.name for _, t_name in graph_captures.values()]
    else:
        graph_captures = concrete_func.graph.function_captures.by_val_internal
        captured_inputs = [t.name for t in graph_captures.values()]
    input_names = [input_tensor.name for input_tensor in concrete_func.inputs
                   if input_tensor.name not in captured_inputs]
    output_names = [output_tensor.name for output_tensor in concrete_func.outputs
                    if output_tensor.dtype != tf.dtypes.resource]

    tensors_to_rename = tensor_names_from_structed(concrete_func, input_names, output_names)
    reverse_lookup = {v: k for k, v in tensors_to_rename.items()}

    if model.output_names:
        # model.output_names is an optional field of Keras models indicating output order. It is None if unused.
        output_names = [reverse_lookup[out] for out in model.output_names]
    elif isinstance(concrete_func.structured_outputs, dict):
        # Other models specify output order using the key order of structured_outputs
        output_names = [reverse_lookup[out] for out in concrete_func.structured_outputs.keys()]

    if old_out_names is not None:
        model.output_names = old_out_names

    with tf.device("/cpu:0"):
        frozen_graph, initialized_tables = \
            tf_loader.from_trackable(model, concrete_func, input_names, output_names, large_model)
        model_proto, external_tensor_storage = _convert_common(
            frozen_graph,
            name=model.name,
            continue_on_error=True,
            target=target,
            opset=opset,
            custom_ops=custom_ops,
            custom_op_handlers=custom_op_handlers,
            optimizers=optimizers,
            custom_rewriter=custom_rewriter,
            extra_opset=extra_opset,
            shape_override=shape_override,
            input_names=input_names,
            output_names=output_names,
            inputs_as_nchw=inputs_as_nchw,
            outputs_as_nchw=outputs_as_nchw,
            large_model=large_model,
            tensors_to_rename=tensors_to_rename,
            initialized_tables=initialized_tables,
            output_path=output_path)

        return model_proto, external_tensor_storage


def from_function(function, input_signature=None, opset=None, custom_ops=None, custom_op_handlers=None,
                  custom_rewriter=None, inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None,
                  shape_override=None, target=None, large_model=False, output_path=None):
    """Returns a ONNX model_proto for a tf.function.

    Args:
        function: the tf.function we want to convert
        input_signature: a tf.TensorSpec or a numpy array defining the shape/dtype of the input
        opset: the opset to be used for the ONNX model, default is the latest
        custom_ops: if a model contains ops not recognized by onnx runtime,
            you can tag these ops with a custom op domain so that the
            runtime can still open the model. Type is a dictionary `{op name: domain}`.
        target: list of workarounds applied to help certain platforms
        custom_op_handlers: dictionary of custom ops handlers
        custom_rewriter: list of custom graph rewriters
        extra_opset: list of extra opset's, for example the opset's used by custom ops
        shape_override: dict with inputs that override the shapes given by tensorflow
        inputs_as_nchw: transpose inputs in list from nhwc to nchw
        outputs_as_nchw: transpose outputs in list from nhwc to nchw
        large_model: use the ONNX external tensor storage format
        output_path: save model to output_path

    Returns:
        An ONNX model_proto and an external_tensor_storage dict.
    """
    if get_tf_version() < Version("2.0"):
        raise NotImplementedError("from_function requires tf-2.0 or newer")

    if input_signature is None:
        raise ValueError("from_function requires input_signature")

    concrete_func = function.get_concrete_function(*input_signature)

    input_names = [input_tensor.name for input_tensor in concrete_func.inputs
                   if input_tensor.dtype != tf.dtypes.resource]
    output_names = [output_tensor.name for output_tensor in concrete_func.outputs
                    if output_tensor.dtype != tf.dtypes.resource]

    initialized_tables = None
    tensors_to_rename = tensor_names_from_structed(concrete_func, input_names, output_names)

    with tf.device("/cpu:0"):
        frozen_graph = tf_loader.from_function(concrete_func, input_names, output_names, large_model=large_model)
        model_proto, external_tensor_storage = _convert_common(
            frozen_graph,
            name=concrete_func.name,
            continue_on_error=True,
            target=target,
            opset=opset,
            custom_ops=custom_ops,
            custom_op_handlers=custom_op_handlers,
            custom_rewriter=custom_rewriter,
            extra_opset=extra_opset,
            shape_override=shape_override,
            input_names=input_names,
            output_names=output_names,
            inputs_as_nchw=inputs_as_nchw,
            outputs_as_nchw=outputs_as_nchw,
            large_model=large_model,
            tensors_to_rename=tensors_to_rename,
            initialized_tables=initialized_tables,
            output_path=output_path)

        return model_proto, external_tensor_storage


def from_graph_def(graph_def, name=None, input_names=None, output_names=None, opset=None, custom_ops=None,
                   custom_op_handlers=None, custom_rewriter=None, inputs_as_nchw=None, outputs_as_nchw=None,
                   extra_opset=None, shape_override=None, target=None, large_model=False,
                   tensors_to_rename=None, output_path=None):
    """Returns a ONNX model_proto for a tensorflow graphdef.

    Args:
        graph_def: the graphdef we want to convert
        input_names: list of input names
        output_names: list of output names
        name: A name for the graph
        opset: the opset to be used for the ONNX model, default is the latest
        custom_ops: if a model contains ops not recognized by onnx runtime,
            you can tag these ops with a custom op domain so that the
            runtime can still open the model. Type is a dictionary `{op name: domain}`.
        target: list of workarounds applied to help certain platforms
        custom_op_handlers: dictionary of custom ops handlers
        custom_rewriter: list of custom graph rewriters
        extra_opset: list of extra opset's, for example the opset's used by custom ops
        shape_override: dict with inputs that override the shapes given by tensorflow
        inputs_as_nchw: transpose inputs in list from nhwc to nchw
        outputs_as_nchw: transpose outputs in list from nhwc to nchw
        large_model: use the ONNX external tensor storage format
        output_path: save model to output_path

    Returns:
        An ONNX model_proto and an external_tensor_storage dict.
    """
    if not input_names:
        raise ValueError("input_names needs to be provided")
    if not output_names:
        raise ValueError("output_names needs to be provided")
    if not name:
        name = "unknown"
    initialized_tables = None

    with tf.device("/cpu:0"):
        with tf.Graph().as_default() as tf_graph:
            with tf_loader.tf_session(graph=tf_graph) as sess:
                tf.import_graph_def(graph_def, name='')
                frozen_graph = tf_loader.freeze_session(sess, input_names=input_names, output_names=output_names)
                input_names = tf_loader.inputs_without_resource(sess, input_names)
                frozen_graph = tf_loader.tf_optimize(input_names, output_names, graph_def)

    model_proto, external_tensor_storage = _convert_common(
        frozen_graph,
        name=name,
        continue_on_error=True,
        target=target,
        opset=opset,
        custom_ops=custom_ops,
        custom_op_handlers=custom_op_handlers,
        custom_rewriter=custom_rewriter,
        extra_opset=extra_opset,
        shape_override=shape_override,
        input_names=input_names,
        output_names=output_names,
        inputs_as_nchw=inputs_as_nchw,
        outputs_as_nchw=outputs_as_nchw,
        large_model=large_model,
        tensors_to_rename=tensors_to_rename,
        initialized_tables=initialized_tables,
        output_path=output_path)

    return model_proto, external_tensor_storage


def from_tflite(tflite_path, input_names=None, output_names=None, opset=None, custom_ops=None, custom_op_handlers=None,
                custom_rewriter=None, inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None, shape_override=None,
                target=None, large_model=False, output_path=None):
    """Returns a ONNX model_proto for a tflite model file.

    Args:
        tflite_path: the tflite model file full path
        input_names: list of input names
        output_names: list of output names
        opset: the opset to be used for the ONNX model, default is the latest
        custom_ops: if a model contains ops not recognized by onnx runtime,
            you can tag these ops with a custom op domain so that the
            runtime can still open the model. Type is a dictionary `{op name: domain}`.
        custom_op_handlers: dictionary of custom ops handlers
        custom_rewriter: list of custom graph rewriters
        inputs_as_nchw: transpose inputs in list from nhwc to nchw
        outputs_as_nchw: transpose outputs in list from nhwc to nchw
        extra_opset: list of extra opset's, for example the opset's used by custom ops
        shape_override: dict with inputs that override the shapes given by tensorflow
        target: list of workarounds applied to help certain platforms
        large_model: use the ONNX external tensor storage format
        output_path: save model to output_path

    Returns:
        An ONNX model_proto and an external_tensor_storage dict.
    """
    if not tflite_path:
        raise ValueError("tflite_path needs to be provided")

    with tf.device("/cpu:0"):
        model_proto, external_tensor_storage = _convert_common(
            None,
            tflite_path=tflite_path,
            name=os.path.splitext(os.path.basename(tflite_path))[0],
            continue_on_error=True,
            target=target,
            opset=opset,
            custom_ops=custom_ops,
            custom_op_handlers=custom_op_handlers,
            custom_rewriter=custom_rewriter,
            extra_opset=extra_opset,
            shape_override=shape_override,
            input_names=input_names,
            output_names=output_names,
            inputs_as_nchw=inputs_as_nchw,
            outputs_as_nchw=outputs_as_nchw,
            large_model=large_model,
            tensors_to_rename=None,
            initialized_tables=None,
            output_path=output_path)

        return model_proto, external_tensor_storage


if __name__ == "__main__":
    main()
