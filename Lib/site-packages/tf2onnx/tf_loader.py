# SPDX-License-Identifier: Apache-2.0


"""Methods to load tensorflow graph from graphdef, checkpoint or saved_model."""

import logging
import uuid
from packaging.version import Version

import tensorflow as tf
from google.protobuf.message import DecodeError
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.ops import lookup_ops
from tensorflow.python.util import compat

from tf2onnx import utils
from tf2onnx.tf_utils import (get_tf_version, tflist_to_onnx, get_hash_table_info, replace_placeholders_with_tables,
                              HashTableInfo)

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,unused-import,no-value-for-parameter,unexpected-keyword-arg,ungrouped-imports
# pylint: disable=missing-function-docstring,import-outside-toplevel,useless-import-alias,missing-docstring


def is_tf2():
    return tf.__version__.startswith("2.")


def _not_implemented_tf_placeholder(name):
    """Creates a placeholder function for missing Tensorflow imports"""

    def not_implemented_tf_placeholder(*args, **kwargs):
        raise NotImplementedError(
            f'Tensorflow verison {tf.__version__} does not implement '
            f'`{name}`, try converting your model with a different version.'
        )

    return not_implemented_tf_placeholder


try:
    from tensorflow.python.framework.function_def_to_graph import function_def_to_graph
except ImportError:
    function_def_to_graph = _not_implemented_tf_placeholder('function_def_to_graph')

try:
    # pylint: disable=protected-access
    from tensorflow.python.saved_model.load import _RestoredResource as TfRestoredResourceType
    from tensorflow.python.ops.lookup_ops import StaticHashTable as TfStaticHashTableType
    from tensorflow.python.training.tracking.base import Trackable as TfTrackableType
except ImportError:
    TfRestoredResourceType = tuple()  # isinstance(x, tuple()) is always false
    TfStaticHashTableType = tuple()
    TfTrackableType = tuple()

if is_tf2():
    convert_variables_to_constants = tf.compat.v1.graph_util.convert_variables_to_constants
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
else:
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    convert_variables_to_constants_v2 = _not_implemented_tf_placeholder('convert_variables_to_constants_v2')

if is_tf2():
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_global_variables = tf.compat.v1.global_variables
    tf_session = tf.compat.v1.Session  # pylint: disable=invalid-name
    tf_graphdef = tf.compat.v1.GraphDef
    tf_import_meta_graph = tf.compat.v1.train.import_meta_graph
    tf_gfile = tf.io.gfile
    tf_placeholder = tf.compat.v1.placeholder
    tf_placeholder_with_default = tf.compat.v1.placeholder_with_default
elif Version(tf.__version__) >= Version("1.13"):
    # 1.13 introduced the compat namespace
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_global_variables = tf.compat.v1.global_variables
    tf_session = tf.compat.v1.Session  # pylint: disable=invalid-name
    tf_graphdef = tf.compat.v1.GraphDef
    tf_import_meta_graph = tf.compat.v1.train.import_meta_graph
    tf_gfile = tf.gfile
    tf_placeholder = tf.compat.v1.placeholder
    tf_placeholder_with_default = tf.compat.v1.placeholder_with_default
else:
    # older than 1.13
    tf_reset_default_graph = tf.reset_default_graph
    tf_global_variables = tf.global_variables
    tf_session = tf.Session  # pylint: disable=invalid-name
    tf_graphdef = tf.GraphDef
    tf_import_meta_graph = tf.train.import_meta_graph
    tf_gfile = tf.gfile
    tf_placeholder = tf.placeholder
    tf_placeholder_with_default = tf.placeholder_with_default


def inputs_without_resource(sess, input_names):
    try:
        new_input_names = []
        for n in input_names:
            t = sess.graph.get_tensor_by_name(n)
            if t.dtype != tf.dtypes.resource:
                new_input_names.append(n)
        input_names = new_input_names
    except:  # pylint: disable=bare-except
        pass
    return input_names


def convert_variables_to_constants_large_model(func):
    # For large models we use internal tf methods as a hack

    if tf.__version__.startswith("2.1.") or tf.__version__.startswith("2.0."):
        from tensorflow.python.framework import convert_to_constants
        orig_fn = convert_to_constants._construct_concrete_function  # pylint: disable=protected-access
        def fake_construct_fn(func, output_graph_def, converted_input_indices):
            # Return graph_def without loading it to avoid crash. Will fix errors in graph_def later.
            return output_graph_def
        convert_to_constants._construct_concrete_function = fake_construct_fn  # pylint: disable=protected-access
        try:
            frozen_graph_def = convert_to_constants.convert_variables_to_constants_v2(func, lower_control_flow=False)
        finally:
            convert_to_constants._construct_concrete_function = orig_fn  # pylint: disable=protected-access
        return frozen_graph_def

    if tf.__version__.startswith("2.2."):
        try:
            from tensorflow.python.framework.convert_to_constants import \
                 _convert_variables_to_constants_v2_impl # pylint: disable=protected-access
        except ImportError:
            _not_implemented_tf_placeholder("_convert_variables_to_constants_v2_impl")()
        frozen_graph_def, _ = \
            _convert_variables_to_constants_v2_impl(func, lower_control_flow=False, aggressive_inlining=True)
        return frozen_graph_def

    try:
        from tensorflow.python.framework.convert_to_constants import \
            _FunctionConverterData, _replace_variables_by_constants # pylint: disable=protected-access
    except ImportError:
        _not_implemented_tf_placeholder("_replace_variables_by_constants")()

    from tensorflow.python.framework import tensor_util, tensor_shape
    make_tensor_proto_original = tensor_util.make_tensor_proto
    # Hack to avoid 2GB check
    def make_tensor_proto_wrapped(values, dtype=None, shape=None, verify_shape=False, allow_broadcast=False):
        try:
            return make_tensor_proto_original(values, dtype, shape, verify_shape, allow_broadcast)
        except ValueError:
            if dtype is None:
                dtype = tf.dtypes.as_dtype(values.dtype).as_datatype_enum
            tensor_proto = tensor_pb2.TensorProto(
                dtype=dtype,
                tensor_shape=tensor_shape.as_shape(values.shape).as_proto())
            tensor_proto.tensor_content = values.tobytes()
            return tensor_proto
    tensor_util.make_tensor_proto = make_tensor_proto_wrapped

    try:
        function_converter = _FunctionConverterData
        if Version(tf.__version__) >= Version("2.6.0"):
            from tensorflow.python.eager import context
            from tensorflow.python.framework.convert_to_constants import _FunctionConverterDataInEager, \
                _FunctionConverterDataInGraph
            if context.executing_eagerly():
                function_converter = _FunctionConverterDataInEager
            else:
                function_converter = _FunctionConverterDataInGraph
        else:
            function_converter = _FunctionConverterData
        converter_data = function_converter(func=func, lower_control_flow=False, aggressive_inlining=True)
        frozen_graph_def, _ = _replace_variables_by_constants(converter_data=converter_data)
    finally:
        tensor_util.make_tensor_proto = make_tensor_proto_original
    return frozen_graph_def


def fix_freezing_errors(graph_def):
    assign_var_ops = []
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op in ["AssignVariableOp", "AssignSubVariableOp"]:
            assign_var_ops.append(graph_def.node.pop(i).name)
            logger.warning("Removed %s %s", graph_def.node[i].op, assign_var_ops[-1])
    names_to_remove = set(assign_var_ops)
    for n in graph_def.node:
        for i in reversed(range(len(n.input))):
            if n.input[i].startswith("^") and n.input[i][1:] in names_to_remove:
                n.input.pop(i)
    return graph_def


def fix_freezing_errors_part2(graph_def):
    # Sometimes tf freezing fails to convert ResourceGather ops in subgraphs
    for f in graph_def.library.function:
        for n in f.node_def:
            if n.op == "ResourceGather":
                # Convert to standard Gather op. Freezing will have replaced resource with constant.
                # Needed because of: https://github.com/tensorflow/tensorflow/issues/51488
                n.op = "Gather"
                n.attr['Tparams'].type = n.attr['dtype'].type
                del n.attr['dtype']
                if 'batch_dims' in n.attr:
                    v = n.attr['batch_dims'].i
                    utils.make_sure(v == 0, "Unsupported batch_dims value of ResourceGather %d", v)
                    del n.attr['batch_dims']
    return graph_def


def from_trackable(trackable, concrete_func, inputs, outputs, large_model):
    err_large_model = "model exceeds maximum protobuf size of 2GB. Try setting large_model."

    # Avoid errors due to bug in TF freezing
    removed_resource_to_placeholder, placeholder_to_resource, graph_captures_copy, func_captures_copy = \
        _remove_non_variable_resources_from_captures(concrete_func)

    try:
        frozen_graph = from_function(concrete_func, inputs, outputs, large_model)
    except ValueError as e:
        if any(msg in str(e) for msg in ["exceeds maximum protobuf size of 2GB", "string too long"]):
            raise ValueError(err_large_model)
        raise e

    # We might be returning the concrete_func so let's put it back in working order
    _restore_captured_resources(concrete_func, graph_captures_copy, func_captures_copy)

    table_info = get_hash_table_info(frozen_graph)
    placeholder_to_table_info = {}
    _get_hash_table_info_from_trackable(trackable, table_info,
                                        removed_resource_to_placeholder, placeholder_to_table_info)

    initialized_tables = {}
    for info in table_info:
        if info.shared_name is not None:
            h = lookup_ops.hash_table_v2(info.key_dtype, info.val_dtype, shared_name=info.shared_name)
            n = info.shared_name
        elif info.resource_input in placeholder_to_resource and info.resource_input not in placeholder_to_table_info:
            # We found a lookup op with no corresponding HashTable op, but we can associate the placeholder input
            # from the op with the resource handle from graph captures and make up a shared_name
            h = placeholder_to_resource[info.resource_input]
            n = str(uuid.uuid4()).encode()
            info.shared_name = n
            placeholder_to_table_info[info.resource_input] = info
        else:
            # Found a lookup op but the corresponding HashTable op has already been found and processed.
            continue
        try:
            k, v = lookup_ops.lookup_table_export_v2(h, info.key_dtype, info.val_dtype)
            initialized_tables[n] = (k.numpy(), v.numpy())
        except Exception:  # pylint: disable=broad-except
            logger.warning("Could not initialize table with shared_name = %r", n)

    for placeholder in removed_resource_to_placeholder.values():
        if placeholder not in placeholder_to_table_info:
            logger.error("Could not find table resource to replace placeholder %s", placeholder)

    replace_placeholders_with_tables(frozen_graph, placeholder_to_table_info)

    return frozen_graph, initialized_tables


def from_function(func, input_names, output_names, large_model=False):
    if large_model:
        return convert_variables_to_constants_large_model(func)

    try:
        if get_tf_version() < Version("2.2"):
            frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
        else:
            frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False, aggressive_inlining=True)
    except ValueError as e:
        if "incompatible with expected resource" in str(e):
            bad_graph_def = convert_variables_to_constants_large_model(func)
            logger.warning("TF freezing failed. Attempting to fix freezing errors.")
            graph_def = fix_freezing_errors(bad_graph_def)
        else:
            raise e
    else:
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    graph_def = fix_freezing_errors_part2(graph_def)

    # output_names = [i.name for i in frozen_func.outputs]
    with tf.Graph().as_default() as tf_graph:
        with tf_session(graph=tf_graph) as sess:
            tf.import_graph_def(graph_def, name='')
            input_names = inputs_without_resource(sess, input_names)
            graph_def = tf_optimize(input_names, output_names, graph_def)
    return graph_def


def freeze_session(sess, input_names=None, output_names=None, get_tables=False):
    """Freezes the state of a session into a pruned computation graph."""
    output_node_names = [i.split(':')[0] for i in output_names]
    keep_var_names = [i.split(':')[0] for i in input_names]
    with sess.graph.as_default():
        output_node_names = output_node_names or []
        output_node_names += [v.op.name for v in tf_global_variables()]
        output_node_names += keep_var_names
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        for node in graph_def.node:
            node.device = ""
        graph_def = convert_variables_to_constants(sess, graph_def, output_node_names)
        table_info = get_hash_table_info(graph_def)
        if get_tables:
            initialized_tables = {}
            tf.tables_initializer().run(session=sess)
            for info in table_info:
                if info.shared_name is None:
                    continue
                h = lookup_ops.hash_table_v2(info.key_dtype, info.val_dtype, shared_name=info.shared_name)
                n = info.shared_name
                try:
                    k, v = lookup_ops.lookup_table_export_v2(h, info.key_dtype, info.val_dtype)
                    k, v = sess.run([k, v])
                    initialized_tables[n] = (k, v)
                except Exception:  # pylint: disable=broad-except
                    logger.warning("Could not initialize table with shared_name = %r", n)
            return graph_def, initialized_tables
    return graph_def


def remove_redundant_inputs(frozen_graph, input_names):
    """Remove redundant inputs not in frozen graph."""
    frozen_inputs = []
    # get inputs in frozen graph
    node_names = set(n.name for n in frozen_graph.node)
    frozen_inputs = [inp for inp in input_names if utils.node_name(inp) in node_names]
    deleted_inputs = list(set(input_names) - set(frozen_inputs))
    if deleted_inputs:
        logger.warning("inputs [%s] is not in frozen graph, delete them", ",".join(deleted_inputs))
    return frozen_inputs


def from_graphdef(model_path, input_names, output_names):
    """Load tensorflow graph from graphdef."""
    # make sure we start with clean default graph
    tf_reset_default_graph()
    with tf_session() as sess:
        graph_def = tf_graphdef()
        with tf_gfile.GFile(model_path, 'rb') as f:
            try:
                content = f.read()
            except Exception as e:
                raise OSError(
                    "Unable to load file '{}'.".format(model_path)) from e
            try:
                graph_def.ParseFromString(content)
            except DecodeError:
                content_as_bytes = compat.as_bytes(content)
                saved_model = saved_model_pb2.SavedModel()
                saved_model.ParseFromString(content_as_bytes)
                graph_def = saved_model.meta_graphs[0].graph_def
            except Exception as e:
                raise RuntimeError(
                    "Unable to parse file '{}'.".format(model_path)) from e
            tf.import_graph_def(graph_def, name='')
        input_names = inputs_without_resource(sess, input_names)
        frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
        input_names = remove_redundant_inputs(frozen_graph, input_names)

    tf_reset_default_graph()
    with tf_session() as sess:
        input_names = inputs_without_resource(sess, input_names)
        frozen_graph = tf_optimize(input_names, output_names, frozen_graph)
    tf_reset_default_graph()
    return frozen_graph, input_names, output_names


def from_checkpoint(model_path, input_names, output_names):
    """Load tensorflow graph from checkpoint."""
    # make sure we start with clean default graph
    tf_reset_default_graph()
    # model_path = checkpoint/checkpoint.meta
    with tf.device("/cpu:0"):
        with tf_session() as sess:
            saver = tf_import_meta_graph(model_path, clear_devices=True)
            # restore from model_path minus the ".meta"
            saver.restore(sess, model_path[:-5])
            input_names = inputs_without_resource(sess, input_names)
            frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
            input_names = remove_redundant_inputs(frozen_graph, input_names)

        tf_reset_default_graph()
        with tf_session() as sess:
            frozen_graph = tf_optimize(input_names, output_names, frozen_graph)
    tf_reset_default_graph()
    return frozen_graph, input_names, output_names


def _from_saved_model_v1(sess, model_path, input_names, output_names, tag, signature_names, use_graph_names):
    """Load tensorflow graph from saved_model."""

    wrn_no_tag = "'--tag' not specified for saved_model. Using --tag serve"
    wrn_empty_tag = "'--tag' value is empty string. Using tags = []"
    wrn_empty_sig = "'--signature_def' not provided. Using all signatures."

    if tag is None:
        tag = [tf.saved_model.tag_constants.SERVING]
        logger.warning(wrn_no_tag)

    if not signature_names:
        logger.warning(wrn_empty_sig)

    if tag == '':
        tag = []
        logger.warning(wrn_empty_tag)

    if not isinstance(tag, list):
        tag = [tag]

    imported = tf.saved_model.loader.load(sess, tag, model_path)
    signatures = []
    for k in imported.signature_def.keys():
        if k in signature_names or (not signature_names and not k.startswith("_")):
            signatures.append(k)
    try:
        from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
        # pylint: disable=unnecessary-lambda
        get_signature_def = lambda meta_graph_def, k: \
            signature_def_utils.get_signature_def_by_key(meta_graph_def, k)
    except ImportError:
        # TF1.12 changed the api
        get_signature_def = lambda meta_graph_def, k: meta_graph_def.signature_def[k]

    tensors_to_rename = {}
    if input_names is None:
        input_names = []
        for k in signatures:
            inputs_tensor_info = get_signature_def(imported, k).inputs
            for structured_name, input_tensor in inputs_tensor_info.items():
                if input_tensor.name not in input_names:
                    input_names.append(input_tensor.name)
                    if not use_graph_names:
                        tensors_to_rename[input_tensor.name] = structured_name
    if output_names is None:
        output_names = []
        for k in signatures:
            outputs_tensor_info = get_signature_def(imported, k).outputs
            for structured_name, output_tensor in outputs_tensor_info.items():
                if output_tensor.name not in output_names:
                    output_names.append(output_tensor.name)
                    if not use_graph_names:
                        tensors_to_rename[output_tensor.name] = structured_name
    frozen_graph, initialized_tables = \
        freeze_session(sess, input_names=input_names, output_names=output_names, get_tables=True)
    return frozen_graph, input_names, output_names, initialized_tables, tensors_to_rename


def _get_hash_table_info_from_trackable(trackable, table_info,
                                        removed_resource_to_placeholder, placeholder_to_table_info):
    # pylint: disable=protected-access
    stack = [trackable]
    visited = set()
    while stack:
        r = stack.pop()
        visited.add(id(r))
        try:
            for trackable_ref in r._checkpoint_dependencies:
                if id(trackable_ref.ref) not in visited:
                    if isinstance(trackable_ref.ref, TfTrackableType):
                        stack.append(trackable_ref.ref)
        except Exception:  # pylint: disable=broad-except
            continue
        for t in r.__dict__.values() if hasattr(r, '__dict__') else []:
            if isinstance(t, TfStaticHashTableType) and hasattr(t, '_shared_name'):
                info = HashTableInfo(t._shared_name.encode(), t.key_dtype.as_datatype_enum,
                                     t.value_dtype.as_datatype_enum)
                table_info.append(info)
                table_handle = id(t.resource_handle)
                if table_handle in removed_resource_to_placeholder:
                    placeholder_to_table_info[removed_resource_to_placeholder[table_handle]] = info
        if isinstance(r, TfRestoredResourceType) and hasattr(r, '_create_resource'):
            try:
                table_handle = id(r.resource_handle)
            except Exception:  # pylint: disable=broad-except
                continue
            initializer = r._create_resource.concrete_functions[0].function_def
            new_table_info = get_hash_table_info(initializer.node_def)
            table_info.extend(new_table_info)
            if table_handle in removed_resource_to_placeholder and len(new_table_info) == 1:
                placeholder_to_table_info[removed_resource_to_placeholder[table_handle]] = new_table_info[0]


def _get_resources_from_captures(concrete_func, graph_captures):
    resource_id_to_placeholder = {}
    placeholder_to_resource = {}
    keys_to_be_removed = []

    # pylint: disable=protected-access
    variable_handles = {id(v.handle) for v in concrete_func.graph.variables}
    for k, v in list(graph_captures.items()):
        val_tensor, name_tensor = v
        if val_tensor.dtype == tf.resource and id(val_tensor) not in variable_handles:
            resource_id_to_placeholder[id(val_tensor)] = name_tensor.name.split(':')[0]
            placeholder_to_resource[name_tensor.name.split(':')[0]] = val_tensor
            keys_to_be_removed.append(k)
            for i in reversed(range(len(concrete_func._captured_inputs))):
                if concrete_func._captured_inputs[i] is val_tensor:
                    concrete_func._captured_inputs.pop(i)

    return resource_id_to_placeholder, placeholder_to_resource, keys_to_be_removed


def _remove_non_variable_resources_from_captures(concrete_func):
    """
    Removes all non-variable resources (such as tables) from a function's captured inputs to prevent tf from
    raising a 'cannot convert dtype resource to numpy' error while freezing the graph.
    """
    # pylint: disable=protected-access
    resource_id_to_placeholder = {}
    placeholder_to_resource = {}
    graph_captures_copy = None
    func_captures_copy = None
    if hasattr(concrete_func.graph, '_captures') and hasattr(concrete_func, '_captured_inputs'):
        graph_captures_copy = concrete_func.graph._captures.copy()
        func_captures_copy = concrete_func._captured_inputs.copy()

        resource_id_to_placeholder, placeholder_to_resource, keys_to_be_removed = \
            _get_resources_from_captures(concrete_func, concrete_func.graph._captures)
        for key in keys_to_be_removed:
            del concrete_func.graph._captures[key]
    elif hasattr(concrete_func.graph, 'function_captures') and hasattr(concrete_func, '_captured_inputs'):
        # Since tensorflow 2.13.0, _captures has been removed and replaced with function_captures
        graph_captures = {}
        for k, v in concrete_func.graph.function_captures.by_val_external.items():
            graph_captures[k] = (v, concrete_func.graph.function_captures.by_val_internal[k])
        graph_captures_copy = graph_captures.copy()
        func_captures_copy = concrete_func._captured_inputs.copy()

        resource_id_to_placeholder, placeholder_to_resource, keys_to_be_removed = \
            _get_resources_from_captures(concrete_func, graph_captures)
        for key in keys_to_be_removed:
            del concrete_func.graph.function_captures.by_val_internal[key]
            del concrete_func.graph.function_captures.by_val_external[key]
    else:
        logger.warning(
            "Could not search for non-variable resources. Concrete function internal representation may have changed.")
    return resource_id_to_placeholder, placeholder_to_resource, graph_captures_copy, func_captures_copy


def _restore_captured_resources(concrete_func, graph_captures_copy, func_captures_copy):
    """Undoes effect of _remove_non_variable_resources_from_captures on concrete_func"""
    # pylint: disable=protected-access
    if hasattr(concrete_func.graph, '_captures') and hasattr(concrete_func, '_captured_inputs'):
        concrete_func.graph._captures = graph_captures_copy
        concrete_func._captured_inputs = func_captures_copy


def _from_saved_model_v2(model_path, input_names, output_names, tag, signature_def,
                         concrete_function_index, large_model, use_graph_names):
    """Load tensorflow graph from saved_model."""

    wrn_no_tag = "'--tag' not specified for saved_model. Using --tag serve"
    wrn_empty_tag = "'--tag' value is empty string. Using tag =[[]]"
    wrn_sig_1 = "'--signature_def' not specified, using first signature: %s"
    err_many_sig = "Cannot load multiple signature defs in TF2.x: %s"
    err_no_call = "Model doesn't contain usable concrete functions under  __call__. Try --signature-def instead."
    err_index = "Invalid concrete_function value: %i. Valid values are [0 to %i]"
    err_no_sig = "No signatures found in model. Try --concrete_function instead."
    err_sig_nomatch = "Specified signature not in model %s"

    if tag is None:
        tag = ['serve']
        logger.warning(wrn_no_tag)

    if tag == '':
        tag = [[]]
        logger.warning(wrn_empty_tag)

    utils.make_sure(len(signature_def) < 2, err_many_sig, str(signature_def))
    imported = tf.saved_model.load(model_path, tags=tag)  # pylint: disable=no-value-for-parameter

    all_sigs = imported.signatures.keys()
    valid_sigs = [s for s in all_sigs if not s.startswith("_")]
    logger.info("Signatures found in model: %s", "[" + ",".join(valid_sigs) + "].")

    concrete_func = None
    if concrete_function_index is not None:
        utils.make_sure(hasattr(imported, "__call__"), err_no_call)
        utils.make_sure(concrete_function_index < len(imported.__call__.concrete_functions),
                        err_index, concrete_function_index, len(imported.__call__.concrete_functions) - 1)
        args, kwargs = imported.__call__.concrete_functions[concrete_function_index].structured_input_signature
        concrete_func = imported.__call__.get_concrete_function(*args, **kwargs)
    elif signature_def:
        utils.make_sure(signature_def[0] in valid_sigs, err_sig_nomatch, signature_def[0])
        concrete_func = imported.signatures[signature_def[0]]
    else:
        utils.make_sure(len(valid_sigs) > 0, err_no_sig)
        logger.warning(wrn_sig_1, valid_sigs[0])
        concrete_func = imported.signatures[valid_sigs[0]]

    tensors_to_rename = {}
    if input_names is None:
        inputs = [tensor.name for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
        if hasattr(concrete_func.graph, '_captures'):
            graph_captures = concrete_func.graph._captures  # pylint: disable=protected-access
            captured_inputs = [t_name.name for _, t_name in graph_captures.values()]
        else:
            graph_captures = concrete_func.graph.function_captures.by_val_internal
            captured_inputs = [t.name for t in graph_captures.values()]
        inputs = [inp for inp in inputs if inp not in captured_inputs]
        if concrete_func.structured_input_signature is not None and not use_graph_names:
            flat_structured_inp = tf.nest.flatten(concrete_func.structured_input_signature)
            structured_inputs = [t.name for t in flat_structured_inp if isinstance(t, tf.TensorSpec)]
            tensors_to_rename.update(zip(inputs, structured_inputs))
    else:
        inputs = input_names

    if output_names is None:
        outputs = [tensor.name for tensor in concrete_func.outputs if tensor.dtype != tf.dtypes.resource]
        if isinstance(concrete_func.structured_outputs, dict) and not use_graph_names:
            # outputs are sorted, sort structured_outputs the same way
            structured_outputs = sorted(concrete_func.structured_outputs.keys())
            tensors_to_rename.update(zip(outputs, structured_outputs))
            logger.info("Output names: %r", structured_outputs)
        else:
            logger.info("Output names: %r", outputs)
    else:
        outputs = output_names

    frozen_graph, initialized_tables = from_trackable(imported, concrete_func, inputs, outputs, large_model)

    return frozen_graph, inputs, outputs, concrete_func, imported, initialized_tables, tensors_to_rename


def from_saved_model(model_path, input_names, output_names, tag=None,
                     signatures=None, concrete_function=None, large_model=False,
                     return_concrete_func=False, return_initialized_tables=False,
                     return_tensors_to_rename=False, use_graph_names=False):
    """Load tensorflow graph from saved_model."""
    if signatures is None:
        signatures = []
    tf_reset_default_graph()
    with tf.device("/cpu:0"):
        if is_tf2():
            frozen_graph, input_names, output_names, concrete_func, imported, initialized_tables, tensors_to_rename = \
                _from_saved_model_v2(model_path, input_names, output_names,
                                     tag, signatures, concrete_function, large_model, use_graph_names)
            result = [frozen_graph, input_names, output_names]
            if return_concrete_func:
                result += [concrete_func, imported]
            if return_initialized_tables:
                result += [initialized_tables]
            if return_tensors_to_rename:
                result += [tensors_to_rename]
        else:
            with tf_session() as sess:
                frozen_graph, input_names, output_names, initialized_tables, tensors_to_rename = \
                    _from_saved_model_v1(sess, model_path, input_names, output_names, tag, signatures, use_graph_names)
                result = [frozen_graph, input_names, output_names]
                if return_initialized_tables:
                    result += [initialized_tables]
                if return_tensors_to_rename:
                    result += [tensors_to_rename]
    tf_reset_default_graph()
    return result


def from_keras(model_path, input_names, output_names):
    """Load keras model - experimental for now."""
    from tensorflow import keras as _keras
    from tensorflow.python.keras.saving import saving_utils as _saving_utils

    # Handles Keras when Eager mode is enabled.
    custom_objects = None
    with tf.device("/cpu:0"):
        if tf.executing_eagerly():
            _keras.backend.clear_session()
            _keras.backend.set_learning_phase(False)
            keras_model = _keras.models.load_model(model_path, custom_objects)

            function = _saving_utils.trace_model_call(keras_model)
            concrete_func = function.get_concrete_function()
            # allow to pass inputs and outputs from caller if we don't want all of them
            input_names = [input_tensor.name for input_tensor in concrete_func.inputs
                           if input_tensor.dtype != tf.dtypes.resource]
            output_names = [output_tensor.name for output_tensor in concrete_func.outputs
                            if output_tensor.dtype != tf.dtypes.resource]
            frozen_graph = from_function(concrete_func, input_names, output_names)
        else:
            # Handles Keras when Eager mode is disabled.
            _keras.backend.clear_session()
            _keras.backend.set_learning_phase(False)
            keras_model = _keras.models.load_model(model_path, custom_objects)
            # allow to pass inputs and outputs from caller if we don't want all of them
            input_names = keras_model.inputs
            output_names = keras_model.outputs
            sess = _keras.backend.get_session()
            input_names = inputs_without_resource(sess, input_names)
            frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
            tf_reset_default_graph()
            with tf_session() as sess:
                frozen_graph = tf_optimize(input_names, output_names, frozen_graph)
            tf_reset_default_graph()
    return frozen_graph, input_names, output_names


def tf_optimize_grappler(input_names, output_names, graph_def):
    from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2, rewriter_config_pb2
    from tensorflow.python.grappler import tf_optimizer as tf_opt

    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    config.graph_options.infer_shapes = True
    # TODO: if we turn on pruning, grappler removes some identities that the tf-1.x lstm rewriter
    # depends on so for now don't turn this on, constfold is always enabled now.
    rewrite_options.optimizers[:] = [
        # 'pruning', 'constfold', 'arithmetic', 'dependency', 'function',
        'constfold', 'function'
    ]

    if is_tf2():
        # add for tf2.x lstm optimization.
        rewrite_options.optimizers.append('dependency')

    if Version(tf.__version__) >= Version("2.5"):
        # This flag disables folding QDQ nodes around constants in the network (eg: around conv/FC weights)
        rewrite_options.experimental_disable_folding_quantization_emulation = True

    meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for t in input_names + output_names:
        fetch_collection.node_list.value.append(t)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
    graph_def = tf_opt.OptimizeGraph(config, meta_graph)
    return graph_def


def tf_optimize(input_names, output_names, graph_def):
    """Extract inference subgraph and optimize graph."""
    assert isinstance(input_names, list)
    assert isinstance(output_names, list)

    want_grappler = is_tf2() or Version(tf.__version__) >= Version("1.15")
    if want_grappler:
        graph_def = tf_optimize_grappler(input_names, output_names, graph_def)
    else:
        # the older transform path
        from tensorflow.tools.graph_transforms import TransformGraph  # pylint: disable=redefined-outer-name
        transforms = [
            "fold_constants(ignore_errors=true)",
            "remove_attribute(attribute_name=_class)",  # remove node colocation attributes
            "fold_batch_norms",
            "fold_old_batch_norms",
        ]
        graph_def = TransformGraph(graph_def, input_names, output_names, transforms)

    return graph_def


def tf_reload_graph(tf_graph):
    """Invoke tensorflow cpp shape inference by reloading graph_def."""
    # invoke c api if tf version is below 1.8
    if get_tf_version() < Version("1.8"):
        logger.debug(
            "On TF < 1.8, graph is constructed by python API, "
            "which doesn't invoke shape inference, please set "
            "TF_C_API_GRAPH_CONSTRUCTION=1 to enable it"
        )

    graph_def = tf_graph.as_graph_def(add_shapes=True)
    with tf.Graph().as_default() as inferred_graph:
        tf.import_graph_def(graph_def, name="")
    return inferred_graph


def is_function(g):
    if is_tf2():
        return 'tensorflow.python.framework.func_graph.FuncGraph' in str(type(g))
    return False


_FUNCTIONS = {}


def resolve_functions(tf_graph):
    def toposort(data):
        while True:
            ordered = set(item for item, dep in data.items() if not dep)
            if not ordered:
                break
            yield ordered
            data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}

    _, _, _, _, _, functions = tflist_to_onnx(tf_graph, {})
    data = {}
    for k, fdef in tf_graph._functions.items():  # pylint: disable=protected-access
        input_shapes = functions.get(k)
        fdef = fdef.definition
        if input_shapes and len(fdef.signature.input_arg) < len(input_shapes):
            input_shapes = input_shapes[:len(fdef.signature.input_arg)]
        try:
            func = function_def_to_graph(fdef, input_shapes=input_shapes)
        except:  # pylint: disable=bare-except
            # if there is a mismatch between caller and function use the functions shape
            logger.warning("shape mismatch between caller and function: %s", k)
            func = function_def_to_graph(fdef)
        _FUNCTIONS[k] = func
        _, _, _, _, _, tfunctions = tflist_to_onnx(func, {})
        functions.update(tfunctions)
        data[k] = set(tfunctions.keys())

    result = []
    for d in toposort(data):
        result.extend(list(d))
    return [_FUNCTIONS[k] for k in result]


def set_function(name, func):
    _FUNCTIONS[name] = func


def find_function(name):
    return _FUNCTIONS.get(name)


def clear_functions():
    _FUNCTIONS.clear()
