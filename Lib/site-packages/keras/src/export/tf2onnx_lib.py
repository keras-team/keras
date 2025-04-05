import copy
import functools
import logging
import traceback

import numpy as np


@functools.lru_cache()
def patch_tf2onnx():
    """Patches `tf2onnx` to ensure compatibility with numpy>=2.0.0."""

    from onnx import AttributeProto
    from onnx import TensorProto

    from keras.src.utils.module_utils import tf2onnx

    logger = logging.getLogger(tf2onnx.__name__)

    def patched_rewrite_constant_fold(g, ops):
        """
        We call tensorflow transform with constant folding but in some cases
        tensorflow does fold all constants. Since there are a bunch of ops in
        onnx that use attributes where tensorflow has dynamic inputs, we badly
        want constant folding to work. For cases where tensorflow missed
        something, make another pass over the graph and fix want we care about.
        """
        func_map = {
            "Add": np.add,
            "GreaterEqual": np.greater_equal,
            "Cast": np.asarray,
            "ConcatV2": np.concatenate,
            "Less": np.less,
            "ListDiff": np.setdiff1d,
            "Mul": np.multiply,
            "Pack": np.stack,
            "Range": np.arange,
            "Sqrt": np.sqrt,
            "Sub": np.subtract,
        }
        ops = list(ops)

        keep_looking = True
        while keep_looking:
            keep_looking = False
            for idx, op in enumerate(ops):
                func = func_map.get(op.type)
                if func is None:
                    continue
                if set(op.output) & set(g.outputs):
                    continue
                try:
                    inputs = []
                    for node in op.inputs:
                        if not node.is_const():
                            break
                        inputs.append(node.get_tensor_value(as_list=False))

                    logger.debug(
                        "op name %s, %s, %s",
                        op.name,
                        len(op.input),
                        len(inputs),
                    )
                    if inputs and len(op.input) == len(inputs):
                        logger.info(
                            "folding node type=%s, name=%s" % (op.type, op.name)
                        )
                        if op.type == "Cast":
                            dst = op.get_attr_int("to")
                            np_type = tf2onnx.utils.map_onnx_to_numpy_type(dst)
                            val = np.asarray(*inputs, dtype=np_type)
                        elif op.type == "ConcatV2":
                            axis = inputs[-1]
                            values = inputs[:-1]
                            val = func(tuple(values), axis)
                        elif op.type == "ListDiff":
                            out_type = op.get_attr_int("out_idx")
                            np_type = tf2onnx.utils.map_onnx_to_numpy_type(
                                out_type
                            )
                            val = func(*inputs)
                            val = val.astype(np_type)
                        elif op.type in ["Pack"]:
                            # handle ops that need input array and axis
                            axis = op.get_attr_int("axis")
                            val = func(inputs, axis=axis)
                        elif op.type == "Range":
                            dtype = op.get_attr_int("Tidx")
                            np_type = tf2onnx.utils.map_onnx_to_numpy_type(
                                dtype
                            )
                            val = func(*inputs, dtype=np_type)
                        else:
                            val = func(*inputs)

                        new_node_name = tf2onnx.utils.make_name(op.name)
                        new_output_name = new_node_name
                        old_output_name = op.output[0]
                        old_node_name = op.name
                        logger.debug(
                            "create const node [%s] replacing [%s]",
                            new_node_name,
                            old_node_name,
                        )
                        ops[idx] = g.make_const(new_node_name, val)

                        logger.debug(
                            "replace old output [%s] with new output [%s]",
                            old_output_name,
                            new_output_name,
                        )
                        # need to re-write the consumers input name to use the
                        # const name
                        consumers = g.find_output_consumers(old_output_name)
                        if consumers:
                            for consumer in consumers:
                                g.replace_input(
                                    consumer, old_output_name, new_output_name
                                )

                        # keep looking until there is nothing we can fold.
                        # We keep the graph in topological order so if we
                        # folded, the result might help a following op.
                        keep_looking = True
                except Exception as ex:
                    tb = traceback.format_exc()
                    logger.info("exception: %s, details: %s", ex, tb)
                    # ignore errors

        return ops

    def patched_get_value_attr(self, external_tensor_storage=None):
        """
        Return onnx attr for value property of node.
        Attr is modified to point to external tensor data stored in
        external_tensor_storage, if included.
        """
        a = self._attr["value"]
        if (
            external_tensor_storage is not None
            and self in external_tensor_storage.node_to_modified_value_attr
        ):
            return external_tensor_storage.node_to_modified_value_attr[self]
        if external_tensor_storage is None or a.type != AttributeProto.TENSOR:
            return a

        def prod(x):
            if hasattr(np, "product"):
                return np.product(x)
            else:
                return np.prod(x)

        if (
            prod(a.t.dims)
            > external_tensor_storage.external_tensor_size_threshold
        ):
            a = copy.deepcopy(a)
            tensor_name = (
                self.name.strip()
                + "_"
                + str(external_tensor_storage.name_counter)
            )
            for c in '~"#%&*:<>?/\\{|}':
                tensor_name = tensor_name.replace(c, "_")
            external_tensor_storage.name_counter += 1
            external_tensor_storage.name_to_tensor_data[tensor_name] = (
                a.t.raw_data
            )
            external_tensor_storage.node_to_modified_value_attr[self] = a
            a.t.raw_data = b""
            a.t.ClearField("raw_data")
            location = a.t.external_data.add()
            location.key = "location"
            location.value = tensor_name
            a.t.data_location = TensorProto.EXTERNAL
        return a

    tf2onnx.tfonnx.rewrite_constant_fold = patched_rewrite_constant_fold
    tf2onnx.graph.Node.get_value_attr = patched_get_value_attr
