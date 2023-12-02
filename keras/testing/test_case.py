import json
import shutil
import tempfile
import unittest

import numpy as np
import tree

from keras import backend
from keras import ops
from keras import utils
from keras.backend.common import is_float_dtype
from keras.backend.common import standardize_dtype
from keras.backend.common.global_state import clear_session
from keras.backend.common.keras_tensor import KerasTensor
from keras.models import Model
from keras.utils import traceback_utils


class TestCase(unittest.TestCase):
    maxDiff = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        # clear global state so that test cases are independent
        # required for the jit enabled torch tests since dynamo has
        # a global cache for guards, compiled fn, etc
        clear_session()
        if traceback_utils.is_traceback_filtering_enabled():
            traceback_utils.disable_traceback_filtering()

    def get_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir))
        return temp_dir

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        if not isinstance(x1, np.ndarray):
            x1 = backend.convert_to_numpy(x1)
        if not isinstance(x2, np.ndarray):
            x2 = backend.convert_to_numpy(x2)
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol)

    def assertNotAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        try:
            self.assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)
        except AssertionError:
            return
        msg = msg or ""
        raise AssertionError(
            f"The two values are close at all elements. \n"
            f"{msg}.\n"
            f"Values: {x1}"
        )

    def assertAlmostEqual(self, x1, x2, decimal=3, msg=None):
        if not isinstance(x1, np.ndarray):
            x1 = backend.convert_to_numpy(x1)
        if not isinstance(x2, np.ndarray):
            x2 = backend.convert_to_numpy(x2)
        np.testing.assert_almost_equal(x1, x2, decimal=decimal)

    def assertAllEqual(self, x1, x2, msg=None):
        self.assertEqual(len(x1), len(x2), msg=msg)
        for e1, e2 in zip(x1, x2):
            if isinstance(e1, (list, tuple)) or isinstance(e2, (list, tuple)):
                self.assertAllEqual(e1, e2, msg=msg)
            else:
                e1 = backend.convert_to_numpy(e1)
                e2 = backend.convert_to_numpy(e2)
                self.assertEqual(e1, e2, msg=msg)

    def assertLen(self, iterable, expected_len, msg=None):
        self.assertEqual(len(iterable), expected_len, msg=msg)

    def run_class_serialization_test(self, instance, custom_objects=None):
        from keras.saving import custom_object_scope
        from keras.saving import deserialize_keras_object
        from keras.saving import serialize_keras_object

        # get_config roundtrip
        cls = instance.__class__
        config = instance.get_config()
        config_json = json.dumps(config, sort_keys=True, indent=4)
        ref_dir = dir(instance)[:]
        with custom_object_scope(custom_objects):
            revived_instance = cls.from_config(config)
        revived_config = revived_instance.get_config()
        revived_config_json = json.dumps(
            revived_config, sort_keys=True, indent=4
        )
        self.assertEqual(config_json, revived_config_json)
        self.assertEqual(set(ref_dir), set(dir(revived_instance)))

        # serialization roundtrip
        serialized = serialize_keras_object(instance)
        serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
        with custom_object_scope(custom_objects):
            revived_instance = deserialize_keras_object(
                json.loads(serialized_json)
            )
        revived_config = revived_instance.get_config()
        revived_config_json = json.dumps(
            revived_config, sort_keys=True, indent=4
        )
        self.assertEqual(config_json, revived_config_json)
        new_dir = dir(revived_instance)[:]
        for lst in [ref_dir, new_dir]:
            if "__annotations__" in lst:
                lst.remove("__annotations__")
        self.assertEqual(set(ref_dir), set(new_dir))
        return revived_instance

    def run_layer_test(
        self,
        layer_cls,
        init_kwargs,
        input_shape=None,
        input_dtype="float32",
        input_sparse=False,
        input_data=None,
        call_kwargs=None,
        expected_output_shape=None,
        expected_output_dtype=None,
        expected_output_sparse=False,
        expected_output=None,
        expected_num_trainable_weights=None,
        expected_num_non_trainable_weights=None,
        expected_num_non_trainable_variables=None,
        expected_num_seed_generators=None,
        expected_num_losses=None,
        supports_masking=None,
        expected_mask_shape=None,
        custom_objects=None,
        run_training_check=True,
        run_mixed_precision_check=True,
    ):
        """Run basic checks on a layer.

        Args:
            layer_cls: The class of the layer to test.
            init_kwargs: Dict of arguments to be used to
                instantiate the layer.
            input_shape: Shape tuple (or list/dict of shape tuples)
                to call the layer on.
            input_dtype: Corresponding input dtype.
            input_sparse: Whether the input is a sparse tensor (this requires
                the backend to support sparse tensors).
            input_data: Tensor (or list/dict of tensors)
                to call the layer on.
            call_kwargs: Dict of arguments to use when calling the
                layer (does not include the first input tensor argument)
            expected_output_shape: Shape tuple
                (or list/dict of shape tuples)
                expected as output.
            expected_output_dtype: dtype expected as output.
            expected_output_sparse: Whether the output is expected to be sparse
                (this requires the backend to support sparse tensors).
            expected_output: Expected output tensor -- only
                to be specified if input_data is provided.
            expected_num_trainable_weights: Expected number
                of trainable weights of the layer once built.
            expected_num_non_trainable_weights: Expected number
                of non-trainable weights of the layer once built.
            expected_num_seed_generators: Expected number of
                SeedGenerators objects of the layer once built.
            expected_num_losses: Expected number of loss tensors
                produced when calling the layer.
            supports_masking: If True, will check that the layer
                supports masking.
            expected_mask_shape: Expected mask shape tuple
                returned by compute_mask() (only supports 1 shape).
            custom_objects: Dict of any custom objects to be
                considered during deserialization.
            run_training_check: Whether to attempt to train the layer
                (if an input shape or input data was provided).
            run_mixed_precision_check: Whether to test the layer with a mixed
                precision dtype policy.
        """
        if input_shape is not None and input_data is not None:
            raise ValueError(
                "input_shape and input_data cannot be passed "
                "at the same time."
            )
        if expected_output_shape is not None and expected_output is not None:
            raise ValueError(
                "expected_output_shape and expected_output cannot be passed "
                "at the same time."
            )
        if expected_output is not None and input_data is None:
            raise ValueError(
                "In order to use expected_output, input_data must be provided."
            )
        if expected_mask_shape is not None and supports_masking is not True:
            raise ValueError(
                """In order to use expected_mask_shape, supports_masking
                must be True."""
            )

        init_kwargs = init_kwargs or {}
        call_kwargs = call_kwargs or {}

        # Serialization test.
        layer = layer_cls(**init_kwargs)
        self.run_class_serialization_test(layer, custom_objects)

        # Basic masking test.
        if supports_masking is not None:
            self.assertEqual(
                layer.supports_masking,
                supports_masking,
                msg="Unexpected supports_masking value",
            )

        def run_build_asserts(layer):
            self.assertTrue(layer.built)
            if expected_num_trainable_weights is not None:
                self.assertLen(
                    layer.trainable_weights,
                    expected_num_trainable_weights,
                    msg="Unexpected number of trainable_weights",
                )
            if expected_num_non_trainable_weights is not None:
                self.assertLen(
                    layer.non_trainable_weights,
                    expected_num_non_trainable_weights,
                    msg="Unexpected number of non_trainable_weights",
                )
            if expected_num_non_trainable_variables is not None:
                self.assertLen(
                    layer.non_trainable_variables,
                    expected_num_non_trainable_variables,
                    msg="Unexpected number of non_trainable_variables",
                )
            if expected_num_seed_generators is not None:
                self.assertLen(
                    layer._seed_generators,
                    expected_num_seed_generators,
                    msg="Unexpected number of _seed_generators",
                )

        def run_output_asserts(layer, output, eager=False):
            if expected_output_shape is not None:
                if isinstance(expected_output_shape, tuple):
                    self.assertEqual(
                        expected_output_shape,
                        output.shape,
                        msg="Unexpected output shape",
                    )
                elif isinstance(expected_output_shape, dict):
                    self.assertIsInstance(output, dict)
                    self.assertEqual(
                        set(output.keys()),
                        set(expected_output_shape.keys()),
                        msg="Unexpected output dict keys",
                    )
                    output_shape = {
                        k: v.shape for k, v in expected_output_shape.items()
                    }
                    self.assertEqual(
                        expected_output_shape,
                        output_shape,
                        msg="Unexpected output shape",
                    )
                elif isinstance(expected_output_shape, list):
                    self.assertIsInstance(output, list)
                    self.assertEqual(
                        len(output),
                        len(expected_output_shape),
                        msg="Unexpected number of outputs",
                    )
                    output_shape = [v.shape for v in expected_output_shape]
                    self.assertEqual(
                        expected_output_shape,
                        output_shape,
                        msg="Unexpected output shape",
                    )
            if expected_output_dtype is not None:
                output_dtype = tree.flatten(output)[0].dtype
                self.assertEqual(
                    expected_output_dtype,
                    backend.standardize_dtype(output_dtype),
                    msg="Unexpected output dtype",
                )
            if expected_output_sparse:
                import tensorflow as tf

                for x in tree.flatten(output):
                    if isinstance(x, KerasTensor):
                        self.assertTrue(x.sparse)
                    else:
                        self.assertIsInstance(x, tf.SparseTensor)
            if eager:
                if expected_output is not None:
                    self.assertEqual(type(expected_output), type(output))
                    for ref_v, v in zip(
                        tree.flatten(expected_output), tree.flatten(output)
                    ):
                        self.assertAllClose(
                            ref_v, v, msg="Unexpected output value"
                        )
                if expected_num_losses is not None:
                    self.assertLen(layer.losses, expected_num_losses)

        def run_training_step(layer, input_data, output_data):
            class TestModel(Model):
                def __init__(self, layer):
                    super().__init__()
                    self.layer = layer

                def call(self, x):
                    return self.layer(x)

            model = TestModel(layer)

            if input_sparse:
                import tensorflow as tf

                if backend.backend() == "torch":
                    # Bring output Torch tensors to CPU before using `tf.data`
                    dataset = tf.data.Dataset.from_tensors(
                        (input_data, backend.convert_to_numpy(output_data))
                    )
                else:
                    dataset = tf.data.Dataset.from_tensors(
                        (input_data, output_data)
                    )
                model.compile(optimizer="sgd", loss="mse", jit_compile=False)
                model.fit(dataset, verbose=0)
            else:
                input_data = tree.map_structure(
                    lambda x: backend.convert_to_numpy(x), input_data
                )
                output_data = tree.map_structure(
                    lambda x: backend.convert_to_numpy(x), output_data
                )
                # test the "default" path for each backend by setting
                # jit_compile="auto.
                # for tensorflow and jax backends auto is jitted
                # for torch backend auto is eager
                #
                # NB: for torch, jit_compile=True turns on torchdynamo
                #  which may not always succeed in tracing depending
                #  on the model. Run your program with these env vars
                #  to get debug traces of dynamo:
                #    TORCH_LOGS="+dynamo"
                #    TORCHDYNAMO_VERBOSE=1
                #    TORCHDYNAMO_REPORT_GUARD_FAILURES=1
                model.compile(optimizer="sgd", loss="mse", jit_compile="auto")
                model.fit(input_data, output_data, verbose=0)

        # Build test.
        if input_data is not None or input_shape is not None:
            if input_shape is None:
                build_shape = tree.map_structure(
                    lambda x: ops.shape(x), input_data
                )
            else:
                build_shape = input_shape
            layer = layer_cls(**init_kwargs)
            if isinstance(build_shape, dict):
                layer.build(**build_shape)
            else:
                layer.build(build_shape)
            run_build_asserts(layer)

            # Symbolic call test.
            if input_shape is None:
                keras_tensor_inputs = tree.map_structure(
                    lambda x: create_keras_tensors(
                        ops.shape(x), x.dtype, input_sparse
                    ),
                    input_data,
                )
            else:
                keras_tensor_inputs = create_keras_tensors(
                    input_shape, input_dtype, input_sparse
                )
            layer = layer_cls(**init_kwargs)
            if isinstance(keras_tensor_inputs, dict):
                keras_tensor_outputs = layer(
                    **keras_tensor_inputs, **call_kwargs
                )
            else:
                keras_tensor_outputs = layer(keras_tensor_inputs, **call_kwargs)
            run_build_asserts(layer)
            run_output_asserts(layer, keras_tensor_outputs, eager=False)

            if expected_mask_shape is not None:
                output_mask = layer.compute_mask(keras_tensor_inputs)
                self.assertEqual(expected_mask_shape, output_mask.shape)

        # Eager call test and compiled training test.
        if input_data is not None or input_shape is not None:
            if input_data is None:
                input_data = create_eager_tensors(
                    input_shape, input_dtype, input_sparse
                )
            layer = layer_cls(**init_kwargs)
            if isinstance(input_data, dict):
                output_data = layer(**input_data, **call_kwargs)
            else:
                output_data = layer(input_data, **call_kwargs)
            run_output_asserts(layer, output_data, eager=True)

            if run_training_check:
                run_training_step(layer, input_data, output_data)

            # Never test mixed precision on torch CPU. Torch lacks support.
            if run_mixed_precision_check and backend.backend() == "torch":
                import torch

                run_mixed_precision_check = torch.cuda.is_available()

            if run_mixed_precision_check:
                layer = layer_cls(**{**init_kwargs, "dtype": "mixed_float16"})
                if isinstance(input_data, dict):
                    output_data = layer(**input_data, **call_kwargs)
                else:
                    output_data = layer(input_data, **call_kwargs)
                for tensor in tree.flatten(output_data):
                    dtype = standardize_dtype(tensor.dtype)
                    if is_float_dtype(dtype):
                        self.assertEqual(dtype, "float16")
                for weight in layer.weights:
                    dtype = standardize_dtype(weight.dtype)
                    if is_float_dtype(dtype):
                        self.assertEqual(dtype, "float32")


def create_keras_tensors(input_shape, dtype, sparse):
    if isinstance(input_shape, tuple):
        return KerasTensor(input_shape, dtype=dtype, sparse=sparse)
    if isinstance(input_shape, list):
        return [KerasTensor(s, dtype=dtype, sparse=sparse) for s in input_shape]
    if isinstance(input_shape, dict):
        return {
            utils.removesuffix(k, "_shape"): KerasTensor(
                v, dtype=dtype, sparse=sparse
            )
            for k, v in input_shape.items()
        }
    raise ValueError(f"Unsupported type for `input_shape`: {type(input_shape)}")


def create_eager_tensors(input_shape, dtype, sparse):
    from keras.backend import random

    if dtype not in [
        "float16",
        "float32",
        "float64",
        "int16",
        "int32",
        "int64",
    ]:
        raise ValueError(
            "dtype must be a standard float or int dtype. "
            f"Received: dtype={dtype}"
        )

    if sparse:
        import tensorflow as tf

        def create_fn(shape, dtype):
            min_dim = min(dim for dim in shape if dim > 1)
            x = random.uniform(shape, dtype="float32") * 3 / min_dim
            x = tf.nn.dropout(x, 1.0 / min_dim)
            x = tf.cast(x, dtype=dtype)
            return tf.sparse.from_dense(x)

    else:

        def create_fn(shape, dtype):
            return ops.cast(
                random.uniform(shape, dtype="float32") * 3, dtype=dtype
            )

    if isinstance(input_shape, tuple):
        return create_fn(input_shape, dtype=dtype)
    if isinstance(input_shape, list):
        return [create_fn(s, dtype=dtype) for s in input_shape]
    if isinstance(input_shape, dict):
        return {
            utils.removesuffix(k, "_shape"): create_fn(v, dtype=dtype)
            for k, v in input_shape.items()
        }
