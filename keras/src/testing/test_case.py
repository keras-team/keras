import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from keras.src import backend
from keras.src import distribution
from keras.src import ops
from keras.src import tree
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils


class TestCase(unittest.TestCase):
    maxDiff = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        # clear global state so that test cases are independent
        # required for the jit enabled torch tests since dynamo has
        # a global cache for guards, compiled fn, etc
        clear_session(free_memory=False)
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
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol, err_msg=msg)

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
        np.testing.assert_almost_equal(x1, x2, decimal=decimal, err_msg=msg)

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

    def assertSparse(self, x, sparse=True):
        if isinstance(x, KerasTensor):
            self.assertEqual(x.sparse, sparse)
        elif backend.backend() == "tensorflow":
            import tensorflow as tf

            if sparse:
                self.assertIsInstance(x, tf.SparseTensor)
            else:
                self.assertNotIsInstance(x, tf.SparseTensor)
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            if sparse:
                self.assertIsInstance(x, jax_sparse.JAXSparse)
            else:
                self.assertNotIsInstance(x, jax_sparse.JAXSparse)
        else:
            self.assertFalse(
                sparse,
                f"Backend {backend.backend()} does not support sparse tensors",
            )

    def assertDType(self, x, dtype, msg=None):
        if hasattr(x, "dtype"):
            x_dtype = backend.standardize_dtype(x.dtype)
        else:
            # If x is a python number
            x_dtype = backend.standardize_dtype(type(x))
        standardized_dtype = backend.standardize_dtype(dtype)
        default_msg = (
            "The dtype of x does not match the expected one. "
            f"Received: x.dtype={x_dtype} and dtype={dtype}"
        )
        msg = msg or default_msg
        self.assertEqual(x_dtype, standardized_dtype, msg=msg)

    def assertFileExists(self, path):
        if not Path(path).is_file():
            raise AssertionError(f"File {path} does not exist")

    def run_class_serialization_test(self, instance, custom_objects=None):
        from keras.src.saving import custom_object_scope
        from keras.src.saving import deserialize_keras_object
        from keras.src.saving import serialize_keras_object

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
        input_dtype=None,
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
        assert_built_after_instantiation=False,
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
            assert_built_after_instantiation: Whether to assert `built=True`
                after the layer's instantiation.
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
                "In order to use expected_mask_shape, supports_masking "
                "must be True."
            )

        init_kwargs = init_kwargs or {}
        call_kwargs = call_kwargs or {}

        if input_shape is not None and input_dtype is not None:
            if isinstance(input_shape, tuple) and is_shape_tuple(
                input_shape[0]
            ):
                self.assertIsInstance(input_dtype, tuple)
                self.assertEqual(
                    len(input_shape),
                    len(input_dtype),
                    msg="The number of input shapes and dtypes does not match",
                )
            elif isinstance(input_shape, dict):
                self.assertIsInstance(input_dtype, dict)
                self.assertEqual(
                    set(input_shape.keys()),
                    set(input_dtype.keys()),
                    msg="The number of input shapes and dtypes does not match",
                )
            elif isinstance(input_shape, list):
                self.assertIsInstance(input_dtype, list)
                self.assertEqual(
                    len(input_shape),
                    len(input_dtype),
                    msg="The number of input shapes and dtypes does not match",
                )
            elif not isinstance(input_shape, tuple):
                raise ValueError("The type of input_shape is not supported")
        if input_shape is not None and input_dtype is None:
            input_dtype = tree.map_shape_structure(
                lambda _: "float32", input_shape
            )

        # Estimate actual number of weights, variables, seed generators if
        # expected ones not set. When using layers uses composition it should
        # build each sublayer manually.
        if input_data is not None or input_shape is not None:
            if input_data is None:
                input_data = create_eager_tensors(
                    input_shape, input_dtype, input_sparse
                )
            layer = layer_cls(**init_kwargs)
            if isinstance(input_data, dict):
                layer(**input_data, **call_kwargs)
            else:
                layer(input_data, **call_kwargs)

            if expected_num_trainable_weights is None:
                expected_num_trainable_weights = len(layer.trainable_weights)
            if expected_num_non_trainable_weights is None:
                expected_num_non_trainable_weights = len(
                    layer.non_trainable_weights
                )
            if expected_num_non_trainable_variables is None:
                expected_num_non_trainable_variables = len(
                    layer.non_trainable_variables
                )
            if expected_num_seed_generators is None:
                expected_num_seed_generators = len(get_seed_generators(layer))

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
                    get_seed_generators(layer),
                    expected_num_seed_generators,
                    msg="Unexpected number of seed_generators",
                )
            if (
                backend.backend() == "torch"
                and expected_num_trainable_weights is not None
                and expected_num_non_trainable_weights is not None
                and expected_num_seed_generators is not None
            ):
                self.assertLen(
                    layer.torch_params,
                    expected_num_trainable_weights
                    + expected_num_non_trainable_weights
                    + expected_num_seed_generators,
                    msg="Unexpected number of torch_params",
                )

        def run_output_asserts(layer, output, eager=False):
            if expected_output_shape is not None:
                if isinstance(expected_output_shape, tuple) and is_shape_tuple(
                    expected_output_shape[0]
                ):
                    self.assertIsInstance(output, tuple)
                    self.assertEqual(
                        len(output),
                        len(expected_output_shape),
                        msg="Unexpected number of outputs",
                    )
                    output_shape = tuple(v.shape for v in output)
                    self.assertEqual(
                        expected_output_shape,
                        output_shape,
                        msg="Unexpected output shape",
                    )
                elif isinstance(expected_output_shape, tuple):
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
                    output_shape = {k: v.shape for k, v in output.items()}
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
                    output_shape = [v.shape for v in output]
                    self.assertEqual(
                        expected_output_shape,
                        output_shape,
                        msg="Unexpected output shape",
                    )
                else:
                    raise ValueError(
                        "The type of expected_output_shape is not supported"
                    )
            if expected_output_dtype is not None:
                if isinstance(expected_output_dtype, tuple):
                    self.assertIsInstance(output, tuple)
                    self.assertEqual(
                        len(output),
                        len(expected_output_dtype),
                        msg="Unexpected number of outputs",
                    )
                    output_dtype = tuple(
                        backend.standardize_dtype(v.dtype) for v in output
                    )
                    self.assertEqual(
                        expected_output_dtype,
                        output_dtype,
                        msg="Unexpected output dtype",
                    )
                elif isinstance(expected_output_dtype, dict):
                    self.assertIsInstance(output, dict)
                    self.assertEqual(
                        set(output.keys()),
                        set(expected_output_dtype.keys()),
                        msg="Unexpected output dict keys",
                    )
                    output_dtype = {
                        k: backend.standardize_dtype(v.dtype)
                        for k, v in output.items()
                    }
                    self.assertEqual(
                        expected_output_dtype,
                        output_dtype,
                        msg="Unexpected output dtype",
                    )
                elif isinstance(expected_output_dtype, list):
                    self.assertIsInstance(output, list)
                    self.assertEqual(
                        len(output),
                        len(expected_output_dtype),
                        msg="Unexpected number of outputs",
                    )
                    output_dtype = [
                        backend.standardize_dtype(v.dtype) for v in output
                    ]
                    self.assertEqual(
                        expected_output_dtype,
                        output_dtype,
                        msg="Unexpected output dtype",
                    )
                else:
                    output_dtype = tree.flatten(output)[0].dtype
                    self.assertEqual(
                        expected_output_dtype,
                        backend.standardize_dtype(output_dtype),
                        msg="Unexpected output dtype",
                    )
            if expected_output_sparse:
                for x in tree.flatten(output):
                    self.assertSparse(x)
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

                def call(self, x, training=False):
                    return self.layer(x, training=training)

            model = TestModel(layer)

            data = (input_data, output_data)
            if backend.backend() == "torch":
                data = tree.map_structure(backend.convert_to_numpy, data)

            def data_generator():
                while True:
                    yield data

            # test the "default" path for each backend by setting
            # jit_compile="auto".
            # for tensorflow and jax backends auto is jitted
            # Note that tensorflow cannot be jitted with sparse tensors
            # for torch backend auto is eager
            #
            # NB: for torch, jit_compile=True turns on torchdynamo
            #  which may not always succeed in tracing depending
            #  on the model. Run your program with these env vars
            #  to get debug traces of dynamo:
            #    TORCH_LOGS="+dynamo"
            #    TORCHDYNAMO_VERBOSE=1
            #    TORCHDYNAMO_REPORT_GUARD_FAILURES=1
            jit_compile = "auto"
            if backend.backend() == "tensorflow" and input_sparse:
                jit_compile = False
            model.compile(optimizer="sgd", loss="mse", jit_compile=jit_compile)
            model.fit(data_generator(), steps_per_epoch=1, verbose=0)

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

            # The stateless layers should be built after instantiation.
            if assert_built_after_instantiation:
                layer = layer_cls(**init_kwargs)
                self.assertTrue(
                    layer.built,
                    msg=(
                        f"{type(layer)} is stateless, so it should be built "
                        "after instantiation."
                    ),
                )

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
                input_spec = tree.map_structure(
                    lambda spec: KerasTensor(
                        spec.shape,
                        dtype=(
                            layer.compute_dtype
                            if layer.autocast
                            and backend.is_float_dtype(spec.dtype)
                            else spec.dtype
                        ),
                    ),
                    keras_tensor_inputs,
                )
                if isinstance(input_data, dict):
                    output_data = layer(**input_data, **call_kwargs)
                    output_spec = layer.compute_output_spec(**input_spec)
                else:
                    output_data = layer(input_data, **call_kwargs)
                    output_spec = layer.compute_output_spec(input_spec)
                for tensor, spec in zip(
                    tree.flatten(output_data), tree.flatten(output_spec)
                ):
                    dtype = standardize_dtype(tensor.dtype)
                    self.assertEqual(
                        dtype,
                        spec.dtype,
                        f"expected output dtype {spec.dtype}, got {dtype}",
                    )
                for weight in layer.weights:
                    dtype = standardize_dtype(weight.dtype)
                    if is_float_dtype(dtype):
                        self.assertEqual(dtype, "float32")


def tensorflow_uses_gpu():
    return backend.backend() == "tensorflow" and uses_gpu()


def jax_uses_gpu():
    return backend.backend() == "jax" and uses_gpu()


def torch_uses_gpu():
    if backend.backend() != "torch":
        return False
    from keras.src.backend.torch.core import get_device

    return get_device() == "cuda"


def uses_gpu():
    # Condition used to skip tests when using the GPU
    devices = distribution.list_devices()
    if any(d.startswith("gpu") for d in devices):
        return True
    return False


def create_keras_tensors(input_shape, dtype, sparse):
    if isinstance(input_shape, dict):
        return {
            utils.removesuffix(k, "_shape"): KerasTensor(
                v, dtype=dtype[k], sparse=sparse
            )
            for k, v in input_shape.items()
        }
    return map_shape_dtype_structure(
        lambda shape, dt: KerasTensor(shape, dtype=dt, sparse=sparse),
        input_shape,
        dtype,
    )


def create_eager_tensors(input_shape, dtype, sparse):
    from keras.src.backend import random

    if set(tree.flatten(dtype)).difference(
        [
            "float16",
            "float32",
            "float64",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
        ]
    ):
        raise ValueError(
            "dtype must be a standard float or int dtype. "
            f"Received: dtype={dtype}"
        )

    if sparse:
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            def create_fn(shape, dt):
                rng = np.random.default_rng(0)
                x = (4 * rng.standard_normal(shape)).astype(dt)
                x = np.multiply(x, rng.random(shape) < 0.7)
                return tf.sparse.from_dense(x)

        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            def create_fn(shape, dt):
                rng = np.random.default_rng(0)
                x = (4 * rng.standard_normal(shape)).astype(dt)
                x = np.multiply(x, rng.random(shape) < 0.7)
                return jax_sparse.BCOO.fromdense(x, n_batch=1)

        else:
            raise ValueError(
                f"Sparse is unsupported with backend {backend.backend()}"
            )

    else:

        def create_fn(shape, dt):
            return ops.cast(
                random.uniform(shape, dtype="float32") * 3, dtype=dt
            )

    if isinstance(input_shape, dict):
        return {
            utils.removesuffix(k, "_shape"): create_fn(v, dtype[k])
            for k, v in input_shape.items()
        }
    return map_shape_dtype_structure(create_fn, input_shape, dtype)


def is_shape_tuple(x):
    return isinstance(x, (list, tuple)) and all(
        isinstance(e, (int, type(None))) for e in x
    )


def map_shape_dtype_structure(fn, shape, dtype):
    """Variant of tree.map_structure that operates on shape tuples."""
    if is_shape_tuple(shape):
        return fn(tuple(shape), dtype)
    if isinstance(shape, list):
        return [
            map_shape_dtype_structure(fn, s, d) for s, d in zip(shape, dtype)
        ]
    if isinstance(shape, tuple):
        return tuple(
            map_shape_dtype_structure(fn, s, d) for s, d in zip(shape, dtype)
        )
    if isinstance(shape, dict):
        return {
            k: map_shape_dtype_structure(fn, v, dtype[k])
            for k, v in shape.items()
        }
    else:
        raise ValueError(
            f"Cannot map function to unknown objects {shape} and {dtype}"
        )


def get_seed_generators(layer):
    """Get a List of all seed generators in the layer recursively."""
    seed_generators = []
    seen_ids = set()
    for sublayer in layer._flatten_layers(True, True):
        for sg in sublayer._seed_generators:
            if id(sg) not in seen_ids:
                seed_generators.append(sg)
                seen_ids.add(id(sg))
    return seed_generators
