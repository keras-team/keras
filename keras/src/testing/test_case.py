import functools
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.losses.loss import Loss
from keras.src.models import Model
from keras.src.testing.test_utils import named_dtype_combinations
from keras.src.utils import traceback_utils

# Input dtypes the dtype checks skip per backend, with the reason.
_SKIPPED_DTYPES = {
    "torch": {
        "uint16": "Not supported by the torch backend",
        "uint32": "Not supported by the torch backend",
    },
    "tensorflow": {"uint32": "uint32 + int promotes to int64, unlike JAX"},
}

# Standard op test inputs: small non-negative ints fit every dtype. `Z`
# broadcasts against `X`. 32-bit, as 64-bit dtypes are not covered yet.
X = np.array([[1, 2, 3], [3, 2, 1]], dtype="int32")
Y = np.array([[4, 5, 6], [3, 2, 1]], dtype="int32")
Z = np.array([[[1, 2, 3], [3, 2, 1]]], dtype="int32")


class TestCase(parameterized.TestCase):
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

    def convert_to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x
        elif backend.ops.is_tensor(x) or isinstance(x, backend.Variable):
            return backend.ops.convert_to_numpy(x)
        return np.array(x)

    def assertAllClose(
        self,
        actual,
        desired,
        atol=1e-6,
        rtol=1e-6,
        tpu_atol=None,
        tpu_rtol=None,
        msg=None,
    ):
        """Assert that two arrays are equal up to given tolerances.

        Given two array-like objects, check that their shapes and all elements
        are equal up to given tolerances.
        """
        if tpu_atol is not None and uses_tpu():
            atol = tpu_atol
        if tpu_rtol is not None and uses_tpu():
            rtol = tpu_rtol
        actual = self.convert_to_numpy(actual)
        desired = self.convert_to_numpy(desired)
        shape_msg = (
            f"Shapes don't match. {msg}" if msg else "Shapes don't match"
        )
        self.assertEqual(actual.shape, desired.shape, shape_msg)
        np.testing.assert_allclose(
            actual, desired, atol=atol, rtol=rtol, err_msg=msg or ""
        )

    def assertNotAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        try:
            self.assertAllClose(x1, x2, atol=atol, rtol=rtol)
        except AssertionError:
            return
        msg = msg or ""
        raise AssertionError(
            f"The two values are close at all elements. \n{msg}.\nValues: {x1}"
        )

    def assertAlmostEqual(
        self, actual, desired, decimal=3, tpu_decimal=None, msg=None
    ):
        """Assert that two arrays are equal up to given decimal places.

        Given two array-like objects, check that their shapes and all elements
        are equal up to given decimal places.
        """
        if tpu_decimal is not None and uses_tpu():
            decimal = tpu_decimal
        msg = msg or ""
        actual = self.convert_to_numpy(actual)
        desired = self.convert_to_numpy(desired)
        shape_msg = (
            f"Shapes don't match. {msg}" if msg else "Shapes don't match"
        )
        self.assertEqual(actual.shape, desired.shape, shape_msg)
        np.testing.assert_almost_equal(
            actual, desired, decimal=decimal, err_msg=msg or ""
        )

    def assertAllEqual(self, actual, desired, msg=None):
        """Assert that two arrays are equal.

        Given two array-like objects, check that their shapes and all elements
        are equal.
        """
        actual = self.convert_to_numpy(actual)
        desired = self.convert_to_numpy(desired)
        shape_msg = (
            f"Shapes don't match. {msg}" if msg else "Shapes don't match"
        )
        self.assertEqual(actual.shape, desired.shape, shape_msg)
        np.testing.assert_array_equal(actual, desired, err_msg=msg or "")

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

    def assertRagged(self, x, ragged=True):
        if isinstance(x, KerasTensor):
            self.assertEqual(x.ragged, ragged)
        elif backend.backend() == "tensorflow":
            import tensorflow as tf

            if ragged:
                self.assertIsInstance(x, tf.RaggedTensor)
            else:
                self.assertNotIsInstance(x, tf.RaggedTensor)
        else:
            self.assertFalse(
                ragged,
                f"Backend {backend.backend()} does not support ragged tensors",
            )

    def assertDType(self, x, dtype, msg=None):
        if hasattr(x, "dtype"):
            x_dtype = standardize_dtype(x.dtype)
        else:
            # If x is a python number
            x_dtype = standardize_dtype(type(x))
        standardized_dtype = standardize_dtype(dtype)
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
        config_json = to_json_with_tuples(config)
        ref_dir = dir(instance)[:]
        with custom_object_scope(custom_objects):
            revived_instance = cls.from_config(config)
        revived_config = revived_instance.get_config()
        revived_config_json = to_json_with_tuples(revived_config)
        self.assertEqual(config_json, revived_config_json)
        self.assertEqual(set(ref_dir), set(dir(revived_instance)))

        # serialization roundtrip
        serialized = serialize_keras_object(instance)
        serialized_json = to_json_with_tuples(serialized)
        with custom_object_scope(custom_objects):
            revived_instance = deserialize_keras_object(
                from_json_with_tuples(serialized_json)
            )
        revived_config = revived_instance.get_config()
        revived_config_json = to_json_with_tuples(revived_config)
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
        input_ragged=False,
        input_data=None,
        call_kwargs=None,
        expected_output_shape=None,
        expected_output_dtype=None,
        expected_output_sparse=False,
        expected_output_ragged=False,
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
        tpu_atol=None,
        tpu_rtol=None,
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
            input_ragged: Whether the input is a ragged tensor (this requires
                the backend to support ragged tensors).
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
            expected_output_ragged: Whether the output is expected to be ragged
                (this requires the backend to support ragged tensors).
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
                "input_shape and input_data cannot be passed at the same time."
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
                    input_shape, input_dtype, input_sparse, input_ragged
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

        def run_output_asserts(
            layer, output, eager=False, tpu_atol=None, tpu_rtol=None
        ):
            if expected_output_shape is not None:

                def verify_shape(expected_shape, x):
                    shape = x.shape
                    if len(shape) != len(expected_shape):
                        return False
                    for expected_dim, dim in zip(expected_shape, shape):
                        if expected_dim is not None and expected_dim != dim:
                            return False
                    return True

                shapes_match = tree.map_structure_up_to(
                    output, verify_shape, expected_output_shape, output
                )
                self.assertTrue(
                    all(tree.flatten(shapes_match)),
                    msg=f"Expected output shapes {expected_output_shape} but "
                    f"received {tree.map_structure(lambda x: x.shape, output)}",
                )
            if expected_output_dtype is not None:

                def verify_dtype(expected_dtype, x):
                    return expected_dtype == standardize_dtype(x.dtype)

                dtypes_match = tree.map_structure(
                    verify_dtype, expected_output_dtype, output
                )
                self.assertTrue(
                    all(tree.flatten(dtypes_match)),
                    msg=f"Expected output dtypes {expected_output_dtype} but "
                    f"received {tree.map_structure(lambda x: x.dtype, output)}",
                )
            if expected_output_sparse:
                for x in tree.flatten(output):
                    self.assertSparse(x)
            if expected_output_ragged:
                for x in tree.flatten(output):
                    self.assertRagged(x)
            if eager:
                if expected_output is not None:
                    self.assertEqual(type(expected_output), type(output))
                    for ref_v, v in zip(
                        tree.flatten(expected_output), tree.flatten(output)
                    ):
                        self.assertAllClose(
                            ref_v,
                            v,
                            msg="Unexpected output value",
                            tpu_atol=tpu_atol,
                            tpu_rtol=tpu_rtol,
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
                data = tree.map_structure(backend.ops.convert_to_numpy, data)

            def data_generator():
                while True:
                    yield data

            # Single op loss to avoid compilation issues with ragged / sparse.
            class TestLoss(Loss):
                def __call__(self, y_true, y_pred, sample_weight=None):
                    return ops.sum(y_pred)

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
            model.compile(
                optimizer="sgd", loss=TestLoss(), jit_compile=jit_compile
            )
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
                        ops.shape(x), x.dtype, input_sparse, input_ragged
                    ),
                    input_data,
                )
            else:
                keras_tensor_inputs = create_keras_tensors(
                    input_shape, input_dtype, input_sparse, input_ragged
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

                # Ensure that the subclass layer doesn't mark itself as built
                # when `build` is overridden.

                class ModifiedBuildLayer(layer_cls):
                    def build(self, *args, **kwargs):
                        pass

                layer = ModifiedBuildLayer(**init_kwargs)
                self.assertFalse(
                    layer.built,
                    msg=(
                        f"The `build` of {type(layer)} is overriden, so it "
                        "should not be built after instantiation."
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
            run_output_asserts(
                layer,
                output_data,
                eager=True,
                tpu_atol=tpu_atol,
                tpu_rtol=tpu_rtol,
            )

            if run_training_check and backend.SUPPORTS_GRADIENT:
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
                            if layer.autocast and is_float_dtype(spec.dtype)
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

    def run_op_eager_check(
        self, op, reference_fn, inputs, kwargs, atol=1e-6, rtol=1e-6
    ):
        """Checks the eager output count and values against `reference_fn`."""
        outputs = tree.flatten(_call_eager(op, inputs, kwargs))
        expected = tree.flatten(reference_fn(*inputs, **kwargs))
        msg = f"Unexpected eager output for inputs {_signature(inputs)}"
        self.assertLen(outputs, len(expected), f"{msg}: output count differs")
        for output, expected_output in zip(outputs, expected):
            self.assertAllClose(
                output, expected_output, atol=atol, rtol=rtol, msg=msg
            )

    def run_op_symbolic_check(self, op, inputs, kwargs, symbolic_shapes=None):
        """Checks the symbolic outputs against the eager ones.

        `symbolic_shapes` has one shape per array input, with `None` for the
        unknown dimensions; by default the arrays' shapes. Each symbolic
        output must be a `KerasTensor` with the eager output's dtype, rank
        and known dimensions.
        """
        arrays = _arrays(inputs)
        if symbolic_shapes is None:
            symbolic_shapes = [x.shape for x in arrays]
        # The dtypes Keras gives the arrays, e.g. float64 -> float32 on JAX.
        dtypes = _dtypes_of([ops.convert_to_tensor(x) for x in arrays])
        keras_tensors = [
            KerasTensor(shape, dtype=dtype)
            for shape, dtype in zip(symbolic_shapes, dtypes, strict=True)
        ]
        symbolic_inputs = _with_arrays(inputs, keras_tensors)
        symbolic = tree.flatten(op(*symbolic_inputs, **kwargs))
        eager = tree.flatten(_call_eager(op, inputs, kwargs))
        msg = (
            f"Output differs: symbolic={_signature(symbolic)}, "
            f"eager={_signature(eager)} "
            f"for inputs {_signature(symbolic_inputs)}"
        )
        self.assertLen(symbolic, len(eager), msg)
        for symbolic_output, eager_output in zip(symbolic, eager):
            if not _symbolic_matches(symbolic_output, eager_output):
                self.fail(msg)

    def run_op_input_dtypes_check(self, op, jax_fn, inputs, kwargs, dtypes):
        """Checks the output dtypes with the array inputs cast to `dtypes`.

        The eager output dtypes must match `jax_fn`'s, and the symbolic
        output dtypes must match the eager ones.
        """
        skipped = _SKIPPED_DTYPES.get(backend.backend(), {})
        for dtype in dtypes:
            if dtype in skipped:
                self.skipTest(f"{dtype}: {skipped[dtype]}")
        arrays = [
            x.astype(d) for x, d in zip(_arrays(inputs), dtypes, strict=True)
        ]
        inputs = _with_arrays(inputs, arrays)
        import jax.numpy as jnp

        jax_inputs = _with_arrays(inputs, [jnp.asarray(x) for x in arrays])
        keras_dtypes = _dtypes_of(_call_eager(op, inputs, kwargs))
        jax_dtypes = _dtypes_of(jax_fn(*jax_inputs, **kwargs))
        self.assertEqual(
            keras_dtypes,
            jax_dtypes,
            f"Output dtype differs from JAX for inputs {_signature(inputs)}",
        )
        self.run_op_symbolic_check(op, inputs, kwargs)


class OpTest(TestCase):
    """Generates the standard tests of an op from class attributes.

    Per case, suffixed with the case name if `cases` is set:
      - `test_eager`: values vs `reference_fn`.
      - `test_eager_class`: the same, on `op_class`.
      - `test_symbolic_static`: symbolic outputs vs eager outputs.
      - `test_symbolic_dynamic`: the same, every input dimension unknown.
      - `test_symbolic_partial`: the same, one unknown dimension at a time.
    Per input dtype combination of the first case:
      - `test_input_dtypes_<dtypes>`: output dtypes vs `jax_fn`, and
        symbolic vs eager.
      - `test_python_types_<dtypes>_<int|float>`: the same, with a Python
        scalar as last input, if `python_scalars`.

    A test that builds on a failed one skips, so one bug is one failure. A
    method defined with a generated name replaces the generated test, e.g.
    to skip a dtype combination with a reason.

    Attributes:
        op: Function form of the op, e.g. `knp.add`.
        op_class: `Operation` class, e.g. `knp.Add`, checked eagerly.
        reference_fn: NumPy reference, e.g. `np.add`.
        jax_fn: JAX reference for the output dtypes. `None` skips them.
        inputs: Tuple of NumPy arrays and Python scalars passed to the op.
        kwargs: Static keyword arguments of the op.
        cases: `{name: (inputs, kwargs)}` to test several inputs instead.
        atol: Absolute tolerance of the eager checks.
        rtol: Relative tolerance of the eager checks.
        run_dynamic_shape_checks: Disable for ops that need static sizes.
        dtypes: Dtype pool(s), see `named_dtype_combinations`.
        ordered: Whether to test every order of the input dtypes.
        python_scalars: Whether to also test a Python scalar as last input.
    """

    op = None
    op_class = None
    reference_fn = None
    jax_fn = None
    inputs = ()
    kwargs = {}
    cases = None
    atol = 1e-6
    rtol = 1e-6
    run_dynamic_shape_checks = True
    dtypes = None
    ordered = False
    python_scalars = False

    def __init_subclass__(cls, **class_kwargs):
        super().__init_subclass__(**class_kwargs)
        for name in ("op", "reference_fn", "jax_fn"):
            fn = cls.__dict__.get(name)
            if callable(fn) and not isinstance(fn, staticmethod):
                setattr(cls, name, staticmethod(fn))  # Or it would bind `self`.
        if cls.op is None:
            return  # A preset such as `UnaryOpTest`.
        if cls.cases is None:
            cls.cases = {"": (cls.inputs, cls.kwargs)}
        for name, test in cls._generated_tests().items():
            if not hasattr(cls, name):  # A hand-written test wins.
                setattr(cls, name, test)

    @classmethod
    def _generated_tests(cls):
        """Returns `{test name: test method}`."""
        tests = {}
        for case, (inputs, _) in cls.cases.items():
            checks = [("eager", cls.check_eager)]
            if cls.op_class is not None:
                checks.append(("eager_class", cls.check_eager_class))
            if _arrays(inputs):  # No symbolic path without array inputs.
                checks += [
                    ("symbolic_static", cls.check_symbolic_static),
                    ("symbolic_dynamic", cls.check_symbolic_dynamic),
                    ("symbolic_partial", cls.check_symbolic_partial),
                ]
            suffix = f"_{case}" if case else ""
            for check_name, check in checks:
                tests[f"test_{check_name}{suffix}"] = _test_method(check, case)
        num_arrays = len(_arrays(cls._first_case()[0]))
        if cls.jax_fn is None or num_arrays == 0:
            return tests
        combinations = named_dtype_combinations(
            num_arrays, cls.ordered, cls.dtypes
        )
        for name, dtypes in combinations:
            tests[f"test_input_dtypes_{name}"] = _test_method(
                cls.check_input_dtypes, dtypes
            )
        if not cls.python_scalars or num_arrays == 1:
            return tests
        combinations = named_dtype_combinations(
            num_arrays - 1, cls.ordered, cls.dtypes
        )
        for name, dtypes in combinations:
            for scalar_type in (int, float):
                tests[f"test_python_types_{name}_{scalar_type.__name__}"] = (
                    _test_method(cls.check_python_types, dtypes, scalar_type)
                )
        return tests

    @classmethod
    def _first_case(cls):
        return next(iter(cls.cases.values()))

    def _skip_unless_passes(self, check, *args, reason):
        """Skips this test when its prerequisite `check(*args)` fails."""
        try:
            check(*args)
        except unittest.SkipTest:
            raise
        except Exception:
            self.skipTest(reason)

    def check_eager(self, case=""):
        inputs, kwargs = self.cases[case]
        self.run_op_eager_check(
            self.op, self.reference_fn, inputs, kwargs, self.atol, self.rtol
        )

    def check_eager_class(self, case=""):
        self._skip_unless_passes(self.check_eager, case, reason="eager failed")
        inputs, kwargs = self.cases[case]
        op = self.op_class(**kwargs)
        reference_fn = functools.partial(self.reference_fn, **kwargs)
        self.run_op_eager_check(
            op, reference_fn, inputs, {}, self.atol, self.rtol
        )

    def check_symbolic_static(self, case=""):
        inputs, kwargs = self.cases[case]
        self._skip_unless_passes(
            _call_eager, self.op, inputs, kwargs, reason="eager failed"
        )
        self.run_op_symbolic_check(self.op, inputs, kwargs)

    def check_symbolic_dynamic(self, case=""):
        shapes = [x.shape for x in _arrays(self.cases[case][0])]
        self._check_unknown_dims(case, [_all_dims_unknown(shapes)])

    def check_symbolic_partial(self, case=""):
        shapes = [x.shape for x in _arrays(self.cases[case][0])]
        self._check_unknown_dims(case, _one_dim_unknown(shapes))

    def _check_unknown_dims(self, case, symbolic_shapes_list):
        if not self.run_dynamic_shape_checks:
            self.skipTest("Disabled by the test")
        self._skip_unless_passes(
            self.check_symbolic_static, case, reason="symbolic_static failed"
        )
        inputs, kwargs = self.cases[case]
        for symbolic_shapes in symbolic_shapes_list:
            try:
                self.run_op_symbolic_check(
                    self.op, inputs, kwargs, symbolic_shapes
                )
            except Exception as e:
                e.add_note(f"Symbolic input shapes: {symbolic_shapes}")
                raise

    def check_input_dtypes(self, dtypes):
        inputs, kwargs = self._first_case()
        self.run_op_input_dtypes_check(
            self.op, self.jax_fn, inputs, kwargs, dtypes
        )

    def check_python_types(self, dtypes, scalar_type):
        inputs, kwargs = self._first_case()
        arrays = _arrays(inputs)
        inputs = _with_arrays(inputs, arrays[:-1] + [scalar_type(1)])
        self.run_op_input_dtypes_check(
            self.op, self.jax_fn, inputs, kwargs, dtypes
        )


class UnaryOpTest(OpTest):
    """`OpTest` of an elementwise unary op, on the standard input `X`."""

    inputs = (X,)


class BinaryOpTest(OpTest):
    """`OpTest` of an elementwise binary op: broadcasting, Python scalars."""

    cases = {"same_shape": ((X, Y), {}), "broadcast": ((X, Z), {})}
    python_scalars = True


class ReductionOpTest(OpTest):
    """`OpTest` of a reduction over `axis`, with and without `keepdims`."""

    cases = {
        f"{name}_keepdims" if keepdims else name: (
            (X,),
            {"axis": axis, "keepdims": keepdims},
        )
        for name, axis in [
            ("all", None),
            ("axis0", 0),
            ("axis1", 1),
            ("last", -1),
            ("axes01", (0, 1)),
        ]
        for keepdims in (False, True)
    }


def _test_method(check, *args):
    """Returns a test method that runs `check(self, *args)`."""

    def test(self):
        check(self, *args)

    return test


def _arrays(inputs):
    """The NumPy arrays among `inputs`; the other inputs are Python scalars."""
    return [x for x in inputs if isinstance(x, np.ndarray)]


def _with_arrays(inputs, arrays):
    """`inputs` with its arrays replaced by `arrays`, in order."""
    arrays = iter(arrays)
    return tuple(
        next(arrays) if isinstance(x, np.ndarray) else x for x in inputs
    )


def _call_eager(op, inputs, kwargs):
    tensors = [ops.convert_to_tensor(x) for x in _arrays(inputs)]
    return op(*_with_arrays(inputs, tensors), **kwargs)


def _dtypes_of(outputs):
    return [standardize_dtype(x.dtype) for x in tree.flatten(outputs)]


def _symbolic_matches(symbolic, eager):
    """Whether a `KerasTensor` has the dtype and known dims of a tensor."""
    return (
        isinstance(symbolic, KerasTensor)
        and standardize_dtype(symbolic.dtype) == standardize_dtype(eager.dtype)
        and len(symbolic.shape) == len(eager.shape)
        and all(s in (None, e) for s, e in zip(symbolic.shape, eager.shape))
    )


def _all_dims_unknown(shapes):
    return [(None,) * len(shape) for shape in shapes]


def _one_dim_unknown(shapes):
    """Every way of making one dimension of one of `shapes` unknown."""
    variants = []
    for i, shape in enumerate(shapes):
        for d in range(len(shape)):
            unknown = shape[:d] + (None,) + shape[d + 1 :]
            variants.append(shapes[:i] + [unknown] + shapes[i + 1 :])
    return variants


def _signature(values):
    """Describes inputs or outputs for messages, e.g. `int32[2, 3], 1.0`."""
    return ", ".join(
        f"{standardize_dtype(x.dtype)}{list(x.shape)}"
        if hasattr(x, "shape")
        else repr(x)
        for x in tree.flatten(values)
    )


def _jax_uses(device_type):
    import jax

    return jax.default_backend() == device_type


def _tensorflow_uses(device_type):
    import tensorflow as tf

    return len(tf.config.list_physical_devices(device_type.upper())) > 0


def _torch_uses(device_type):
    if device_type == "gpu":
        from keras.src.backend.torch.ops.core import get_device

        return get_device() == "cuda"
    return device_type == "cpu"


def uses_gpu():
    if not hasattr(uses_gpu, "_value"):
        if backend.backend() == "tensorflow":
            uses_gpu._value = _tensorflow_uses("gpu")
        elif backend.backend() == "jax":
            uses_gpu._value = _jax_uses("gpu")
        elif backend.backend() == "torch":
            uses_gpu._value = _torch_uses("gpu")
        else:
            uses_gpu._value = False
    return uses_gpu._value


def uses_tpu():
    if not hasattr(uses_tpu, "_value"):
        if backend.backend() == "tensorflow":
            uses_tpu._value = _tensorflow_uses("tpu")
        elif backend.backend() == "jax":
            uses_tpu._value = _jax_uses("tpu")
        else:
            uses_tpu._value = False
    return uses_tpu._value


def jax_uses_gpu():
    return backend.backend() == "jax" and uses_gpu()


def jax_uses_tpu():
    return backend.backend() == "jax" and uses_tpu()


def tensorflow_uses_gpu():
    return backend.backend() == "tensorflow" and uses_gpu()


def tensorflow_uses_tpu():
    return backend.backend() == "tensorflow" and uses_tpu()


def torch_uses_gpu():
    return backend.backend() == "torch" and uses_gpu()


def create_keras_tensors(input_shape, dtype, sparse, ragged):
    if isinstance(input_shape, dict):
        return {
            utils.removesuffix(k, "_shape"): KerasTensor(
                v, dtype=dtype[k], sparse=sparse, ragged=ragged
            )
            for k, v in input_shape.items()
        }
    return map_shape_dtype_structure(
        lambda shape, dt: KerasTensor(
            shape, dtype=dt, sparse=sparse, ragged=ragged
        ),
        input_shape,
        dtype,
    )


def create_eager_tensors(input_shape, dtype, sparse, ragged):
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

    elif ragged:
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            def create_fn(shape, dt):
                rng = np.random.default_rng(0)
                x = (4 * rng.standard_normal(shape)).astype(dt)
                x = np.multiply(x, rng.random(shape) < 0.7)
                return tf.RaggedTensor.from_tensor(x, padding=0)

        else:
            raise ValueError(
                f"Ragged is unsupported with backend {backend.backend()}"
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


def to_json_with_tuples(value):
    def _tuple_encode(obj):
        if isinstance(obj, tuple):
            return {"__class__": "tuple", "__value__": list(obj)}
        if isinstance(obj, list):
            return [_tuple_encode(e) for e in obj]
        if isinstance(obj, dict):
            return {key: _tuple_encode(value) for key, value in obj.items()}
        return obj

    class _PreserveTupleJsonEncoder(json.JSONEncoder):
        def encode(self, obj):
            obj = _tuple_encode(obj)
            return super().encode(obj)

    return _PreserveTupleJsonEncoder(sort_keys=True, indent=4).encode(value)


def from_json_with_tuples(value):
    def _tuple_decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "__class__" not in obj or "__value__" not in obj:
            return obj
        return tuple(obj["__value__"])

    return json.loads(value, object_hook=_tuple_decode)
