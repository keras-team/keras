import json
import shutil
import tempfile
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
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils

# Input dtypes `run_op_input_dtypes_check` skips per backend, with the reason
# shown in the test report.
_SKIPPED_DTYPES = {
    "torch": {
        "uint16": "Not supported by the torch backend",
        "uint32": "Not supported by the torch backend",
    },
    "tensorflow": {
        "uint32": "TensorFlow promotes uint32 + int* to int64, which JAX "
        "gives as int32"
    },
}


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
                    return expected_dtype == backend.standardize_dtype(x.dtype)

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

    def run_op_test(
        self,
        op,
        reference_fn,
        inputs,
        kwargs=None,
        run_dynamic_shape_checks=True,
        atol=1e-6,
        rtol=1e-6,
    ):
        """Runs the standard checks on an op.

        The checks are `eager` (values vs `reference_fn`), `symbolic_static`,
        `symbolic_dynamic` (every input dimension unknown) and
        `symbolic_partial` (one input dimension unknown at a time, plus the
        first dimension of every input for 2+ inputs), see
        `run_op_symbolic_check`. Each check runs in its own `subTest`, so one
        failing check doesn't hide the others; a check that refines a failed
        one is skipped instead, so one bug is one failure.

        The symbolic checks need a function with array inputs: an `Operation`
        instance (`knp.Square()`) is already covered through its function, so
        it only runs `eager` (its `call`), and an op without array inputs
        (`knp.ones`, `knp.blackman(5)`) has nothing symbolic to test.

        Args:
            op: The op to test, e.g. `knp.square` or `knp.Square()`.
            reference_fn: Reference implementation, e.g. `np.square`.
            inputs: Tuple of inputs passed to the op positionally: NumPy
                arrays (may be nested) and Python scalars, which are passed
                through as-is.
            kwargs: Static arguments, passed to the op by keyword.
            run_dynamic_shape_checks: Whether to run `symbolic_dynamic` and
                `symbolic_partial`. Disable for ops that need static sizes,
                e.g. `squeeze`. Disabled checks are reported as skipped,
                unlike the inapplicable ones above, which aren't run at all.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
        """
        kwargs = kwargs or {}
        arrays = _arrays(inputs)
        # Computed once and shared by every check. An op that doesn't run
        # eagerly fails here, before any check.
        outputs = _call_eager(op, inputs, kwargs)
        with self.subTest(check="eager"):
            self.run_op_eager_check(
                op, reference_fn, inputs, kwargs, atol, rtol, outputs
            )
        if not arrays or isinstance(op, Operation):
            return  # Nothing symbolic to test, see the docstring.
        static = [x.shape for x in arrays]
        dynamic, partial = _unknown_dim_variants(static)
        disabled = None if run_dynamic_shape_checks else "Disabled by the test"
        # Check name, its symbolic shape variants, reason to skip it or None.
        groups = [
            ("symbolic_static", [static], None),
            ("symbolic_dynamic", dynamic, disabled),
            ("symbolic_partial", partial, disabled),
        ]
        static_failed = False
        for name, variants, skip_reason in groups:
            if static_failed and name != "symbolic_static":
                skip_reason = "symbolic_static failed"  # It refines static.
            if skip_reason:
                with self.subTest(check=name):
                    self.skipTest(skip_reason)
                continue
            for shapes in variants:
                shapes_id = " x ".join(str(shape) for shape in shapes)
                with self.subTest(check=name, shapes=shapes_id):
                    try:
                        self.run_op_symbolic_check(
                            op, inputs, kwargs, shapes, outputs
                        )
                    except Exception:
                        static_failed |= name == "symbolic_static"
                        raise

    def run_op_eager_check(
        self,
        op,
        reference_fn,
        inputs,
        kwargs,
        atol=1e-6,
        rtol=1e-6,
        eager_outputs=None,
    ):
        """Checks the eager output count and values against `reference_fn`.

        Args:
            op: The op to test.
            reference_fn: Reference implementation.
            inputs: Tuple of inputs, as in `run_op_test`.
            kwargs: Static arguments, passed to both functions by keyword.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            eager_outputs: The op's outputs for `inputs`, if already computed.
        """
        if eager_outputs is None:
            eager_outputs = _call_eager(op, inputs, kwargs)
        actual = tree.flatten(eager_outputs)
        expected = tree.flatten(reference_fn(*inputs, **kwargs))
        msg = f"Unexpected eager output for inputs {_signature(inputs)}"
        self.assertLen(actual, len(expected), f"{msg}: output count differs")
        for a, e in zip(actual, expected):
            self.assertAllClose(a, e, atol=atol, rtol=rtol, msg=msg)

    def run_op_symbolic_check(
        self, op, inputs, kwargs, symbolic_shapes=None, eager_outputs=None
    ):
        """Checks the symbolic output against the eager output.

        Compares output count, rank, known dimensions and dtype. Symbolic
        inputs mirror the converted eager tensors, so both paths see the
        same input dtypes.

        Args:
            op: The op to test.
            inputs: Tuple of inputs, as in `run_op_test`.
            kwargs: Static arguments, passed to the op by keyword.
            symbolic_shapes: One shape per array input, in flattening order,
                with `None` for the dimensions that are unknown symbolically.
                Defaults to the arrays' shapes. An output dimension may be
                `None` only if it depends on an unknown input dimension: one
                that doesn't change when the unknown input dimensions are
                resized must be known.
            eager_outputs: The op's outputs for `inputs`, if already computed.
        """
        arrays = _arrays(inputs)
        if symbolic_shapes is None:
            symbolic_shapes = [x.shape for x in arrays]
        if eager_outputs is None:
            eager_outputs = _call_eager(op, inputs, kwargs)
        eager = tree.flatten(eager_outputs)
        where = f"for inputs {_signature(inputs)}"

        symbolic = _call_symbolic(op, inputs, kwargs, symbolic_shapes)
        if len(symbolic) != len(eager):
            raise AssertionError(
                f"Output count differs: symbolic={len(symbolic)}, "
                f"eager={len(eager)} {where}"
            )
        for i, (sym, eag) in enumerate(zip(symbolic, eager)):
            out = f"Output {i}" if len(eager) > 1 else "Output"
            if not isinstance(sym, KerasTensor):
                raise AssertionError(
                    f"{out} is a {type(sym).__name__}, not a KerasTensor "
                    f"{where}"
                )
            if not _symbolic_matches(sym, eag):
                raise AssertionError(
                    f"{out} differs: symbolic={_signature([sym])}, "
                    f"eager={_signature([eag])} {where}"
                )
        if not any(None in shape for shape in symbolic_shapes):
            return
        # Precision: an output dim that doesn't change when the unknown input
        # dims are resized can't depend on them, so it must be known.
        resized = [
            tuple(
                (2 if e == 1 else 1) if s is None else s
                for s, e in zip(shape, x.shape)
            )
            for shape, x in zip(symbolic_shapes, arrays)
        ]
        try:
            probe = _call_symbolic(op, inputs, kwargs, resized)
        except Exception:  # The resized shapes may be invalid for the op.
            return
        if [len(p.shape) for p in probe] != [len(e.shape) for e in eager]:
            return
        for sym, eag, prb in zip(symbolic, eager, probe):
            for d, (s, e, p) in enumerate(zip(sym.shape, eag.shape, prb.shape)):
                if s is None and p == e:
                    raise AssertionError(
                        f"Output dim {d} is None but does not depend on the "
                        f"unknown input dims {where}"
                    )

    def run_op_input_dtypes_check(
        self, op, reference_fn, jax_fn, inputs, kwargs, dtypes
    ):
        """Checks the output dtypes for one input dtype combination.

        Casts the array inputs to `dtypes` and checks, in two `subTest`s,
        that the symbolic output dtypes match the eager ones (`symbolic`) and
        that the eager output dtypes match `jax_fn` on the same inputs
        (`jax_dtype`; default JAX mode, so no 64-bit dtypes). If JAX rejects
        the combination, the check fails.

        Skipped if the backend doesn't support one of `dtypes` (see
        `_SKIPPED_DTYPES`).

        Args:
            op: The op to test, e.g. `knp.add`.
            reference_fn: Reference implementation, e.g. `np.add`. Not used
                yet: a value check per dtype combination is planned.
            jax_fn: JAX reference for the output dtypes, e.g. `jnp.add`.
            inputs: Tuple of inputs, as in `run_op_test`. Array values
                should be small non-negative integers (e.g. `0` to `5`), so
                they fit every dtype.
            kwargs: Static arguments, passed to all three functions.
            dtypes: One dtype per array input, in flattening order, e.g.
                `("int8", "uint32")`.
        """
        skipped = _SKIPPED_DTYPES.get(backend.backend(), {})
        for dtype in dtypes:
            if dtype in skipped:
                self.skipTest(f"{dtype}: {skipped[dtype]}")
        cast = [
            x.astype(d) for x, d in zip(_arrays(inputs), dtypes, strict=True)
        ]
        inputs = _replace_arrays(inputs, cast)
        outputs = _call_eager(op, inputs, kwargs)
        with self.subTest(check="symbolic"):
            self.run_op_symbolic_check(
                op, inputs, kwargs, eager_outputs=outputs
            )
        with self.subTest(check="jax_dtype"):
            import jax.numpy as jnp

            jax_outputs = jax_fn(*_map_arrays(jnp.asarray, inputs), **kwargs)
            keras_dtypes = _dtypes_of(outputs)
            jax_dtypes = _dtypes_of(jax_outputs)
            if keras_dtypes != jax_dtypes:
                raise AssertionError(
                    "Output dtype differs from JAX: "
                    f"keras={', '.join(keras_dtypes)}, "
                    f"jax={', '.join(jax_dtypes)} "
                    f"for inputs {_signature(inputs)}"
                )


def _is_scalar(x):
    return isinstance(x, (int, float))


def _arrays(inputs):
    """The array inputs in flattening order, without the Python scalars."""
    return [x for x in tree.flatten(inputs) if not _is_scalar(x)]


def _map_arrays(fn, inputs):
    """Applies `fn` to the array inputs, passing Python scalars through."""
    return tree.map_structure(lambda x: x if _is_scalar(x) else fn(x), inputs)


def _replace_arrays(inputs, arrays):
    """Returns `inputs` with its arrays replaced by `arrays`, in order."""
    arrays = iter(arrays)
    return _map_arrays(lambda x: next(arrays), inputs)


def _call_eager(op, inputs, kwargs):
    """Calls `op` with the arrays converted to backend tensors."""
    return op(*_map_arrays(ops.convert_to_tensor, inputs), **kwargs)


def _call_symbolic(op, inputs, kwargs, shapes):
    """Calls `op` with the arrays replaced by `KerasTensor`s of `shapes`.

    Their dtypes mirror the converted eager tensors, so both paths see the
    same input dtypes (e.g. NumPy `int64` is `int32` on JAX).
    """
    dtypes = _dtypes_of([ops.convert_to_tensor(x) for x in _arrays(inputs)])
    tensors = [KerasTensor(s, dtype=d) for s, d in zip(shapes, dtypes)]
    return tree.flatten(op(*_replace_arrays(inputs, tensors), **kwargs))


def _dtypes_of(xs):
    """The standardized dtypes of the flattened `xs`."""
    return [standardize_dtype(x.dtype) for x in tree.flatten(xs)]


def _symbolic_matches(sym, eag):
    """Whether a `KerasTensor` is compatible with an eager tensor."""
    shape = tuple(eag.shape)
    return (
        len(sym.shape) == len(shape)
        and all(s in (None, e) for s, e in zip(sym.shape, shape))
        and standardize_dtype(sym.dtype) == standardize_dtype(eag.dtype)
    )


def _unknown_dim_variants(shapes):
    """Variants of `shapes` with unknown (`None`) dimensions to test.

    Returns `(dynamic, partial)`: `dynamic` has every dimension unknown;
    `partial` has one dimension unknown at a time, plus the first dimension
    of every array for 2+ arrays (a shared batch dimension). Each variant is
    a list with one shape per array, e.g. `[(None, 3), (2, 3)]`.
    """

    def with_unknown(positions):
        return [
            tuple(
                None if (i, d) in positions else s for d, s in enumerate(shape)
            )
            for i, shape in enumerate(shapes)
        ]

    dims = [(i, d) for i, shape in enumerate(shapes) for d in range(len(shape))]
    partial = [with_unknown({p}) for p in dims]
    batch = {(i, 0) for i, shape in enumerate(shapes) if shape}
    if len(batch) > 1:
        partial.append(with_unknown(batch))
    return [with_unknown(set(dims))], partial


def _signature(inputs):
    """Describes the inputs for messages, e.g. `int64[2, 3], float32[2]`."""
    return ", ".join(
        repr(x)
        if _is_scalar(x)
        else f"{standardize_dtype(x.dtype)}{list(x.shape)}"
        for x in tree.flatten(inputs)
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
