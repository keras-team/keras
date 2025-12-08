import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import export
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import quantizers
from keras.src import random
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.gptq_config import GPTQConfig


class EinsumDenseTest(testing.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "_1d_end_weight",
            "equation": "ab,b->a",
            "bias_axes": None,
            "input_shape": (2, 32),
            "output_shape": (),
            "expected_kernel_shape": (32,),
            "expected_bias_shape": None,
            "expected_output_shape": (2,),
        },
        {
            "testcase_name": "_2d_middle_weight",
            "equation": "ab,bc->ac",
            "bias_axes": None,
            "input_shape": (2, 32),
            "output_shape": (64),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 64),
        },
        {
            "testcase_name": "_3d_bert",
            "equation": "abc,cde->abde",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (1, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_3_bias",
            "equation": "abc,cde->abde",
            "bias_axes": "e",
            "input_shape": (2, 1, 2),
            "output_shape": (1, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (4,),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_2_bias",
            "equation": "abc,cde->abde",
            "bias_axes": "d",
            "input_shape": (2, 1, 2),
            "output_shape": (1, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (3, 1),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_1_3_bias",
            "equation": "abc,cde->abde",
            "bias_axes": "be",
            "input_shape": (2, 7, 2),
            "output_shape": (7, 3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (7, 1, 4),
            "expected_output_shape": (2, 7, 3, 4),
        },
        {
            "testcase_name": "_3d_bert_projection",
            "equation": "BFNH,NHD->BFD",
            "bias_axes": None,
            "input_shape": (2, 1, 2, 3),
            "output_shape": (1, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 4),
        },
        {
            "testcase_name": "_2d_bert",
            "equation": "abc,cd->abd",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (1, 4),
            "expected_kernel_shape": (2, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 4),
        },
        {
            "testcase_name": "_embedding_1d",
            "equation": "i,d->id",
            "bias_axes": None,
            "input_shape": (2,),
            "output_shape": (2,),
            "expected_kernel_shape": (2,),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 2),
        },
        {
            "testcase_name": "_xlnet_lm",
            "equation": "ibd,nd->ibn",
            "bias_axes": None,
            "input_shape": (2, 2, 1),
            "output_shape": (2, 2),
            "expected_kernel_shape": (2, 1),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 2, 2),
        },
        {
            "testcase_name": "_2d_precast",
            "equation": "...b,bc->...c",
            "bias_axes": None,
            "input_shape": (2, 32),
            "output_shape": (64,),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 64),
        },
        {
            "testcase_name": "_2d_precast_elided_input_used_in_output",
            "equation": "...bc,bc->...b",
            "bias_axes": None,
            "input_shape": (2, 32, 64),
            "output_shape": (32,),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 32),
        },
        {
            "testcase_name": "_2d_precast_multiple_elided_dims",
            "equation": "...b,bc->...c",
            "bias_axes": None,
            "input_shape": (2, 3, 32),
            "output_shape": (64,),
            "expected_kernel_shape": (32, 64),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 3, 64),
        },
        {
            "testcase_name": "_3d_precast",
            "equation": "...c,cde->...de",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_precast_3_bias",
            "equation": "...c,cde->...de",
            "bias_axes": "e",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (4,),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_precast_2_bias",
            "equation": "...c,cde->...de",
            "bias_axes": "d",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (3, 1),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_3d_precast_2_3_bias",
            "equation": "...c,cde->...de",
            "bias_axes": "de",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (2, 3, 4),
            "expected_bias_shape": (3, 4),
            "expected_output_shape": (2, 1, 3, 4),
        },
        {
            "testcase_name": "_2d_postcast",
            "equation": "bc...,cd->bd...",
            "bias_axes": None,
            "input_shape": (2, 1, 2, 3),
            "output_shape": (4,),
            "expected_kernel_shape": (1, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 4, 2, 3),
        },
        {
            "testcase_name": "_3d_postcast",
            "equation": "bc...,cde->bde...",
            "bias_axes": None,
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": None,
            "expected_output_shape": (2, 3, 4, 2),
        },
        {
            "testcase_name": "_3d_postcast_1_bias",
            "equation": "bc...,cde->bde...",
            "bias_axes": "d",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": (3, 1, 1),
            "expected_output_shape": (2, 3, 4, 2),
        },
        {
            "testcase_name": "_3d_postcast_2_bias",
            "equation": "bc...,cde->bde...",
            "bias_axes": "e",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": (4, 1),
            "expected_output_shape": (2, 3, 4, 2),
        },
        {
            "testcase_name": "_3d_postcast_1_2_bias",
            "equation": "bc...,cde->bde...",
            "bias_axes": "de",
            "input_shape": (2, 1, 2),
            "output_shape": (3, 4),
            "expected_kernel_shape": (1, 3, 4),
            "expected_bias_shape": (3, 4, 1),
            "expected_output_shape": (2, 3, 4, 2),
        },
    )
    @pytest.mark.requires_trainable_backend
    def test_einsum_dense_basics(
        self,
        equation,
        bias_axes,
        input_shape,
        output_shape,
        expected_kernel_shape,
        expected_bias_shape,
        expected_output_shape,
    ):
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": equation,
                "output_shape": output_shape,
                "bias_axes": bias_axes,
            },
            input_shape=input_shape,
            expected_output_shape=expected_output_shape,
            expected_num_trainable_weights=(
                2 if expected_bias_shape is not None else 1
            ),
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        layer = layers.EinsumDense(
            equation, output_shape=output_shape, bias_axes=bias_axes
        )
        layer.build(input_shape)
        self.assertEqual(layer.kernel.shape, expected_kernel_shape)
        if expected_bias_shape is not None:
            self.assertEqual(layer.bias.shape, expected_bias_shape)

    def test_einsum_dense_constraints(self):
        layer = layers.EinsumDense(
            "abc,cde->abde", (1, 3, 4), kernel_constraint="non_neg"
        )
        layer.build((2, 1, 2))
        self.assertIsInstance(layer.kernel.constraint, constraints.NonNeg)
        layer = layers.EinsumDense(
            "ab,b->a", (1, 3, 4), bias_axes="a", bias_constraint="non_neg"
        )
        layer.build((2, 1, 2))
        self.assertIsInstance(layer.bias.constraint, constraints.NonNeg)

    @pytest.mark.requires_trainable_backend
    def test_enable_lora(self):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes=None,
        )
        layer.build((None, 3))
        layer.enable_lora(2)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 3)
        # Try eager call
        x = np.random.random((64, 3))
        y = np.random.random((64, 8, 32))
        _ = layer(x[:2])

        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()

        # Try calling fit()
        model = models.Sequential(
            [
                layer,
            ]
        )
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        final_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        final_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_kernel_value - final_lora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_kernel_value - final_lora_b_kernel_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "lora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-lora model
        new_model = models.Sequential(
            [
                layers.EinsumDense(
                    equation="ab,bcd->acd",
                    output_shape=(8, 32),
                    bias_axes=None,
                ),
            ]
        )
        new_model.build((None, 3))
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @pytest.mark.requires_trainable_backend
    def test_enable_lora_with_alpha(self):
        # Use a simple equation that mimics a `Dense` layer behavior.
        equation = "ab,bc->ac"
        output_shape = 3  # This means the kernel shape will be (input_dim, 3).
        bias_axes = None

        # Create and build the `EinsumDense` layer
        # with an input shape (None, 2).
        layer = layers.EinsumDense(
            equation=equation, output_shape=output_shape, bias_axes=bias_axes
        )
        # Build the layer with an input shape of (batch, 2).
        layer.build((None, 2))

        # Set the base kernel weights to a known value.
        base_kernel = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        layer._kernel.assign(base_kernel)

        # Enable LoRA with `rank`=2 and a custom `lora_alpha`=3.0.
        layer.enable_lora(rank=2, lora_alpha=3.0)
        self.assertEqual(layer.lora_rank, 2)
        self.assertEqual(layer.lora_alpha, 3.0)

        # The expected shapes are:
        #   `base_kernel`: (2, 3)
        #   `lora_kernel_a`: (2, 2) and `lora_kernel_b`: (2, 3)
        a_val = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        b_val = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32)
        layer.lora_kernel_a.assign(a_val)
        layer.lora_kernel_b.assign(b_val)

        # Compute expected effective kernel.
        # Scaling factor is `lora_alpha / lora_rank` = 3.0 / 2 = 1.5
        expected_delta = 1.5 * np.matmul(a_val, b_val)
        expected_kernel = base_kernel + expected_delta

        # Verify that the effective kernel property returns the expected value.
        actual_kernel = ops.convert_to_numpy(layer.kernel)
        self.assertAllClose(
            actual_kernel, expected_kernel, tpu_atol=1e-3, tpu_rtol=1e-3
        )

    @pytest.mark.requires_trainable_backend
    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": "ab,bcd->acd",
                "output_shape": (8, 32),
                "bias_axes": None,
                "lora_rank": 2,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 8, 32),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    # Test quantization-related methods.

    @parameterized.named_parameters(
        ("int8", "int8", 1e-3),
        ("int4", "int4", 3e-3),
    )
    def test_quantize_int(self, mode, error_threshold):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        x = np.random.random((2, 3))
        y_float = layer(x)
        layer.quantize(mode)

        # Verify weights dtype
        self.assertEqual(backend.standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(
            backend.standardize_dtype(layer.kernel_scale.dtype),
            layer.variable_dtype,
        )

        # Try eager call and verify output correctness
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, error_threshold)  # A weak correctness test

        # Try saving and reloading the model
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Try building with quantized dtype policy
        layer = layers.EinsumDense(
            equation="abcde,afce->acdbf",  # Test reduce and transpose
            output_shape=(2, 4, 8, 16),
            bias_axes="d",
            dtype=f"{mode}_from_mixed_bfloat16",
        )
        layer.build((1, 8, 2, 4, 32))
        self.assertEqual(backend.standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(
            backend.standardize_dtype(layer.kernel_scale.dtype), "float32"
        )
        layer = layers.EinsumDense(
            equation="a,b->ab",  # Test expand
            output_shape=(4,),
            dtype=f"{mode}_from_float32",
        )
        layer.build((None,))
        self.assertEqual(backend.standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(
            backend.standardize_dtype(layer.kernel_scale.dtype), "float32"
        )
        layer = layers.EinsumDense(
            equation="ab,ab->a",  # Test squeeze
            output_shape=(2,),
            dtype="int8_from_float32",
        )
        layer.build((2, 4))
        self.assertEqual(backend.standardize_dtype(layer._kernel.dtype), "int8")
        self.assertEqual(
            backend.standardize_dtype(layer.kernel_scale.dtype), "float32"
        )

    @parameterized.named_parameters(
        (
            "int8_btnh,nhd->btd",
            "int8",
            "btnh,nhd->btd",
            (None, 8),
            (1, 2, 2, 4),
            1e-3,
        ),
        (
            "int8_btd,ndh->btnh",
            "int8",
            "btd,ndh->btnh",
            (None, 2, 8),
            (1, 2, 4),
            1e-3,
        ),
        ("int8_btd,df->btf", "int8", "btd,df->btf", (None, 4), (1, 2, 4), 1e-3),
        (
            "int4_btnh,nhd->btd",
            "int4",
            "btnh,nhd->btd",
            (None, 8),
            (1, 2, 2, 4),
            3e-3,
        ),
        (
            "int4_btd,ndh->btnh",
            "int4",
            "btd,ndh->btnh",
            (None, 2, 8),
            (1, 2, 4),
            3e-3,
        ),
        (
            "int4_btd,df->btf",
            "int4",
            "btd,df->btf",
            (None, 4),
            (1, 2, 4),
            3e-3,
        ),
    )
    def test_quantize_with_specific_equations(
        self,
        quantization_mode,
        equation,
        output_shape,
        input_shape,
        error_threshold,
    ):
        layer = layers.EinsumDense(equation=equation, output_shape=output_shape)
        layer.build(input_shape)
        x = ops.random.uniform(input_shape)
        y_float = layer(x)

        layer.quantize(quantization_mode)
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, error_threshold)  # A weak correctness test

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
        ("int4", "int4"),
    )
    def test_quantize_on_unbuilt_layer(self, mode):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            layer.quantize(mode)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
        ("int4", "int4"),
    )
    def test_quantize_on_subclass(self, mode):
        class MyEinsumDense(layers.EinsumDense):
            pass

        layer = MyEinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        with self.assertRaises(NotImplementedError):
            layer.quantize(mode)

        layer.quantize(mode, type_check=False)  # No error

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
        ("int4", "int4"),
    )
    def test_quantize_when_already_quantized(self, mode):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 16),
            bias_axes="d",
        )
        layer.build((None, 3))
        layer.quantize(mode)
        for m in ["int8", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 16),
            bias_axes="d",
            dtype=f"{mode}_from_float32",
        )
        layer.build((None, 3))
        for m in ["int8", "float8"]:
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", 3),
        ("float8", "float8_from_float32", 8),
        ("int4", "int4_from_float32", 3),
    )
    def test_quantize_by_setting_dtype_policy(
        self, policy, expected_num_variables
    ):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        layer.dtype_policy = policy
        self.assertLen(layer.variables, expected_num_variables)

    @parameterized.named_parameters(
        ("int7", "int7"),
        ("float7", "float7"),
        ("int3", "int3"),
    )
    def test_quantize_invalid_mode(self, mode):
        layer = layers.EinsumDense(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer.build((None, 3))
        x = np.random.random((1, 3))
        # dtype_policy should not be altered by failed quantization
        original_dtype_policy = layer.dtype_policy

        # Test quantize
        with self.assertRaisesRegex(ValueError, "Invalid quantization mode."):
            layer.quantize(mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_build
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            layer.quantized_build((None, 2), mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_call
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            # Explicitly set quantization_mode
            layer._dtype_policy._quantization_mode = mode
            layer.quantized_call(x)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

    @parameterized.named_parameters(
        ("int8", "int8_from_mixed_bfloat16", 1, 2),
        ("float8", "float8_from_mixed_bfloat16", 8, 0),
        ("int4", "int4_from_mixed_bfloat16", 1, 2),
    )
    @pytest.mark.requires_trainable_backend
    def test_quantize_dtype_argument(
        self, dtype, num_trainable_weights, num_non_trainable_weights
    ):
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": "ab,bcd->acd",
                "output_shape": (8, 32),
                "bias_axes": "d",
                "dtype": dtype,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 8, 32),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.named_parameters(
        ("int8_ab,bcd->acd", "int8", "ab,bcd->acd", (64, 3), (64, 8, 32)),
        (
            "int8_btd,ndh->btnh",
            "int8",
            "btd,ndh->btnh",
            (1, 4, 32),
            (1, 4, 8, 16),
        ),
        ("int4_ab,bcd->acd", "int4", "ab,bcd->acd", (64, 3), (64, 8, 32)),
        (
            "int4_btd,ndh->btnh",
            "int4",
            "btd,ndh->btnh",
            (1, 4, 32),
            (1, 4, 8, 16),
        ),
    )
    @pytest.mark.requires_trainable_backend
    def test_quantize_lora_integration(
        self, quantization_mode, equation, input_shape, output_shape
    ):
        config = dict(
            equation=equation, output_shape=output_shape[1:], bias_axes=None
        )
        layer = layers.EinsumDense(**config)
        layer.build(input_shape)
        layer.enable_lora(2)
        layer.quantize(quantization_mode)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(layer.non_trainable_weights, 2)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 4)

        # Try calling fit()
        init_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        init_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        x = np.random.random(input_shape)
        y = np.random.random(output_shape)
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        final_lora_a_kernel_value = layer.lora_kernel_a.numpy()
        final_lora_b_kernel_value = layer.lora_kernel_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_kernel_value - final_lora_a_kernel_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_kernel_value - final_lora_b_kernel_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.EinsumDense(**config)])
        new_model.build(input_shape)
        new_model.quantize(quantization_mode)
        new_model.load_weights(temp_filepath)
        self.assertFalse(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal(input_shape)
            ref_output = model(ref_input)
            model.export(temp_filepath, format="tf_saved_model")
            reloaded_layer = export.TFSMLayer(temp_filepath)
            self.assertAllClose(
                reloaded_layer(ref_input), ref_output, atol=1e-7
            )
            self.assertLen(reloaded_layer.weights, len(model.weights))
            self.assertLen(
                reloaded_layer.trainable_weights, len(model.trainable_weights)
            )
            self.assertLen(
                reloaded_layer.non_trainable_weights,
                len(model.non_trainable_weights),
            )

    @pytest.mark.requires_trainable_backend
    def test_quantize_float8(self):
        import ml_dtypes

        from keras.src import quantizers

        layer = layers.EinsumDense(
            "ab,bc->ac",
            output_shape=[32],
            bias_axes="c",
        )
        layer.build((None, 16))
        layer.quantize("float8")
        optimizer = optimizers.AdamW(learning_rate=0.1)
        optimizer.build(layer.trainable_variables)

        def loss_fn(x, dy):
            y = layer(x, training=True)
            loss = y * ops.cast(dy, y.dtype)
            return ops.sum(loss)

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            @tf.function(jit_compile=True)
            def train_one_step(x, dy):
                with tf.GradientTape() as tape:
                    loss = loss_fn(x, dy)
                grads = tape.gradient(loss, layer.trainable_variables)
                optimizer.apply(grads, layer.trainable_variables)

        elif backend.backend() == "jax":
            import jax

            def stateless_loss_fn(trainable_variables, x, dy):
                y = layer.stateless_call(
                    trainable_variables, [], x, training=True
                )[0]
                loss = y * ops.cast(dy, y.dtype)
                return ops.sum(loss)

            grad_fn = jax.jit(jax.grad(stateless_loss_fn))

            def train_one_step(x, dy):
                trainable_variables = [
                    v.value for v in layer.trainable_variables
                ]
                optimizer_variables = [v.value for v in optimizer.variables]
                grads = grad_fn(trainable_variables, x, dy)
                trainable_variables, optimizer_variables = (
                    optimizer.stateless_apply(
                        optimizer_variables, grads, trainable_variables
                    )
                )
                for variable, value in zip(
                    layer.trainable_variables, trainable_variables
                ):
                    variable.assign(value)
                for variable, value in zip(
                    optimizer.variables, optimizer_variables
                ):
                    variable.assign(value)

        elif backend.backend() == "torch":

            def train_one_step(x, dy):
                layer.zero_grad()
                loss = loss_fn(x, dy)
                loss.backward()
                grads = [v.value.grad for v in layer.trainable_variables]
                optimizer.apply(grads, layer.trainable_variables)

        scale_x, amax_history_x = ops.ones(()), ops.zeros((1024,))
        scale_k, amax_history_k = ops.ones(()), ops.zeros((1024,))
        scale_g, amax_history_g = ops.ones(()), ops.zeros((1024,))
        e4m3_max = ops.cast(
            float(ml_dtypes.finfo("float8_e4m3fn").max), "float32"
        )
        e5m2_max = ops.cast(
            float(ml_dtypes.finfo("float8_e5m2").max), "float32"
        )

        for _ in range(3):
            x = random.normal((16, 16), dtype="float32")
            g = random.normal((16, 32), dtype="float32")
            k = ops.convert_to_tensor(layer._kernel)

            # Manually compute the expected amax history and scaling factors.
            amax_from_history_x = ops.max(amax_history_x)
            amax_from_history_k = ops.max(amax_history_k)
            amax_from_history_g = ops.max(amax_history_g)
            scale_x = quantizers.compute_float8_scale(
                amax_from_history_x, scale_x, e4m3_max
            )
            scale_k = quantizers.compute_float8_scale(
                amax_from_history_k, scale_k, e4m3_max
            )
            scale_g = quantizers.compute_float8_scale(
                amax_from_history_g, scale_g, e5m2_max
            )
            amax_history_x = quantizers.compute_float8_amax_history(
                x, amax_history_x
            )
            amax_history_k = quantizers.compute_float8_amax_history(
                k, amax_history_k
            )
            amax_history_g = quantizers.compute_float8_amax_history(
                g, amax_history_g
            )

            train_one_step(x, g)

            self.assertAllClose(layer.inputs_amax_history, amax_history_x)
            self.assertAllClose(layer.kernel_amax_history, amax_history_k)
            self.assertAllClose(layer.outputs_grad_amax_history, amax_history_g)
            self.assertAllClose(layer.inputs_scale, scale_x)
            self.assertAllClose(layer.kernel_scale, scale_k)
            self.assertAllClose(layer.outputs_grad_scale, scale_g)

    @pytest.mark.requires_trainable_backend
    def test_quantize_float8_fitting(self):
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize("float8")
        self.assertLen(layer.trainable_weights, 8)
        self.assertLen(layer.non_trainable_weights, 0)

        # Try calling fit()
        x = np.random.random((64, 3))
        y = np.random.random((64, 8, 32))
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=2)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.EinsumDense(**config)])
        new_model.build((None, 3))
        new_model.quantize("float8")
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal((2, 3))
            ref_output = model(ref_input)
            model.export(temp_filepath, format="tf_saved_model")
            reloaded_layer = export.TFSMLayer(temp_filepath)
            self.assertAllClose(reloaded_layer(ref_input), ref_output)
            self.assertLen(reloaded_layer.weights, len(model.weights))
            self.assertLen(
                reloaded_layer.trainable_weights, len(model.trainable_weights)
            )
            self.assertLen(
                reloaded_layer.non_trainable_weights,
                len(model.non_trainable_weights),
            )

    def test_quantize_float8_inference(self):
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize("float8")

        # Try calling with `training=False` and the result must match
        # `training=True` because there is no update.
        x = np.random.random((64, 3))
        y_inference = layer(x, training=False)
        y_training = layer(x, training=True)
        self.assertAllClose(y_inference, y_training)

    def test_gptq_serialization(self):
        """Test that a GPTQ-quantized layer can be serialized and deserialized
        correctly."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        config = layer.get_config()
        new_layer = layers.EinsumDense.from_config(config)
        new_layer.build((None, 3))
        self.assertEqual(new_layer.quantization_mode, "gptq")

    def test_int4_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 kernel."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(2,),
        )
        layer.build((None, 2))
        layer.quantize("int4")
        packed_kernel = layer._kernel
        self.assertAllClose(
            layer.kernel, quantizers.unpack_int4(packed_kernel, 2)
        )

    def test_legacy_load_own_variables(self):
        # In previous versions, `load_own_variables` accepted a store with
        # numeric keys.
        float32_store = {
            "0": np.random.random((3, 8, 32)).astype("float32"),
            "1": np.random.random((32,)).astype("float32"),
        }
        int8_store = {
            "0": np.random.randint(-128, 127, size=(3, 8, 32), dtype="int8"),
            "1": np.random.random((32,)).astype("float32"),
            "2": np.random.random((1, 8, 32)).astype("float32"),
        }
        int4_store = {
            "0": np.random.randint(-128, 127, size=(2, 8, 32), dtype="int8"),
            "1": np.random.random((32,)).astype("float32"),
            "2": np.random.random((1, 8, 32)).astype("float32"),
        }
        float8_store = {
            "0": np.random.random((3, 8, 32)).astype("float32"),
            "1": np.random.random((32,)).astype("float32"),
            # inputs_scale.
            "2": np.random.random(()).astype("float32"),
            # inputs_amax_history.
            "3": np.random.random((1024,)).astype("float32"),
            # kernel_scale.
            "4": np.random.random(()).astype("float32"),
            # kernel_amax_history.
            "5": np.random.random((1024,)).astype("float32"),
            # outputs_grad_scale.
            "6": np.random.random(()).astype("float32"),
            # outputs_grad_amax_history.
            "7": np.random.random((1024,)).astype("float32"),
        }
        gptq_store = {
            # bias
            "0": np.random.random((32,)).astype("float32"),
            # quantized_kernel
            "1": np.random.randint(0, 16, size=(16, 24), dtype="uint8"),
            # kernel_scale.
            "2": np.random.random((32, 3)).astype("float32"),
            # kernel_zero
            "3": np.random.random((32, 3)).astype("uint8"),
            # g_idx
            "4": np.random.random((24,)).astype("float32"),
        }
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )

        # Test float32 layer.
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))
        layer.load_own_variables(float32_store)
        self.assertAllClose(layer._kernel, float32_store["0"])
        self.assertAllClose(layer.bias, float32_store["1"])

        # Test int8-quantized layer.
        layer = layers.EinsumDense(**config, dtype="int8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(int8_store)
        self.assertAllClose(layer._kernel, int8_store["0"])
        self.assertAllClose(layer.bias, int8_store["1"])
        self.assertAllClose(layer.kernel_scale, int8_store["2"])

        # Test int4-quantized layer.
        layer = layers.EinsumDense(**config, dtype="int4_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(int4_store)
        self.assertAllClose(layer._kernel, int4_store["0"])
        self.assertAllClose(layer.bias, int4_store["1"])
        self.assertAllClose(layer.kernel_scale, int4_store["2"])

        # Test float8-quantized layer.
        layer = layers.EinsumDense(**config, dtype="float8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(float8_store)
        self.assertAllClose(layer._kernel, float8_store["0"])
        self.assertAllClose(layer.bias, float8_store["1"])
        self.assertAllClose(layer.inputs_scale, float8_store["2"])
        self.assertAllClose(layer.inputs_amax_history, float8_store["3"])
        self.assertAllClose(layer.kernel_scale, float8_store["4"])
        self.assertAllClose(layer.kernel_amax_history, float8_store["5"])
        self.assertAllClose(layer.outputs_grad_scale, float8_store["6"])
        self.assertAllClose(layer.outputs_grad_amax_history, float8_store["7"])

        # Test gptq-quantized layer.
        layer = layers.EinsumDense(**config, dtype="gptq/4/8_from_float32")
        layer.build((None, 3))
        layer.load_own_variables(gptq_store)
        self.assertTrue(layer.is_gptq_calibrated)
        self.assertAllClose(layer.bias, gptq_store["0"])
        self.assertAllClose(layer.quantized_kernel, gptq_store["1"])
        self.assertAllClose(layer.kernel_scale, gptq_store["2"])
        self.assertAllClose(layer.kernel_zero, gptq_store["3"])
        self.assertAllClose(layer.g_idx, gptq_store["4"])

    def test_int4_gptq_kernel_returns_unpacked_form(self):
        """Test that the `kernel` property returns the unpacked int4 GPTQ
        kernel."""
        layer = layers.EinsumDense(
            equation="ab,bc->ac",
            output_shape=(2,),
        )
        layer.build((None, 2))
        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )
        layer.is_gptq_calibrated = True  # Bypass calibration check
        packed_kernel = layer.quantized_kernel
        self.assertAllClose(
            layer.kernel, quantizers.unpack_int4(packed_kernel, 2)
        )

    def test_gptq_kernel_packing(self):
        """Validates that 4-bit GPTQ packing reduces the kernel size."""
        config = dict(
            equation="ab,bcd->acd",
            output_shape=(8, 32),
            bias_axes="d",
        )
        layer = layers.EinsumDense(**config)
        layer.build((None, 3))

        original_kernel_params = ops.prod(layer._kernel.shape)

        layer.quantize(
            "gptq",
            config=GPTQConfig(
                dataset=None, tokenizer=None, weight_bits=4, group_size=8
            ),
        )

        quantized_kernel_params = ops.prod(layer.quantized_kernel.shape)
        self.assertEqual(
            quantized_kernel_params,
            original_kernel_params // 2,
        )
