import numpy as np
import pytest
from absl.testing import parameterized

from keras import layers
from keras import testing
from keras.layers.core import einsum_dense
from keras.models import Sequential


class EinsumDenseTest(testing.TestCase, parameterized.TestCase):
    """Test the EinsumDense layer."""

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
        """Basic properties and shapes test for EinsumDense layer."""
        self.run_layer_test(
            layers.EinsumDense,
            init_kwargs={
                "equation": equation,
                "output_shape": output_shape,
                "bias_axes": bias_axes,
            },
            input_shape=input_shape,
            expected_output_shape=expected_output_shape,
            expected_num_trainable_weights=2
            if expected_bias_shape is not None
            else 1,
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

    def test_einsum_dense_activation(self):
        """Test activation functionality of EinsumDense layer"""
        equation = "ab,bc->ac"
        output_shape = 4
        bias_axes = None
        activation = "relu"

        layer = layers.EinsumDense(
            equation,
            output_shape=output_shape,
            bias_axes=bias_axes,
            activation=activation,
        )
        model = Sequential([layer])

        input_data = np.array([[-1, 2, -3], [4, -5, 6]])

        output_data = model.predict(input_data)

        assert np.all(
            output_data >= 0
        ), "ReLU activation was not applied correctly!"

    def test_analyze_einsum_string_kernel_shape_with_ellipses_on_right(self):
        equation = "ab0,bc->ac0"
        bias_axes = "c"
        input_shape = (2, 3, 4)
        output_shape = (3, 5)

        expected_kernel_shape = (3, 5)

        kernel_shape, _, _ = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(
            kernel_shape,
            expected_kernel_shape,
            f"Expected {expected_kernel_shape}, but got {kernel_shape}",
        )

    def test_analyze_einsum_string_bias_shape_with_ellipses_on_right(self):
        equation = "ab0,bc->ac0"
        bias_axes = "c"
        input_shape = (2, 3, 4)
        output_shape = (3, 5)

        expected_bias_shape = (5, 1)

        _, bias_shape, _ = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(
            bias_shape,
            expected_bias_shape,
            f"Expected {expected_bias_shape}, but got {bias_shape}",
        )

    def test_analyze_einsum_string_full_output_shape_with_ellipses_on_right(
        self,
    ):
        equation = "ab0,bc->ac0"
        bias_axes = "c"
        input_shape = (2, 3, 4)
        output_shape = (3, 5)

        expected_full_output_shape = (2, 3, 5)

        _, _, full_output_shape = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(
            full_output_shape,
            expected_full_output_shape,
            f"Expected {expected_full_output_shape}, but got "
            f"{full_output_shape}",
        )

    def test_analyze_einsum_string_invalid_equation(self):
        """Test the _analyze_einsum_string with an invalid equation"""
        equation = "invalid_equation"
        bias_axes = "c"
        input_shape = (2, 3)
        output_shape = (4, 5)

        with self.assertRaisesRegex(ValueError, "Invalid einsum equation"):
            einsum_dense._analyze_einsum_string(
                equation, bias_axes, input_shape, output_shape
            )

    def test_analyze_einsum_string_basic(self):
        """Test the _analyze_einsum_string with a basic equation."""
        equation = "ab,bc->ac"
        bias_axes = "c"
        input_shape = (2, 3)
        output_shape = 4
        expected_kernel_shape = (3, 4)
        expected_bias_shape = (4,)
        expected_full_output_shape = (2, 4)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )
        self.assertEqual(tuple(kernel_shape), expected_kernel_shape)
        self.assertEqual(tuple(bias_shape), expected_bias_shape)
        self.assertEqual(tuple(full_output_shape), expected_full_output_shape)

    def test_analyze_einsum_string_ellipses_left(self):
        """Test the _analyze_einsum_string with ellipses on the left."""
        equation = "...ab,bc->...ac"
        bias_axes = "c"
        input_shape = (10, 10, 3)
        output_shape = 4
        expected_kernel_shape = (3, 4)
        expected_bias_shape = (4,)
        expected_full_output_shape = (10, 10, 4)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )
        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_ellipses_right_multiple_dims(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "ab0,bc->ac01"
        bias_axes = "c"
        input_shape = (2, 3, 4, 5)
        output_shape = (4, 6)
        expected_kernel_shape = (3, 4)
        expected_bias_shape = (4, 6)
        expected_full_output_shape = (2, 4, 4, 6)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )
        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_no_bias(self):
        """Test the _analyze_einsum_string with no bias."""
        equation = "ab,bc->ac"
        bias_axes = None
        input_shape = (2, 3)
        output_shape = 4
        expected_kernel_shape = (3, 4)
        expected_bias_shape = None
        expected_full_output_shape = (2, 4)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )
        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_simple_dense(self):
        """Test the _analyze_einsum_string with a simple dense layer."""
        equation = "ab,bc->ac"
        bias_axes = "c"
        input_shape = (5, 6)
        output_shape = (7,)

        expected_kernel_shape = (6, 7)
        expected_bias_shape = (7,)
        expected_full_output_shape = (5, 7)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_ellipses_on_left(self):
        """Test the _analyze_einsum_string with ellipses on the left."""
        equation = "...ab,bc->...ac"
        bias_axes = "c"
        input_shape = (8, 9, 6)
        output_shape = (9, 7)

        expected_kernel_shape = (6, 7)
        expected_bias_shape = (7, 1)
        expected_full_output_shape = (8, 9, 7)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_complex_ellipses_on_right(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "ab0,bc->ac0"
        bias_axes = "c"
        input_shape = (3, 5, 7)
        output_shape = (5, 8)

        expected_kernel_shape = (5, 8)
        expected_bias_shape = (8, 1)
        expected_full_output_shape = (3, 5, 8)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_complex_no_ellipses(self):
        """Test the _analyze_einsum_string with no ellipses."""
        equation = "abc,bd->acd"
        bias_axes = "d"
        input_shape = (4, 5, 6)
        output_shape = (4, 7)

        expected_kernel_shape = (5, 7)
        expected_bias_shape = (7,)
        expected_full_output_shape = (4, 4, 7)
        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_4d_ellipses_on_right(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "abc0,bd->adc0"
        bias_axes = "d"
        input_shape = (3, 5, 4, 6)
        output_shape = (5, 7, 8)

        expected_kernel_shape = (4, 8)
        expected_bias_shape = (8, 1, 1)
        expected_full_output_shape = (3, 5, 7, 8)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_large_output_ellipses_on_right(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "ab0,bc->ac0"
        bias_axes = "c"
        input_shape = (5, 7, 6)
        output_shape = (7, 10)

        expected_kernel_shape = (7, 10)
        expected_bias_shape = (10, 1)
        expected_full_output_shape = (5, 7, 10)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_complex_input_ellipses_on_right(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "ab0,bc->ac0"
        bias_axes = "c"
        input_shape = (9, 7, 11)
        output_shape = (9, 13)

        expected_kernel_shape = (7, 13)
        expected_bias_shape = (13, 1)
        expected_full_output_shape = (9, 9, 13)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_single_output_dim_ellipses_on_right(
        self,
    ):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "a0,b->a0"
        bias_axes = "a"
        input_shape = (6, 7)
        output_shape = (8,)

        expected_kernel_shape = (7, 8)
        expected_bias_shape = (8,)
        expected_full_output_shape = (6, 8)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_ellipses_on_right_simple_case(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "a0,b->a0"
        bias_axes = None
        input_shape = (3, 5)
        output_shape = 3
        expected_kernel_shape = (5, 3)
        expected_bias_shape = None
        expected_full_output_shape = (3, 5)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_ellipses_on_right_multiple_dims(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "ab0,cd->ab0"
        bias_axes = None
        input_shape = (4, 5, 6)
        output_shape = (4, 5)
        expected_kernel_shape = (6, 4, 5)
        expected_bias_shape = None
        expected_full_output_shape = (4, 5, 6)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_ellipses_on_right_diff_dims(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "ab0,cd->ad0"
        bias_axes = None
        input_shape = (3, 4, 5)
        output_shape = (3, 6)
        expected_kernel_shape = (4, 5, 6)
        expected_bias_shape = None
        expected_full_output_shape = (3, 4, 6)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_ellipses_on_right_single_dim(self):
        """Test the _analyze_einsum_string with ellipses on the right."""
        equation = "a0,b->a0"
        bias_axes = None
        input_shape = (4, 5)
        output_shape = 4
        expected_kernel_shape = (5, 4)
        expected_bias_shape = None
        expected_full_output_shape = (4, 5)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_einsum_string_invalid_format(self):
        """Test _analyze_einsum_string with an invalid equation format"""
        equation = "ab,bc,ac->ac"  # Invalid equation format
        bias_axes = None
        input_shape = (2, 3)
        output_shape = 4

        with self.assertRaisesRegex(
            ValueError,
            "Invalid einsum equation.*Equations must be in the form",
        ):
            einsum_dense._analyze_einsum_string(
                equation, bias_axes, input_shape, output_shape
            )

    def test_analyze_einsum_string_with_split_string(
        self,
    ):
        """Test when equation is split into components."""
        # Equation format that leads to split_string being non-empty
        equation = "ab,bc->ac"
        bias_axes = "c"
        input_shape = (2, 3)
        output_shape = 4
        expected_kernel_shape = (3, 4)
        expected_bias_shape = (4,)
        expected_full_output_shape = (2, 4)

        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_einsum_string(
            equation, bias_axes, input_shape, output_shape
        )

        self.assertEqual(kernel_shape, expected_kernel_shape)
        self.assertEqual(bias_shape, expected_bias_shape)
        self.assertEqual(full_output_shape, expected_full_output_shape)

    def test_analyze_split_string_with_integer_output_shape(self):
        """Test _analyze_split_string when output_shape is an integer"""
        split_string = ["ab", "bc", "ac"]
        bias_axes = "c"
        input_shape = (2, 3)
        output_shape = 4
        (
            kernel_shape,
            bias_shape,
            full_output_shape,
        ) = einsum_dense._analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

        self.assertEqual(full_output_shape, [output_shape])
