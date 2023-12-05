import pytest
from absl.testing import parameterized

from keras import constraints
from keras import layers
from keras import testing


class EinsumDenseTest(testing.TestCase, parameterized.TestCase):
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
