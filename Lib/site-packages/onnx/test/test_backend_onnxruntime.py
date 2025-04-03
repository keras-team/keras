# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import unittest
from typing import Any

import numpy
from packaging.version import Version

import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx.backend.base import Device, DeviceType

try:
    import onnxruntime as ort

    ort_version = Version(ort.__version__)
except ImportError:
    # onnxruntime is not installed, all tests are skipped.
    ort: Any = None  # type: ignore[no-redef]
    ort_version: Any = None  # type: ignore[no-redef]


# The following just executes a backend based on InferenceSession through the backend test


class InferenceSessionBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        del kwargs  # Unused
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            input_names = [i.name for i in self._session.get_inputs()]
            input_shapes = [i.shape for i in self._session.get_inputs()]
            if len(inputs) == len(input_names):
                feeds = dict(zip(input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, shape in zip(input_names, input_shapes):
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)
        return outs


def _create_inference_session(model: onnx.ModelProto, device: str):
    if device == "CPU":
        providers = ("CPUExecutionProvider",)
    elif device == "CUDA":
        providers = ("CUDAExecutionProvider",)
    else:
        raise ValueError(f"Unexpected device {device!r}.")
    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    except Exception as e:
        raise RuntimeError(
            f"Unable to create inference session. Model is:\n\n{onnx.printer.to_text(model)}"
        ) from e
    return session


class InferenceSessionBackend(onnx.backend.base.Backend):
    @classmethod
    def supports_device(cls, device: str) -> bool:
        providers = set(ort.get_available_providers())
        d = Device(device)
        if d.type == DeviceType.CPU and "CPUExecutionProvider" in providers:
            return True
        if d.type == DeviceType.CUDA and "CUDAExecutionProvider" in providers:
            return True
        return False

    @classmethod
    def prepare(
        cls, model: onnx.ModelProto, device: str = "CPU", **kwargs: Any
    ) -> InferenceSessionBackendRep:
        del kwargs  # Unused
        if not isinstance(model, (str, bytes, onnx.ModelProto)):
            raise TypeError(f"Unexpected type {type(model)} for model.")

        session = _create_inference_session(model, device)
        return InferenceSessionBackendRep(session)

    @classmethod
    def run_model(cls, model: onnx.ModelProto, inputs, device=None, **kwargs):
        return super().run_model(model, inputs, device=device, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


if ort is not None:
    backend_test = onnx.backend.test.BackendTest(InferenceSessionBackend, __name__)

    if platform.architecture()[0] == "32bit":
        backend_test.exclude("(test_vgg19|test_zfnet|test_bvlc_alexnet)")
    if platform.system() == "Windows":
        backend_test.exclude("test_sequence_model")

    # The following tests cannot pass because they consists in generating random number.
    backend_test.exclude("(test_bernoulli)")

    # The following tests are not supported by onnxruntime.
    backend_test.exclude(
        "("
        "test_adagrad"
        "|test_adam"
        "|test_add_uint8"
        "|bitshift_left_uint16"
        "|bitshift_right_uint16"
        "|cast_BFLOAT16_to_FLOAT"
        "|cast_FLOAT_to_BFLOAT16"
        "|castlike_BFLOAT16_to_FLOAT"
        "|castlike_FLOAT_to_BFLOAT16"
        "|clip_default_int8_min_expanded"
        "|clip_default_int8_max_expanded"
        "|div_uint8"
        "|gru_batchwise"  # Batchwise recurrent operations (layout == 1) are not supported.
        "|loop16_seq_none"  # The graph is missing type information needed to construct the ORT tensor.
        "|lstm_batchwise"  # Batchwise recurrent operations (layout == 1) are not supported.
        "|m(in|ax)_u?int(16|8)"
        "|momentum"
        "|mul_uint8"
        "|pow_types_float32_uint32"
        "|pow_types_float32_uint64"
        "|simple_rnn_batchwise"  # Batchwise recurrent operations (layout == 1) are not supported.
        "|sub_uint8"
        "|gradient_of_add"
        "|test_batchnorm_epsilon_training_mode"  # Training mode does not support BN opset 14 (or higher) yet.
        "|test_batchnorm_example_training_mode"  # Training mode does not support BN opset 14 (or higher) yet.
        "|_to_FLOAT8E4M3FN"  # No corresponding Numpy type for Tensor Type.
        "|_to_FLOAT8E5M2"  # No corresponding Numpy type for Tensor Type.
        "|cast_FLOAT8E"  # No corresponding Numpy type for Tensor Type.
        "|castlike_FLOAT8E"  # No corresponding Numpy type for Tensor Type.
        "|test_dequantizelinear_axis"  # y_scale must be a scalar or 1D tensor of size 1.
        "|test_dequantizelinear"  # No corresponding Numpy type for Tensor Type.
        "|test_quantizelinear_axis"  # y_scale must be a scalar or 1D tensor of size 1.
        "|test_quantizelinear"  # No corresponding Numpy type for Tensor Type.
        "|test_affine_grid_"  # new IR version 9 and opset version 20 not supported yet.
        "|test_quantizelinear_uint4"  # No corresponding Numpy type for Tensor Type.
        "|test_quantizelinear_int4"  # No corresponding Numpy type for Tensor Type.
        "|test_dequantizelinear_uint4"  # No corresponding Numpy type for Tensor Type.
        "|test_dequantizelinear_int4"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_UINT4_to_FLOAT"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_INT4_to_FLOAT"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_UINT4_to_FLOAT16"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_INT4_to_FLOAT16"  # No corresponding Numpy type for Tensor Type.
        "|test_maxpool_2d_ceil_output_size_reduce_by_one"  # TODO: remove after https://github.com/microsoft/onnxruntime/pull/18377 in Ort release.
        ")"
    )

    # Exclude all tests that require IR10 until onnxruntime aligns
    # TODO: Unwaive tests once onnxruntime supports Opset21/IR10 https://github.com/onnx/onnx/issues/5840
    backend_test.exclude(
        "("
        "test_cast_"
        "|test_castlike_"
        "|test_constant"
        "|test_edge_pad_cpu"
        "|test_flatten_"
        "|test_identity"
        "|test_reflect_pad"
        "|test_reshape_"
        "|test_shape_"
        "|test_size_"
        "|test_squeeze_"
        "|test_transpose_"
        "|test_unsqueeze_"
        "|test_wrap_pad_"
        "|test_acos_cpu"
        "|test_acos_example_cpu"
        "|test_acosh_cpu"
        "|test_acosh_example_cpu"
        "|test_asin_cpu"
        "|test_asin_example_cpu"
        "|test_asinh_cpu"
        "|test_asinh_example_cpu"
        "|test_atan_cpu"
        "|test_atan_example_cpu"
        "|test_atanh_cpu"
        "|test_atanh_example_cpu"
        "|test_averagepool_1d_default_cpu"
        "|test_averagepool_2d_ceil_cpu"
        "|test_averagepool_2d_default_cpu"
        "|test_averagepool_2d_dilations_cpu"
        "|test_averagepool_2d_pads_count_include_pad_cpu"
        "|test_averagepool_2d_pads_cpu"
        "|test_averagepool_2d_precomputed_pads_count_include_pad_cpu"
        "|test_averagepool_2d_precomputed_pads_cpu"
        "|test_averagepool_2d_precomputed_same_upper_cpu"
        "|test_averagepool_2d_precomputed_strides_cpu"
        "|test_averagepool_2d_same_lower_cpu"
        "|test_averagepool_2d_same_upper_cpu"
        "|test_averagepool_2d_strides_cpu"
        "|test_averagepool_3d_default_cpu"
        "|test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False_cpu"
        "|test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True_cpu"
        "|test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False_cpu"
        "|test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True_cpu"
        "|test_averagepool_3d_dilations_small_cpu"
        "|test_basic_conv_with_padding_cpu"
        "|test_basic_conv_without_padding_cpu"
        "|test_conv_with_autopad_same_cpu"
        "|test_conv_with_strides_and_asymmetric_padding_cpu"
        "|test_conv_with_strides_no_padding_cpu"
        "|test_conv_with_strides_padding_cpu"
        "|test_convtranspose_1d_cpu"
        "|test_convtranspose_3d_cpu"
        "|test_convtranspose_autopad_same_cpu"
        "|test_convtranspose_cpu"
        "|test_convtranspose_dilations_cpu"
        "|test_convtranspose_kernel_shape_cpu"
        "|test_convtranspose_output_shape_cpu"
        "|test_convtranspose_pad_cpu"
        "|test_convtranspose_pads_cpu"
        "|test_cos_cpu"
        "|test_cos_example_cpu"
        "|test_cosh_cpu"
        "|test_cosh_example_cpu"
        "|test_det_2d_cpu"
        "|test_det_nd_cpu"
        "|test_dropout_default_cpu"
        "|test_dropout_default_mask_cpu"
        "|test_dropout_default_mask_ratio_cpu"
        "|test_dropout_default_ratio_cpu"
        "|test_elu_cpu"
        "|test_elu_default_cpu"
        "|test_elu_example_cpu"
        "|test_eyelike_populate_off_main_diagonal_cpu"
        "|test_eyelike_with_dtype_cpu"
        "|test_eyelike_without_dtype_cpu"
        "|test_globalaveragepool_cpu"
        "|test_globalaveragepool_precomputed_cpu"
        "|test_gridsample_aligncorners_true_cpu"
        "|test_gridsample_bicubic_align_corners_0_additional_1_cpu"
        "|test_gridsample_bicubic_align_corners_1_additional_1_cpu"
        "|test_gridsample_bicubic_cpu"
        "|test_gridsample_bilinear_align_corners_0_additional_1_cpu"
        "|test_gridsample_bilinear_align_corners_1_additional_1_cpu"
        "|test_gridsample_bilinear_cpu"
        "|test_gridsample_border_padding_cpu"
        "|test_gridsample_cpu"
        "|test_gridsample_nearest_align_corners_0_additional_1_cpu"
        "|test_gridsample_nearest_align_corners_1_additional_1_cpu"
        "|test_gridsample_nearest_cpu"
        "|test_gridsample_reflection_padding_cpu"
        "|test_gridsample_volumetric_bilinear_align_corners_0_cpu"
        "|test_gridsample_volumetric_bilinear_align_corners_1_cpu"
        "|test_gridsample_volumetric_nearest_align_corners_0_cpu"
        "|test_gridsample_volumetric_nearest_align_corners_1_cpu"
        "|test_gridsample_zeros_padding_cpu"
        "|test_gru_defaults_cpu"
        "|test_gru_seq_length_cpu"
        "|test_gru_with_initial_bias_cpu"
        "|test_hardsigmoid_cpu"
        "|test_hardsigmoid_default_cpu"
        "|test_hardsigmoid_example_cpu"
        "|test_hardswish_cpu"
        "|test_hardswish_expanded_cpu"
        "|test_lppool_1d_default_cpu"
        "|test_lppool_2d_default_cpu"
        "|test_lppool_2d_dilations_cpu"
        "|test_lppool_2d_pads_cpu"
        "|test_lppool_2d_same_lower_cpu"
        "|test_lppool_2d_same_upper_cpu"
        "|test_lppool_2d_strides_cpu"
        "|test_lppool_3d_default_cpu"
        "|test_lstm_defaults_cpu"
        "|test_lstm_with_initial_bias_cpu"
        "|test_lstm_with_peepholes_cpu"
        "|test_maxpool_1d_default_cpu"
        "|test_maxpool_2d_ceil_cpu"
        "|test_maxpool_2d_default_cpu"
        "|test_maxpool_2d_dilations_cpu"
        "|test_maxpool_2d_pads_cpu"
        "|test_maxpool_2d_precomputed_pads_cpu"
        "|test_maxpool_2d_precomputed_same_upper_cpu"
        "|test_maxpool_2d_precomputed_strides_cpu"
        "|test_maxpool_2d_same_lower_cpu"
        "|test_maxpool_2d_same_upper_cpu"
        "|test_maxpool_2d_strides_cpu"
        "|test_maxpool_2d_uint8_cpu"
        "|test_maxpool_3d_default_cpu"
        "|test_maxpool_3d_dilations_cpu"
        "|test_maxpool_3d_dilations_use_ref_impl_cpu"
        "|test_maxpool_3d_dilations_use_ref_impl_large_cpu"
        "|test_maxpool_with_argmax_2d_precomputed_pads_cpu"
        "|test_maxpool_with_argmax_2d_precomputed_strides_cpu"
        "|test_maxunpool_export_without_output_shape_cpu"
        "|test_mish_cpu"
        "|test_mish_expanded_cpu"
        "|test_nllloss_NC_cpu"
        "|test_nllloss_NC_expanded_cpu"
        "|test_nllloss_NCd1_cpu"
        "|test_nllloss_NCd1_expanded_cpu"
        "|test_nllloss_NCd1_ii_cpu"
        "|test_nllloss_NCd1_ii_expanded_cpu"
        "|test_nllloss_NCd1_mean_weight_negative_ii_cpu"
        "|test_nllloss_NCd1_mean_weight_negative_ii_expanded_cpu"
        "|test_nllloss_NCd1_weight_cpu"
        "|test_nllloss_NCd1_weight_expanded_cpu"
        "|test_nllloss_NCd1_weight_ii_cpu"
        "|test_nllloss_NCd1_weight_ii_expanded_cpu"
        "|test_nllloss_NCd1d2_cpu"
        "|test_nllloss_NCd1d2_expanded_cpu"
        "|test_nllloss_NCd1d2_no_weight_reduction_mean_ii_cpu"
        "|test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded_cpu"
        "|test_nllloss_NCd1d2_reduction_mean_cpu"
        "|test_nllloss_NCd1d2_reduction_mean_expanded_cpu"
        "|test_nllloss_NCd1d2_reduction_sum_cpu"
        "|test_nllloss_NCd1d2_reduction_sum_expanded_cpu"
        "|test_nllloss_NCd1d2_with_weight_cpu"
        "|test_nllloss_NCd1d2_with_weight_expanded_cpu"
        "|test_nllloss_NCd1d2_with_weight_reduction_mean_cpu"
        "|test_nllloss_NCd1d2_with_weight_reduction_mean_expanded_cpu"
        "|test_nllloss_NCd1d2_with_weight_reduction_sum_cpu"
        "|test_nllloss_NCd1d2_with_weight_reduction_sum_expanded_cpu"
        "|test_nllloss_NCd1d2_with_weight_reduction_sum_ii_cpu"
        "|test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded_cpu"
        "|test_nllloss_NCd1d2d3_none_no_weight_negative_ii_cpu"
        "|test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded_cpu"
        "|test_nllloss_NCd1d2d3_sum_weight_high_ii_cpu"
        "|test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded_cpu"
        "|test_nllloss_NCd1d2d3d4d5_mean_weight_cpu"
        "|test_nllloss_NCd1d2d3d4d5_mean_weight_expanded_cpu"
        "|test_nllloss_NCd1d2d3d4d5_none_no_weight_cpu"
        "|test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded_cpu"
        "|test_rnn_seq_length_cpu"
        "|test_roialign_aligned_false_cpu"
        "|test_roialign_aligned_true_cpu"
        "|test_roialign_mode_max_cpu"
        "|test_round_cpu"
        "|test_selu_cpu"
        "|test_selu_default_cpu"
        "|test_selu_example_cpu"
        "|test_simple_rnn_defaults_cpu"
        "|test_simple_rnn_with_initial_bias_cpu"
        "|test_sin_cpu"
        "|test_sin_example_cpu"
        "|test_sinh_cpu"
        "|test_sinh_example_cpu"
        "|test_softplus_cpu"
        "|test_softplus_example_cpu"
        "|test_softsign_cpu"
        "|test_softsign_example_cpu"
        "|test_tan_cpu"
        "|test_tan_example_cpu"
        "|test_thresholdedrelu_cpu"
        "|test_thresholdedrelu_default_cpu"
        "|test_thresholdedrelu_example_cpu"
        "|test_resize_downsample_scales_cubic_A_n0p5_exclude_outside_cpu"
        "|test_resize_downsample_scales_cubic_antialias_cpu"
        "|test_resize_downsample_scales_cubic_cpu"
        "|test_resize_downsample_scales_linear_antialias_cpu"
        "|test_resize_downsample_scales_linear_cpu"
        "|test_resize_downsample_scales_linear_half_pixel_symmetric_cpu"
        "|test_resize_downsample_scales_nearest_cpu"
        "|test_resize_downsample_sizes_cubic_antialias_cpu"
        "|test_resize_downsample_sizes_cubic_cpu"
        "|test_resize_downsample_sizes_linear_antialias_cpu"
        "|test_resize_downsample_sizes_linear_pytorch_half_pixel_cpu"
        "|test_resize_downsample_sizes_nearest_cpu"
        "|test_resize_downsample_sizes_nearest_not_larger_cpu"
        "|test_resize_downsample_sizes_nearest_not_smaller_cpu"
        "|test_resize_tf_crop_and_resize_axes_2_3_cpu"
        "|test_resize_tf_crop_and_resize_axes_3_2_cpu"
        "|test_resize_tf_crop_and_resize_cpu"
        "|test_resize_upsample_scales_cubic_A_n0p5_exclude_outside_cpu"
        "|test_resize_upsample_scales_cubic_align_corners_cpu"
        "|test_resize_upsample_scales_cubic_asymmetric_cpu"
        "|test_resize_upsample_scales_cubic_cpu"
        "|test_resize_upsample_scales_linear_align_corners_cpu"
        "|test_resize_upsample_scales_linear_cpu"
        "|test_resize_upsample_scales_linear_half_pixel_symmetric_cpu"
        "|test_resize_upsample_scales_nearest_axes_2_3_cpu"
        "|test_resize_upsample_scales_nearest_axes_3_2_cpu"
        "|test_resize_upsample_scales_nearest_cpu"
        "|test_resize_upsample_sizes_cubic_cpu"
        "|test_resize_upsample_sizes_nearest_axes_2_3_cpu"
        "|test_resize_upsample_sizes_nearest_axes_3_2_cpu"
        "|test_resize_upsample_sizes_nearest_ceil_half_pixel_cpu"
        "|test_resize_upsample_sizes_nearest_cpu"
        "|test_resize_upsample_sizes_nearest_floor_align_corners_cpu"
        "|test_resize_upsample_sizes_nearest_not_larger_cpu"
        "|test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric_cpu"
        "|test_qlinearmatmul_2D_uint8_float32_cuda"
        "|test_qlinearmatmul_2D_int8_float32_cpu"
        "|test_image_decoder_decode_jpeg_rgb_cpu"
        "|test_basic_deform_conv_without_padding_cuda"
        "|test_qlinearmatmul_3D_int8_float16_cuda"
        "|test_image_decoder_decode_bmp_rgb_cuda"
        "|test_qlinearmatmul_2D_uint8_float16_cpu"
        "|test_image_decoder_decode_jpeg2k_rgb_cuda"
        "|test_image_decoder_decode_jpeg_bgr_cuda"
        "|test_qlinearmatmul_3D_uint8_float32_cpu"
        "|test_qlinearmatmul_3D_uint8_float16_cuda"
        "|test_deform_conv_with_mask_bias_cpu"
        "|test_qlinearmatmul_2D_int8_float16_cuda"
        "|test_image_decoder_decode_jpeg_grayscale_cpu"
        "|test_basic_deform_conv_without_padding_cpu"
        "|test_qlinearmatmul_3D_int8_float32_cuda"
        "|test_qlinearmatmul_3D_int8_float16_cpu"
        "|test_qlinearmatmul_2D_int8_float32_cuda"
        "|test_deform_conv_with_mask_bias_cuda"
        "|test_image_decoder_decode_tiff_rgb_cuda"
        "|test_image_decoder_decode_jpeg2k_rgb_cpu"
        "|test_image_decoder_decode_jpeg_rgb_cuda"
        "|test_image_decoder_decode_jpeg_grayscale_cuda"
        "|test_qlinearmatmul_3D_uint8_float32_cuda"
        "|test_image_decoder_decode_png_rgb_cpu"
        "|test_image_decoder_decode_png_rgb_cuda"
        "|test_image_decoder_decode_bmp_rgb_cpu"
        "|test_qlinearmatmul_3D_uint8_float16_cpu"
        "|test_deform_conv_with_multiple_offset_groups_cuda"
        "|test_image_decoder_decode_webp_rgb_cpu"
        "|test_basic_deform_conv_with_padding_cpu"
        "|test_qlinearmatmul_2D_uint8_float16_cuda"
        "|test_image_decoder_decode_webp_rgb_cuda"
        "|test_basic_deform_conv_with_padding_cuda"
        "|test_image_decoder_decode_pnm_rgb_cpu"
        "|test_qlinearmatmul_3D_int8_float32_cpu"
        "|test_image_decoder_decode_jpeg_bgr_cpu"
        "|test_qlinearmatmul_2D_int8_float16_cpu"
        "|test_image_decoder_decode_pnm_rgb_cuda"
        "|test_deform_conv_with_multiple_offset_groups_cpu"
        "|test_qlinearmatmul_2D_uint8_float32_cpu"
        "|test_image_decoder_decode_tiff_rgb_cpu"
        "|test_globalmaxpool_cpu"
        "|test_globalmaxpool_precomputed_cpu"
        "|test_instancenorm_example_cpu"
        "|test_instancenorm_epsilon_cpu"
        ")"
    )

    # The following tests fail due to small discrepancies.
    backend_test.exclude("(cast_FLOAT_to_STRING|castlike_FLOAT_to_STRING|stft)")

    # The following tests fail due to huge discrepancies.
    backend_test.exclude(
        "("
        "resize_downsample_scales_cubic_align_corners"
        "|resize_downsample_scales_linear_align_corners"
        "|training_dropout"
        ")"
    )

    # The followiing tests fail due to a bug in onnxruntime in handling reduction
    # ops that perform reduction over an empty set of values.
    backend_test.exclude(
        "("
        "test_reduce_sum_empty_set"
        "|test_reduce_prod_empty_set"
        "|test_reduce_min_empty_set"
        "|test_reduce_max_empty_set"
        "|test_reduce_sum_square_empty_set"
        "|test_reduce_log_sum_empty_set"
        "|test_reduce_log_sum_exp_empty_set"
        "|test_reduce_l1_empty_set"
        "|test_reduce_l2_empty_set"
        ")"
    )

    # The following tests fail for no obvious reason.
    backend_test.exclude(
        "("
        "maxunpool_export_with_output_shape"  # not the same expected output
        "|softplus_example_expanded"  # Could not find an implementation for Exp(1) node with name ''
        "|softplus_expanded"  # Could not find an implementation for Exp(1) node with name ''
        "|AvgPool[1-3]d"  # Could not find an implementation for AveragePool(1) node with name ''
        "|BatchNorm1d_3d_input_eval"  # Could not find an implementation for BatchNormalization(6) node with name ''
        "|BatchNorm[2-3]d_eval"  # Could not find an implementation for BatchNormalization(6) node with name ''
        "|GLU"  # Could not find an implementation for Mul(6) node with name ''
        "|Linear"  # Could not find an implementation for Gemm(6) node with name ''
        "|PReLU"  # Could not find an implementation for PRelu(6) node with name ''
        "|PoissonNLL"  # Could not find an implementation for Mul(6) node with name ''
        "|Softsign"  # Could not find an implementation for Gemm(6) node with name ''
        "|operator_add_broadcast"  # Could not find an implementation for Gemm(6) node with name ''
        "|operator_add_size1"  # Could not find an implementation for Gemm(6) node with name ''
        "|operator_addconstant"  # Could not find an implementation for Gemm(6) node with name ''
        "|operator_addmm"  # Could not find an implementation for Gemm(6) node with name ''
        "|operator_basic"  # Could not find an implementation for Add(6) node with name ''
        "|operator_mm"  # Could not find an implementation for Gemm(6) node with name ''
        "|operator_non_float_params"  # Could not find an implementation for Add(6) node with name ''
        "|operator_params"  # Could not find an implementation for Add(6) node with name ''
        "|operator_pow"  # Could not find an implementation for Pow(1) node with name ''
        ")"
    )

    # The following tests are new with opset 19 and 20, or ai.onnx.ml 4
    if ort_version is not None and ort_version < Version("1.16"):
        backend_test.exclude(
            "("
            "averagepool"
            "|_pad_"
            "|_resize_"
            "|_size_"
            "|cast"
            "|castlike"
            "|equal_string_broadcast"
            "|equal_string"
            "|equal"
            "|half_pixel_symmetric"
            "|identity"
            "|reshape"
            ")"
        )
    if ort_version is not None and ort_version < Version("1.17"):
        backend_test.exclude(
            "("
            "deform_conv"
            "|dequantizelinear_uint16"
            "|dequantizelinear_int16"
            "|quantizelinear_uint16"
            "|quantizelinear_int16"
            "|dft"
            "|gelu"
            "|gridsample"
            "|group_normalization"
            "|identity_opt"
            "|image_decoder"
            "|isinf_float16"
            "|label_encoder"
            "|optional_get_element_optional_sequence"
            "|qlinearmatmul_2D_int8"
            "|qlinearmatmul_2D_uint8_float16"
            "|qlinearmatmul_3D_int8"
            "|qlinearmatmul_3D_uint8_float16"
            "|reduce_max_bool_inputs"
            "|reduce_min_bool_inputs"
            "|regex_full_match"
            "|string_concat"
            "|string_split"
            "|constantofshape_float_ones"
            "|constantofshape_int_shape_zero"
            "|constantofshape_int_zeros"
            "|isinf"
            "|isinf_negative"
            "|isinf_positive"
            "|isnan"
            "|isnan_float16"
            "|qlinearmatmul_2D_uint8_float32"
            "|qlinearmatmul_3D_uint8_float32"
            ")"
        )
    if ort_version is not None and ort_version < Version("1.18"):
        # when adding new tests to the list, please add a comment with the reason for exclusion
        # for tests that "not supported by onnxruntime 1.17", it will be solved in the next
        # onnxruntime release with ONNX 1.16.0 integrated. The work is covered in ONNX integration procedure.
        backend_test.exclude(
            "("
            "deform_conv"  # deform_conv is not supported in onnxruntime
            "|group_normalization"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|identity_opt"  # fixed in ort 1.18 (https://github.com/microsoft/onnxruntime/pull/19273)
            "|image_decoder"  # image_decoder is not supported in onnxruntime
            "|optional_get_element_optional_sequence"  # fixed in ort 1.18 (https://github.com/microsoft/onnxruntime/pull/19273)
            "|qlinearmatmul_2D_int8"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_2D_uint8_float16"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_int8"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_uint8_float16"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_2D_uint8_float32"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_uint8_float32"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|tree_ensemble"  # tree_ensemble not yet implemented in ort
            ")"
        )

    if ort_version is not None and ort_version < Version("1.20"):
        backend_test.exclude(
            "("
            "tree_ensemble_set_membership"
            "|tree_ensemble_single_tree"
            "|convtranspose_group_2"
            "|dft"
            ")"
        )

    # Import all test cases at global scope to make them visible to python.unittest
    globals().update(backend_test.test_cases)


if __name__ == "__main__":
    unittest.main()
