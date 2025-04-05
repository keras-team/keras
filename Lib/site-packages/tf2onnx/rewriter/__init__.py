# SPDX-License-Identifier: Apache-2.0

"""tf2onnx.rewriter module."""

from tf2onnx.rewriter.cond_rewriter import rewrite_cond
from tf2onnx.rewriter.conv2d_with_pad_rewriter import rewrite_conv2d_with_pad
from tf2onnx.rewriter.conv_dilations_rewriter import rewrite_conv_dilations
from tf2onnx.rewriter.dropout_rewriter import rewrite_dropout
from tf2onnx.rewriter.eye_rewriter import rewrite_eye
from tf2onnx.rewriter.flatten_rewriter import rewrite_flatten
from tf2onnx.rewriter.gemm_rewriter import rewrite_gemm
from tf2onnx.rewriter.leakyrelu_rewriter import rewrite_leakyrelu
from tf2onnx.rewriter.random_normal_rewriter import rewrite_random_normal
from tf2onnx.rewriter.random_uniform import rewrite_random_uniform, rewrite_random_uniform_fold_const
from tf2onnx.rewriter.rnn import rewrite_single_direction_lstm, rewrite_bi_direction_lstm, \
    rewrite_single_direction_gru, rewrite_bi_direction_gru, \
    rewrite_custom_rnn_cell, rewrite_generic_loop
from tf2onnx.rewriter.thresholded_relu_rewriter import rewrite_thresholded_relu
from tf2onnx.rewriter.transpose_rewriter import rewrite_transpose
from tf2onnx.rewriter.conv2d_with_add_rewriter import rewrite_biasadd_with_conv2d
from tf2onnx.rewriter.quantization_ops_rewriter import rewrite_quantize_and_dequantize
from tf2onnx.rewriter.layer_normalization_rewriter import rewrite_layer_normalization
from tf2onnx.rewriter.ragged_variant_shape_rewriter import rewrite_ragged_variant_shape
from tf2onnx.rewriter.lstm_tf2_rewriter import rewriter_lstm_tf2
from tf2onnx.rewriter.gru_tf2_rewriter import rewrite_gru_tf2
from tf2onnx.rewriter.fused_op_rewriter import rewrite_fused_ops


__all__ = [
    "rewrite_cond",
    "rewrite_dropout",
    "rewrite_eye",
    "rewrite_flatten",
    "rewrite_gemm",
    "rewrite_leakyrelu",
    "rewrite_random_normal",
    "rewrite_random_uniform",
    "rewrite_random_uniform_fold_const",
    "rewrite_thresholded_relu",
    "rewrite_transpose",
    "rewrite_single_direction_lstm",
    "rewrite_bi_direction_lstm",
    "rewrite_single_direction_gru",
    "rewrite_bi_direction_gru",
    "rewrite_custom_rnn_cell",
    "rewrite_generic_loop",
    "rewrite_biasadd_with_conv2d",
    "rewrite_quantize_and_dequantize",
    "rewrite_layer_normalization",
    "rewrite_conv_dilations",
    "rewrite_conv2d_with_pad",
    "rewrite_ragged_variant_shape",
    "rewriter_lstm_tf2",
    "rewrite_gru_tf2",
    "rewrite_fused_ops",
]
