# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import torch
from torch._decomp.decompositions import aten, pw_cast_for_opmath
from torch._decomp import register_decomposition, get_decompositions


@register_decomposition(aten.convolution_backward)
@pw_cast_for_opmath
def convolution_backward(
    grad_output,
    inp,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    if stride == [2, 2]:
        output_padding = [1, 1]

    # Compute the gradient of the input tensor
    grad_input = torch.nn.functional.conv_transpose2d(
        grad_output, weight, stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
    )

    # Compute the gradient of the weight tensor
    grad_weight = torch.nn.functional.conv_transpose2d(
        inp, weight.transpose(0, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
    )

    # Compute the gradient of the bias tensor
    if bias is not None:
        grad_bias = grad_output.sum([0, 2, 3], keepdim=True)
    else:
        grad_bias = None

    return grad_input, grad_weight, grad_bias


if len(get_decompositions([aten._scaled_dot_product_flash_attention.default])) == 0:

    @register_decomposition(aten._scaled_dot_product_flash_attention.default)
    def scaled_dot_product_flash_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
        *,
        return_debug_mask=False,
        scale=None,
    ):
        batch_size, num_head, q_size, head_size = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        )

        logsumexp = torch.empty([batch_size, q_size, num_head, head_size], dtype=torch.float)
        cum_seq_q, cum_seq_k = torch.empty([], dtype=torch.long), torch.empty(
            [], dtype=torch.long
        )
        max_q, max_k = 0, 0
        philox_seed, philox_offset = torch.empty([], dtype=torch.long), torch.empty(
            [], dtype=torch.long
        )
        debug_attn_mask = torch.empty(
            [],
            dtype=query.dtype,
            device=query.device,
            requires_grad=query.requires_grad,
        )
        output, _ = aten._scaled_dot_product_attention_math.default(
            query, key, value, None, dropout_p, is_causal, None, scale=scale
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        logsumexp = torch.logsumexp(scores, dim=-1)

        output = output.transpose(1, 2).contiguous(memory_format=torch.contiguous_format)
        return (
            output.transpose(1, 2),
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        )


def get_aot_decomposition_list():
    return [
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._softmax.default,
        torch.ops.aten._softmax_backward_data.default,
        torch.ops.aten.convolution_backward.default,
        torch.ops.aten.gelu_backward.default,
        torch.ops.aten.native_group_norm.default,
        torch.ops.aten.native_group_norm_backward.default,
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.native_layer_norm_backward.default,
        torch.ops.aten.slice_backward.default,
    ]


def get_inf_decomposition_list():
    return [torch.ops.aten.nll_loss_forward.default]


def get_export_decomposition_list():
    # List of decompositions from torch._decomp.core_aten_decompositions
    # removed _backward ops and ops supported without decomposition
    decomp = [
        torch.ops.aten.addcdiv,
        torch.ops.aten.addcdiv_,
        torch.ops.aten.addcmul,
        torch.ops.aten.addcmul_,
        torch.ops.aten.addr,
        torch.ops.aten.affine_grid_generator,
        torch.ops.aten.all,
        torch.ops.aten.aminmax,
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.baddbmm,
        torch.ops.aten.binary_cross_entropy,
        torch.ops.aten.binary_cross_entropy_with_logits,
        torch.ops.aten.block_diag,
        torch.ops.aten.celu,
        torch.ops.aten.celu_,
        torch.ops.aten.clamp_max,
        torch.ops.aten.clamp_min,
        torch.ops.aten.count_nonzero,
        torch.ops.aten.linalg_cross,
        torch.ops.aten.cudnn_batch_norm,
        torch.ops.aten.deg2rad,
        torch.ops.aten.deg2rad_,
        torch.ops.aten.detach,
        torch.ops.aten.diag_embed,
        torch.ops.aten.dot,
        torch.ops.aten.vdot,
        torch.ops.aten.elu,
        torch.ops.aten.elu_,
        torch.ops.aten._embedding_bag,
        torch.ops.aten.empty_like,
        torch.ops.aten._euclidean_dist.default,
        torch.ops.aten.expand_as,
        torch.ops.aten.eye,
        torch.ops.aten.fill,
        torch.ops.aten.fill_,
        torch.ops.aten.floor_divide,
        torch.ops.aten.frac,
        torch.ops.aten.frac_,
        torch.ops.aten._fused_moving_avg_obs_fq_helper,
        torch.ops.aten.gelu_,
        torch.ops.aten.glu,
        torch.ops.aten.hardshrink,
        torch.ops.aten.hardsigmoid,
        torch.ops.aten.hardsigmoid_,
        torch.ops.aten.hardswish,
        torch.ops.aten.hardswish_,
        torch.ops.aten.hardtanh_,
        torch.ops.aten.heaviside,
        torch.ops.aten.heaviside_,
        torch.ops.aten.huber_loss,
        torch.ops.aten.im2col,
        torch.ops.aten.index_add,
        torch.ops.aten.index_add_,
        torch.ops.aten.index_copy,
        torch.ops.aten.index_copy_,
        torch.ops.aten.index_fill,
        torch.ops.aten.index_fill_,
        torch.ops.aten.isin,
        torch.ops.aten.isneginf,
        torch.ops.aten.isposinf,
        torch.ops.aten.l1_loss,
        torch.ops.aten.leaky_relu_,
        torch.ops.aten.lerp,
        torch.ops.aten.lerp_,
        torch.ops.aten.linspace,
        torch.ops.aten.logaddexp,
        torch.ops.aten.logaddexp2,
        torch.ops.aten.logit,
        torch.ops.aten.logit_,
        torch.ops.aten.log_sigmoid_forward,
        torch.ops.aten.logspace,
        torch.ops.aten.logsumexp.default,
        torch.ops.aten.masked_fill,
        torch.ops.aten.masked_fill_,
        torch.ops.aten.mish,
        torch.ops.aten.mish_,
        torch.ops.aten.mse_loss,
        torch.ops.aten.multi_margin_loss,
        torch.ops.aten.multilabel_margin_loss_forward,
        torch.ops.aten.mv,
        torch.ops.aten.mvlgamma,
        torch.ops.aten.mvlgamma_,
        torch.ops.aten.nansum,
        torch.ops.aten.nan_to_num,
        torch.ops.aten.nan_to_num_,
        torch.ops.aten.narrow,
        torch.ops.aten.new_empty,
        torch.ops.aten.new_full,
        torch.ops.aten.new_ones,
        torch.ops.aten.new_zeros,
        torch.ops.aten.nll_loss_forward,
        torch.ops.aten.norm,
        torch.ops.aten.ones,
        torch.ops.aten.ones_like,
        torch.ops.aten._prelu_kernel,
        torch.ops.aten._reshape_alias,
        torch.ops.aten.rad2deg,
        torch.ops.aten.rad2deg_,
        torch.ops.aten.reflection_pad1d,
        torch.ops.aten.reflection_pad2d,
        torch.ops.aten.reflection_pad3d,
        torch.ops.aten.replication_pad1d,
        torch.ops.aten.replication_pad2d,
        torch.ops.aten.replication_pad3d,
        torch.ops.aten.renorm,
        torch.ops.aten.renorm_,
        torch.ops.aten.resize_as,
        torch.ops.aten.roll,
        torch.ops.aten.rot90,
        torch.ops.aten.rrelu_with_noise,
        torch.ops.aten.rrelu_with_noise_,
        torch.ops.aten.rsub,
        torch.ops.aten.select_scatter,
        torch.ops.aten.sgn,
        torch.ops.aten.sgn_,
        torch.ops.aten.silu,
        torch.ops.aten.silu_,
        torch.ops.aten.sinc,
        torch.ops.aten.sinc_,
        torch.ops.aten.smooth_l1_loss,
        torch.ops.aten.soft_margin_loss,
        torch.ops.aten.softplus,
        torch.ops.aten.softshrink,
        torch.ops.aten.special_entr,
        torch.ops.aten.special_log_ndtr,
        torch.ops.aten.special_xlog1py,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes_copy,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.std,
        torch.ops.aten.std_mean,
        torch.ops.aten.stack,
        torch.ops.aten.sum.default,
        torch.ops.aten.sum.out,
        torch.ops.aten.t,
        torch.ops.aten.take,
        torch.ops.aten.threshold,
        torch.ops.aten.threshold_,
        torch.ops.aten.trace,
        torch.ops.aten.transpose.int,
        torch.ops.aten.tril,
        torch.ops.aten.tril_,
        torch.ops.aten.triu,
        torch.ops.aten.triu_,
        torch.ops.aten.unbind,
        torch.ops.aten.unfold_copy,
        torch.ops.aten._unsafe_index,
        torch.ops.aten.unsafe_split.Tensor,
        torch.ops.aten.unsafe_split_with_sizes,
        torch.ops.aten._unsafe_view,
        torch.ops.aten.view_as_complex,
        torch.ops.aten.xlogy,
        torch.ops.aten.xlogy_,
        torch.ops.aten.zero,
        torch.ops.aten.zero_,
        torch.ops.aten.zeros,
        torch.ops.aten.zeros_like,
        torch.ops.aten._weight_norm_interface,
    ]
    try:
        from packaging import version
        if version.parse(torch.__version__) >= version.parse("2.3"):
            decomp += [
                torch.ops.aten._lazy_clone,
                torch.ops.aten._test_parallel_materialize,
                torch.ops.aten._chunk_cat,
            ]
    except ImportError:
        pass
    return decomp
