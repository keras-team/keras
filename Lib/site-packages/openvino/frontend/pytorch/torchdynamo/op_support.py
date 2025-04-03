# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from typing import Dict

import torch
from torch.nn import Module
from torch._ops import OpOverload

from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_disabled_ops

import typing as t
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class OperatorSupport(OperatorSupport):
    """
    Operator support for OpenVINO backend.
    """

    def __init__(self, options):
        support_dict = {
            "_operator.add": None,
            "_operator.floordiv": None,
            "_operator.getitem": None,
            "_operator.mul": None,
            "_operator.sub": None,
            "torch.ops.aten.sym_size.int": None,
            "torch.ops.aten._adaptive_avg_pool1d.default": None,
            "torch.ops.aten._adaptive_avg_pool2d.default": None,
            "torch.ops.aten._adaptive_avg_pool3d.default": None,
            "torch.ops.aten._convolution.default": None,
            "torch.ops.aten._embedding_bag.default": None,
            "torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default": None,
            "torch.ops.aten._local_scalar_dense.default": None,
            "torch.ops.aten._log_softmax.default": None,
            "torch.ops.aten._native_batch_norm_legit.default": None,
            "torch.ops.aten._native_batch_norm_legit.no_stats": None,
            "torch.ops.aten._native_batch_norm_legit_functional.default": None,
            "torch.ops.aten._native_batch_norm_legit_no_training.default": None,
            "torch.ops.aten._scaled_dot_product_flash_attention.default": None,
            "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default": None,
            "torch.ops.aten._softmax.default": None,
            "torch.ops.aten._to_copy.default": None,
            "torch.ops.aten._unsafe_view.default": None,
            "torch.ops.aten.abs.default": None,
            "torch.ops.aten.acos.default": None,
            "torch.ops.aten.acosh.default": None,
            "torch.ops.aten.adaptive_max_pool1d.default": None,
            "torch.ops.aten.adaptive_max_pool2d.default": None,
            "torch.ops.aten.adaptive_max_pool3d.default": None,
            "torch.ops.aten.add.Scalar": None,
            "torch.ops.aten.add.Tensor": None,
            "torch.ops.aten.add_.Tensor": None,
            "torch.ops.aten.addcmul.default": None,
            "torch.ops.aten.addmm.default": None,
            "torch.ops.aten.alias.default": None,
            "torch.ops.aten.all.default": None,
            "torch.ops.aten.amax.default": None,
            "torch.ops.aten.amin.default": None,
            "torch.ops.aten.any.default": None,
            "torch.ops.aten.any.dim": None,
            "torch.ops.aten.arange.default": None,
            "torch.ops.aten.arange.start": None,
            "torch.ops.aten.arange.start_step": None,
            "torch.ops.aten.argmax.default": None,
            "torch.ops.aten.argmin.default": None,
            "torch.ops.aten.as_strided.default": None,
            "torch.ops.aten.as_strided_.default": None,
            "torch.ops.aten.asin.default": None,
            "torch.ops.aten.asinh.default": None,
            "torch.ops.aten.asinh.default": None,
            "torch.ops.aten.atanh.default": None,
            "torch.ops.aten.avg_pool2d.default": None,
            "torch.ops.aten.avg_pool3d.default": None,
            "torch.ops.aten.baddbmm.default": None,
            "torch.ops.aten.bitwise_and.Scalar": None,
            "torch.ops.aten.bitwise_and.Tensor": None,
            "torch.ops.aten.bitwise_not.default": None,
            "torch.ops.aten.bitwise_or.Tensor": None,
            "torch.ops.aten.bitwise_xor.Tensor": None,
            "torch.ops.aten.bmm.default": None,
            "torch.ops.aten.cat.default": None,
            "torch.ops.aten.ceil.default": None,
            "torch.ops.aten.clamp.default": None,
            "torch.ops.aten.clamp_max.default": None,
            "torch.ops.aten.clamp_max.Tensor": None,
            "torch.ops.aten.clamp_min.default": None,
            "torch.ops.aten.clamp_min.Tensor": None,
            "torch.ops.aten.clone.default": None,
            "torch.ops.aten.constant_pad_nd.default": None,
            "torch.ops.aten.convolution.default": None,
            "torch.ops.aten.copy.default": None,
            "torch.ops.aten.copy_.default": None,
            "torch.ops.aten.cos.default": None,
            "torch.ops.aten.cosh.default": None,
            "torch.ops.aten.cumsum.default": None,
            "torch.ops.aten.detach.default": None,
            "torch.ops.aten.detach_.default": None,
            "torch.ops.aten.div.Scalar": None,
            "torch.ops.aten.div.Tensor": None,
            "torch.ops.aten.div.Tensor_mode": None,
            "torch.ops.aten.div_.Tensor": None,
            "torch.ops.aten.elu.default": None,
            "torch.ops.aten.elu_.default": None,
            "torch.ops.aten.embedding.default": None,
            "torch.ops.aten.empty.memory_format": None,
            "torch.ops.aten.eq.Scalar": None,
            "torch.ops.aten.eq.Tensor": None,
            "torch.ops.aten.erf.default": None,
            "torch.ops.aten.exp.default": None,
            "torch.ops.aten.expand.default": None,
            "torch.ops.aten.fake_quantize_per_channel_affine_cachemask.default": None,
            "torch.ops.aten.fill.Scalar": None,
            "torch.ops.aten.fill_.Scalar": None,
            "torch.ops.aten.fill.Tensor": None,
            "torch.ops.aten.fill_.Tensor": None,
            "torch.ops.aten.flip.default": None,
            "torch.ops.aten.floor.default": None,
            "torch.ops.aten.floor.default": None,
            "torch.ops.aten.fmod.Scalar": None,
            "torch.ops.aten.fmod.Tensor": None,
            "torch.ops.aten.full.default": None,
            "torch.ops.aten.full.names": None,
            "torch.ops.aten.full_like.default": None,
            "torch.ops.aten.gather.default": None,
            "torch.ops.aten.ge.Scalar": None,
            "torch.ops.aten.ge.Tensor": None,
            "torch.ops.aten.gelu.default": None,
            "torch.ops.aten.glu.default": None,
            "torch.ops.aten.grid_sampler_2d.default": None,
            "torch.ops.aten.gt.Scalar": None,
            "torch.ops.aten.gt.Tensor": None,
            "torch.ops.aten.hardsigmoid.default": None,
            "torch.ops.aten.hardswish.default": None,
            "torch.ops.aten.hardswish_.default": None,
            "torch.ops.aten.hardtanh.default": None,
            "torch.ops.aten.hardtanh_.default": None,
            "torch.ops.aten.index.Tensor": None,
            "torch.ops.aten._unsafe_index.Tensor": None,
            "torch.ops.aten.index_select.default": None,
            "torch.ops.aten.isfinite.default": None,
            "torch.ops.aten.isinf.default": None,
            "torch.ops.aten.isnan.default": None,
            "torch.ops.aten.le.Scalar": None,
            "torch.ops.aten.le.Tensor": None,
            "torch.ops.aten.leaky_relu.default": None,
            "torch.ops.aten.leaky_relu_.default": None,
            "torch.ops.aten.lift_fresh_copy.default": None,
            "torch.ops.aten.linalg_vector_norm.default": None,
            "torch.ops.aten.log.default": None,
            "torch.ops.aten.log_sigmoid_forward.default": None,
            "torch.ops.aten.log10.default": None,
            "torch.ops.aten.log1p.default": None,
            "torch.ops.aten.log2.default": None,
            "torch.ops.aten.logical_not.default": None,
            "torch.ops.aten.logsumexp.default": None,
            "torch.ops.aten.lt.Scalar": None,
            "torch.ops.aten.lt.Tensor": None,
            "torch.ops.aten.masked_fill.Scalar": None,
            "torch.ops.aten.masked_fill.Tensor": None,
            "torch.ops.aten.masked_fill_.Scalar": None,
            "torch.ops.aten.masked_fill_.Tensor": None,
            "torch.ops.aten.max.default": None,
            "torch.ops.aten.max.dim": None,
            "torch.ops.aten.max_pool2d_with_indices.default": None,
            "torch.ops.aten.max_pool3d_with_indices.default": None,
            "torch.ops.aten.maximum.default": None,
            "torch.ops.aten.mean.default": None,
            "torch.ops.aten.mean.dim": None,
            "torch.ops.aten.min.default": None,
            "torch.ops.aten.min.dim": None,
            "torch.ops.aten.minimum.default": None,
            "torch.ops.aten.mm.default": None,
            "torch.ops.aten.mul.Scalar": None,
            "torch.ops.aten.mul.Tensor": None,
            "torch.ops.aten.mul_.Tensor": None,
            "torch.ops.aten.native_batch_norm.default": None,
            "torch.ops.aten.native_dropout.default": None,
            "torch.ops.aten.native_group_norm.default": None,
            "torch.ops.aten.native_layer_norm.default": None,
            "torch.ops.aten.ne.Scalar": None,
            "torch.ops.aten.ne.Tensor": None,
            "torch.ops.aten.neg.default": None,
            "torch.ops.aten.new_full.default": None,
            "torch.ops.aten.new_ones.default": None,
            "torch.ops.aten.ones_like.default": None,
            "torch.ops.aten.new_zeros.default": None,
            "torch.ops.aten.ones.default": None,
            "torch.ops.aten.permute.default": None,
            "torch.ops.aten.pow.Scalar": None,
            "torch.ops.aten.pow.Tensor_Scalar": None,
            "torch.ops.aten.pow.Tensor_Tensor": None,
            "torch.ops.aten.rand.default": None,
            "torch.ops.aten.reflection_pad2d.default": None,
            "torch.ops.aten.reciprocal.default": None,
            "torch.ops.aten.relu.default": None,
            "torch.ops.aten.relu_.default": None,
            "torch.ops.aten.repeat.default": None,
            "torch.ops.aten.roll.default": None,
            "torch.ops.aten.rsqrt.default": None,
            "torch.ops.aten.rsub.Scalar": None,
            "torch.ops.aten.rsub.Tensor": None,
            "torch.ops.aten.scalar_tensor.default": None,
            "torch.ops.aten.scatter.src": None,
            "torch.ops.aten.scatter.value": None,
            "torch.ops.aten.select.int": None,
            "torch.ops.aten.select_scatter.default": None,
            "torch.ops.aten.sigmoid.default": None,
            "torch.ops.aten.sigmoid_.default": None,
            "torch.ops.aten.sign.default": None,
            "torch.ops.aten.silu.default": None,
            "torch.ops.aten.silu_.default": None,
            "torch.ops.aten.sin.default": None,
            "torch.ops.aten.sinh.default": None,
            "torch.ops.aten.slice.Tensor": None,
            "torch.ops.aten.slice_scatter.default": None,
            "torch.ops.aten.sort.default": None,
            "torch.ops.aten.split.Tensor": None,
            "torch.ops.aten.split_with_sizes.default": None,
            "torch.ops.aten.sqrt.default": None,
            "torch.ops.aten.squeeze.dim": None,
            "torch.ops.aten.squeeze.dims": None,
            "torch.ops.aten.stack.default": None,
            "torch.ops.aten.std.correction": None,
            "torch.ops.aten.sub.default": None,
            "torch.ops.aten.sub.Tensor": None,
            "torch.ops.aten.sum.default": None,
            "torch.ops.aten.sum.dim_IntList": None,
            "torch.ops.aten.t.default": None,
            "torch.ops.aten.tan.default": None,
            "torch.ops.aten.tanh.default": None,
            "torch.ops.aten.topk.default": None,
            "torch.ops.aten.transpose.int": None,
            "torch.ops.aten.tril.default": None,
            "torch.ops.aten.tril_.default": None,
            "torch.ops.aten.triu.default": None,
            "torch.ops.aten.unbind.int": None,
            "torch.ops.aten.unfold.default": None,
            "torch.ops.aten.unsqueeze.default": None,
            "torch.ops.aten.upsample_nearest2d.default": None,
            "torch.ops.aten.var.correction": None,
            "torch.ops.aten.var_mean.correction": None,
            "torch.ops.aten.view.default": None,
            "torch.ops.aten.where.self": None,
            "torch.ops.aten.zeros.default": None,
            "torch.ops.aten.zeros_like.default": None,
            "torch.ops.torchvision.deform_conv2d.default": None,
            "torch.ops.torchvision.roi_align.default": None,
            "torch.ops.quantized_decomposed.quantize_per_tensor.default": None,
            "torch.ops.quantized_decomposed.quantize_per_channel.default": None,
            "torch.ops.quantized_decomposed.dequantize_per_tensor.default": None,
            "torch.ops.quantized_decomposed.dequantize_per_channel.default": None

        }
            
        self.enabled_op_names = []

        for op in _get_disabled_ops(options):
            del support_dict[op]

        super().__init__(support_dict)

    def enable_by_name(self, node: Node):
        self.enabled_op_names.append(node.name)

    def is_node_supported(self, submodules: t.Mapping[str, Module], node: Node) -> bool:
        # OpenVINO FX subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            return False

        # ops in supported_dict doesn't have overload name
        # use overloadpacket's qualified_name for OpOverload
        if isinstance(node.target, OpOverload):
            target = _get_qualified_name(node.target.overloadpacket)

            if target in self._support_dict:
                return True

        if node.name in self.enabled_op_names:
            return True

        return super().is_node_supported(submodules, node)
