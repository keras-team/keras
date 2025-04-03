# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
from openvino.frontend.pytorch import ModuleExtension, gptq
from openvino.frontend.pytorch.patch_model import patch_model, unpatch_model


def detect_quantized_model(model: torch.nn.Module) -> Optional[str]:
    """Detects the quantization method used in a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to check for quantization.

    Returns:
        str: The quantization method if available, otherwise None.
    """
    if (model and getattr(model, "config", None)
            and getattr(model.config, "quantization_config", None)):
        return model.config.quantization_config.quant_method
    if getattr(model, "model", None):
        return detect_quantized_model(model.model)
    return None


def patch_quantized(model: torch.nn.Module) -> None:
    """Patches a model based on its quantization type ("awq" or "gptq").

    Args:
        model (torch.nn.Module): The model to patch.

    Raises:
        RuntimeError: If the quantization type is unknown.
    """
    quant_type = detect_quantized_model(model)
    if quant_type == "awq":
        extensions = {}
        try:
            from awq.modules.linear import WQLinear_GEMM
            extensions[WQLinear_GEMM] = ModuleExtension(
                WQLinear_GEMM, "ov_ext::awq_gemm",
                convert=lambda module, target_op, *args, **kwargs: target_op(
                    args[0], module.qweight, module.qzeros, module.scales,
                    torch.tensor(module.group_size),
                    torch.tensor(module.w_bit), module.bias),
                evaluate=lambda module, *args, **kwargs: torch.full(
                    list(args[0].shape[:-1]) + [module.out_features], 0.5,
                    dtype=torch.float32))  # type: ignore
        except ImportError:
            pass
        patch_model(model, extensions,
                    "_openvino_quantized_patch_orig_forward")  # type: ignore
    elif quant_type == "gptq":
        model._openvino_gptq_patched = True
        gptq.patch_model(model)  # type: ignore
    else:
        raise RuntimeError(f"Unknown quantization type: {quant_type}.")


def unpatch_quantized(model: torch.nn.Module) -> None:
    """Reverts the patching applied to a quantized PyTorch model.

    Args:
        model (torch.nn.Module): The model to unpatch.
    """
    if getattr(model, "_openvino_gptq_patched", False):
        gptq.unpatch_model(model)  # type: ignore
        del model._openvino_gptq_patched
    else:
        unpatch_model(model,
                      "_openvino_quantized_patch_orig_forward")  # type: ignore
