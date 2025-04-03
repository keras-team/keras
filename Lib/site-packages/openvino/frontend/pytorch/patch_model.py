# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import functools
import logging
import torch
from openvino.frontend.pytorch import ModuleExtension

log = logging.getLogger(__name__)


def patch_model(model, module_extensions, orig_forward_name):
    def module_patcher(m, name):
        extension = None
        if m in module_extensions:
            extension = module_extensions[m]
        elif m.__class__ in module_extensions:
            extension = module_extensions[m.__class__]
        elif name in module_extensions:
            extension = module_extensions[name]

        if extension:
            log.debug("Patching module %s", m)
            # The Trampoline class is instantiated for every module replacement, so we can use
            # class members individually for each module.

            class Trampoline(torch.autograd.Function):
                # required to be saved in class
                target_extension = extension

                @staticmethod
                @torch.jit.ignore
                def forward(ctx, *args, **kwargs):
                    # Temporarily restore the original forward function of `module` to avoid
                    # recursion issues in `evaluate`, then revert it back.
                    patched_forward = m.forward
                    # set original forward for the module
                    m.forward = getattr(m, orig_forward_name)
                    # call user code
                    results = extension.evaluate(m, *args, **kwargs)
                    m.forward = patched_forward  # return patched forward back
                    return results

            def new_forward(*args, **kwargs):
                return extension.convert(m, Trampoline.apply, *args, **kwargs)

            # make signature of new_forward same as of forward
            new_forward = functools.wraps(m.forward)(new_forward)
            setattr(m, orig_forward_name, m.forward)
            m.forward = new_forward

    for name, m in model.named_modules():
        if hasattr(m, orig_forward_name):
            # already patched, skipping. It may happen when patching applied for same module twice
            log.debug("Unexpectedly found already patched module %s while applying "
                      "ModuleExtension during PyTorch model conversion. "
                      "Result of the conversion maybe broken. Depending on the exact issue "
                      "it may lead to broken original model.", name)
            continue

        module_patcher(m, name)


def unpatch_model(model, orig_forward_name):
    for _, m in model.named_modules():
        if hasattr(m, orig_forward_name):
            try:
                m.forward = getattr(m, orig_forward_name)
                delattr(m, orig_forward_name)
            except Exception as error:
                log.warning("Exception raised during model unpatching. "
                            "Depending on the exact issue it may lead to broken original model.\n"
                            "Original exception details:\n%s", error)


def __make_16bit_traceable(model: torch.nn.Module):
    """
    Prepare a 16-bit PyTorch model for tracing with OpenVINO.
     - Replace known list of modules with ModuleExtension.
     - Convert other modules with weights to FP32.
    """
    extensions = {
        torch.nn.Linear: ModuleExtension(
            torch.nn.Linear, "ov_ext::linear",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: torch.full(
                list(args[0].shape[:-1]) + [module.out_features], 0.5, dtype=torch.float32)),
        torch.nn.Embedding: ModuleExtension(
            torch.nn.Embedding, "ov_ext::embedding",
            convert=lambda module, target_op, *args, **kwargs: target_op(module.weight,
                                                                         args[0],
                                                                         module.padding_idx,
                                                                         module.scale_grad_by_freq,
                                                                         module.sparse),
            evaluate=lambda module, *args, **kwargs: torch.full(
                list(args[1].shape) + [module.embedding_dim], 0.5, dtype=torch.float32)),
    }
    try:
        from transformers.pytorch_utils import Conv1D
        extensions[Conv1D] = ModuleExtension(
            Conv1D, "ov_ext::conv1d",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: torch.full(
                list(args[0].shape[:-1]) + [module.nf], 0.5, dtype=torch.float32))
    except ImportError:
        pass
    patch_model(model, extensions,
                "_openvino_module_extension_patch_orig_forward")
    dtype_to_patch = [torch.float16, torch.bfloat16]
    for _, module in model.named_modules():
        if (module.__class__ not in extensions and
            (any(p.dtype in dtype_to_patch for p in module.parameters(False))
             or any(b.dtype in dtype_to_patch for b in module.buffers(False)))):
            log.debug("Casting module %s to float32", module)
            module.float()
