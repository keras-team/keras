# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

class ModuleExtension:
    def __init__(self, module, target_op, evaluate=None, convert=None):
        """
        Creates an extension that replaces entire PyTorch module by a single operation.
        This functionality works with PyTorch models only. A module can be identified by
        module type (e.g. torch.nn.Linear), module instance in the model or module name.

        Args:
            module (str, torch.nn.Module, type(torch.nn.Module)): PyTorch module to replace

            target_op (str): a target operation that will be used as a replacer for the module,
                            could be a name of the extension operation or existing PyTorch operation
                            (with prim:: or aten:: prefix following TorchScript syntax).

            evaluate (callable with args module, *args, **kwargs): a callable that will replace a target
                            module in model execution it is responsible for producing valid output for
                            the module to allow correct model tracing. By default it calls original module
                            forward with the same arguments. The provided code will not be a part of the final
                            traced model, it is used only to produce valid results in the tracing.

            convert (callable with args target_op, *args, **kwargs): a callable that will be traced and become
                            a part of the final model instead of the target module. It accepts target_op as
                            the first parameter, target_op is callable that will appear as a single node in the
                            graph, the type of the node is target_op provided as another argument above.
        """
        self.module = module
        self.target_op = target_op
        self.evaluate = evaluate
        if self.evaluate is None:
            self.evaluate = lambda module, *args, **kwargs: module(*args, **kwargs)
        self.convert = convert
        if self.convert is None:
            self.convert = lambda module, target_op, *args, **kwargs: target_op(*args, **kwargs)
