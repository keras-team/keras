# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# type: ignore
from openvino._pyopenvino.passes import Manager as ManagerBase
from openvino._pyopenvino.passes import PassBase


class Manager(ManagerBase):
    """Manager that additionally holds transformations objects."""

    def __init__(self) -> None:
        super().__init__()
        self.passes_list = []  # need to keep python instances alive

    def register_pass(self, *args, **kwargs) -> PassBase:
        """Register transformation for further execution."""
        for arg in args:
            if isinstance(arg, PassBase):
                self.passes_list.append(arg)

        for arg in kwargs.values():
            if isinstance(arg, PassBase):
                self.passes_list.append(arg)

        return super().register_pass(*args, **kwargs)
