# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# type: ignore
from openvino._pyopenvino.passes import MatcherPass
from openvino._pyopenvino.passes import GraphRewrite as GraphRewriteBase
from openvino._pyopenvino.passes import BackwardGraphRewrite as BackwardGraphRewriteBase


class GraphRewrite(GraphRewriteBase):
    """GraphRewrite that additionally holds python transformations objects."""

    def __init__(self) -> None:
        super().__init__()
        self.passes_list = []  # need to keep python instances alive

    def add_matcher(self, transformation: MatcherPass) -> MatcherPass:
        """Append MatcherPass instance to the end of execution list."""
        self.passes_list.append(transformation)
        return super().add_matcher(transformation)


class BackwardGraphRewrite(BackwardGraphRewriteBase):
    """BackwardGraphRewriteBase that additionally holds python transformations objects."""

    def __init__(self) -> None:
        super().__init__()
        self.passes_list = []  # need to keep python instances alive

    def add_matcher(self, transformation: MatcherPass) -> MatcherPass:
        """Append MatcherPass instance to the end of execution list."""
        self.passes_list.append(transformation)
        return super().add_matcher(transformation)
