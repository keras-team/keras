# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class RegexFullMatch(Base):
    @staticmethod
    def export_basic() -> None:
        node = onnx.helper.make_node(
            "RegexFullMatch",
            inputs=["X"],
            outputs=["Y"],
            pattern=r"www\.[\w.-]+\.\bcom\b",
        )

        x = np.array(["www.google.com", "www.facebook.com", "www.bbc.co.uk"]).astype(
            object
        )
        result = np.array([True, True, False])
        expect(node, inputs=[x], outputs=[result], name="test_regex_full_match_basic")

    @staticmethod
    def export_match_email_domain() -> None:
        node = onnx.helper.make_node(
            "RegexFullMatch",
            inputs=["X"],
            outputs=["Y"],
            pattern=r"(\W|^)[\w.\-]{0,25}@(yahoo|gmail)\.com(\W|$)",
        )

        x = np.array(
            [
                ["account@gmail.com", "account@hotmail.com"],
                ["not email", "account2@yahoo.com"],
            ]
        ).astype(object)
        result = np.array([[True, False], [False, True]])
        expect(
            node,
            inputs=[x],
            outputs=[result],
            name="test_regex_full_match_email_domain",
        )

    @staticmethod
    def export_match_empty() -> None:
        node = onnx.helper.make_node(
            "RegexFullMatch",
            inputs=["X"],
            outputs=["Y"],
            pattern=r"(\W|^)[\w.\-]{0,25}@(yahoo|gmail)\.com(\W|$)",
        )

        x = np.array([[], []]).astype(object)
        result = np.array([[], []]).astype(bool)
        expect(
            node,
            inputs=[x],
            outputs=[result],
            name="test_regex_full_match_empty",
        )
