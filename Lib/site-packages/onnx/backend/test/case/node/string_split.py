# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class StringSplit(Base):
    @staticmethod
    def export_basic() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["substrings", "length"],
            delimiter=".",
            maxsplit=None,
        )

        x = np.array(["abc.com", "def.net"]).astype(object)

        substrings = np.array([["abc", "com"], ["def", "net"]]).astype(object)

        length = np.array([2, 2], dtype=np.int64)

        expect(
            node,
            inputs=[x],
            outputs=[substrings, length],
            name="test_string_split_basic",
        )

    @staticmethod
    def export_maxsplit() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["substrings", "length"],
            maxsplit=2,
        )

        x = np.array(
            [["hello world", "def.net"], ["o n n x", "the quick brown fox"]]
        ).astype(object)

        substrings = np.array(
            [
                [["hello", "world", ""], ["def.net", "", ""]],
                [["o", "n", "n x"], ["the", "quick", "brown fox"]],
            ]
        ).astype(object)

        length = np.array([[2, 1], [3, 3]], np.int64)

        expect(
            node,
            inputs=[x],
            outputs=[substrings, length],
            name="test_string_split_maxsplit",
        )

    @staticmethod
    def export_consecutive_delimiters() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["substrings", "length"],
            delimiter="-",
            maxsplit=None,
        )

        x = np.array(["o-n-n--x-", "o-n----nx"]).astype(object)

        substrings = np.array(
            [["o", "n", "n", "", "x", ""], ["o", "n", "", "", "", "nx"]]
        ).astype(object)

        length = np.array([6, 6], dtype=np.int64)

        expect(
            node,
            inputs=[x],
            outputs=[substrings, length],
            name="test_string_split_consecutive_delimiters",
        )

    @staticmethod
    def export_empty_string_delimiter() -> None:
        for delimiter, test_name in (
            ("", "test_string_split_empty_string_delimiter"),
            (None, "test_string_split_no_delimiter"),
        ):
            node = onnx.helper.make_node(
                "StringSplit",
                inputs=["x"],
                outputs=["substrings", "length"],
                delimiter=delimiter,
                maxsplit=None,
            )

            x = np.array(
                ["hello world !", "  hello   world !", " hello world   ! "]
            ).astype(object)

            substrings = np.array(
                [
                    ["hello", "world", "!"],
                    ["hello", "world", "!"],
                    ["hello", "world", "!"],
                ]
            ).astype(object)

            length = np.array([3, 3, 3], dtype=np.int64)

            expect(
                node,
                inputs=[x],
                outputs=[substrings, length],
                name=test_name,
            )

    @staticmethod
    def export_empty_string_split() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["substrings", "length"],
            delimiter=None,
            maxsplit=None,
        )

        x = np.array([]).astype(object)

        substrings = np.array([]).astype(object).reshape(0, 0)

        length = np.array([], dtype=np.int64)

        expect(
            node,
            inputs=[x],
            outputs=[substrings, length],
            name="test_string_split_empty_tensor",
            output_type_protos=[
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.STRING, (0, None)),
                None,
            ],
        )
