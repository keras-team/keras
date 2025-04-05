# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from onnx import NodeProto, checker, load


def check_model() -> None:
    parser = argparse.ArgumentParser("check-model")
    parser.add_argument("model_pb", type=argparse.FileType("rb"))
    args = parser.parse_args()

    model = load(args.model_pb)
    checker.check_model(model)


def check_node() -> None:
    parser = argparse.ArgumentParser("check-node")
    parser.add_argument("node_pb", type=argparse.FileType("rb"))
    args = parser.parse_args()

    node = NodeProto()
    node.ParseFromString(args.node_pb.read())
    checker.check_node(node)
