# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from typing import Dict

import torch
from torch.nn import Module
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition

from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from openvino.frontend.pytorch.torchdynamo.op_support import OperatorSupport
from openvino.frontend.pytorch.torchdynamo.backend_utils import _is_testing

import typing as t
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PatternNode:
    op_types = {}

    def __init__(self):
        self.op_types = {}


class Partitioner:
    def __init__(self, options):
        self.supported_ops = OperatorSupport(options)

    def fx_serialize(self, graph_module: GraphModule, *args, **kwargs):
        fx_gm = make_fx(graph_module)(*args)
        return fx_gm

    def add_get_attr_inputs(self, partitions: t.List[Partition]):
        # TODO: Find a more efficient way to include input
        # "get_attr" nodes to the partitions.
        getattr_to_merge: Dict[Node, Node] = {}
        for partition in partitions:
            for pnode in partition.nodes:
                for pnode_input in pnode.all_input_nodes:
                    if pnode_input.op in ["get_attr"] and pnode_input.op not in getattr_to_merge:
                        getattr_to_merge[pnode_input] = partition
        for getattr_node, getattr_part in getattr_to_merge.items():
            getattr_part.add_node(getattr_node)

    def check_fully_supported(self, graph_module: GraphModule) -> bool:
        num_fused = 0
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and "fused_" in node.name:
                num_fused += 1
            elif node.op != "placeholder" and node.op != "output":
                return False
        if num_fused == 1:
            return True
        return False

    def check_pattern(self, node: torch.fx.Node, pattern: PatternNode, enabled_ops: list) -> bool:
        if node.op == "call_function":
            if ("call_function" + ":" + str(node.target)) in pattern.op_types:
                pt_input_nodes = node.all_input_nodes
                pattern_input_ops = pattern.op_types["call_function" + ":" + str(node.target)]
                if pattern_input_ops is None:
                    enabled_ops.append(node)
                    return True
                if len(pt_input_nodes) != len(pattern_input_ops):
                    return False
                for i in range(len(pt_input_nodes)):
                    if not self.check_pattern(pt_input_nodes[i], pattern_input_ops[i], enabled_ops):
                        return False
                enabled_ops.append(node)
                return True
        elif node.op == "get_attr":
            if "get_attr" in pattern.op_types:
                return True
            else:
                return False
        return False

    def capture_gptq_patterns(self, graph_module: GraphModule):
        const_0_node = PatternNode
        const_0_node.op_types["get_attr"] = None
        unsqueeze_0_node = PatternNode
        unsqueeze_0_node.op_types["call_function:aten.unsqueeze.default"] = [const_0_node,]
        expand_node = PatternNode
        expand_node.op_types["call_function:aten.expand.default"] = [unsqueeze_0_node,]
        const_1_node = PatternNode
        const_1_node.op_types["get_attr"] = None
        unsqueeze_1_node = PatternNode
        unsqueeze_1_node.op_types["call_function:aten.unsqueeze.default"] = [const_1_node,]
        bitwise_right_shift_node = PatternNode
        bitwise_right_shift_node.op_types["call_function:aten.bitwise_right_shift.Tensor"] = [expand_node, unsqueeze_1_node]
        to_copy_node = PatternNode
        to_copy_node.op_types["call_function:aten._to_copy.default"] = [bitwise_right_shift_node,]
        add_or_to_copy_node = PatternNode
        add_or_to_copy_node.op_types["call_function:aten._to_copy.default"] = [bitwise_right_shift_node,]
        add_or_to_copy_node.op_types["call_function:aten.add.Tensor"] = [to_copy_node,]
        bitwise_and_node = PatternNode
        bitwise_and_node.op_types["call_function:aten.bitwise_and.Scalar"] = [add_or_to_copy_node,]

        for node in graph_module.graph.nodes:
            if str(node.op) == "call_function" and str(node.target) == "aten.bitwise_and.Scalar":
                enabled_ops = []
                pattern_match = self.check_pattern(node, bitwise_and_node, enabled_ops)
                if pattern_match:
                    for pattern_op in enabled_ops:
                        self.supported_ops.enable_by_name(pattern_op)

    def capture_nncf_patterns(self, graph_module: GraphModule):
        const_node = PatternNode
        const_node.op_types["get_attr"] = None
        bitwise_right_shift_node = PatternNode
        bitwise_right_shift_node.op_types["call_function:aten.bitwise_right_shift.Tensor_Scalar"] = [const_node]
        bitwise_and_node = PatternNode
        bitwise_and_node.op_types["call_function:aten.bitwise_and.Scalar"] = [const_node,]
        stack_node = PatternNode
        stack_node.op_types["call_function:aten.stack.default"] = [bitwise_and_node, bitwise_right_shift_node]

        for node in graph_module.graph.nodes:
            if str(node.op) == "call_function" and str(node.target) == "aten.stack.default":
                enabled_ops = []
                pattern_match = self.check_pattern(node, bitwise_and_node, enabled_ops)
                if pattern_match:
                    for pattern_op in enabled_ops:
                        self.supported_ops.enable_by_name(pattern_op)

    def make_partitions(self, graph_module: GraphModule, options) -> GraphModule:
        allow_single_node_partition = _is_testing(options)
        self.capture_gptq_patterns(graph_module)
        self.capture_nncf_patterns(graph_module)
        partitioner = CapabilityBasedPartitioner(
            graph_module, self.supported_ops, allows_single_node_partition=allow_single_node_partition)
        partitions = partitioner.propose_partitions()
        self.add_get_attr_inputs(partitions)
        fused_graph_module = partitioner.fuse_partitions(partitions)

        return fused_graph_module
