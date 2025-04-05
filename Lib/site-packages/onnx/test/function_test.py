# SPDX-License-Identifier: Apache-2.0

# Copyright (c) ONNX Project Contributors
from __future__ import annotations

import unittest

import onnx
from onnx import checker, utils


class TestFunction(unittest.TestCase):
    def _verify_function_set(self, extracted_model, function_set, func_domain):  # type: ignore
        checker.check_model(extracted_model)
        self.assertEqual(len(extracted_model.functions), len(function_set))
        for function in function_set:
            self.assertIsNotNone(
                next(
                    (
                        f
                        for f in extracted_model.functions
                        if f.name == function and f.domain == func_domain
                    ),
                    None,
                )
            )

    def test_extract_model_with_local_function(self) -> None:
        r"""#   1. build a model with graph below. extract models with output combinations
        #   2. validate extracted models' local functions
        #
        # model graph:
        #      i0                    i1                 i2
        #      |   __________________|__________________/_________
        #      |  |                  |             |   /          |
        #      |  |                  |             |  /           |
        #   func_add        func_identity          add         identity
        #    |  ___\___________\____________________|_________    |
        #    | |    \           \                   |  _______|___|
        #    | |     \           \                  | |       |   |
        #    add     function_nested_identity_add   add     function_nested_identity_add
        #     |                 |                    |              |
        #     |                 |                    |              |
        #   o_func_add      o_all_func0           o_no_func     o_all_func1
        #
        # where function_nested_identity_add is a function that is defined with functions:
        #       a               b
        #       |               |
        #   func_identity   func_identity
        #             \       /
        #             func_add
        #                |
        #                c
        #
        """
        # function common
        func_domain = "local"
        func_opset_imports = [onnx.helper.make_opsetid("", 14)]
        func_nested_opset_imports = [
            onnx.helper.make_opsetid("", 14),
            onnx.helper.make_opsetid(func_domain, 1),
        ]

        # add function
        func_add_name = "func_add"
        func_add_inputs = ["a", "b"]
        func_add_outputs = ["c"]
        func_add_nodes = [onnx.helper.make_node("Add", ["a", "b"], ["c"])]
        func_add = onnx.helper.make_function(
            func_domain,
            func_add_name,
            func_add_inputs,
            func_add_outputs,
            func_add_nodes,
            func_opset_imports,
        )

        # identity function
        func_identity_name = "func_identity"
        func_identity_inputs = ["a"]
        func_identity_outputs = ["b"]
        func_identity_nodes = [onnx.helper.make_node("Identity", ["a"], ["b"])]
        func_identity = onnx.helper.make_function(
            func_domain,
            func_identity_name,
            func_identity_inputs,
            func_identity_outputs,
            func_identity_nodes,
            func_opset_imports,
        )

        # nested identity/add function
        func_nested_identity_add_name = "func_nested_identity_add"
        func_nested_identity_add_inputs = ["a", "b"]
        func_nested_identity_add_outputs = ["c"]
        func_nested_identity_add_nodes = [
            onnx.helper.make_node("func_identity", ["a"], ["a1"], domain=func_domain),
            onnx.helper.make_node("func_identity", ["b"], ["b1"], domain=func_domain),
            onnx.helper.make_node("func_add", ["a1", "b1"], ["c"], domain=func_domain),
        ]
        func_nested_identity_add = onnx.helper.make_function(
            func_domain,
            func_nested_identity_add_name,
            func_nested_identity_add_inputs,
            func_nested_identity_add_outputs,
            func_nested_identity_add_nodes,
            func_nested_opset_imports,
        )

        # create graph nodes
        node_func_add = onnx.helper.make_node(
            func_add_name, ["i0", "i1"], ["t0"], domain=func_domain
        )
        node_add0 = onnx.helper.make_node("Add", ["i1", "i2"], ["t2"])
        node_add1 = onnx.helper.make_node("Add", ["t0", "t2"], ["o_func_add"])
        node_func_identity = onnx.helper.make_node(
            func_identity_name, ["i1"], ["t1"], domain=func_domain
        )
        node_identity = onnx.helper.make_node("Identity", ["i1"], ["t3"])
        node_add2 = onnx.helper.make_node("Add", ["t3", "t2"], ["o_no_func"])
        node_func_nested0 = onnx.helper.make_node(
            func_nested_identity_add_name,
            ["t0", "t1"],
            ["o_all_func0"],
            domain=func_domain,
        )
        node_func_nested1 = onnx.helper.make_node(
            func_nested_identity_add_name,
            ["t3", "t2"],
            ["o_all_func1"],
            domain=func_domain,
        )

        graph_name = "graph_with_imbedded_functions"
        ir_version = 8
        opset_imports = [
            onnx.helper.make_opsetid("", 14),
            onnx.helper.make_opsetid("local", 1),
        ]
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=2, shape=[5])

        graph = onnx.helper.make_graph(
            [
                node_func_add,
                node_add0,
                node_add1,
                node_func_identity,
                node_identity,
                node_func_nested0,
                node_func_nested1,
                node_add2,
            ],
            graph_name,
            [
                onnx.helper.make_value_info(name="i0", type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name="i1", type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name="i2", type_proto=tensor_type_proto),
            ],
            [
                onnx.helper.make_value_info(
                    name="o_no_func", type_proto=tensor_type_proto
                ),
                onnx.helper.make_value_info(
                    name="o_func_add", type_proto=tensor_type_proto
                ),
                onnx.helper.make_value_info(
                    name="o_all_func0", type_proto=tensor_type_proto
                ),
                onnx.helper.make_value_info(
                    name="o_all_func1", type_proto=tensor_type_proto
                ),
            ],
        )

        meta = {
            "ir_version": ir_version,
            "opset_imports": opset_imports,
            "producer_name": "test_extract_model_with_local_function",
            "functions": [func_identity, func_add, func_nested_identity_add],
        }
        model = onnx.helper.make_model(graph, **meta)

        checker.check_model(model)
        extracted_with_no_funcion = utils.Extractor(model).extract_model(
            ["i0", "i1", "i2"], ["o_no_func"]
        )
        self._verify_function_set(extracted_with_no_funcion, {}, func_domain)

        extracted_with_add_funcion = utils.Extractor(model).extract_model(
            ["i0", "i1", "i2"], ["o_func_add"]
        )
        self._verify_function_set(
            extracted_with_add_funcion, {func_add_name}, func_domain
        )

        extracted_with_o_all_funcion0 = utils.Extractor(model).extract_model(
            ["i0", "i1", "i2"], ["o_all_func0"]
        )
        self._verify_function_set(
            extracted_with_o_all_funcion0,
            {func_add_name, func_identity_name, func_nested_identity_add_name},
            func_domain,
        )

        extracted_with_o_all_funcion1 = utils.Extractor(model).extract_model(
            ["i0", "i1", "i2"], ["o_all_func1"]
        )
        self._verify_function_set(
            extracted_with_o_all_funcion1,
            {func_add_name, func_identity_name, func_nested_identity_add_name},
            func_domain,
        )

        extracted_with_o_all_funcion2 = utils.Extractor(model).extract_model(
            ["i0", "i1", "i2"],
            ["o_no_func", "o_func_add", "o_all_func0", "o_all_func1"],
        )
        self._verify_function_set(
            extracted_with_o_all_funcion2,
            {func_add_name, func_identity_name, func_nested_identity_add_name},
            func_domain,
        )


if __name__ == "__main__":
    unittest.main()
