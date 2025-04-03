# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from typing import Callable, Sequence

import numpy as np

from onnx import (
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
    ValueInfoProto,
    checker,
    compose,
    helper,
    parser,
    version_converter,
)


def _load_model(m_def: str) -> ModelProto:
    """Parses a model from a string representation, including checking the model for correctness"""
    m = parser.parse_model(m_def)
    checker.check_model(m)
    return m


def _prefixed(prefix: str, s: str) -> str:
    """Prefixes a string (if not empty)"""
    return prefix + s if len(s) > 0 else s


def _get_shape(value_info: ValueInfoProto) -> list[int]:
    """Returns a list of integers representing the shape of the provided ValueInfoProto"""
    return [
        value_info.type.tensor_type.shape.dim[d].dim_value
        for d in range(len(value_info.type.tensor_type.shape.dim))
    ]


def _make_sparse_tensor(name: str) -> SparseTensorProto:
    dense_shape = [3, 3]
    linear_indices = [2, 3, 5]
    sparse_values = [1.7, 0.4, 0.9]
    values_tensor = helper.make_tensor(
        name=name + "_values",
        data_type=TensorProto.FLOAT,
        dims=[len(sparse_values)],
        vals=np.array(sparse_values).astype(np.float32),
        raw=False,
    )

    indices_tensor = helper.make_tensor(
        name=name + "_idx",
        data_type=TensorProto.INT64,
        dims=[len(linear_indices)],
        vals=np.array(linear_indices).astype(np.int64),
        raw=False,
    )
    return helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)


M1_DEF = """
    <
        ir_version: 7,
        opset_import: [ "": 10, "com.microsoft": 1]
    >
    agraph (float[N, M] A0, float[N, M] A1, float[N, M] _A) => (float[N, M] B00, float[N, M] B10, float[N, M] B20)
    {
        B00 = Add(A0, A1)
        B10 = Sub(A0, A1)
        B20 = Mul(A0, A1)
    }
    """

M2_DEF = """
    <
        ir_version: 7,
        opset_import: [ "": 10, "com.microsoft": 1]
    >
    agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21) => (float[N, M] D0)
    {
        C0 = Add(B01, B11)
        C1 = Sub(B11, B21)
        D0 = Mul(C0, C1)
    }
    """


class TestComposeFunctions(unittest.TestCase):
    def _test_merge_models(
        self,
        m1def: str,
        m2def: str,
        io_map: list[tuple[str, str]],
        check_expectations: Callable[[GraphProto, GraphProto, GraphProto], None],
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        prefix1: str | None = None,
        prefix2: str | None = None,
    ) -> None:
        m1, m2 = _load_model(m1def), _load_model(m2def)
        g3 = compose.merge_graphs(
            m1.graph,
            m2.graph,
            io_map=io_map,
            inputs=inputs,
            outputs=outputs,
            prefix1=prefix1,
            prefix2=prefix2,
        )
        checker.check_graph(g3)
        check_expectations(m1.graph, m2.graph, g3)
        m3 = compose.merge_models(
            m1,
            m2,
            io_map=io_map,
            inputs=inputs,
            outputs=outputs,
            prefix1=prefix1,
            prefix2=prefix2,
        )
        checker.check_model(m3)
        check_expectations(m1.graph, m2.graph, m3.graph)

    def test_case_connect_all_no_name_collision(self) -> None:
        """Tests a simple scenario where two models without overlapping names are merged by
        connecting all the outputs in the first models to all the inputs in the second model
        """

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(g3.output, g2.output)
            self.assertEqual(
                ["Add", "Sub", "Mul", "Add", "Sub", "Mul"],
                [item.op_type for item in g3.node],
            )

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)

    def test_case_connect_same_output_twice(self) -> None:
        """Tests a scenario where we merge two models by connecting a single output in the first model
        to all the inputs in the second
        """

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            del g2  # Unused
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(["B10", "B20", "D0"], [elem.name for elem in g3.output])
            self.assertEqual(
                ["Add", "Sub", "Mul", "Add", "Sub", "Mul"],
                [item.op_type for item in g3.node],
            )

        io_map = [("B00", "B01"), ("B00", "B11"), ("B00", "B21")]
        self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)

    def test_case_connect_same_output_drop_outputs(self) -> None:
        """Tests a scenario where we merge two models by connecting a single output in the first model
        to all the inputs in the second, while dropping the rest of the outputs in the first model
        """

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            del g2  # Unused
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(["D0"], [elem.name for elem in g3.output])
            self.assertEqual(
                ["Add", "Add", "Sub", "Mul"], [item.op_type for item in g3.node]
            )

        io_map = [("B00", "B01"), ("B00", "B11"), ("B00", "B21")]
        outputs = ["D0"]
        self._test_merge_models(
            M1_DEF, M2_DEF, io_map, check_expectations, outputs=outputs
        )

    def test_case_connect_same_input_output_name(self) -> None:
        """Tests a scenario where we merge two models, where the inputs/outputs connected
        are named exactly the same
        """
        m1_def = """
            <
                ir_version: 7,
                opset_import: [ "": 10]
            >
            agraph (float[N, M] A) => (float[N, M] B)
            {
                B = Add(A, A)
            }
            """
        m2_def = """
            <
                ir_version: 7,
                opset_import: [ "": 10]
            >
            agraph (float[N, M] B) => (float[N, M] C)
            {
                C = Add(B, B)
            }
            """
        io_map = [("B", "B")]

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            del g1, g2  # Unused

            self.assertEqual(["A"], [elem.name for elem in g3.input])
            self.assertEqual(["C"], [elem.name for elem in g3.output])

        self._test_merge_models(m1_def, m2_def, io_map, check_expectations)

    def test_case_drop_inputs_outputs(self) -> None:
        """Tests a scenario where we merge two models, not including some of the inputs/outputs"""
        m1_def = """
            <
                ir_version: 7,
                opset_import: [ "": 10]
            >
            agraph (float[N] A0, float[N] B0) => (float[N] A1, float[N] B1)
            {
                A1 = Add(A0, A0)
                B1 = Sub(B0, B0)
            }
            """
        m2_def = """
            <
                ir_version: 7,
                opset_import: [ "": 10]
            >
            agraph (float[N] A2, float[N] B2) => (float[N] A3, float[N] B3)
            {
                A3 = Add(A2, A2)
                B3 = Sub(B2, B2)
            }
            """
        io_map = [("A1", "B2")]

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            del g1, g2  # Unused

            self.assertEqual(["A0"], [elem.name for elem in g3.input])
            self.assertEqual(["B3"], [elem.name for elem in g3.output])
            self.assertEqual(["Add", "Sub"], [elem.op_type for elem in g3.node])

        inputs = ["A0"]
        outputs = ["B3"]
        self._test_merge_models(
            m1_def, m2_def, io_map, check_expectations, inputs=inputs, outputs=outputs
        )

    def test_case_name_collision_prefix(self) -> None:
        """Tests a scenario where we merge two models that have name collisions, but they
        are avoided by prefixing the models model.
        """
        m1_def = """
            <
                ir_version: 7,
                opset_import: [ "": 10]
            >
            agraph (float[N] A, float[N] B) => (float[N] C)
            {
                C = Add(A, B)
            }
            """
        io_map = [("C", "A")]

        def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
            del g1, g2  # Unused

            self.assertEqual(["m1/A", "m1/B", "m2/B"], [elem.name for elem in g3.input])
            self.assertEqual(["m2/C"], [elem.name for elem in g3.output])
            self.assertEqual(["Add", "Add"], [elem.op_type for elem in g3.node])

        self._test_merge_models(
            m1_def, m1_def, io_map, check_expectations, prefix1="m1/", prefix2="m2/"
        )

    def test_case_connect_partially_no_name_collision(self) -> None:
        """Tests a scenario where two models without overlapping names are merged by
        connecting some outputs from the first model to some inputs in the second.
        The remaining inputs/outputs should be present in the combined model
        """

        def check_expectations(g1: GraphProto, g2: GraphProto, g4: GraphProto) -> None:
            del g1, g2  # Unused

            # B20 <-> B21 not connected. They should still be present
            # in the inputs and outputs of the combined graph
            self.assertEqual(
                ["A0", "A1", "_A", "B21"], [elem.name for elem in g4.input]
            )
            self.assertEqual(["B20", "D0"], [elem.name for elem in g4.output])

        io_map = [("B00", "B01"), ("B10", "B11")]
        self._test_merge_models(M1_DEF, M2_DEF, io_map, check_expectations)

    def test_merge_models_with_metadata_props(self) -> None:
        m1 = _load_model(M1_DEF)
        helper.set_model_props(m1, {"p1": "v1", "p2": "v2"})

        m2 = _load_model(M2_DEF)
        helper.set_model_props(m2, {"p3": "v3", "p4": "v4"})

        io_map = [("B00", "B01")]
        m3 = compose.merge_models(m1, m2, io_map=io_map)
        assert len(m3.metadata_props) == 4

        # Overlap, but same value
        helper.set_model_props(m2, {"p1": "v1", "p4": "v4"})
        m3 = compose.merge_models(m1, m2, io_map=io_map)
        assert len(m3.metadata_props) == 3

        # Same keys but not same value. Error
        helper.set_model_props(m2, {"p1": "v5", "p4": "v4"})
        self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map=io_map)

    def test_error_wrong_input_output_name(self) -> None:
        """Tests that providing a non existing output/input name in the io_map argument produces an error."""
        m1, m2 = _load_model(M1_DEF), _load_model(M2_DEF)

        self.assertRaises(
            ValueError,
            compose.merge_models,
            m1,
            m2,
            io_map=[("wrong_outname", "B01"), ("B10", "B11"), ("B20", "B21")],
        )

        # Wrong output name
        self.assertRaises(
            ValueError,
            compose.merge_models,
            m1,
            m2,
            io_map=[("B00", "wrong_input"), ("B10", "B11"), ("B20", "B21")],
        )

    def test_error_ir_version_mismatch(self) -> None:
        m1 = _load_model(
            """
    <
        ir_version: 7,
        opset_import: [ "": 13]
    >
    agraph (float[N, M] X0) => (float[N, M] Y0)
    {
        Y0 = Add(X0, X0)
    }
    """
        )

        m2 = _load_model(
            """
    <
        ir_version: 6,
        opset_import: [ "": 13]
    >
    agraph (float[N, M] X1) => (float[N, M] Y1)
    {
        Y1 = Add(X1, X1)
    }
    """
        )
        # Wrong IR version name
        self.assertRaises(
            ValueError, compose.merge_models, m1, m2, io_map=[("Y0", "X1")]
        )

    def test_error_opset_import_mismatch(self) -> None:
        """Tests that providing models with different operator set imported produces an error."""
        m1, m2 = _load_model(M1_DEF), _load_model(M2_DEF)
        m1 = helper.make_model(
            m1.graph, producer_name="test", opset_imports=[helper.make_opsetid("", 10)]
        )
        m2 = helper.make_model(
            m2.graph, producer_name="test", opset_imports=[helper.make_opsetid("", 15)]
        )

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map)

        # Converting to the same Operator set version, should work
        m1 = version_converter.convert_version(m1, 15)
        m3 = compose.merge_models(m1, m2, io_map=io_map)
        checker.check_model(m3)

    # FIXME: This function should be removed, as tests should not contain a copy of the tested logic.
    def _test_add_prefix(
        self,
        rename_nodes: bool = False,
        rename_edges: bool = False,
        rename_inputs: bool = False,
        rename_outputs: bool = False,
        rename_initializers: bool = False,
        rename_value_infos: bool = False,
        inplace: bool = False,
    ) -> None:
        m1 = _load_model(M1_DEF)

        prefix = "pre/"

        if inplace:
            m2 = ModelProto()
            m2.CopyFrom(m1)
            compose.add_prefix(
                m2,
                prefix,
                rename_nodes=rename_nodes,
                rename_edges=rename_edges,
                rename_inputs=rename_inputs,
                rename_outputs=rename_outputs,
                rename_initializers=rename_initializers,
                rename_value_infos=rename_value_infos,
                inplace=True,
            )
        else:
            m2 = compose.add_prefix(
                m1,
                prefix,
                rename_nodes=rename_nodes,
                rename_edges=rename_edges,
                rename_inputs=rename_inputs,
                rename_outputs=rename_outputs,
                rename_initializers=rename_initializers,
                rename_value_infos=rename_value_infos,
            )
        g_in = m1.graph
        g_out = m2.graph

        if (
            rename_edges
            or rename_inputs
            or rename_outputs
            or rename_initializers
            or rename_value_infos
        ):
            name_mapping = {}

            # Rename inputs/outputs/edges. Propagate name changes from and to edges
            if rename_edges:
                for n in g_in.node:
                    for e in n.input:
                        name_mapping[e] = _prefixed(prefix, e)
                    for e in n.output:
                        name_mapping[e] = _prefixed(prefix, e)
            if rename_inputs:
                for elem in g_in.input:
                    name_mapping[elem.name] = _prefixed(prefix, elem.name)
            if rename_outputs:
                for elem in g_in.output:
                    name_mapping[elem.name] = _prefixed(prefix, elem.name)

            if rename_initializers:
                for init in g_in.initializer:
                    name_mapping[init.name] = _prefixed(prefix, init.name)
                for sparse_init in g_in.sparse_initializer:
                    name_mapping[sparse_init.values.name] = _prefixed(
                        prefix, sparse_init.values.name
                    )
                    name_mapping[sparse_init.indices.name] = _prefixed(
                        prefix, sparse_init.indices.name
                    )

            if rename_value_infos:
                for value_info in g_in.output:
                    name_mapping[value_info.name] = _prefixed(prefix, value_info.name)

            for n1, n0 in zip(g_out.node, g_in.node):
                for e1, e0 in zip(n1.input, n0.input):
                    self.assertEqual(name_mapping.get(e0, e0), e1)
                for e1, e0 in zip(n1.output, n0.output):
                    self.assertEqual(name_mapping.get(e0, e0), e1)
            for i1, i0 in zip(g_out.input, g_in.input):
                self.assertEqual(name_mapping.get(i0.name, i0.name), i1.name)
            for o1, o0 in zip(g_out.output, g_in.output):
                self.assertEqual(name_mapping.get(o0.name, o0.name), o1.name)

            for init1, init0 in zip(g_out.initializer, g_in.initializer):
                self.assertEqual(name_mapping.get(init0.name, init0.name), init1.name)

            for sparse_init1, sparse_init0 in zip(
                g_out.sparse_initializer, g_in.sparse_initializer
            ):
                self.assertEqual(
                    name_mapping.get(
                        sparse_init0.values.name, sparse_init0.values.name
                    ),
                    sparse_init1.values.name,
                )
                self.assertEqual(
                    name_mapping.get(
                        sparse_init0.indices.name, sparse_init0.indices.name
                    ),
                    sparse_init1.indices.name,
                )

            for vi1, vi0 in zip(g_out.value_info, g_in.value_info):
                self.assertEqual(name_mapping.get(vi0.name, vi0.name), vi1.name)

            if rename_nodes:
                for n1, n0 in zip(g_out.node, g_in.node):
                    self.assertEqual(_prefixed(prefix, n0.name), n1.name)

    def test_add_prefix_nodes(self) -> None:
        """Tests renaming nodes only"""
        self._test_add_prefix(rename_nodes=True)

    def test_add_prefix_edges(self) -> None:
        """Tests prefixing nodes edges. This will also rename inputs/outputs, since the names are shared"""
        self._test_add_prefix(rename_edges=True)

    def test_add_prefix_inputs(self) -> None:
        """Tests prefixing graph inputs only. Relevant node edges should be renamed as well"""
        self._test_add_prefix(rename_inputs=True)

    def test_add_prefix_outputs(self) -> None:
        """Tests prefixing graph outputs only. Relevant node edges should be renamed as well"""
        self._test_add_prefix(rename_outputs=True)

    def test_add_prefix_attribute_subgraph(self) -> None:
        """Tests prefixing attribute's subgraph. Relevant subgraph should be renamed as well"""
        C = helper.make_tensor_value_info("C", TensorProto.BOOL, [1])
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [None, 1])
        Out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [None, 1])

        XY = helper.make_node("Mul", inputs=["X", "Y"], outputs=["XY"])
        add = helper.make_node("Add", inputs=["XY", "Z"], outputs=["Out"])
        sub = helper.make_node("Sub", inputs=["XY", "Z"], outputs=["Out"])

        cond = helper.make_node(
            "If",
            inputs=["C"],
            outputs=["Out"],
            then_branch=helper.make_graph(
                nodes=[add], name="then", inputs=[], outputs=[Out]
            ),
            else_branch=helper.make_graph(
                nodes=[sub], name="else", inputs=[], outputs=[Out]
            ),
        )
        graph = helper.make_graph(
            nodes=[XY, cond], name="graph", inputs=[C, X, Y, Z], outputs=[Out]
        )
        prefix = "prefix."
        prefixed_graph = compose.add_prefix_graph(graph, prefix)
        checker.check_graph(prefixed_graph)
        for n1, n0 in zip(prefixed_graph.node, graph.node):
            self.assertEqual(_prefixed(prefix, n0.name), n1.name)
            for attribute1, attribute0 in zip(n1.attribute, n0.attribute):
                if attribute1.g:
                    for subgraph_n1, subgraph_n0 in zip(
                        attribute1.g.node, attribute0.g.node
                    ):
                        for input_n1, input_n0 in zip(
                            subgraph_n1.input, subgraph_n0.input
                        ):
                            self.assertEqual(_prefixed(prefix, input_n0), input_n1)
                        for output_n1, output_n0 in zip(
                            subgraph_n1.output, subgraph_n0.output
                        ):
                            self.assertEqual(_prefixed(prefix, output_n0), output_n1)

    def test_add_prefix_all(self) -> None:
        """Tests prefixing all names in the graph"""
        self._test_add_prefix(True, True, True, True, True, True)

    def test_add_prefix_inplace(self) -> None:
        """Tests prefixing inplace"""
        self._test_add_prefix(inplace=True)

    def test_expand_out_dim(self) -> None:
        """Tests expanding output dimensions. The resulting graph should have the same output names,
        but with one more dimension at the specified index.
        """
        m1 = _load_model(M1_DEF)

        def _check_model(m1: ModelProto, m2: ModelProto, dim_idx: int) -> None:
            for out_g2, out_g1 in zip(m2.graph.output, m1.graph.output):
                self.assertEqual(out_g2.name, out_g1.name)
                self.assertEqual(
                    out_g2.type.tensor_type.elem_type, out_g1.type.tensor_type.elem_type
                )
                expected_out_shape = _get_shape(out_g1)
                expected_out_shape.insert(dim_idx, 1)
                self.assertEqual(_get_shape(out_g2), expected_out_shape)

        for dim_idx in [0, 2, -1, -3]:
            m2 = compose.expand_out_dim(m1, dim_idx)
            _check_model(m1, m2, dim_idx)

        # Test inplace
        m2 = ModelProto()
        m2.CopyFrom(m1)
        dim_idx = 0
        compose.expand_out_dim(m2, dim_idx, inplace=True)
        _check_model(m1, m2, dim_idx)

    def _test_overlapping_names(
        self,
        inputs0: Sequence[str] = ("i0", "i1"),
        inputs1: Sequence[str] = ("i2", "i3"),
        outputs0: Sequence[str] = ("o0", "o1"),
        outputs1: Sequence[str] = ("o2", "o3"),
        value_info0: Sequence[str] = ("v0", "v1"),
        value_info1: Sequence[str] = ("v2", "v3"),
        initializer0: Sequence[str] = ("init0", "init1"),
        initializer1: Sequence[str] = ("init2", "init3"),
        sparse_initializer0: Sequence[str] = ("sparse_init0", "sparse_init1"),
        sparse_initializer1: Sequence[str] = ("sparse_init2", "sparse_init3"),
    ) -> None:
        n0 = [
            helper.make_node("Identity", inputs=[inputs0[i]], outputs=[outputs0[i]])
            for i in range(len(inputs0))
        ]
        i0 = [
            helper.make_tensor_value_info(inputs0[i], TensorProto.FLOAT, [])
            for i in range(len(inputs0))
        ]
        o0 = [
            helper.make_tensor_value_info(outputs0[i], TensorProto.FLOAT, [])
            for i in range(len(outputs0))
        ]
        vi0 = [
            helper.make_tensor_value_info(value_info0[i], TensorProto.FLOAT, [])
            for i in range(len(value_info0))
        ]
        init0 = [
            helper.make_tensor(
                name=initializer0[i], data_type=TensorProto.INT64, dims=(), vals=[1]
            )
            for i in range(len(initializer0))
        ]

        sparse_init0 = [
            _make_sparse_tensor(sparse_initializer0[i])
            for i in range(len(sparse_initializer0))
        ]

        n1 = [
            helper.make_node("Identity", inputs=[inputs1[i]], outputs=[outputs1[i]])
            for i in range(len(inputs1))
        ]
        i1 = [
            helper.make_tensor_value_info(inputs1[i], TensorProto.FLOAT, [])
            for i in range(len(inputs1))
        ]
        o1 = [
            helper.make_tensor_value_info(outputs1[i], TensorProto.FLOAT, [])
            for i in range(len(outputs1))
        ]
        vi1 = [
            helper.make_tensor_value_info(value_info1[i], TensorProto.FLOAT, [])
            for i in range(len(value_info1))
        ]
        init1 = [
            helper.make_tensor(
                name=initializer1[i], data_type=TensorProto.INT64, dims=(), vals=[1]
            )
            for i in range(len(initializer1))
        ]
        sparse_init1 = [
            _make_sparse_tensor(sparse_initializer1[i])
            for i in range(len(sparse_initializer1))
        ]

        ops = [helper.make_opsetid("", 10)]
        m0 = helper.make_model(
            helper.make_graph(
                nodes=n0,
                name="g0",
                inputs=i0,
                outputs=o0,
                value_info=vi0,
                initializer=init0,
                sparse_initializer=sparse_init0,
            ),
            producer_name="test",
            opset_imports=ops,
        )
        m1 = helper.make_model(
            helper.make_graph(
                nodes=n1,
                name="g1",
                inputs=i1,
                outputs=o1,
                value_info=vi1,
                initializer=init1,
                sparse_initializer=sparse_init1,
            ),
            producer_name="test",
            opset_imports=ops,
        )

        overlap = compose.check_overlapping_names(m0.graph, m1.graph)
        i = 0

        overlapping_inputs = list(set(inputs0) & set(inputs1))
        overlapping_outputs = list(set(outputs0) & set(outputs1))
        overlapping_edges = list(set(overlapping_inputs + overlapping_outputs))
        if overlapping_edges:
            self.assertEqual(overlap[i], ("edge", overlapping_edges))
            i += 1

        overlapping_vis = list(set(value_info0) & set(value_info1))
        if overlapping_vis:
            self.assertEqual(overlap[i], ("value_info", overlapping_vis))
            i += 1

        overlapping_init = list(set(initializer0) & set(initializer1))
        if overlapping_init:
            self.assertEqual(overlap[i], ("initializer", overlapping_init))
            i += 1

        overlapping_sparse_init = list(
            set(sparse_initializer0) & set(sparse_initializer1)
        )
        if overlapping_sparse_init:
            expected_overlap = []
            for overlapping_name in overlapping_sparse_init:
                expected_overlap.append(overlapping_name + "_values")
                expected_overlap.append(overlapping_name + "_idx")
            self.assertEqual(overlap[i], ("sparse_initializer", expected_overlap))
            i += 1

        m0_new = compose.add_prefix(m0, prefix="g0/")
        overlap = compose.check_overlapping_names(m0_new.graph, m1.graph)
        self.assertEqual(0, len(overlap))

    def test_overlapping_input_names(self) -> None:
        """Tests error checking when the name of the inputs overlaps"""
        self._test_overlapping_names(inputs0=["i0", "i1"], inputs1=["i1", "i2"])

    def test_overlapping_output_names(self) -> None:
        """Tests error checking when the name of the output overlaps"""
        self._test_overlapping_names(outputs0=["o0", "o1"], outputs1=["o1", "o2"])

    def test_overlapping_value_info_names(self) -> None:
        """Tests error checking when the name of value_info entries overlaps"""
        self._test_overlapping_names(
            value_info0=["vi0", "vi1"], value_info1=["vi1", "vi2"]
        )

    def test_overlapping_initializer_names(self) -> None:
        """Tests error checking when the name of initializer entries overlaps"""
        self._test_overlapping_names(
            initializer0=["init0", "init1"], initializer1=["init1", "init2"]
        )

    def test_overlapping_sparse_initializer_names(self) -> None:
        """Tests error checking when the name of sparse_initializer entries overlaps"""
        self._test_overlapping_names(
            sparse_initializer0=["sparse_init0", "sparse_init1"],
            sparse_initializer1=["sparse_init1", "sparse_init2"],
        )

    def test_overlapping_function_names(self) -> None:
        """Tests error checking when the name of local function entries overlaps"""
        ops = [helper.make_opsetid("", 10), helper.make_opsetid("local", 10)]

        def _make_function(
            domain: str,
            fname: str,
            inputs: list[str],
            outputs: list[str],
            nodes: list[NodeProto],
        ) -> FunctionProto:
            f = FunctionProto()
            f.domain = domain
            f.name = fname
            f.input.extend(inputs)
            f.output.extend(outputs)
            f.node.extend(nodes)
            f.opset_import.extend(ops)
            return f

        ops = [helper.make_opsetid("", 10), helper.make_opsetid("local", 10)]

        g = GraphProto()
        g.input.extend(
            [
                helper.make_tensor_value_info("x0", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("x1", TensorProto.FLOAT, []),
            ]
        )
        g.output.extend(
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, []),
            ]
        )
        g.node.extend(
            [helper.make_node("f1", domain="local", inputs=["x0", "x1"], outputs=["y"])]
        )

        g1 = GraphProto()
        g1.CopyFrom(g)
        g1.name = "g1"
        m1 = helper.make_model(g1, producer_name="test", opset_imports=ops)
        m1.functions.extend(
            [
                _make_function(
                    "local",
                    "f1",
                    ["x0", "x1"],
                    ["y"],
                    [helper.make_node("Add", inputs=["x0", "x1"], outputs=["y"])],
                )
            ]
        )
        checker.check_model(m1)

        g2 = GraphProto()
        g2.CopyFrom(g)
        g2.name = "g2"
        m2 = helper.make_model(g2, producer_name="test", opset_imports=ops)
        m2.functions.extend(
            [
                _make_function(
                    "local",
                    "f1",
                    ["x0", "x1"],
                    ["y"],
                    [helper.make_node("Mul", inputs=["x0", "x1"], outputs=["y"])],
                )
            ]
        )
        checker.check_model(m2)

        m = compose.merge_models(
            m1, m2, io_map=[("y", "x0"), ("y", "x1")], prefix1="m1/", prefix2="m2/"
        )
        checker.check_model(m)

        nodes = [n.op_type for n in m.graph.node]
        self.assertEqual(["m1/f1", "m2/f1"], nodes)

        functions = [f.name for f in m.functions]
        self.assertEqual(["m1/f1", "m2/f1"], functions)

        g3 = GraphProto()
        g3.CopyFrom(g)
        g3.name = "g3"
        g3.node[0].op_type = "f2"
        m3 = helper.make_model(g3, producer_name="test", opset_imports=ops)
        m3.functions.extend(
            [
                _make_function(
                    "local",
                    "f1",
                    ["x0", "x1"],
                    ["y"],
                    [
                        helper.make_node("Add", inputs=["x0", "x1"], outputs=["y0"]),
                        helper.make_node("Mul", inputs=["x0", "x1"], outputs=["y1"]),
                        helper.make_node("Add", inputs=["y0", "y1"], outputs=["y"]),
                    ],
                ),
                _make_function(
                    "local",
                    "f2",
                    ["x0", "x1"],
                    ["y"],
                    [
                        helper.make_node(
                            "f1", domain="local", inputs=["x0", "x1"], outputs=["y0"]
                        ),
                        helper.make_node("Mul", inputs=["x0", "x1"], outputs=["y1"]),
                        helper.make_node("Add", inputs=["y0", "y1"], outputs=["y"]),
                    ],
                ),
            ]
        )
        checker.check_model(m3)

        m = compose.merge_models(
            m1, m3, io_map=[("y", "x0"), ("y", "x1")], prefix1="m1/", prefix2="m3/"
        )
        checker.check_model(m)

        nodes = [n.op_type for n in m.graph.node]
        self.assertEqual(["m1/f1", "m3/f2"], nodes)

        functions = [f.name for f in m.functions]
        self.assertEqual(["m1/f1", "m3/f1", "m3/f2"], functions)

        self.assertEqual(["Add"], [n.op_type for n in m.functions[0].node])
        self.assertEqual(
            ["Add", "Mul", "Add"], [n.op_type for n in m.functions[1].node]
        )
        self.assertEqual(
            ["m3/f1", "Mul", "Add"], [n.op_type for n in m.functions[2].node]
        )

    def test_merge_drop_unnecessary_initializers_and_value_info(self) -> None:
        """Tests automatic removal of initializers when merging graphs"""
        ops = [helper.make_opsetid("", 10)]

        g = GraphProto()
        g.input.extend([helper.make_tensor_value_info("x", TensorProto.FLOAT, [])])
        g.output.extend([helper.make_tensor_value_info("y", TensorProto.FLOAT, [])])
        g.node.extend([helper.make_node("Identity", inputs=["x"], outputs=["y"])])

        g1 = GraphProto()
        g1.CopyFrom(g)
        g1.name = "g1"
        m1 = helper.make_model(g1, producer_name="test", opset_imports=ops)
        checker.check_model(m1)

        g2 = GraphProto()
        g2.CopyFrom(g)
        g2.name = "g2"
        g2.initializer.extend(
            [
                helper.make_tensor(
                    name="x", data_type=TensorProto.FLOAT, dims=(), vals=[0]
                )
            ]
        )
        m2 = helper.make_model(g2, producer_name="test", opset_imports=ops)
        checker.check_model(m2)

        g3 = GraphProto()
        g3.CopyFrom(g)
        g3.name = "g3"
        g3.sparse_initializer.extend([_make_sparse_tensor("x")])
        m3 = helper.make_model(g3, producer_name="test", opset_imports=ops)
        checker.check_model(m3)

        g4 = GraphProto()
        g4.CopyFrom(g)
        g4.name = "g3"
        g4.value_info.extend(
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [])]
        )
        m4 = helper.make_model(g4, producer_name="test", opset_imports=ops)
        checker.check_model(m4)

        # Initializer 'x' from m1 is removed, because there is no longer an input with that name
        out_m1 = compose.merge_models(m1, m2, prefix1="m1/", io_map=[("y", "x")])
        self.assertEqual(0, len(out_m1.graph.initializer))

        # Sparse initializer 'x' from m1 is removed, because there is no longer an input with that name
        out_m2 = compose.merge_models(m1, m3, prefix1="m1/", io_map=[("y", "x")])
        self.assertEqual(0, len(out_m2.graph.initializer))

        # Value info 'x' from m1 is removed, because there is no longer an input with that name
        out_m3 = compose.merge_models(m1, m4, prefix1="m1/", io_map=[("y", "x")])
        self.assertEqual(0, len(out_m3.graph.value_info))


if __name__ == "__main__":
    unittest.main()
