# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import unittest
from typing import Sequence

import parameterized

import onnx
from onnx import defs


class TestSchema(unittest.TestCase):
    def test_get_schema(self) -> None:
        defs.get_schema("Relu")

    def test_typecheck(self) -> None:
        defs.get_schema("Conv")

    def test_attr_default_value(self) -> None:
        v = defs.get_schema("BatchNormalization").attributes["epsilon"].default_value
        self.assertEqual(type(v), onnx.AttributeProto)
        self.assertEqual(v.type, onnx.AttributeProto.FLOAT)

    def test_function_body(self) -> None:
        self.assertEqual(
            type(defs.get_schema("Selu").function_body), onnx.FunctionProto
        )


class TestOpSchema(unittest.TestCase):
    def test_init(self):
        # Test that the constructor creates an OpSchema object
        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertIsInstance(schema, defs.OpSchema)

    def test_init_with_inputs(self) -> None:
        op_schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            inputs=[defs.OpSchema.FormalParameter("input1", "T")],
            type_constraints=[("T", ["tensor(int64)"], "")],
        )
        self.assertEqual(op_schema.name, "test_op")
        self.assertEqual(op_schema.domain, "test_domain")
        self.assertEqual(op_schema.since_version, 1)
        self.assertEqual(len(op_schema.inputs), 1)
        self.assertEqual(op_schema.inputs[0].name, "input1")
        self.assertEqual(op_schema.inputs[0].type_str, "T")
        self.assertEqual(len(op_schema.type_constraints), 1)
        self.assertEqual(op_schema.type_constraints[0].type_param_str, "T")
        self.assertEqual(
            op_schema.type_constraints[0].allowed_type_strs, ["tensor(int64)"]
        )

    def test_init_creates_multi_input_output_schema(self) -> None:
        op_schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            inputs=[
                defs.OpSchema.FormalParameter("input1", "T"),
                defs.OpSchema.FormalParameter("input2", "T"),
            ],
            outputs=[
                defs.OpSchema.FormalParameter("output1", "T"),
                defs.OpSchema.FormalParameter("output2", "T"),
            ],
            type_constraints=[("T", ["tensor(int64)"], "")],
            attributes=[
                defs.OpSchema.Attribute(
                    "attr1", defs.OpSchema.AttrType.INTS, "attr1 description"
                )
            ],
        )
        self.assertEqual(len(op_schema.inputs), 2)
        self.assertEqual(op_schema.inputs[0].name, "input1")
        self.assertEqual(op_schema.inputs[0].type_str, "T")
        self.assertEqual(op_schema.inputs[1].name, "input2")
        self.assertEqual(op_schema.inputs[1].type_str, "T")
        self.assertEqual(len(op_schema.outputs), 2)
        self.assertEqual(op_schema.outputs[0].name, "output1")
        self.assertEqual(op_schema.outputs[0].type_str, "T")
        self.assertEqual(op_schema.outputs[1].name, "output2")
        self.assertEqual(op_schema.outputs[1].type_str, "T")
        self.assertEqual(len(op_schema.type_constraints), 1)
        self.assertEqual(op_schema.type_constraints[0].type_param_str, "T")
        self.assertEqual(
            op_schema.type_constraints[0].allowed_type_strs, ["tensor(int64)"]
        )
        self.assertEqual(len(op_schema.attributes), 1)
        self.assertEqual(op_schema.attributes["attr1"].name, "attr1")
        self.assertEqual(
            op_schema.attributes["attr1"].type, defs.OpSchema.AttrType.INTS
        )
        self.assertEqual(op_schema.attributes["attr1"].description, "attr1 description")

    def test_init_without_optional_arguments(self) -> None:
        op_schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(op_schema.name, "test_op")
        self.assertEqual(op_schema.domain, "test_domain")
        self.assertEqual(op_schema.since_version, 1)
        self.assertEqual(len(op_schema.inputs), 0)
        self.assertEqual(len(op_schema.outputs), 0)
        self.assertEqual(len(op_schema.type_constraints), 0)

    def test_name(self):
        # Test that the name parameter is required and is a string
        with self.assertRaises(TypeError):
            defs.OpSchema(domain="test_domain", since_version=1)  # type: ignore
        with self.assertRaises(TypeError):
            defs.OpSchema(123, "test_domain", 1)  # type: ignore

        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(schema.name, "test_op")

    def test_domain(self):
        # Test that the domain parameter is required and is a string
        with self.assertRaises(TypeError):
            defs.OpSchema(name="test_op", since_version=1)  # type: ignore
        with self.assertRaises(TypeError):
            defs.OpSchema("test_op", 123, 1)  # type: ignore

        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(schema.domain, "test_domain")

    def test_since_version(self):
        # Test that the since_version parameter is required and is an integer
        with self.assertRaises(TypeError):
            defs.OpSchema("test_op", "test_domain")  # type: ignore

        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(schema.since_version, 1)

    def test_doc(self):
        schema = defs.OpSchema("test_op", "test_domain", 1, doc="test_doc")
        self.assertEqual(schema.doc, "test_doc")

    def test_inputs(self):
        # Test that the inputs parameter is optional and is a sequence of FormalParameter tuples
        inputs = [
            defs.OpSchema.FormalParameter(
                name="input1", type_str="T", description="The first input."
            )
        ]
        schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            inputs=inputs,
            type_constraints=[("T", ["tensor(int64)"], "")],
        )

        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(schema.inputs[0].name, "input1")
        self.assertEqual(schema.inputs[0].type_str, "T")
        self.assertEqual(schema.inputs[0].description, "The first input.")

    def test_outputs(self):
        # Test that the outputs parameter is optional and is a sequence of FormalParameter tuples
        outputs = [
            defs.OpSchema.FormalParameter(
                name="output1", type_str="T", description="The first output."
            )
        ]

        schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            outputs=outputs,
            type_constraints=[("T", ["tensor(int64)"], "")],
        )
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].name, "output1")
        self.assertEqual(schema.outputs[0].type_str, "T")
        self.assertEqual(schema.outputs[0].description, "The first output.")


class TestFormalParameter(unittest.TestCase):
    def test_init(self):
        name = "input1"
        type_str = "tensor(float)"
        description = "The first input."
        param_option = defs.OpSchema.FormalParameterOption.Single
        is_homogeneous = True
        min_arity = 1
        differentiation_category = defs.OpSchema.DifferentiationCategory.Unknown
        formal_parameter = defs.OpSchema.FormalParameter(
            name,
            type_str,
            description,
            param_option=param_option,
            is_homogeneous=is_homogeneous,
            min_arity=min_arity,
            differentiation_category=differentiation_category,
        )

        self.assertEqual(formal_parameter.name, name)
        self.assertEqual(formal_parameter.type_str, type_str)
        self.assertEqual(formal_parameter.description, description)
        self.assertEqual(formal_parameter.option, param_option)
        self.assertEqual(formal_parameter.is_homogeneous, is_homogeneous)
        self.assertEqual(formal_parameter.min_arity, min_arity)
        self.assertEqual(
            formal_parameter.differentiation_category, differentiation_category
        )


class TestTypeConstraintParam(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("single_type", "T", ["tensor(float)"], "Test description"),
            (
                "double_types",
                "T",
                ["tensor(float)", "tensor(int64)"],
                "Test description",
            ),
            ("tuple", "T", ("tensor(float)", "tensor(int64)"), "Test description"),
        ]
    )
    def test_init(
        self,
        _: str,
        type_param_str: str,
        allowed_types: Sequence[str],
        description: str,
    ) -> None:
        type_constraint = defs.OpSchema.TypeConstraintParam(
            type_param_str, allowed_types, description
        )
        self.assertEqual(type_constraint.description, description)
        self.assertEqual(type_constraint.allowed_type_strs, list(allowed_types))
        self.assertEqual(type_constraint.type_param_str, type_param_str)


class TestAttribute(unittest.TestCase):
    def test_init(self):
        name = "test_attr"
        type_ = defs.OpSchema.AttrType.STRINGS
        description = "Test attribute"
        attribute = defs.OpSchema.Attribute(name, type_, description)

        self.assertEqual(attribute.name, name)
        self.assertEqual(attribute.type, type_)
        self.assertEqual(attribute.description, description)

    def test_init_with_default_value(self):
        default_value = (
            defs.get_schema("BatchNormalization").attributes["epsilon"].default_value
        )
        self.assertIsInstance(default_value, onnx.AttributeProto)
        attribute = defs.OpSchema.Attribute("attr1", default_value, "attr1 description")
        self.assertEqual(default_value, attribute.default_value)
        self.assertEqual("attr1", attribute.name)
        self.assertEqual("attr1 description", attribute.description)


@parameterized.parameterized_class(
    [
        # register to exist domain
        {
            "op_type": "CustomOp",
            "op_version": 5,
            "op_domain": "",
            "trap_op_version": [1, 2, 6, 7],
        },
        # register to new domain
        {
            "op_type": "CustomOp",
            "op_version": 5,
            "op_domain": "test",
            "trap_op_version": [1, 2, 6, 7],
        },
    ]
)
class TestOpSchemaRegister(unittest.TestCase):
    op_type: str
    op_version: int
    op_domain: str
    # register some fake schema to check behavior
    trap_op_version: list[int]

    def setUp(self) -> None:
        # Ensure the schema is unregistered
        self.assertFalse(onnx.defs.has(self.op_type, self.op_domain))

    def tearDown(self) -> None:
        # Clean up the registered schema
        for version in [*self.trap_op_version, self.op_version]:
            with contextlib.suppress(onnx.defs.SchemaError):
                onnx.defs.deregister_schema(self.op_type, version, self.op_domain)

    def test_register_multi_schema(self):
        for version in [*self.trap_op_version, self.op_version]:
            op_schema = defs.OpSchema(
                self.op_type,
                self.op_domain,
                version,
            )
            onnx.defs.register_schema(op_schema)
            self.assertTrue(onnx.defs.has(self.op_type, version, self.op_domain))
        for version in [*self.trap_op_version, self.op_version]:
            # Also make sure the `op_schema` is accessible after register
            registered_op = onnx.defs.get_schema(
                op_schema.name, version, op_schema.domain
            )
            op_schema = defs.OpSchema(
                self.op_type,
                self.op_domain,
                version,
            )
            self.assertEqual(str(registered_op), str(op_schema))

    def test_using_the_specified_version_in_onnx_check(self):
        input = f"""
            <
                ir_version: 7,
                opset_import: [
                    "{self.op_domain}" : {self.op_version}
                ]
            >
            agraph (float[N, 128] X, int32 Y) => (float[N] Z)
            {{
                Z = {self.op_domain}.{self.op_type}<attr1=[1,2]>(X, Y)
            }}
           """
        model = onnx.parser.parse_model(input)
        op_schema = defs.OpSchema(
            self.op_type,
            self.op_domain,
            self.op_version,
            inputs=[
                defs.OpSchema.FormalParameter("input1", "T"),
                defs.OpSchema.FormalParameter("input2", "int32"),
            ],
            outputs=[
                defs.OpSchema.FormalParameter("output1", "T"),
            ],
            type_constraints=[("T", ["tensor(float)"], "")],
            attributes=[
                defs.OpSchema.Attribute(
                    "attr1", defs.OpSchema.AttrType.INTS, "attr1 description"
                )
            ],
        )
        with self.assertRaises(onnx.checker.ValidationError):
            onnx.checker.check_model(model, check_custom_domain=True)
        onnx.defs.register_schema(op_schema)
        # The fake schema will raise check exception if selected in checker
        for version in self.trap_op_version:
            onnx.defs.register_schema(
                defs.OpSchema(
                    self.op_type,
                    self.op_domain,
                    version,
                    outputs=[
                        defs.OpSchema.FormalParameter("output1", "int32"),
                    ],
                )
            )
        onnx.checker.check_model(model, check_custom_domain=True)

    def test_register_schema_raises_error_when_registering_a_schema_twice(self):
        op_schema = defs.OpSchema(
            self.op_type,
            self.op_domain,
            self.op_version,
        )
        onnx.defs.register_schema(op_schema)
        with self.assertRaises(onnx.defs.SchemaError):
            onnx.defs.register_schema(op_schema)

    def test_deregister_the_specified_schema(self):
        for version in [*self.trap_op_version, self.op_version]:
            op_schema = defs.OpSchema(
                self.op_type,
                self.op_domain,
                version,
            )
            onnx.defs.register_schema(op_schema)
            self.assertTrue(onnx.defs.has(op_schema.name, version, op_schema.domain))
        onnx.defs.deregister_schema(op_schema.name, self.op_version, op_schema.domain)
        for version in self.trap_op_version:
            self.assertTrue(onnx.defs.has(op_schema.name, version, op_schema.domain))
        # Maybe has lesser op version in trap list
        if onnx.defs.has(op_schema.name, self.op_version, op_schema.domain):
            schema = onnx.defs.get_schema(
                op_schema.name, self.op_version, op_schema.domain
            )
            self.assertLess(schema.since_version, self.op_version)

    def test_deregister_schema_raises_error_when_opschema_does_not_exist(self):
        with self.assertRaises(onnx.defs.SchemaError):
            onnx.defs.deregister_schema(self.op_type, self.op_version, self.op_domain)

    def test_legacy_schema_accessible_after_deregister(self):
        op_schema = defs.OpSchema(
            self.op_type,
            self.op_domain,
            self.op_version,
        )
        onnx.defs.register_schema(op_schema)
        schema_a = onnx.defs.get_schema(
            op_schema.name, op_schema.since_version, op_schema.domain
        )
        schema_b = onnx.defs.get_schema(op_schema.name, op_schema.domain)

        def filter_schema(schemas):
            return [op for op in schemas if op.name == op_schema.name]

        schema_c = filter_schema(onnx.defs.get_all_schemas())
        schema_d = filter_schema(onnx.defs.get_all_schemas_with_history())
        self.assertEqual(len(schema_c), 1)
        self.assertEqual(len(schema_d), 1)
        # Avoid memory residue and access storage as much as possible
        self.assertEqual(str(schema_a), str(op_schema))
        self.assertEqual(str(schema_b), str(op_schema))
        self.assertEqual(str(schema_c[0]), str(op_schema))
        self.assertEqual(str(schema_d[0]), str(op_schema))


if __name__ == "__main__":
    unittest.main()
