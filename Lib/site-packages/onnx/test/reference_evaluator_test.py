# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# type: ignore

"""You can run a specific test by using the following syntax.

::

    python onnx/test/reference_evaluator_test.py TestReferenceEvaluator.test_function_attribute_nested_graph
"""

from __future__ import annotations

import itertools
import math
import sys
import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from textwrap import dedent
from typing import Sequence

import numpy as np
import parameterized
import version_utils
from numpy.testing import assert_allclose, assert_almost_equal

import onnx._custom_element_types as custom
from onnx import (
    AttributeProto,
    FunctionProto,
    ModelProto,
    TensorProto,
    checker,
    parser,
    subbyte,
)
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    float32_to_bfloat16,
    float32_to_float8e4m3,
    float32_to_float8e5m2,
    make_function,
    make_graph,
    make_model,
    make_model_gen_version,
    make_node,
    make_operatorsetid,
    make_opsetid,
    make_sequence_type_proto,
    make_tensor,
    make_tensor_sequence_value_info,
    make_tensor_value_info,
    make_value_info,
)
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32, from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun, OpRunExpand
from onnx.reference.ops import load_op
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
from onnx.reference.ops._op_list import Cast_19, Celu
from onnx.reference.ops.aionnx_preview_training._op_list import Adam
from onnx.reference.ops.op_celu import _vcelu1
from onnx.reference.ops.op_col2im import (
    _col2im_naive_implementation_2d,
    col2im_naive_implementation,
)
from onnx.reference.ops.op_conv import Conv, _conv_implementation
from onnx.reference.ops_optimized import Conv as ConvOptimized
from onnx.reference.ops_optimized.op_conv_optimized import _conv_implementation_im2col

# TODO (https://github.com/microsoft/onnxruntime/issues/14932): Get max supported version from onnxruntime directly
# For now, bump the version in CIs whenever there is a new onnxruntime release
ORT_MAX_IR_SUPPORTED_VERSION = int(getenv("ORT_MAX_IR_SUPPORTED_VERSION", "8"))
ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION = int(
    getenv("ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION", "18")
)


def skip_if_no_onnxruntime(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import onnxruntime

            del onnxruntime
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed") from None
        fn(*args, **kwargs)

    return wrapper


def skip_if_no_torch(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import torch

            del torch
        except ImportError:
            raise unittest.SkipTest("torch not installed") from None
        fn(*args, **kwargs)

    return wrapper


def skip_if_no_torchvision(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import torchvision

            del torchvision
        except ImportError:
            raise unittest.SkipTest("torchvision not installed") from None
        fn(*args, **kwargs)

    return wrapper


def skip_if_no_ml_dtypes(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import ml_dtypes

            del ml_dtypes
        except ImportError:
            raise unittest.SkipTest("ml-dtypes not installed") from None
        fn(*args, **kwargs)

    return wrapper


def make_sequence_value_info(name, elem_type, shape):
    if isinstance(elem_type, int):
        return make_tensor_sequence_value_info(name, elem_type, shape)
    s_type = make_sequence_type_proto(elem_type)
    return make_value_info(name, s_type, shape)


def run_ort_inference(onnx_model):
    import onnxruntime as ort

    onnx_domain_opset = ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION
    for opset in onnx_model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            onnx_domain_opset = opset.version
            break
    # The new IR or opset version is not supported by onnxruntime yet
    if (
        onnx_model.ir_version > ORT_MAX_IR_SUPPORTED_VERSION
        or onnx_domain_opset > ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION
    ):
        return None

    return ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def im2col_naive_implementation(data, kernel_shape, dilations, pads, strides):  # type: ignore
    """Naive implementation for `im2col`.

    Args:
        data: image (float)
        kernel_shape: kernel shape
        dilations: dilations
        pads: pads
        strides: strides

    Returns:
        result
    """
    if not isinstance(kernel_shape, tuple):
        raise TypeError(f"Unexpected type {type(kernel_shape)!r} for kernel_shape.")
    if len(data.shape) != len(kernel_shape):
        raise ValueError(f"Shape mismatch {data.shape!r} and {kernel_shape!r}.")
    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    list_output_shape = list(data.shape + kernel_shape)
    for d in range(n_dims):
        kd = kernel_shape[d] + (kernel_shape[d] - 1) * (dilations[d] - 1)
        nd = int(
            ((list_output_shape[d] - kd + new_pads[d][0] + new_pads[d][1]) / strides[d])
            + 1
        )
        list_output_shape[d] = nd
    output_shape = tuple(list_output_shape)

    res = np.zeros(output_shape, dtype=data.dtype)
    kernel_size = np.prod(kernel_shape)
    res_size = np.prod(res.shape[:-n_dims])
    for i in range(res_size):
        i_res = _get_indices(i, res.shape[:-n_dims])
        t_res = tuple(i_res)
        for j in range(kernel_size):
            i_kernel = _get_indices(j, kernel_shape)
            t_kernel = tuple(i_kernel)

            i_img = i_res * strides - new_pads[:, 0] + i_kernel * dilations
            t_img = tuple(i_img)
            if _is_out(t_img, data.shape):
                res[t_res + t_kernel] = 0
            else:
                res[t_res + t_kernel] = data[tuple(t_img)]
    return res


def im2col(
    img: np.ndarray,
    kernel_shape: tuple[int, ...],
    dilations: Sequence[int],
    pads: Sequence[int],
    strides: Sequence[int],
) -> np.ndarray:
    res = None
    for n in range(img.shape[0]):
        for c in range(img.shape[1]):
            out = im2col_naive_implementation(
                img[n, c, ...], kernel_shape, dilations, pads, strides
            )
            if res is None:
                new_shape = img.shape[:2] + out.shape
                res = np.empty(new_shape, dtype=img.dtype)
            res[n, c, ...] = out
    new_shape = res.shape[: -len(kernel_shape)] + (-1,)  # type: ignore
    return res.reshape(new_shape)  # type: ignore


class TestReferenceEvaluator(unittest.TestCase):
    m2_def = """
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

    @staticmethod
    def _load_model(m_def: str) -> ModelProto:
        """Parses a model from a string representation, including checking
        the model for correctness
        """
        m = parser.parse_model(m_def)
        checker.check_model(m)
        return m

    @staticmethod
    def _linear_regression(clip=False, opset=None, min_value=-1.0, max_value=1.0):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        if clip:
            node2 = make_node("Add", ["XA", "B"], ["Y_clip"])
            if opset is not None and opset < 11:
                if min_value:
                    if max_value:
                        node3 = make_node(
                            "Clip", ["Y_clip"], ["Y"], min=min_value, max=max_value
                        )
                    else:
                        node3 = make_node("Clip", ["Y_clip"], ["Y"], min=min_value)
                elif max_value:
                    node3 = make_node("Clip", ["Y_clip"], ["Y"], max=max_value)
                else:
                    node3 = make_node("Clip", ["Y_clip"], ["Y"])
                graph = make_graph([node1, node2, node3], "lr", [X, A, B], [Y])
            else:
                mi = (
                    from_array(np.array([min_value], dtype=np.float32), name="mi")
                    if min_value
                    else None
                )
                ma = (
                    from_array(np.array([max_value], dtype=np.float32), name="ma")
                    if max_value
                    else None
                )
                inputs = ["Y_clip", "mi" if mi else "", "ma" if ma else ""]
                node3 = make_node("Clip", inputs, ["Y"])
                initializer = [_ for _ in [mi, ma] if _]
                graph = make_graph(
                    [node1, node2, node3], "lr", [X, A, B], [Y], initializer=initializer
                )

            def f(x, a, b):  # noqa: ARG001
                return np.clip(a @ a + b, min_value, max_value)

        else:
            node2 = make_node("Add", ["XA", "B"], ["Y"])
            graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
            f = lambda x, a, b: a @ a + b  # noqa: ARG005, E731
        if opset is None:
            onnx_model = make_model(graph)
        else:
            onnx_model = make_model(graph, opset_imports=[make_opsetid("", opset)])
        try:
            check_model(onnx_model)
        except Exception as e:
            raise AssertionError(f"checker fails for\n{onnx_model}") from e
        return onnx_model, f

    def test_reference_evaluator_exceptions(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        with self.assertRaises(TypeError):
            ReferenceEvaluator(X)

    def test_reference_evaluator_no_attribute(self):
        m = TestReferenceEvaluator._load_model(TestReferenceEvaluator.m2_def)
        checker.check_model(m)
        sess = ReferenceEvaluator(m)
        self.assertEqual(sess.input_names, ["B01", "B11", "B21"])
        self.assertEqual(sess.output_names, ["D0"])
        self.assertEqual(sess.opsets, {"": 10, "com.microsoft": 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {"B01": x, "B11": y, "B21": z})[0]
        expected = (x + y) * (y - z)
        assert_allclose(expected, res)

    def test_reference_evaluator_no_attribute_intermediate(self):
        m = TestReferenceEvaluator._load_model(TestReferenceEvaluator.m2_def)
        checker.check_model(m)
        sess = ReferenceEvaluator(m)
        self.assertEqual(sess.input_names, ["B01", "B11", "B21"])
        self.assertEqual(sess.output_names, ["D0"])
        self.assertEqual(sess.opsets, {"": 10, "com.microsoft": 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {"B01": x, "B11": y, "B21": z}, intermediate=True)
        self.assertIsInstance(res, dict)
        expected = (x + y) * (y - z)
        assert_allclose(expected, res["D0"])

    def test_reference_evaluator_no_attribute_bytes(self):
        m = TestReferenceEvaluator._load_model(TestReferenceEvaluator.m2_def)
        checker.check_model(m)
        sess = ReferenceEvaluator(m.SerializeToString())
        self.assertEqual(sess.input_names, ["B01", "B11", "B21"])
        self.assertEqual(sess.output_names, ["D0"])
        self.assertEqual(sess.opsets, {"": 10, "com.microsoft": 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {"B01": x, "B11": y, "B21": z})[0]
        expected = (x + y) * (y - z)
        assert_allclose(expected, res)

    def test_reference_evaluator_no_attribute_verbose(self):
        m = TestReferenceEvaluator._load_model(TestReferenceEvaluator.m2_def)
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)

        with self.subTest(level=2):
            sess = ReferenceEvaluator(m, verbose=2)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = "Add(B01, B11) -> C0\nSub(B11, B21) -> C1\nMul(C0, C1) -> D0\n"
            self.assertEqual(log, out)

        with self.subTest(level=3):
            sess = ReferenceEvaluator(m, verbose=3)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = dedent(
                """
                 +I B01: float32:(2, 2) in [0.0, 3.0]
                 +I B11: float32:(2, 2) in [4.0, 7.0]
                 +I B21: float32:(2, 2) in [-7.0, -4.0]
                Add(B01, B11) -> C0
                 + C0: float32:(2, 2) in [4.0, 10.0]
                Sub(B11, B21) -> C1
                 + C1: float32:(2, 2) in [8.0, 14.0]
                Mul(C0, C1) -> D0
                 + D0: float32:(2, 2) in [32.0, 140.0]
                """
            ).lstrip("\n")
            self.assertEqual(log, out)

        with self.subTest(level=4):
            sess = ReferenceEvaluator(m, verbose=4)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = dedent(
                """
                 +I B01: float32:(2, 2):[0.0, 1.0, 2.0, 3.0]
                 +I B11: float32:(2, 2):[4.0, 5.0, 6.0, 7.0]
                 +I B21: float32:(2, 2):[-4.0, -5.0, -6.0, -7.0]
                Add(B01, B11) -> C0
                 + C0: float32:(2, 2):[4.0, 6.0, 8.0, 10.0]
                Sub(B11, B21) -> C1
                 + C1: float32:(2, 2):[8.0, 10.0, 12.0, 14.0]
                Mul(C0, C1) -> D0
                 + D0: float32:(2, 2):[32.0, 60.0, 96.0, 140.0]
                """
            ).lstrip("\n")
            self.assertEqual(log, out)

        with self.subTest(level=15):
            sess = ReferenceEvaluator(m, verbose=15)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = dedent(
                """
                 +I B01: float32:(2, 2):[0.0, 1.0, 2.0, 3.0]
                 +I B11: float32:(2, 2):[4.0, 5.0, 6.0, 7.0]
                 +I B21: float32:(2, 2):[-4.0, -5.0, -6.0, -7.0]
                Add(B01, B11) -> C0
                -- begin Add.run(2 inputs)
                -- done Add.run -> 1 outputs
                 + C0: float32:(2, 2):[4.0, 6.0, 8.0, 10.0]
                Sub(B11, B21) -> C1
                -- begin Sub.run(2 inputs)
                -- done Sub.run -> 1 outputs
                 + C1: float32:(2, 2):[8.0, 10.0, 12.0, 14.0]
                Mul(C0, C1) -> D0
                -- begin Mul.run(2 inputs)
                -- done Mul.run -> 1 outputs
                 + D0: float32:(2, 2):[32.0, 60.0, 96.0, 140.0]
                """
            ).lstrip("\n")
            self.assertEqual(log, out)

    def test_reference_evaluator_lr(self):
        lr, f = TestReferenceEvaluator._linear_regression()
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        a = np.array([1, 1], dtype=np.float32)
        b = np.array([11], dtype=np.float32)
        expected = f(x, a, b)
        sess = ReferenceEvaluator(lr)
        got = sess.run(None, {"X": a, "A": a, "B": b})[0]
        assert_allclose(expected, got)

    def test_reference_evaluator_lr_clip(self):
        with self.subTest(opt="min+max"):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="max"):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, min_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="min"):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, max_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

    def test_reference_evaluator_lr_clip_6(self):
        with self.subTest(opt="min+max"):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, opset=10)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.min, -1)
            self.assertEqual(last_node.max, 1)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="max"):
            lr, f = TestReferenceEvaluator._linear_regression(
                clip=True, opset=10, min_value=None
            )
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.max, 1)
            self.assertEqual(last_node.min, -3.4028234663852886e38)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="min"):
            lr, f = TestReferenceEvaluator._linear_regression(
                clip=True, opset=10, max_value=None
            )
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.min, -1)
            self.assertEqual(last_node.max, 3.4028234663852886e38)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

    def test_nested_local_functions(self):
        m = parser.parse_model(
            """
            <
              ir_version: 8,
              opset_import: [ "" : 14, "local" : 1],
              producer_name: "test",
              producer_version: "1.0",
              model_version: 1,
              doc_string: "Test preprocessing model"
            >
            agraph (uint8[H, W, C] x) => (uint8[H, W, C] x_processed)
            {
                x_processed = local.func(x)
            }

            <
              opset_import: [ "" : 14 ],
              domain: "local",
              doc_string: "function 1"
            >
            f1 (x) => (y) {
                y = Identity(x)
            }

            <
              opset_import: [ "" : 14 ],
              domain: "local",
              doc_string: "function 2"
            >
            f2 (x) => (y) {
                y = Identity(x)
            }

            <
              opset_import: [ "" : 14, "local" : 1 ],
              domain: "local",
              doc_string: "Preprocessing function."
            >
            func (x) => (y) {
                x1 = local.f1(x)
                y = local.f2(x1)
            }
        """
        )

        sess = ReferenceEvaluator(m)
        x = np.array([0, 1, 3], dtype=np.uint8).reshape((1, 1, 3))
        result = sess.run(None, {"x": x})[0]
        expected = x
        assert_allclose(expected, result)

    def test_reduce_sum_11(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X"], ["Y"], axes=[1], keepdims=1)
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 11)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        expected = x.sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_square_11(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSumSquare", ["X"], ["Y"], axes=[1], keepdims=1)
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 11)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        expected = (x * x).sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_13(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X", "A"], ["Y"], keepdims=1)
        graph = make_graph([node1], "rs", [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([1], dtype=np.int64)
        expected = x.sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_attribute(self):
        opset = onnx_opset_version()
        new_domain = "custom"
        opset_imports = [make_opsetid("", opset), make_opsetid(new_domain, 1)]

        node = make_node("ReduceSum", ["X", "axis"], ["Y"])
        att = AttributeProto()
        att.name = "keepdims"
        att.ref_attr_name = "keepdims"
        att.type = AttributeProto.INT
        node.attribute.append(att)

        my_reduce_sum = make_function(
            new_domain,
            "MyReduceSum",
            ["X", "axis"],
            ["Y"],
            [node],
            opset_imports,
            ["keepdims"],
        )

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        axis = make_tensor_value_info("axis", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = make_graph(
            [
                make_node(
                    "MyReduceSum",
                    ["X", "axis"],
                    ["Y"],
                    domain=new_domain,
                    keepdims=1,
                ),
            ],
            "example",
            [X, axis],
            [Y],
        )

        onnx_model = make_model(
            graph, opset_imports=opset_imports, functions=[my_reduce_sum]
        )
        sess = ReferenceEvaluator(onnx_model)
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([-1], dtype=np.int64)
        result = sess.run(None, {"X": x, "axis": a})[0]
        expected = x.sum(axis=-1, keepdims=1)
        assert_allclose(expected, result)

    def test_reduce_sum_square_18(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSumSquare", ["X", "A"], ["Y"], keepdims=1)
        graph = make_graph([node1], "rs", [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([1], dtype=np.int64)
        expected = (x * x).sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_13_empty_axes(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X", "A"], ["Y"], keepdims=1)
        graph = make_graph([node1], "rs", [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([], dtype=np.int64)
        expected = x.sum(keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_square_18_empty_axes(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSumSquare", ["X", "A"], ["Y"], keepdims=1)
        graph = make_graph([node1], "rs", [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([], dtype=np.int64)
        expected = (x * x).sum(keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_13_empty_axes_noop(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X"], ["Y"], keepdims=1, noop_with_empty_axes=1)
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(x, got)

    def test_reduce_sum_square_18_empty_axes_noop(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node(
            "ReduceSumSquare", ["X"], ["Y"], keepdims=1, noop_with_empty_axes=1
        )
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(x * x, got)

    def test_greater(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node1 = make_node("Greater", ["X", "Y"], ["Z"])
        graph = make_graph([node1], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x > y
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x, "Y": y})[0]
        assert_allclose(expected, got)

    def test_node_proto(self):
        node1 = make_node("Greater", ["X", "Y"], ["Z"])
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x > y
        sess = ReferenceEvaluator(node1)
        got = sess.run(None, {"X": x, "Y": y})[0]
        assert_allclose(expected, got)

    def test_greater_or_equal(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node1 = make_node("GreaterOrEqual", ["X", "Y"], ["Z"])
        graph = make_graph([node1], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x >= y
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": x, "Y": y})[0]
        assert_allclose(expected, got)

    def test_if(self):
        C = make_tensor_value_info("C", TensorProto.FLOAT, [None])
        bthen = make_node(
            "Constant",
            [],
            ["C"],
            value_floats=from_array(np.array([1], dtype=np.float32)),
        )
        bthen_body = make_graph([bthen], "gthen", [], [C])

        C = make_tensor_value_info("C", TensorProto.FLOAT, [None])
        belse = make_node(
            "Constant",
            [],
            ["C"],
            value_floats=from_array(np.array([0], dtype=np.float32)),
        )
        belse_body = make_graph([belse], "gelse", [], [C])

        zero = from_array(np.array([0], dtype=np.float32), name="zero")
        greater = make_node("Greater", ["X", "zero"], ["G"])
        node_if = make_node(
            "If",
            ["G"],
            ["Z"],
            then_branch=bthen_body,
            else_branch=belse_body,
        )
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        graph = make_graph([greater, node_if], "g", [X], [Z], initializer=[zero])
        model_def = make_model(graph)

        sess = ReferenceEvaluator(model_def)
        self.assertEqual(str(sess), "ReferenceEvaluator(X) -> Z")

        x = np.array([1], dtype=np.float32)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(np.array([1], dtype=np.float32), got)

        x = np.array([-1], dtype=np.float32)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(np.array([0], dtype=np.float32), got)

    def test_if_function(self):
        then_out = make_tensor_value_info("then_out", TensorProto.FLOAT, [5])
        else_out = make_tensor_value_info("else_out", TensorProto.FLOAT, [5])

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

        then_const_node = make_node(
            "Constant", inputs=[], outputs=["then_out"], value=from_array(x)
        )
        else_const_node = make_node(
            "Constant", inputs=[], outputs=["else_out"], value=from_array(y)
        )
        then_body = make_graph([then_const_node], "then_body", [], [then_out])
        else_body = make_graph([else_const_node], "else_body", [], [else_out])
        if_node = make_node(
            "If",
            inputs=["f_cond"],
            outputs=["f_res"],
            then_branch=then_body,
            else_branch=else_body,
        )

        f = FunctionProto()
        f.domain = "custom"
        f.name = "fn"
        f.input.extend(["f_cond"])
        f.output.extend(["f_res"])
        f.node.extend([if_node])
        opset = onnx_opset_version()
        f.opset_import.extend([make_opsetid("", opset)])

        graph = make_graph(
            nodes=[make_node("fn", domain="custom", inputs=["cond"], outputs=["res"])],
            name="graph",
            inputs=[make_tensor_value_info("cond", TensorProto.BOOL, [])],
            outputs=[make_tensor_value_info("res", TensorProto.FLOAT, [5])],
        )

        m = make_model(
            graph,
            producer_name="test",
            opset_imports=[make_opsetid("", opset), make_opsetid("custom", 1)],
        )
        m.functions.extend([f])

        sess = ReferenceEvaluator(m)
        result = sess.run(None, {"cond": np.array(True)})
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        assert_allclose(expected, result[0])

    def test_function_attribute(self):
        opset = onnx_opset_version()
        new_domain = "custom"
        opset_imports = [make_opsetid("", opset), make_opsetid(new_domain, 1)]
        cst = make_node("Constant", [], ["B"])

        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias"
        att.type = AttributeProto.TENSOR
        cst.attribute.append(att)

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        linear_regression = make_function(
            new_domain,
            "LinearRegression",
            ["X", "A"],
            ["Y"],
            [cst, node1, node2],
            opset_imports,
            ["bias"],
        )

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = make_graph(
            [
                make_node(
                    "LinearRegression",
                    ["X", "A"],
                    ["Y1"],
                    domain=new_domain,
                    bias=make_tensor("former_B", TensorProto.FLOAT, [1], [0.67]),
                ),
                make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [X, A],
            [Y],
        )

        onnx_model = make_model(
            graph, opset_imports=opset_imports, functions=[linear_regression]
        )
        sess = ReferenceEvaluator(onnx_model)
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)
        result = sess.run(None, {"X": x, "A": a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_allclose(expected, result)

    def test_function_attribute_nested_graph(self):
        opset = onnx_opset_version()
        new_domain = "custom"
        opset_imports = [make_opsetid("", opset), make_opsetid(new_domain, 1)]

        cst1 = make_node("Constant", [], ["B1"])
        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias1"
        att.type = AttributeProto.TENSOR
        cst1.attribute.append(att)

        cst2 = make_node("Constant", [], ["B2"])
        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias2"
        att.type = AttributeProto.TENSOR
        cst2.attribute.append(att)

        then_out = make_tensor_value_info("B1", TensorProto.FLOAT, [None])
        else_out = make_tensor_value_info("B2", TensorProto.FLOAT, [None])
        then_body = make_graph([cst1], "then_body", [], [then_out])
        else_body = make_graph([cst2], "else_body", [], [else_out])

        zero = make_node(
            "Constant",
            inputs=[],
            outputs=["zero"],
            value=from_array(np.array([0], dtype=np.float32)),
        )
        mini = make_node("ReduceMin", ["X"], ["Xmin"])
        f_cond = make_node("Greater", ["Xmin", "zero"], ["f_cond"])
        if_node = make_node(
            "If",
            inputs=["f_cond"],
            outputs=["B"],
            then_branch=then_body,
            else_branch=else_body,
        )

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        linear_regression = make_function(
            new_domain,
            "LinearRegression",
            ["X", "A"],
            ["Y"],
            [zero, mini, f_cond, if_node, node1, node2],
            opset_imports,
            ["bias1", "bias2"],
        )

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = make_graph(
            [
                make_node(
                    "LinearRegression",
                    ["X", "A"],
                    ["Y1"],
                    domain=new_domain,
                    bias1=make_tensor("former_B1", TensorProto.FLOAT, [1], [0.67]),
                    bias2=make_tensor("former_B2", TensorProto.FLOAT, [1], [777]),
                ),
                make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [X, A],
            [Y],
        )

        onnx_model = make_model(
            graph, opset_imports=opset_imports, functions=[linear_regression]
        )
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)

        self.assertEqual(sess.rt_nodes_[0].__class__.__name__, "OpFunction")
        self.assertEqual(
            sess.rt_nodes_[0].impl_.__class__.__name__, "ReferenceEvaluator"
        )
        fct = sess.rt_nodes_[0].impl_
        checked = False
        for node in fct.rt_nodes_:
            if node.__class__.__name__.startswith("If"):
                if not node.has_linked_attribute:
                    raise AssertionError(
                        f"Nested node {type(node)} declares no linked attribute "
                        f"but a subgraph does."
                    )
                checked = True
        if not checked:
            raise AssertionError(
                "No node 'If' was found, has_linked_attribute could not be checked."
            )

        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)

        result = sess.run(None, {"X": x + 1, "A": a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_allclose(expected, result)

        result = sess.run(None, {"X": x - 10, "A": a})[0]
        expected = np.abs(x @ a + 777)
        assert_allclose(expected, result)

    def test_function_attribute_nested_nested_graph(self):
        opset = onnx_opset_version()
        new_domain = "custom"
        opset_imports = [make_opsetid("", opset), make_opsetid(new_domain, 1)]

        # first If
        cst1 = make_node("Constant", [], ["B1"])
        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias1"
        att.type = AttributeProto.TENSOR
        cst1.attribute.append(att)

        cst2 = make_node("Constant", [], ["B2"])
        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias2"
        att.type = AttributeProto.TENSOR
        cst2.attribute.append(att)

        then_out = make_tensor_value_info("B1", TensorProto.FLOAT, [None])
        else_out = make_tensor_value_info("B2", TensorProto.FLOAT, [None])
        then_body1 = make_graph([cst1], "then_body", [], [then_out])
        else_body1 = make_graph([cst2], "else_body", [], [else_out])

        # sub graph 2
        c100 = make_node(
            "Constant",
            inputs=[],
            outputs=["c100"],
            value=from_array(np.array([100], dtype=np.float32)),
        )
        f_cond = make_node("Greater", ["Xmin", "c100"], ["f_cond_100"])
        if_node = make_node(
            "If",
            inputs=["f_cond_100"],
            outputs=["B4"],
            then_branch=then_body1,
            else_branch=else_body1,
        )

        # second If
        cst3 = make_node("Constant", [], ["B3"])
        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias3"
        att.type = AttributeProto.TENSOR
        cst3.attribute.append(att)

        then_out = make_tensor_value_info("B3", TensorProto.FLOAT, [None])
        then_body2 = make_graph([cst3], "then_body", [], [then_out])
        else_out = make_tensor_value_info("B4", TensorProto.FLOAT, [None])
        else_body2 = make_graph([c100, f_cond, if_node], "else_body", [], [else_out])

        # function
        zero = make_node(
            "Constant",
            inputs=[],
            outputs=["zero"],
            value=from_array(np.array([0], dtype=np.float32)),
        )
        mini = make_node("ReduceMin", ["X"], ["Xmin"])
        f_cond = make_node("Less", ["Xmin", "zero"], ["f_cond_zero"])
        if_node = make_node(
            "If",
            inputs=["f_cond_zero"],
            outputs=["B"],
            then_branch=then_body2,
            else_branch=else_body2,
        )
        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        linear_regression = make_function(
            new_domain,
            "LinearRegression",
            ["X", "A"],
            ["Y"],
            [zero, mini, f_cond, if_node, node1, node2],
            opset_imports,
            ["bias1", "bias2", "bias3"],
        )

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = make_graph(
            [
                make_node(
                    "LinearRegression",
                    ["X", "A"],
                    ["Y1"],
                    domain=new_domain,
                    bias1=make_tensor("former_B1", TensorProto.FLOAT, [1], [0.67]),
                    bias2=make_tensor("former_B2", TensorProto.FLOAT, [1], [777]),
                    bias3=make_tensor("former_B3", TensorProto.FLOAT, [1], [-888]),
                ),
                make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [X, A],
            [Y],
        )
        onnx_model = make_model(
            graph, opset_imports=opset_imports, functions=[linear_regression]
        )
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)

        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)

        result = sess.run(None, {"X": x + 1, "A": a})[0]
        expected = np.abs(x @ a + 777)
        assert_allclose(expected, result)

        result = sess.run(None, {"X": x - 10, "A": a})[0]
        expected = np.abs(x @ a - 888)
        assert_allclose(expected, result)

        result = sess.run(None, {"X": x + 1000, "A": a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_allclose(expected, result)

    def test_custom_node(self):
        class _InvAlpha:
            op_domain = "custom"

            def __init__(self, onnx_node, run_params):  # type: ignore
                self.onnx_node = onnx_node
                self.run_params = run_params

            def _run(self, x):  # type: ignore
                return (1 / (x + self.alpha),)

        class InvAlpha2(OpRun):
            def _run(self, x):  # type: ignore
                return (1 / (x + self.alpha),)

        class InvAlpha(OpRun):
            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore
                alpha = alpha or self.alpha  # type: ignore
                return (1 / (x + alpha),)

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        with self.assertRaises(NotImplementedError):
            ReferenceEvaluator(onnx_model)

        node1 = make_node("_InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        with self.assertRaises(TypeError):
            ReferenceEvaluator(onnx_model, new_ops=[_InvAlpha])

        node1 = make_node("InvAlpha2", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        with self.assertRaises(NotImplementedError):
            ReferenceEvaluator(onnx_model, new_ops=[InvAlpha2])

        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        sess = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha, InvAlpha])
        got = sess.run(None, {"X": x})[0]
        expected = 1 / (x + 0.5)
        assert_allclose(expected, got)

    def test_custom_no_output_tuple(self):
        class InvAlpha(OpRun):
            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore
                alpha = alpha or self.alpha  # type: ignore
                return 1 / (x + alpha)

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(TypeError):
            ref.run(None, {"X": x})

    def test_custom_empty_output(self):
        class InvAlpha(OpRun):
            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore  # noqa: ARG002
                return tuple()

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(ValueError):
            ref.run(None, {"X": x})

    def test_custom_tuple_tuple(self):
        class InvAlpha(OpRun):
            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore
                alpha = alpha or self.alpha  # type: ignore
                res = tuple([tuple([1 / (x + alpha)])])  # noqa: C409
                assert isinstance(res, tuple)
                assert isinstance(res[0], tuple)
                return res

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(TypeError):
            ref.run(None, {"X": x})

    def test_custom_tuple_unexpected_type(self):
        class CustomType:
            pass

        class InvAlpha(OpRun):
            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore  # noqa: ARG002
                res = tuple([CustomType()])  # noqa: C409
                assert isinstance(res, tuple)
                assert isinstance(res[0], CustomType)
                return res

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(TypeError):
            ref.run(None, {"X": x})

    def test_loop(self):
        # Given a tensor x of values [x1, ..., xN],
        # Return a sequence of tensors of
        #   [[x1], [x1, x2], ..., [x1, ..., xN]]

        cond_in = make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = make_tensor_value_info("iter_count", TensorProto.INT64, [])
        seq_in = make_tensor_sequence_value_info("seq_in", TensorProto.FLOAT, None)
        seq_out = make_tensor_sequence_value_info("seq_out", TensorProto.FLOAT, None)

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        x_const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=make_tensor(
                name="const_tensor_x",
                data_type=TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            ),
        )

        one_const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=make_tensor(
                name="const_tensor_one",
                data_type=TensorProto.INT64,
                dims=(),
                vals=[1],
            ),
        )

        zero_const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["slice_start"],
            value=make_tensor(
                name="const_tensor_zero",
                data_type=TensorProto.INT64,
                dims=(1,),
                vals=[0],
            ),
        )

        axes_node = make_node(
            "Constant",
            inputs=[],
            outputs=["axes"],
            value=make_tensor(
                name="const_tensor_axes",
                data_type=TensorProto.INT64,
                dims=(),
                vals=[0],
            ),
        )

        add_node = make_node("Add", inputs=["iter_count", "one"], outputs=["end"])

        end_unsqueeze_node = make_node(
            "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
        )

        slice_node = make_node(
            "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
        )

        insert_node = make_node(
            "SequenceInsert", inputs=["seq_in", "slice_out"], outputs=["seq_out"]
        )

        identity_node = make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

        loop_body = make_graph(
            [
                identity_node,
                x_const_node,
                one_const_node,
                zero_const_node,
                add_node,
                axes_node,
                end_unsqueeze_node,
                slice_node,
                insert_node,
            ],
            "loop_body",
            [iter_count, cond_in, seq_in],
            [cond_out, seq_out],
        )

        node = make_node(
            "Loop",
            inputs=["trip_count", "cond", "seq_empty"],
            outputs=["seq_res"],
            body=loop_body,
        )
        node_concat = make_node(
            "ConcatFromSequence",
            inputs=["seq_res"],
            outputs=["res"],
            axis=0,
            new_axis=0,
        )

        trip_count = np.array(5).astype(np.int64)
        seq_empty = []  # type: List[Any]
        # seq_res = [x[:int(i)] for i in x]
        cond = np.array(1).astype(np.bool_)

        model_def = make_model(
            graph=make_graph(
                name="loop_test",
                inputs=[
                    make_tensor_value_info(
                        "trip_count", TensorProto.INT64, trip_count.shape
                    ),
                    make_tensor_value_info("cond", TensorProto.BOOL, cond.shape),
                    make_sequence_value_info("seq_empty", TensorProto.FLOAT, []),
                ],
                outputs=[make_tensor_value_info("res", TensorProto.FLOAT, None)],
                nodes=[node, node_concat],
            )
        )

        expected = np.array(
            [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            dtype=np.float32,
        )
        oinf = ReferenceEvaluator(model_def)
        inputs = {"trip_count": trip_count, "cond": cond, "seq_empty": seq_empty}
        got = oinf.run(None, inputs)
        assert_allclose(expected, got[0])

    def test_onnxt_runtime_bernoulli(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("Bernoulli", ["X"], ["Y"], seed=0.0)
        graph = make_graph([node1], "g", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": np.zeros((2, 4), dtype=np.float32) + 0.5})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), -1e-5)
        self.assertLess(got.max(), 1 + 1e-5)

    def test_onnxt_runtime_random_uniform(self):
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomUniform", [], ["Y"], seed=0.0, shape=[2, 4])
        graph = make_graph([node1], "g", [], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), 0)
        self.assertLess(got.max(), 1)

    def test_onnxt_runtime_random_uniform_like(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomUniformLike", ["X"], ["Y"], seed=0.0)
        graph = make_graph([node1], "g", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": np.zeros((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), 0)
        self.assertLess(got.max(), 1)

    def test_onnxt_runtime_random_normal(self):
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomNormal", [], ["Y"], seed=0.0, shape=[2, 4])
        graph = make_graph([node1], "g", [], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)

    def test_onnxt_runtime_random_normal_like(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomNormalLike", ["X"], ["Y"], seed=0.0)
        graph = make_graph([node1], "g", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {"X": np.zeros((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)

    def test_eval_celu(self):
        inst = Celu.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Celu.eval(x, alpha=0.5)
        expected = _vcelu1(x, alpha=0.5)
        assert_allclose(expected, y)

    def test_eval_cast(self):
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Cast_19.eval(x, to=TensorProto.FLOAT8E4M3FN)
        dy = Cast_19.eval(y, to=TensorProto.FLOAT)
        expected = x
        assert_allclose(expected, dy)

    def test_eval_celu_load_op(self):
        celu = load_op("", "Celu")
        self.assertEqual(celu.op_domain, "")
        inst = celu.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = celu.eval(x, alpha=0.5)
        expected = _vcelu1(x, alpha=0.5)
        assert_allclose(expected, y)

    def test_create_adam(self):
        inst = Adam.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)

    @skip_if_no_onnxruntime
    def test_conv(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
        node = make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            pads=[1, 1, 1, 1],
            dilations=[1, 1],
            strides=[2, 2],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model_gen_version(graph, opset_imports=[make_opsetid("", 16)])
        sess1 = run_ort_inference(onnx_model)
        if sess1 is None:
            return
        sess2 = ReferenceEvaluator(onnx_model, optimized=False)
        self.assertIsInstance(sess2.rt_nodes_[0], Conv)
        sess3 = ReferenceEvaluator(onnx_model, new_ops=[ConvOptimized], optimized=False)
        self.assertIsInstance(sess3.rt_nodes_[0], ConvOptimized)
        sess4 = ReferenceEvaluator(onnx_model, optimized=True)
        self.assertIsInstance(sess4.rt_nodes_[0], ConvOptimized)

        sH, sW = 5, 6
        for i in range(sH):
            for j in range(sW):
                X = np.zeros((1, 1, sH, sW), dtype=np.float32)
                X[0, 0, i, j] = 1.0
                W = np.zeros((1, 1, 3, 3), dtype=np.float32)
                W[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 256)

                B = np.array([[[[0]]]], dtype=np.float32)
                expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
                got = sess2.run(None, {"X": X, "W": W, "B": B})[0]
                assert_allclose(expected, got)
                got3 = sess3.run(None, {"X": X, "W": W, "B": B})[0]
                assert_allclose(expected, got3)
                got4 = sess4.run(None, {"X": X, "W": W, "B": B})[0]
                assert_allclose(expected, got4)

    @skip_if_no_onnxruntime
    def test_qlinearconv(self):
        x = make_tensor_value_info("x", TensorProto.UINT8, [None, None, None, None])
        w = make_tensor_value_info("w", TensorProto.UINT8, [None, None, None, None])
        y = make_tensor_value_info("y", TensorProto.UINT8, [None, None, None, None])
        x_scale = make_tensor_value_info("x_scale", TensorProto.FLOAT, [None])
        w_scale = make_tensor_value_info("w_scale", TensorProto.FLOAT, [None])
        y_scale = make_tensor_value_info("y_scale", TensorProto.FLOAT, [None])
        x_zero_point = make_tensor_value_info("x_zero_point", TensorProto.UINT8, [None])
        w_zero_point = make_tensor_value_info("w_zero_point", TensorProto.UINT8, [None])
        y_zero_point = make_tensor_value_info("y_zero_point", TensorProto.UINT8, [None])

        node = make_node(
            "QLinearConv",
            [
                "x",
                "x_scale",
                "x_zero_point",
                "w",
                "w_scale",
                "w_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            ["y"],
        )
        graph = make_graph(
            [node],
            "g",
            [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point],
            [y],
        )
        onnx_model = make_model_gen_version(graph, opset_imports=[make_opsetid("", 16)])

        sess1 = run_ort_inference(onnx_model)
        if sess1 is None:
            return
        sess2 = ReferenceEvaluator(onnx_model)

        sH, sW = 3, 3
        for i in range(sH):
            for j in range(sW):
                x = np.zeros((1, 1, sH, sW), dtype=np.uint8)
                x[0, 0, i, j] = 1.0
                with self.subTest(w="1x1", i=i, j=j):
                    w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                    w[0, 0, :, :] = 1
                    feeds = {
                        "x": x,
                        "x_scale": np.array([1], dtype=np.float32),
                        "x_zero_point": np.array([0], dtype=np.uint8),
                        "w": w,
                        "w_scale": np.array([1], dtype=np.float32),
                        "w_zero_point": np.array([0], dtype=np.uint8),
                        "y_scale": np.array([1], dtype=np.float32),
                        "y_zero_point": np.array([0], np.uint8),
                    }
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    assert_allclose(expected, got)
                with self.subTest(w="3x3", i=i, j=j):
                    w = np.zeros((1, 1, 3, 3), dtype=np.uint8)
                    w[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 128)
                    feeds = {
                        "x": x,
                        "x_scale": np.array([1], dtype=np.float32),
                        "x_zero_point": np.array([0], dtype=np.uint8),
                        "w": w,
                        "w_scale": np.array([1], dtype=np.float32),
                        "w_zero_point": np.array([0], dtype=np.uint8),
                        "y_scale": np.array([1], dtype=np.float32),
                        "y_zero_point": np.array([0], np.uint8),
                    }
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    assert_allclose(expected, got)
                with self.subTest(w="1x1", i=i, j=j):
                    w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                    w[0, 0, :, :] = 0
                    feeds = {
                        "x": x,
                        "x_scale": np.array([0.00369204697], dtype=np.float32),
                        "x_zero_point": np.array([132], dtype=np.uint8),
                        "w": w,
                        "w_scale": np.array([100.001727945750], dtype=np.float32),
                        "w_zero_point": np.array([255], dtype=np.uint8),
                        "y_scale": np.array([0.00162681262], dtype=np.float32),
                        "y_zero_point": np.array([132], np.uint8),
                    }
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    assert_allclose(expected, got)

        x = np.array(
            [
                [255, 174, 162, 25, 203, 168, 58],
                [15, 59, 237, 95, 129, 0, 64],
                [56, 242, 153, 221, 168, 12, 166],
                [232, 178, 186, 195, 237, 162, 237],
                [188, 39, 124, 77, 80, 102, 43],
                [127, 230, 21, 83, 41, 40, 134],
                [255, 154, 92, 141, 42, 148, 247],
            ],
            dtype=np.uint8,
        ).reshape((1, 1, 7, 7))
        x_scale = np.array([0.00369204697], dtype=np.float32)
        x_zero_point = np.array([132], dtype=np.uint8)
        w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))
        w_scale = np.array([0.00172794575], dtype=np.float32)
        w_zero_point = np.array([255], dtype=np.uint8)
        y_scale = np.array([0.00162681262], dtype=np.float32)
        y_zero_point = np.array([123], dtype=np.uint8)

        feeds = {
            "x": x,
            "x_scale": x_scale,
            "x_zero_point": x_zero_point,
            "w": w,
            "w_scale": w_scale,
            "w_zero_point": w_zero_point,
            "y_scale": y_scale,
            "y_zero_point": y_zero_point,
        }
        expected = sess1.run(None, feeds)[0]
        got = sess2.run(None, feeds)[0]
        assert_allclose(expected, got)

    def common_test_im2col(self, kernel_shape, pads, strides, dilations):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        Y1 = make_tensor_value_info("Y1", TensorProto.FLOAT, [None, None, None, None])
        Y2 = make_tensor_value_info("Y2", TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
        node = make_node(
            "Conv", ["X", "W"], ["Y1"], pads=pads, strides=strides, dilations=dilations
        )
        node_shape = make_node("Shape", ["W"], ["shape"])
        node_im = make_node(
            "Im2Col",
            ["X", "shape"],
            ["xim"],
            pads=pads,
            strides=strides,
            dilations=dilations,
            domain="experimental",
        )
        node_flat = make_node("Flatten", ["W"], ["wflat"])
        node_gem = make_node("MatMul", ["wflat", "xim"], ["Y2"])
        graph = make_graph(
            [node, node_shape, node_im, node_flat, node_gem],
            "g",
            [X, W],
            [Y1, Y2],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 16), make_opsetid("experimental", 1)]
        )
        graph_conv = make_graph([node], "g", [X, W], [Y1])
        onnx_model_conv = make_model_gen_version(
            graph_conv, opset_imports=[make_opsetid("", 16)]
        )
        sess = ReferenceEvaluator(onnx_model)

        try:
            sess_conv = run_ort_inference(onnx_model_conv)
            if sess_conv is None:
                return
        except ImportError:
            sess_conv = None

        sH, sW = 7, 7
        nker = np.prod(kernel_shape)
        for i in range(sH):
            for j in range(sW):
                X = np.zeros((1, 1, sH, sW), dtype=np.float32)
                X[0, 0, i, j] = 1.0
                W = np.zeros(
                    (1, 1, *kernel_shape),
                    dtype=np.float32,
                )
                W[0, 0, :, :] = np.minimum(
                    2 ** np.arange(nker).reshape((kernel_shape[0], -1)), 256
                )

                got = sess.run(None, {"X": X, "W": W})
                if sess_conv is not None:
                    ort_res = sess_conv.run(None, {"X": X, "W": W})[0]
                    assert_allclose(got[1].ravel(), ort_res.ravel())
                try:
                    assert_allclose(got[0].ravel(), got[1].ravel())
                except AssertionError as e:
                    raise AssertionError(
                        f"Discrepancies: pads={pads}, dilations={dilations}, strides={strides}, "
                        f"kernel_shape={kernel_shape}"
                        f"\n{got[0]}\n!=\n{got[1]}"
                    ) from e

    def test_im2col_1x1(self):
        self.common_test_im2col(
            (1, 1), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_2x2(self):
        self.common_test_im2col(
            (2, 2), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_3x3(self):
        self.common_test_im2col(
            (3, 3), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_3x3_pads(self):
        self.common_test_im2col(
            (3, 3), pads=[0, 1, 2, 3], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_3x3_strides(self):
        self.common_test_im2col(
            (3, 3), pads=[0, 1, 1, 1], strides=[1, 2], dilations=[1, 1]
        )

    def test_im2col_5x5(self):
        self.common_test_im2col(
            (5, 5), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    @skip_if_no_torch
    def test_col2im(self):
        import torch

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info("I", TensorProto.INT64, [None])
        BS = make_tensor_value_info("B", TensorProto.INT64, [None])
        node = make_node(
            "Col2Im",
            ["X", "I", "B"],
            ["Y"],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
        )
        graph = make_graph([node], "g", [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        sess = ReferenceEvaluator(onnx_model)

        X = np.array(
            [
                [
                    [1.0, 6.0, 11.0, 16.0, 21.0],
                    [2.0, 7.0, 12.0, 17.0, 22.0],
                    [3.0, 8.0, 13.0, 18.0, 23.0],
                    [4.0, 9.0, 14.0, 19.0, 24.0],
                    [5.0, 0.0, 15.0, 20.0, 25.0],
                ]
            ]
        ).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)

        fold = torch.nn.Fold(output_size=tuple(image_shape), kernel_size=block_shape)

        got = sess.run(None, {"X": X, "B": block_shape, "I": image_shape})
        output = fold(torch.from_numpy(X)).numpy()
        assert_allclose(output, got[0])

    def common_test_col2im(
        self, size, image_shape, block_shape, pads, strides, dilations
    ):
        import torch

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info("I", TensorProto.INT64, [None])
        BS = make_tensor_value_info("B", TensorProto.INT64, [None])
        node = make_node(
            "Col2Im",
            ["X", "I", "B"],
            ["Y"],
            pads=pads,
            strides=strides,
            dilations=dilations,
        )
        graph = make_graph([node], "g", [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        sess = ReferenceEvaluator(onnx_model)

        fold = torch.nn.Fold(
            output_size=tuple(image_shape),
            kernel_size=tuple(block_shape),
            dilation=tuple(dilations),
            padding=min(pads),
            stride=tuple(strides),
        )

        nker = np.prod(block_shape)
        for i in range(nker):
            for j in range(size):
                X = np.zeros((1, nker, size), dtype=np.float32)
                X[0, i, j] = 1.0
                i_shape = np.array(image_shape, dtype=np.int64)
                b_shape = np.array(block_shape, dtype=np.int64)

                output = fold(torch.from_numpy(X)).numpy()
                got = sess.run(None, {"X": X, "B": b_shape, "I": i_shape})
                # print(output)
                # print(got)
                assert_allclose(output, got[0])

    @skip_if_no_torch
    def test_col2im_2x3(self):
        self.common_test_col2im(
            10, (6, 4), (2, 3), pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1]
        )

    @skip_if_no_torch
    def test_col2im_2x3_pads(self):
        self.common_test_col2im(
            28, (6, 4), (2, 3), pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1]
        )

    def test_col2im_2d(self):
        data = np.zeros([6, 28], dtype=np.float32)
        data[0][0] = 1.0
        image_shape, kernel_shape, dilations, pads, stride = (
            np.array([6, 4]),
            (2, 3),
            np.array([1, 1]),
            np.array([1, 1, 1, 1]),
            np.array([1, 1]),
        )
        r1 = _col2im_naive_implementation_2d(
            data, image_shape, kernel_shape, dilations, pads, stride
        )
        r2 = col2im_naive_implementation(
            data, image_shape, kernel_shape, dilations, pads, stride
        )
        assert_allclose(r1, r2)

    def test_conv_im2col_group4(self):
        # model 1
        X = make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 6, 6])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [4, 1, 3, 3])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [4])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 6, 6])

        node = make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            group=4,
            dilations=[1, 1],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

        feeds = {
            "X": np.arange(2 * 4 * 6 * 6).reshape((2, 4, 6, 6)).astype(np.float32),
            "W": np.array(
                [
                    [
                        [
                            [
                                -0.026239916682243347,
                                0.07565222680568695,
                                -0.03209298849105835,
                            ],
                            [
                                -0.08708783239126205,
                                0.0961190015077591,
                                0.13418219983577728,
                            ],
                            [
                                0.1598859578371048,
                                0.03840477764606476,
                                -0.13170936703681946,
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                -0.0689004510641098,
                                0.1408083587884903,
                                -0.03717087209224701,
                            ],
                            [
                                0.030967697501182556,
                                0.0263785719871521,
                                -0.0899493545293808,
                            ],
                            [
                                0.07828782498836517,
                                -0.06266771256923676,
                                0.10750330984592438,
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                0.020227551460266113,
                                -0.04353883117437363,
                                -0.10938453674316406,
                            ],
                            [
                                -0.14101561903953552,
                                -0.03393106162548065,
                                0.12139306962490082,
                            ],
                            [
                                0.02838282287120819,
                                0.13864465057849884,
                                -0.06065710633993149,
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                -0.06511610746383667,
                                -0.05987360328435898,
                                -0.008047685027122498,
                            ],
                            [
                                0.07340313494205475,
                                0.0326494425535202,
                                0.012516498565673828,
                            ],
                            [
                                0.13260947167873383,
                                -0.022225692868232727,
                                -0.11167611926794052,
                            ],
                        ]
                    ],
                ],
                dtype=np.float32,
            ),
            "B": np.array(
                [
                    -0.1457933485507965,
                    -0.07481209933757782,
                    -0.05890338122844696,
                    -0.11964251846075058,
                ],
                dtype=np.float32,
            ),
        }
        feeds["B"][:] = 0

        # model 2
        X = feeds["X"]
        W = feeds["W"]
        B = feeds["B"]
        Y = np.empty((2, 4, 6, 6), dtype=X.dtype)
        for b in range(X.shape[0]):
            for g in range(4):
                x = X[b : b + 1, g : g + 1]
                w = W[g]
                c2 = im2col(x, (3, 3), [1, 1], [1, 1, 1, 1], [1, 1])
                mul = np.matmul(c2, w.flatten())
                mul = mul + B[g]
                Y[b, g, :, :] = mul

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)

        assert_allclose(Y, got1[0], atol=1e-5)

    def test_conv_strides(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 6, 6])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [2, 3, 3, 3])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [2])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            group=1,
            dilations=[1, 1],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

        feeds = {
            "X": np.arange(1 * 3 * 6 * 6).reshape((1, 3, 6, 6)).astype(np.float32) + 1,
            "W": np.zeros((2, 3, 3, 3), dtype=np.float32),
            "B": np.zeros((2,), dtype=np.float32),
        }
        feeds["W"][0, 0, 0, 1] = 1

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [7.0, 9.0, 11.0], [19.0, 21.0, 23.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ],
            dtype=np.float32,
        )

        assert_allclose(expected, got1[0])

    def test_max_pool_2d_1(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node(
            "MaxPool",
            ["X"],
            ["Y"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )
        graph = make_graph([node], "g", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

        feeds = {"X": np.arange(49)[::-1].reshape((1, 1, 7, 7)).astype(np.float32)}
        expected = np.array(
            [
                [
                    [
                        [48.0, 47.0, 45.0, 43.0],
                        [41.0, 40.0, 38.0, 36.0],
                        [27.0, 26.0, 24.0, 22.0],
                        [13.0, 12.0, 10.0, 8.0],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    def test_max_pool_2d_2(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node(
            "MaxPool",
            ["X"],
            ["Y"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )
        graph = make_graph([node], "g", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

        feeds = {
            "X": np.array(
                [
                    [
                        [
                            [683, 358, 726, 578, 650, 946, 200],
                            [679, 260, 264, 5, 240, 255, 582],
                            [322, 66, 687, 632, 852, 698, 428],
                            [111, 452, 627, 332, 751, 842, 685],
                            [472, 52, 956, 81, 807, 827, 360],
                            [972, 574, 81, 799, 646, 499, 486],
                            [892, 758, 75, 833, 972, 415, 736],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        }
        expected = np.array(
            [
                [
                    [
                        [683.0, 726.0, 946.0, 946.0],
                        [679.0, 687.0, 852.0, 842.0],
                        [972.0, 956.0, 842.0, 842.0],
                        [972.0, 833.0, 972.0, 736.0],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    def test_scatter_elements(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Ind = make_tensor_value_info("I", TensorProto.INT64, [None, None])
        U = make_tensor_value_info("U", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

        node = make_node(
            "ScatterElements",
            ["X", "I", "U"],
            ["Y"],
            axis=1,
            reduction="min",
        )
        graph = make_graph([node], "g", [X, Ind, U], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        feeds = {
            "X": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
            "I": np.array([[1, 1]]),
            "U": np.array([[1.1, 2.1]], dtype=np.float32),
        }

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([[1.0, 1.1, 3.0, 4.0, 5.0]], dtype=np.float32)
        assert_allclose(expected, got1[0])

    def test_scatternd(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Ind = make_tensor_value_info("I", TensorProto.INT64, [None, None])
        U = make_tensor_value_info("U", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

        node = make_node(
            "ScatterND",
            ["X", "I", "U"],
            ["Y"],
        )
        graph = make_graph([node], "g", [X, Ind, U], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        feeds = {
            "X": np.array([[1.0, 2.0]], dtype=np.float32),
            "I": np.array([[0, 0]]),
            "U": np.array([3.0], dtype=np.float32),
        }

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([[3.0, 2.0]], dtype=np.float32)
        assert_allclose(expected, got1[0])

    def test_col2im_impl(self):
        def get_im2col_indices(
            x_shape, field_height, field_width, padding=None, stride=1
        ):
            # source: https://stackoverflow.com/questions/51703367/col2im-implementation-in-convnet
            N, C, H, W = x_shape
            del N  # Unused
            assert (H + padding[0] + padding[2] - field_height) % stride == 0
            assert (W + padding[1] + padding[3] - field_height) % stride == 0
            out_height = (H + padding[0] + padding[2] - field_height) // stride + 1
            out_width = (W + padding[1] + padding[3] - field_width) // stride + 1

            i0 = np.repeat(np.arange(field_height), field_width)
            i0 = np.tile(i0, C)
            i1 = stride * np.repeat(np.arange(out_height), out_width)
            j0 = np.tile(np.arange(field_width), field_height * C)
            j1 = stride * np.tile(np.arange(out_width), out_height)
            i = i0.reshape(-1, 1) + i1.reshape(1, -1)
            j = j0.reshape(-1, 1) + j1.reshape(1, -1)

            k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

            return (k, i, j)

        def col2im_indices(
            cols, x_shape, field_height=3, field_width=3, padding=None, stride=1
        ):
            # source: https://stackoverflow.com/questions/51703367/col2im-implementation-in-convnet
            N, C, H, W = x_shape
            H_padded, W_padded = (
                H + padding[0] + padding[2],
                W + padding[1] + padding[3],
            )
            x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
            k, i, j = get_im2col_indices(
                x_shape, field_height, field_width, padding, stride
            )
            cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
            cols_reshaped = cols_reshaped.transpose(2, 0, 1)
            np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
            padding = padding.copy()
            if padding[2] == 0:
                padding[2] += x_padded.shape[2]
            elif padding[2] > 0:
                padding[2] *= -1
            if padding[3] == 0:
                padding[3] += x_padded.shape[3]
            elif padding[3] > 0:
                padding[3] *= -1
            res = x_padded[:, :, padding[0] : padding[2], padding[1] : padding[3]]
            return res

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info("IS", TensorProto.INT64, [None])
        BS = make_tensor_value_info("BS", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node("Col2Im", ["X", "IS", "BS"], ["Y"], pads=[0, 1, 0, 1])
        graph = make_graph([node], "g", [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        feeds = {
            "X": np.arange(5 * 15).astype(np.float32).reshape((1, 5, 15)),
            "IS": np.array([5, 5]),
            "BS": np.array([1, 5]),
        }

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = col2im_indices(
            feeds["X"],
            (1, 1, 5, 5),
            field_height=1,
            field_width=5,
            padding=[0, 1, 0, 1],
        )
        assert_allclose(expected, got1[0])

    def test_conv_transpose_2d(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node(
            "ConvTranspose",
            ["X", "W", "B"],
            ["Y"],
            dilations=[1, 1],
            kernel_shape=[3, 3],
            output_padding=[0, 0],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        feeds = {
            "X": np.arange(1 * 3 * 5 * 4).reshape((1, 3, 5, 4)).astype(np.float32),
            "W": np.arange(3 * 1 * 3 * 3).reshape((3, 1, 3, 3)).astype(np.float32),
            "B": np.array([0, 0, 0, 0], dtype=np.float32),
        }

        # import torch
        # ex = torch.nn.functional.conv_transpose2d(
        #     torch.Tensor(feeds["X"]), torch.Tensor(feeds["W"]),
        #     bias=None, stride=1, padding=1, output_padding=0, groups=1, dilation=1)
        # print(ex)

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array(
            [
                [
                    [
                        [4371, 6855, 7062, 4929],
                        [7524, 11781, 12132, 8451],
                        [8424, 13185, 13536, 9423],
                        [9324, 14589, 14940, 10395],
                        [7197, 11229, 11490, 7971],
                    ],
                ]
            ],
            dtype=np.float32,
        )
        assert_allclose(expected, got1[0])

        feeds["X"] *= 0
        feeds["X"][0, 0, 0, 0] = 1

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array(
            [
                [
                    [
                        [4, 5, 0, 0],
                        [7, 8, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        assert_allclose(expected, got1[0])

    def test_conv_transpose_2d_upper(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node(
            "ConvTranspose",
            ["X", "W", "B"],
            ["Y"],
            auto_pad="SAME_UPPER",
            strides=[2, 2],
            # output_shape=[6, 6],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        feeds = {
            "X": np.arange(1 * 1 * 3 * 3).reshape((1, 1, 3, 3)).astype(np.float32),
            "W": np.arange(1 * 2 * 3 * 3).reshape((1, 2, 3, 3)).astype(np.float32),
            "B": np.array([0, 0, 0, 0], dtype=np.float32),
        }

        expected = np.array(
            [
                [
                    [
                        [0, 0, 0, 1, 2, 2],
                        [0, 0, 3, 4, 11, 8],
                        [0, 3, 12, 11, 28, 19],
                        [9, 12, 27, 16, 35, 20],
                        [18, 27, 60, 35, 76, 43],
                        [18, 24, 51, 28, 59, 32],
                    ],
                    [
                        [0, 0, 9, 10, 29, 20],
                        [0, 0, 12, 13, 38, 26],
                        [27, 30, 84, 56, 136, 82],
                        [36, 39, 90, 52, 116, 65],
                        [99, 108, 240, 134, 292, 160],
                        [72, 78, 168, 91, 194, 104],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        # import onnxruntime
        # ref0 = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
        # got0 = ref0.run(None, feeds)

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    @unittest.skipIf(
        version_utils.numpy_older_than("1.21.5"),
        "op_dft and op_stft requires numpy >= 1.21.5",
    )
    def test_stft(self):
        signal = make_tensor_value_info("signal", TensorProto.FLOAT, [None, None, None])
        frame_step = make_tensor_value_info("frame_step", TensorProto.INT64, [None])
        frame_length = make_tensor_value_info("frame_length", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node(
            "STFT",
            ["signal", "frame_step", "", "frame_length"],
            ["Y"],
        )
        graph = make_graph([node], "g", [signal, frame_step, frame_length], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 17)])
        feeds = {
            "signal": np.arange(128).reshape((1, 128, 1)).astype(np.float32),
            "frame_step": np.array(8, dtype=np.int64),
            "frame_length": np.array(16, dtype=np.int64),
        }

        signal = feeds["signal"]
        frame_length = int(feeds["frame_length"])
        frame_step = int(feeds["frame_step"])
        onesided_length = (frame_length // 2) + 1
        nstfts = ((feeds["signal"].shape[1] - frame_length) // frame_step) + 1
        # [batch_size][frames][frame_length][2]
        expected = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
        for i in range(nstfts):
            start = i * frame_step
            stop = i * frame_step + frame_length
            complex_out = np.fft.fft(signal[0, start:stop, 0])
            c_out = complex_out[0:onesided_length]
            expected[0, i] = np.stack((c_out.real, c_out.imag), axis=1)

        # import torch
        # correspondance with torch
        # hop_length = frame_step
        # window = np.ones((frame_length,), dtype=np.float32)
        # ex = torch.stft(
        #      torch.Tensor(feeds["signal"][:, :, 0]),
        #      n_fft=frame_length, window=torch.Tensor(window),
        #      hop_length=hop_length, win_length=frame_length,
        #      onesided=True, return_complex=True, center=False,
        #      normalized=False)
        # ex = np.transpose(ex.numpy(), [0, 2, 1])

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    @unittest.skipIf(
        version_utils.numpy_older_than("1.21.5"),
        "op_dft and op_stft requires numpy >= 1.21.5",
    )
    def test_stft_with_window(self):
        signal = make_tensor_value_info("signal", TensorProto.FLOAT, [None, None, None])
        frame_step = make_tensor_value_info("frame_step", TensorProto.INT64, [None])
        window = make_tensor_value_info("window", TensorProto.FLOAT, [None])
        frame_length = make_tensor_value_info("frame_length", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])

        node = make_node(
            "STFT",
            ["signal", "frame_step", "window", "frame_length"],
            ["Y"],
        )
        graph = make_graph([node], "g", [signal, frame_step, window, frame_length], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 17)])
        feeds = {
            "signal": np.arange(128).reshape((1, 128, 1)).astype(np.float32),
            "frame_step": np.array(8, dtype=np.int64),
            "window": 0.5
            + 0.5 * np.cos(2 * np.pi * np.arange(0, 16, 1, dtype=np.float32) / 16),
            "frame_length": np.array(16, dtype=np.int64),
        }

        signal = feeds["signal"]
        frame_length = int(feeds["frame_length"])
        window = feeds["window"]
        frame_step = int(feeds["frame_step"])
        onesided_length = (frame_length // 2) + 1
        nstfts = 1 + (signal.shape[1] - window.shape[0]) // 8
        # [batch_size][frames][frame_length][2]
        expected = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
        for i in range(nstfts):
            start = i * frame_step
            stop = i * frame_step + frame_length
            complex_out = np.fft.fft(signal[0, start:stop, 0] * window)[
                0:onesided_length
            ]
            c_out = complex_out[0:onesided_length]
            expected[0, i] = np.stack((c_out.real, c_out.imag), axis=1)

        # import torch
        # hop_length = frame_step
        # ex = torch.stft(
        #      torch.Tensor(feeds["signal"][:, :, 0]),
        #      n_fft=frame_length, window=torch.Tensor(window),
        #      hop_length=hop_length, win_length=frame_length,
        #      onesided=True, return_complex=True, center=False,
        #      normalized=False)
        # ex = np.transpose(ex.numpy(), [0, 2, 1])

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    def get_roi_align_model(self, mode):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        rois = make_tensor_value_info("rois", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])
        IS = make_tensor_value_info("I", TensorProto.INT64, [None])
        node = make_node(
            "RoiAlign",
            ["X", "rois", "I"],
            ["Y"],
            output_height=5,
            output_width=5,
            sampling_ratio=2,
            spatial_scale=1.0,
            coordinate_transformation_mode="output_half_pixel",
            mode=mode,
        )
        graph = make_graph([node], "g", [X, rois, IS], [Y])
        return make_model_gen_version(graph, opset_imports=[make_opsetid("", 17)])

    def common_test_roi_align(self, mode):
        onnx_model = self.get_roi_align_model(mode)
        X, batch_indices, rois = get_roi_align_input_values()
        feeds = {"X": X, "rois": rois, "I": batch_indices}
        sess = run_ort_inference(onnx_model)
        if sess is None:
            return
        expected = sess.run(None, feeds)
        ref = ReferenceEvaluator(onnx_model)
        got = ref.run(None, feeds)
        assert_allclose(expected[0], got[0], atol=1e-5)

    @skip_if_no_onnxruntime
    def test_roi_align(self):
        with self.subTest(mode="avg"):
            self.common_test_roi_align("avg")
        # max does not have example in the backend
        with self.subTest(mode="max"):
            self.common_test_roi_align("max")

    def common_test_roi_align_torch(self, mode):
        import torch
        from torchvision.ops import RoIAlign

        onnx_model = self.get_roi_align_model(mode)
        sess = ReferenceEvaluator(onnx_model)
        X, batch_indices, rois = get_roi_align_input_values()
        got = sess.run(None, {"X": X, "rois": rois, "I": batch_indices})

        a = RoIAlign((5, 5), spatial_scale=1.0, sampling_ratio=2)
        expected = a(torch.from_numpy(X), [torch.from_numpy(rois)])
        assert_allclose(expected, got[0], atol=1e-5)

    @skip_if_no_torch
    @skip_if_no_torchvision
    def test_roi_align_torch(self):
        with self.subTest(mode="avg"):
            self.common_test_roi_align_torch("avg")
        # not implemented in torch
        # with self.subTest(mode="max"):
        #     self.common_test_roi_align_torch("max")

    def test_split(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y1 = make_tensor_value_info("Y1", TensorProto.FLOAT, [None])
        Y2 = make_tensor_value_info("Y2", TensorProto.FLOAT, [None])
        Y3 = make_tensor_value_info("Y3", TensorProto.FLOAT, [None])
        Y4 = make_tensor_value_info("Y4", TensorProto.FLOAT, [None])

        node = make_node("Split", ["X"], ["Y1", "Y2", "Y3", "Y4"], num_outputs=4)
        graph = make_graph([node], "g", [X], [Y1, Y2, Y3, Y4])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        feeds = {"X": np.arange(10).astype(np.float32)}

        expected = [
            np.array([0, 1, 2], dtype=np.float32),
            np.array([3, 4, 5], dtype=np.float32),
            np.array([6, 7, 8], dtype=np.float32),
            np.array([9], dtype=np.float32),
        ]

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])

    def test_split_2(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y1 = make_tensor_value_info("Y1", TensorProto.FLOAT, [None])
        Y2 = make_tensor_value_info("Y2", TensorProto.FLOAT, [None])
        Y3 = make_tensor_value_info("Y3", TensorProto.FLOAT, [None])
        Y4 = make_tensor_value_info("Y4", TensorProto.FLOAT, [None])

        node = make_node("Split", ["X", "split"], ["Y1", "Y2", "Y3", "Y4"])
        graph = make_graph([node], "g", [X], [Y1, Y2, Y3, Y4])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        feeds = {
            "X": np.arange(10).astype(np.float32),
            "split": np.array([3, 3, 2, 2], dtype=np.int64),
        }

        expected = [
            np.array([0, 1, 2], dtype=np.float32),
            np.array([3, 4, 5], dtype=np.float32),
            np.array([6, 7], dtype=np.float32),
            np.array([8, 9], dtype=np.float32),
        ]

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])

    def test_split_num_outputs_4(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y1 = make_tensor_value_info("Y1", TensorProto.FLOAT, [None])
        Y2 = make_tensor_value_info("Y2", TensorProto.FLOAT, [None])
        Y3 = make_tensor_value_info("Y3", TensorProto.FLOAT, [None])
        Y4 = make_tensor_value_info("Y4", TensorProto.FLOAT, [None])

        node = make_node("Split", ["X"], ["Y1", "Y2", "Y3", "Y4"], num_outputs=4)
        graph = make_graph([node], "g", [X], [Y1, Y2, Y3, Y4])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])

        # case 1
        feeds = {"X": np.arange(10).astype(np.float32)}
        expected = [
            np.array([0, 1, 2], dtype=np.float32),
            np.array([3, 4, 5], dtype=np.float32),
            np.array([6, 7, 8], dtype=np.float32),
            np.array([9], dtype=np.float32),
        ]

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])

        # case 2
        feeds = {"X": np.arange(9).astype(np.float32)}
        expected = [
            np.array([0, 1, 2], dtype=np.float32),
            np.array([3, 4, 5], dtype=np.float32),
            np.array([6, 7, 8], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])

    def test_argmin(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.INT64, [None])
        node = make_node("ArgMin", ["X"], ["Y"], axis=1)
        graph = make_graph([node], "g", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        feeds = {"X": np.arange(12).reshape((3, 4)).astype(np.float32)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([0, 0, 0], dtype=np.int64).reshape((-1, 1))
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_argmax(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.INT64, [None])
        node = make_node("ArgMax", ["X"], ["Y"], axis=1)
        graph = make_graph([node], "g", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        feeds = {"X": np.arange(12).reshape((3, 4)).astype(np.float32)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([3, 3, 3], dtype=np.int64).reshape((-1, 1))
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_slice_squeeze(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        starts = make_tensor_value_info("starts", TensorProto.INT64, [None])
        ends = make_tensor_value_info("ends", TensorProto.INT64, [None])
        axes = make_tensor_value_info("axes", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.INT64, [None])
        nodes = [
            make_node("Slice", ["X", "starts", "ends", "axes"], ["T"]),
            make_node("Squeeze", ["T", "axes"], ["Y"]),
        ]
        graph = make_graph(nodes, "g", [X, starts, ends, axes], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        feeds = {
            "X": np.array([[0]], dtype=np.int64),
            "starts": np.array([0], dtype=np.int64),
            "ends": np.array([1], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
        }
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([0], dtype=np.int64)
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_slice_squeeze_6(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.INT64, [None])
        nodes = [
            make_node("Slice", ["X"], ["T"], axes=[0], starts=[0], ends=[1]),
            make_node("Squeeze", ["T"], ["Y"], axes=[0]),
        ]
        graph = make_graph(nodes, "g", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 6)])
        feeds = {"X": np.array([[0]], dtype=np.int64)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([0], dtype=np.int64)
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_onnxrt_reduce_mean(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node("ReduceMean", ["X"], ["Y"])
        graph = make_graph([node1], "g", [X], [Y])

        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 17)])
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        cls = sess.rt_nodes_[0]
        self.assertEqual(cls.__class__.__name__, "ReduceMean_1")
        got = sess.run(None, {"X": np.ones((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (1, 1))
        self.assertEqual(got[0, 0], 1)

        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        cls = sess.rt_nodes_[0]
        self.assertEqual(cls.__class__.__name__, "ReduceMean_18")
        got = sess.run(None, {"X": np.ones((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (1, 1))
        self.assertEqual(got[0, 0], 1)

    @staticmethod
    def _cdist_model(opset, reduce_op="ReduceSumSquare"):
        # subgraph
        initializers = []

        inputs = [
            make_tensor_value_info("next_in", TensorProto.FLOAT, [None, 4]),
            make_tensor_value_info("next", TensorProto.FLOAT, [None]),
        ]

        outputs = [
            make_tensor_value_info("next_out", TensorProto.FLOAT, [None, None]),
            make_tensor_value_info("scan_out", TensorProto.FLOAT, [None]),
        ]

        if opset >= 18:
            initializers.append(
                from_array(np.array([1], dtype=np.int64), name="axis_red")
            )
            node_reduce = make_node(
                reduce_op,
                ["cdistdf_17_C0", "axis_red"],
                ["cdistdf_17_reduced0"],
                name="cdistdf_17_ReduceSumSquare",
                keepdims=0,
            )
        else:
            node_reduce = make_node(
                reduce_op,
                ["cdistdf_17_C0"],
                ["cdistdf_17_reduced0"],
                name="cdistdf_17_ReduceSumSquare",
                axes=[1],
                keepdims=0,
            )

        nodes = [
            make_node("Identity", ["next_in"], ["next_out"], name="cdistd_17_Identity"),
            make_node(
                "Sub", ["next_in", "next"], ["cdistdf_17_C0"], name="cdistdf_17_Sub"
            ),
            node_reduce,
            make_node(
                "Identity",
                ["cdistdf_17_reduced0"],
                ["scan_out"],
                name="cdistdf_17_Identity",
            ),
        ]
        graph = make_graph(nodes, "OnnxIdentity", inputs, outputs, initializers)

        # main graph
        initializers = []

        list_value = [
            1.1394007205963135,
            -0.6848101019859314,
            -1.234825849533081,
            0.4023416340351105,
            0.17742614448070526,
            0.46278226375579834,
            -0.4017809331417084,
            -1.630198359489441,
            -0.5096521973609924,
            0.7774903774261475,
            -0.4380742907524109,
            -1.2527953386306763,
            -1.0485529899597168,
            1.950775384902954,
            -1.420017957687378,
            -1.7062702178955078,
            1.8675580024719238,
            -0.15135720372200012,
            -0.9772778749465942,
            0.9500884413719177,
            -2.5529897212982178,
            -0.7421650290489197,
            0.653618574142456,
            0.8644362092018127,
            1.5327792167663574,
            0.37816253304481506,
            1.4693588018417358,
            0.154947429895401,
            -0.6724604368209839,
            -1.7262825965881348,
            -0.35955315828323364,
            -0.8131462931632996,
            -0.8707971572875977,
            0.056165341287851334,
            -0.5788496732711792,
            -0.3115525245666504,
            1.2302906513214111,
            -0.302302747964859,
            1.202379822731018,
            -0.38732680678367615,
            2.269754648208618,
            -0.18718385696411133,
            -1.4543657302856445,
            0.04575851559638977,
            -0.9072983860969543,
            0.12898291647434235,
            0.05194539576768875,
            0.7290905714035034,
            1.4940791130065918,
            -0.8540957570075989,
            -0.2051582634449005,
            0.3130677044391632,
            1.764052391052246,
            2.2408931255340576,
            0.40015721321105957,
            0.978738009929657,
            0.06651721894741058,
            -0.3627411723136902,
            0.30247190594673157,
            -0.6343221068382263,
            -0.5108051300048828,
            0.4283318817615509,
            -1.18063223361969,
            -0.02818222902715206,
            -1.6138978004455566,
            0.38690251111984253,
            -0.21274028718471527,
            -0.8954665660858154,
            0.7610377073287964,
            0.3336743414402008,
            0.12167501449584961,
            0.44386324286460876,
            -0.10321885347366333,
            1.4542734622955322,
            0.4105985164642334,
            0.14404356479644775,
            -0.8877857327461243,
            0.15634897351264954,
            -1.980796456336975,
            -0.34791216254234314,
        ]
        initializers.append(
            from_array(
                np.array(list_value, dtype=np.float32).reshape((20, 4)),
                name="Sc_Scancst",
            )
        )
        initializers.append(
            from_array(np.array([2], dtype=np.int64), name="To_TopKcst")
        )

        inputs = [make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])]
        outputs = [
            make_tensor_value_info("values", TensorProto.FLOAT, [None, 2]),
            make_tensor_value_info("indices", TensorProto.INT64, [None, 2]),
        ]

        # nodes

        nodes = [
            make_node(
                "Scan",
                ["input", "Sc_Scancst"],
                ["UU032UU", "UU033UU"],
                name="Sc_Scan",
                body=graph,
                num_scan_inputs=1,
            ),
            make_node(
                "Transpose",
                ["UU033UU"],
                ["Tr_transposed0"],
                name="Tr_Transpose",
                perm=[1, 0],
            ),
            make_node("Sqrt", ["Tr_transposed0"], ["Sq_Y0"], name="Sq_Sqrt"),
            make_node(
                "TopK",
                ["Sq_Y0", "To_TopKcst"],
                ["values", "indices"],
                name="To_TopK",
                largest=0,
                sorted=1,
            ),
        ]

        graph = make_graph(nodes, "dummy", inputs, outputs, initializers)

        # model
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", opset)])
        return onnx_model

    @parameterized.parameterized.expand(
        itertools.product(
            [
                (
                    "ReduceMin",
                    [
                        np.array(
                            [[np.nan, np.nan], [14.422706, 18.80527]], dtype=np.float32
                        ),
                        np.array([[2, 15], [10, 4]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceL1",
                    [
                        np.array(
                            [[2.2367053, 2.3516612], [4.076292, 4.2970634]],
                            dtype=np.float32,
                        ),
                        np.array([[18, 6], [13, 6]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceL2",
                    [
                        np.array(
                            [[1.80155, 1.8169948], [2.9928076, 3.1205883]],
                            dtype=np.float32,
                        ),
                        np.array([[11, 18], [13, 6]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceLogSum",
                    [
                        np.array(
                            [[0.9497848, 1.1872643], [1.6764175, 1.70759]],
                            dtype=np.float32,
                        ),
                        np.array([[6, 18], [13, 6]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceLogSumExp",
                    [
                        np.array(
                            [[1.6005973, 1.7445935], [2.5616229, 2.6539795]],
                            dtype=np.float32,
                        ),
                        np.array([[13, 6], [13, 6]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceMax",
                    [
                        np.array(
                            [[1.4217108, 1.5069536], [2.453826, 2.5041783]],
                            dtype=np.float32,
                        ),
                        np.array([[13, 11], [13, 11]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceMean",
                    [
                        np.array(
                            [[0.39247903, 0.78497636], [2.038146, 2.1485317]],
                            dtype=np.float32,
                        ),
                        np.array([[13, 6], [13, 6]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceSumSquare",
                    [
                        np.array(
                            [[3.2455828, 3.3014696], [8.956896, 9.7380705]],
                            dtype=np.float32,
                        ),
                        np.array([[11, 18], [13, 6]], dtype=np.int64),
                    ],
                ),
                (
                    "ReduceProd",
                    [
                        np.array(
                            [[np.nan, np.nan], [14.422706, 18.80527]], dtype=np.float32
                        ),
                        np.array([[2, 15], [13, 6]], dtype=np.int64),
                    ],
                ),
            ],
            [17, 18],
        )
    )
    def test_op_reduce(self, reduce_op_expected, opset: int):
        reduce_op, expected = reduce_op_expected
        X = np.arange(8).reshape((-1, 4)).astype(np.float32)

        results = {}

        model = self._cdist_model(opset, reduce_op)
        sess = ReferenceEvaluator(model)
        got = sess.run(None, {"input": X})
        results["ref", opset] = got

        cl = [
            n
            for n in sess.rt_nodes_[0].body.rt_nodes_
            if n.__class__.__name__.startswith(reduce_op)
        ]
        schema = cl[0]._schema
        new_cl = type(reduce_op, (cl[0].__class__,), {"op_schema": schema})
        sess = ReferenceEvaluator(model, new_ops=[new_cl])
        got = sess.run(None, {"input": X})
        results["ref_cl", opset] = got

        baseline = "constant"
        for k, v in results.items():
            for a, b in zip(reversed(expected), reversed(v)):
                if a.shape != b.shape:
                    raise AssertionError(
                        f"Shape mismatch for {reduce_op!r}, {baseline}:{a.shape} != {k}:{b.shape}."
                    )
                diff = np.abs(a - b).max()
                if diff > 1e-6:
                    raise AssertionError(
                        f"Discrepancies (max={diff}) for {reduce_op!r}, {baseline} != {k}\n{a}\n!=\n{b}"
                    )

    @parameterized.parameterized.expand(
        [
            (13,),
            (17,),
            (18,),
        ]
    )
    def test_mvn(self, opset: int, ref_opset: int = 13):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])
        nodes = [
            make_node("MeanVarianceNormalization", ["X"], ["Y"]),
        ]
        graph = make_graph(nodes, "g", [X], [Y])
        x = np.random.rand(3, 3, 3, 1).astype(np.float32)

        onnx_model = make_model(graph, opset_imports=[make_opsetid("", opset)])
        ref = ReferenceEvaluator(onnx_model)
        got = ref.run(None, {"X": x})[0]

        ref_onnx_model = make_model(graph, opset_imports=[make_opsetid("", ref_opset)])
        ref_expected = ReferenceEvaluator(ref_onnx_model)
        expected = ref_expected.run(None, {"X": x})[0]

        self.assertEqual(expected.shape, got.shape)
        assert_allclose(expected, got)

    def test_concat_in_a_function(self):
        def create_model():
            nodes = []
            inputs = []
            outputs = []
            functions = []

            opsets = {"": onnx_opset_version(), "custom_domain": 1}
            nodes_fct = []
            node = make_node("Concat", ["x:0", "x:1"], ["r__0"], axis=0, domain="")
            nodes_fct.append(node)

            opset_imports_fct = [
                make_opsetid(domain, 1 if version is None else version)
                for domain, version in opsets.items()
            ]
            fct = make_function(
                "custom_domain",
                "concat_2",
                ["x:0", "x:1"],
                ["r__0"],
                nodes_fct,
                opset_imports_fct,
            )
            functions.append(fct)

            inputs.append(make_tensor_value_info("I__0", TensorProto.DOUBLE, []))
            inputs.append(make_tensor_value_info("I__1", TensorProto.DOUBLE, []))
            inputs.append(make_tensor_value_info("I__2", TensorProto.DOUBLE, []))
            outputs.append(make_tensor_value_info("r__4", TensorProto.DOUBLE, []))

            node = make_node(
                "concat_2", ["I__0", "I__1"], ["r__3"], axis=0, domain="custom_domain"
            )
            nodes.append(node)
            node = make_node(
                "concat_2", ["I__2", "r__3"], ["r__4"], axis=0, domain="custom_domain"
            )
            nodes.append(node)
            opset_imports = [
                make_opsetid(domain, 1 if version is None else version)
                for domain, version in opsets.items()
            ]

            graph = make_graph(nodes, "numpyx", inputs, outputs)

            onnx_model = make_model(
                graph, opset_imports=opset_imports, functions=functions
            )
            return onnx_model

        onnx_model = create_model()
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        z = np.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onnx_model)
        feeds = {"I__2": x1, "I__0": x2, "I__1": x3}
        got = ref.run(None, feeds)
        assert_allclose(z, got[0])

    def test_cast_float_to_string(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.STRING, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["Y"], to=TensorProto.STRING),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([1.152512, -0.152612, 0.0, np.nan])
        got = ref.run(None, {"X": data})[0]
        self.assertTrue(
            (got == np.array([1.152512, -0.152612, 0.0, np.nan]).astype(np.str_)).all()
        )

    def test_cast_float_to_string_and_back(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["Z"], to=TensorProto.STRING),
                    make_node("Cast", ["Z"], ["Y"], to=TensorProto.FLOAT),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([1.152512, -0.152612, 0.0, np.nan])
        got = ref.run(None, {"X": data})[0]
        assert_allclose(got, np.array([1.152512, -0.152612, 0.0, np.nan]))

    def test_split_to_sequence(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        Y = make_tensor_value_info("Y", TensorProto.INT64, None)
        Z = make_tensor_value_info("Z", TensorProto.UNDEFINED, None)
        nodes = [make_node("SplitToSequence", ["X", "Y"], ["Z"], axis=2)]
        model = make_model(make_graph(nodes, "g", [X, Y], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.arange(18).reshape((1, 3, 6)).astype(np.float32)
        indices = np.array(2, dtype=np.int64)
        got = ref.run(None, {"X": data, "Y": indices})
        expected = [
            [
                np.array([[[0.0, 1.0], [6.0, 7.0], [12.0, 13.0]]], dtype=np.float32),
                np.array([[[2.0, 3.0], [8.0, 9.0], [14.0, 15.0]]], dtype=np.float32),
                np.array([[[4.0, 5.0], [10.0, 11.0], [16.0, 17.0]]], dtype=np.float32),
            ]
        ]
        self.assertEqual(len(expected[0]), len(got[0]))
        for a, b in zip(expected[0], got[0]):
            assert_allclose(a, b)

    def test_split_to_sequence_1d(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        Y = make_tensor_value_info("Y", TensorProto.INT64, None)
        Z = make_tensor_value_info("Z", TensorProto.UNDEFINED, None)
        nodes = [make_node("SplitToSequence", ["X", "Y"], ["Z"], axis=2)]
        model = make_model(make_graph(nodes, "g", [X, Y], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.arange(18).reshape((1, 3, 6)).astype(np.float32)
        indices = np.array([2, 2, 2], dtype=np.int64)
        got = ref.run(None, {"X": data, "Y": indices})
        expected = [
            [
                np.array([[[0.0, 1.0], [6.0, 7.0], [12.0, 13.0]]], dtype=np.float32),
                np.array([[[2.0, 3.0], [8.0, 9.0], [14.0, 15.0]]], dtype=np.float32),
                np.array([[[4.0, 5.0], [10.0, 11.0], [16.0, 17.0]]], dtype=np.float32),
            ]
        ]
        self.assertEqual(len(expected[0]), len(got[0]))
        for a, b in zip(expected[0], got[0]):
            assert_allclose(a, b)

    def test_split_to_sequence_nokeepdims_noinput(self):
        # keepdims is ignored in that case
        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        Z = make_tensor_value_info("Z", TensorProto.UNDEFINED, None)
        nodes = [make_node("SplitToSequence", ["X"], ["Z"], axis=2, keepdims=0)]
        model = make_model(make_graph(nodes, "g", [X], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.arange(18).reshape((1, 3, 6)).astype(np.float32)
        got = ref.run(None, {"X": data})
        expected = [[data[:, :, i] for i in range(data.shape[2])]]
        self.assertEqual(len(expected[0]), len(got[0]))
        for a, b in zip(expected[0], got[0]):
            assert_allclose(a, b)

    def test_cast_float8(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        F1 = make_tensor_value_info("F1", TensorProto.FLOAT, [None])
        F2 = make_tensor_value_info("F2", TensorProto.FLOAT, [None])
        F3 = make_tensor_value_info("F3", TensorProto.FLOAT, [None])
        F4 = make_tensor_value_info("F4", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["f81"], to=TensorProto.FLOAT8E4M3FN),
                    make_node("Cast", ["X"], ["f82"], to=TensorProto.FLOAT8E5M2),
                    make_node(
                        "Constant",
                        [],
                        ["C1"],
                        value=make_tensor(
                            "C1", TensorProto.FLOAT8E4M3FN, [5], [0, 1, 2, 5e-2, 200]
                        ),
                    ),
                    make_node(
                        "Constant",
                        [],
                        ["C2"],
                        value=make_tensor(
                            "C2", TensorProto.FLOAT8E5M2, [5], [0, 1, 2, 5e-2, 200]
                        ),
                    ),
                    make_node("Cast", ["f81"], ["F1"], to=TensorProto.FLOAT),
                    make_node("Cast", ["f82"], ["F2"], to=TensorProto.FLOAT),
                    make_node("Cast", ["C1"], ["F3"], to=TensorProto.FLOAT),
                    make_node("Cast", ["C2"], ["F4"], to=TensorProto.FLOAT),
                ],
                "g",
                [X],
                [F1, F2, F3, F4],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 5e-2, 200], dtype=np.float32)
        print([float32_to_float8e4m3(x) for x in data])
        expected1 = np.array(
            [float8e4m3_to_float32(float32_to_float8e4m3(x)) for x in data]
        )
        expected2 = np.array(
            [float8e5m2_to_float32(float32_to_float8e5m2(x)) for x in data]
        )
        got = ref.run(None, {"X": data})
        assert_allclose(expected1, got[0])
        assert_allclose(expected2, got[1])
        assert_allclose(expected1, got[2])
        assert_allclose(expected2, got[3])

    def test_cast_like_float8(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["f8"], to=TensorProto.FLOAT8E4M3FNUZ),
                    make_node("CastLike", ["X", "f8"], ["f32"], saturate=0),
                    make_node("Cast", ["f32"], ["Y"], to=TensorProto.FLOAT),
                ],
                "g",
                [X],
                [Y],
            )
        )
        data = np.array([0, 1e7], dtype=np.float32)
        expected = np.array(
            [
                float8e4m3_to_float32(
                    float32_to_float8e4m3(x, uz=True, saturate=False), uz=True
                )
                for x in data
            ]
        )
        ref = ReferenceEvaluator(model)
        got = ref.run(None, {"X": data})
        assert_allclose(got[0], expected)

        # Forces ReferenceEvaluator to not use the associated implementation for CastLike
        # but its implementation as a function instead.
        class CastLike(OpRunExpand):
            op_domain = ""

        ref = ReferenceEvaluator(model, new_ops=[CastLike])
        got = ref.run(None, {"X": data})
        assert_allclose(got[0], expected)

    def test_cast_float8_output(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        F1 = make_tensor_value_info("F1", TensorProto.FLOAT8E4M3FN, [None])
        F2 = make_tensor_value_info("F2", TensorProto.FLOAT8E5M2, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["F1"], to=TensorProto.FLOAT8E4M3FN),
                    make_node("Cast", ["X"], ["F2"], to=TensorProto.FLOAT8E5M2),
                ],
                "g",
                [X],
                [F1, F2],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 5e-2, 200], dtype=np.float32)
        expected1 = np.array([float32_to_float8e4m3(x) for x in data])
        expected2 = np.array([float32_to_float8e5m2(x) for x in data])
        got = ref.run(None, {"X": data})
        self.assertEqual(expected1.tolist(), got[0].tolist())
        self.assertEqual(expected2.tolist(), got[1].tolist())

    def test_float8_4_types(self):
        x = np.array(
            [
                0.4068359375,
                352,
                416,
                336,
                304,
                272,
                -248,
                -100,
                1e-4,
                1e-2,
                416,
                432,
                1e5,
                np.inf,
                -np.inf,
                np.nan,
            ],
            dtype=np.float32,
        )
        expected = {
            TensorProto.FLOAT8E4M3FN: np.array(
                [
                    0.40625,
                    352.0,
                    416.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0,
                    0.009765625,
                    416.0,
                    448.0,
                    448.0,
                    448.0,
                    -448.0,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E4M3FNUZ: np.array(
                [
                    0.40625,
                    240.0,
                    240.0,
                    240.0,
                    240.0,
                    240.0,
                    -240.0,
                    -96.0,
                    0.0,
                    0.009765625,
                    240.0,
                    240.0,
                    240.0,
                    240.0,
                    -240.0,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2: np.array(
                [
                    0.4375,
                    384.0,
                    384.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0001068115234375,
                    0.009765625,
                    384.0,
                    448.0,
                    57344.0,
                    57344.0,
                    -57344.0,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2FNUZ: np.array(
                [
                    4.3750000e-01,
                    3.8400000e02,
                    3.8400000e02,
                    3.2000000e02,
                    3.2000000e02,
                    2.5600000e02,
                    -2.5600000e02,
                    -9.6000000e01,
                    1.0681152e-04,
                    9.7656250e-03,
                    3.8400000e02,
                    4.4800000e02,
                    5.7344000e04,
                    5.7344000e04,
                    -5.7344000e04,
                    np.nan,
                ],
                dtype=np.float32,
            ),
        }

        def model_cast_cast(to):
            X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
            Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
            node1 = make_node("Cast", ["X"], ["T"], to=to)
            node2 = make_node("Cast", ["T"], ["Y"], to=TensorProto.FLOAT)
            graph = make_graph([node1, node2], "lr", [X], [Y])
            onnx_model = make_model(graph)
            check_model(onnx_model)
            return onnx_model

        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = model_cast_cast(to)
                ref = ReferenceEvaluator(onnx_model)
                y = ref.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    def test_cast_bfloat16_output(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.BFLOAT16, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["Y"], to=TensorProto.BFLOAT16),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 1e5, 200], dtype=np.float32)
        expected1 = np.array([float32_to_bfloat16(x) for x in data])
        got = ref.run(None, {"X": data})
        self.assertEqual(expected1.tolist(), got[0].tolist())

    def test_quantize_linear_e4m3(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node(
                        "Constant",
                        [],
                        ["scale"],
                        value=make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    ),
                    make_node(
                        "Constant",
                        [],
                        ["zero"],
                        value=make_tensor("zero", TensorProto.FLOAT8E4M3FN, [1], [0.0]),
                    ),
                    make_node("QuantizeLinear", ["X", "scale", "zero"], ["T"]),
                    make_node("DequantizeLinear", ["T", "scale"], ["Y"], axis=0),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 1e5, 200], dtype=np.float32)
        expected = np.array([0, 1, 2, 896, 192], dtype=np.float32)
        got = ref.run(None, {"X": data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_e4m3_initializer(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node("QuantizeLinear", ["X", "scale", "zero"], ["T"]),
                    make_node("DequantizeLinear", ["T", "scale"], ["Y"], axis=0),
                ],
                "g",
                [X],
                [Y],
                [
                    make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    make_tensor("zero", TensorProto.FLOAT8E4M3FN, [1], [0.0]),
                ],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 1e5, 200], dtype=np.float32)
        expected = np.array([0, 1, 2, 896, 192], dtype=np.float32)
        got = ref.run(None, {"X": data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_e5m2(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node(
                        "Constant",
                        [],
                        ["scale"],
                        value=make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    ),
                    make_node(
                        "Constant",
                        [],
                        ["zero"],
                        value=make_tensor("zero", TensorProto.FLOAT8E5M2, [1], [0.0]),
                    ),
                    make_node("QuantizeLinear", ["X", "scale", "zero"], ["T"]),
                    make_node("DequantizeLinear", ["T", "scale"], ["Y"], axis=0),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 1e5, 200], dtype=np.float32)
        expected = np.array([0, 1, 2, 98304, 192], dtype=np.float32)
        got = ref.run(None, {"X": data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_uint16(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.UINT16, [None])
        model = make_model(
            make_graph(
                [
                    make_node("QuantizeLinear", ["X", "scale", "zero"], ["Y"]),
                ],
                "g",
                [X],
                [Y],
                [
                    make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    make_tensor("zero", TensorProto.UINT16, [1], [32767]),
                ],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array(
            [
                # rounding half to even
                0.0,
                -128.0,
                3.0,
                -3.0,
                # round < .5
                2.9,
                -2.9,
                # round > .5
                3.1,
                -3.1,
                # critical point
                65536.0,
                -65534.0,
                # saturate case
                70000.0,
                -70000.0,
            ],
            dtype=np.float32,
        )
        expected = np.array(
            [
                32767,
                32703,
                32769,
                32765,
                32768,
                32766,
                32769,
                32765,
                65535,
                0,
                65535,
                0,
            ],
            dtype=np.uint16,
        )
        got = ref.run(None, {"X": data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_int16(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.INT16, [None])
        model = make_model(
            make_graph(
                [
                    make_node("QuantizeLinear", ["X", "scale", "zero"], ["Y"]),
                ],
                "g",
                [X],
                [Y],
                [
                    make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    make_tensor("zero", TensorProto.INT16, [1], [256]),
                ],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array(
            [
                # rounding half to even
                0.0,
                -514.0,
                3.0,
                -3.0,
                # round < .5
                2.9,
                -2.9,
                # round > .5
                3.1,
                -3.1,
                # critical point
                65022.0,
                -66046.0,
                65023.0,
                -66047.0,
                65024.0,
                -66048.0,
                # saturate case
                70000.0,
                -70000.0,
            ],
            dtype=np.float32,
        )
        expected = np.array(
            [
                256,
                -1,
                258,
                254,
                257,
                255,
                258,
                254,
                32767,
                -32767,
                32767,
                -32768,
                32767,
                -32768,
                32767,
                -32768,
            ],
            dtype=np.int16,
        )
        got = ref.run(None, {"X": data})
        assert_allclose(expected, got[0])

    def test_dequantize_linear_uint16(self):
        X = make_tensor_value_info("X", TensorProto.UINT16, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node(
                        "DequantizeLinear", ["X", "scale", "zero"], ["Y"], axis=0
                    ),
                ],
                "g",
                [X],
                [Y],
                [
                    make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    make_tensor("zero", TensorProto.UINT16, [1], [32767]),
                ],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([30000, 31000, 32768, 33000], dtype=np.uint16)
        expected = np.array([-5534.0, -3534.0, 2.0, 466.0], dtype=np.float32)
        got = ref.run(None, {"X": data})
        assert_allclose(expected, got[0])

    def test_dequantize_linear_int16(self):
        X = make_tensor_value_info("X", TensorProto.INT16, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node(
                        "DequantizeLinear", ["X", "scale", "zero"], ["Y"], axis=0
                    ),
                ],
                "g",
                [X],
                [Y],
                [
                    make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    make_tensor("zero", TensorProto.INT16, [1], [-1024]),
                ],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([-300, -30, -1025, 1270], dtype=np.int16)
        expected = np.array([1448.0, 1988.0, -2.0, 4588.0], dtype=np.float32)
        got = ref.run(None, {"X": data})
        assert_allclose(expected, got[0])

    @parameterized.parameterized.expand(
        [
            (
                4 * np.arange(12).reshape(3, 4),
                np.arange(1, 7).reshape(3, 2),
                np.zeros((3, 2)),
                1,
                2,
                [[0, 4, 4, 6], [5, 7, 6, 7], [6, 7, 7, 7]],
            ),
            (
                4 * np.arange(12).reshape(3, 4),
                np.arange(1, 7).reshape(3, 2),
                np.ones((3, 2)),
                1,
                2,
                [[1, 5, 5, 7], [6, 8, 7, 8], [7, 8, 8, 8]],
            ),
            (
                np.arange(24).reshape(3, 8),
                [[0.25, 0.5, 1], [0.25, 0.5, 1], [0.25, 0.5, 1]],
                np.zeros((3, 3)),
                1,
                3,
                [
                    [0, 4, 8, 6, 8, 10, 6, 7],
                    [32, 36, 40, 22, 24, 26, 14, 15],
                    [64, 68, 72, 38, 40, 42, 22, 23],
                ],
            ),
            (
                np.arange(6),
                [0.25, 0.5],
                [-1, -2],
                0,
                3,
                [-1, 3, 7, 4, 6, 8],
            ),
            (
                np.ones((9, 12)),
                np.ones((3, 4)),
                np.zeros((3, 4)),
                0,
                3,
                None,  # Blocked quantization is defined for 1-D blocks only
            ),
            (
                np.ones((3, 4, 5, 6)),
                np.ones((3, 4)),
                np.zeros((3, 4)),
                2,
                2,
                None,  # Scale and ZP must have the same rank as the input
            ),
        ]
    )
    def test_blocked_quantize_linear(
        self, x, scale, zero_point, axis, block_size, expected
    ):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.INT8, [None])

        scale_data = np.array(scale, dtype=np.float32)
        zp_data = np.array(zero_point, dtype=np.int8)
        model = make_model(
            make_graph(
                [
                    make_node(
                        "QuantizeLinear",
                        ["X", "scale", "zero"],
                        ["Y"],
                        axis=axis,
                        block_size=block_size,
                    ),
                ],
                "g",
                [X],
                [Y],
                [
                    make_tensor(
                        "scale", TensorProto.FLOAT, scale_data.shape, scale_data
                    ),
                    make_tensor("zero", TensorProto.INT8, scale_data.shape, zp_data),
                ],
            )
        )
        ref = ReferenceEvaluator(model)

        data = np.array(x, dtype=np.float32)

        if expected is not None:
            expected = np.array(expected, dtype=np.int8)
            got = ref.run(None, {"X": data})
            assert_allclose(expected, got[0])
        else:
            with self.assertRaises(ValueError):
                ref.run(None, {"X": data})

    @parameterized.parameterized.expand(
        [
            (
                np.arange(12).reshape(3, 4),
                np.arange(1, 7).reshape(3, 2),
                np.zeros((3, 2)),
                1,
                2,
                [[0, 1, 4, 6], [12, 15, 24, 28], [40, 45, 60, 66]],
            ),
            (
                np.arange(12).reshape(3, 4),
                np.arange(1, 7).reshape(3, 2),
                np.ones((3, 2)),
                1,
                2,
                [[-1, 0, 2, 4], [9, 12, 20, 24], [35, 40, 54, 60]],
            ),
            (
                np.dstack([np.arange(4).reshape(2, 2)] * 4),
                np.dstack([np.array([[1, 1], [2, 3]]), np.array([[4, 5], [6, 7]])]),
                np.zeros((2, 2, 2)),
                2,
                2,
                [[[0, 0, 0, 0], [1, 1, 5, 5]], [[4, 4, 12, 12], [9, 9, 21, 21]]],
            ),
            (
                np.arange(24).reshape(3, 8),
                [[2, 1, 3], [2, 1, 3], [2, 1, 3]],
                np.zeros((3, 3)),
                1,
                3,
                [
                    [0, 2, 4, 3, 4, 5, 18, 21],
                    [16, 18, 20, 11, 12, 13, 42, 45],
                    [32, 34, 36, 19, 20, 21, 66, 69],
                ],
            ),
            (
                np.arange(
                    6,
                ),
                [2, 3],
                [1, 2],
                0,
                3,
                [-2, 0, 2, 3, 6, 9],
            ),
            (
                np.ones((9, 12)),
                np.ones((3, 4)),
                np.zeros((3, 4)),
                0,
                3,
                None,  # Blocked quantization is defined for 1-D blocks only
            ),
            (
                np.ones((3, 4, 5, 6)),
                np.ones((3, 4)),
                np.zeros((3, 4)),
                2,
                2,
                None,  # Scale and ZP must have the same rank as the input
            ),
        ]
    )
    def test_blocked_dequantize_linear(
        self, x, scale, zero_point, axis, block_size, expected
    ):
        X = make_tensor_value_info("X", TensorProto.INT8, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        scale_data = np.array(scale, dtype=np.float32)
        zp_data = np.array(zero_point, dtype=np.int8)
        model = make_model(
            make_graph(
                [
                    make_node(
                        "DequantizeLinear",
                        ["X", "scale", "zero"],
                        ["Y"],
                        axis=axis,
                        block_size=block_size,
                    ),
                ],
                "g",
                [X],
                [Y],
                [
                    make_tensor(
                        "scale", TensorProto.FLOAT, scale_data.shape, scale_data
                    ),
                    make_tensor("zero", TensorProto.INT8, scale_data.shape, zp_data),
                ],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array(x, dtype=np.int8)

        if expected is not None:
            expected = np.array(expected, dtype=np.float32)
            got = ref.run(None, {"X": data})
            assert_allclose(expected, got[0])
        else:
            with self.assertRaises(ValueError):
                ref.run(None, {"X": data})

    def test_lrn(self):
        def _expected(x, alpha, beta, bias, size):
            square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
            for n, c, h, w in np.ndindex(x.shape):
                square_sum[n, c, h, w] = sum(
                    x[
                        n,
                        max(0, c - int(math.floor((size - 1) / 2))) : min(
                            5, c + int(math.ceil((size - 1) / 2)) + 1
                        ),
                        h,
                        w,
                    ]
                    ** 2
                )
            y = x / ((bias + (alpha / size) * square_sum) ** beta)
            return y

        # keepdims is ignored in that case
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        size = 3
        X = make_tensor_value_info("X", TensorProto.FLOAT, [5, 5, 50, 50])
        Z = make_tensor_value_info("Z", TensorProto.UNDEFINED, None)
        nodes = [
            make_node("LRN", ["X"], ["Z"], alpha=alpha, beta=beta, bias=bias, size=size)
        ]
        model = make_model(make_graph(nodes, "g", [X], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.random.rand(5, 5, 5, 5).astype(np.float32)
        got = ref.run(None, {"X": data})
        expected = _expected(data, alpha, beta, bias, size)
        self.assertEqual(len(expected), len(got[0]))

    def test_conv_im2col_1d(self):
        feeds = {
            "X": np.arange(1 * 1 * 11).reshape((1, 1, 11)).astype(np.float32) + 1,
            "W": np.arange(3).reshape((1, 1, 3)).astype(np.float32),
            "B": np.zeros((1,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[1],
            kernel_shape=[3],
            pads=[1, 1],
            strides=[1],
            auto_pad="NOTSET",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_1d_pad0(self):
        feeds = {
            "X": np.arange(2 * 4 * 3).reshape((2, 4, -1)).astype(np.float32) + 1,
            "W": np.arange(2 * 4 * 3).reshape((-1, 4, 3)).astype(np.float32),
            "B": np.zeros((1,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[1],
            kernel_shape=[3],
            pads=[0, 0],
            strides=[1],
            auto_pad="NOTSET",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d(self):
        feeds = {
            "X": np.arange(1 * 1 * 11 * 23).reshape((1, 1, 11, 23)).astype(np.float32)
            + 1,
            "W": np.arange(9).reshape((1, 1, 3, 3)).astype(np.float32),
            "B": np.zeros((1,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[1, 1],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            auto_pad="NOTSET",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_pad0(self):
        feeds = {
            "X": np.arange(2 * 3 * 5 * 2).reshape((2, 3, 5, -1)).astype(np.float32) + 1,
            "W": 2
            ** np.arange(3 * 3 * 1 * 2).reshape((-1, 3, 1, 2)).astype(np.float32),
            "B": np.zeros((1,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[1, 1],
            kernel_shape=[1, 2],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            auto_pad="NOTSET",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_autopad(self):
        feeds = {
            "X": np.arange(5 * 5).reshape((1, 1, 5, -1)).astype(np.float32) + 1,
            "W": 2 ** np.arange(3 * 3).reshape((1, 1, 3, 3)).astype(np.float32),
            "B": np.zeros((1,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[1, 1],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=None,
            auto_pad="SAME_LOWER",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_3d(self):
        feeds = {
            "X": np.arange(1 * 1 * 11 * 5 * 13)
            .reshape((1, 1, 11, 5, 13))
            .astype(np.float32)
            + 1,
            "W": np.arange(27).reshape((1, 1, 3, 3, 3)).astype(np.float32),
            "B": np.zeros((1,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[1, 1, 1],
            kernel_shape=[3, 3, 3],
            pads=[1, 1, 1, 1, 1, 1],
            strides=[1, 1, 1],
            auto_pad="NOTSET",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_strides(self):
        feeds = {
            "X": np.arange(1 * 3 * 6 * 6).reshape((1, 3, 6, 6)).astype(np.float32) + 1,
            "W": np.arange(2 * 3 * 3 * 3).reshape((2, 3, 3, 3)).astype(np.float32),
            "B": np.zeros((2,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[1, 1],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
            auto_pad="NOTSET",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_dilations(self):
        feeds = {
            "X": np.arange(1 * 3 * 6 * 6).reshape((1, 3, 6, 6)).astype(np.float32) + 1,
            "W": np.arange(2 * 3 * 3 * 3).reshape((2, 3, 3, 3)).astype(np.float32),
            "B": np.zeros((2,), dtype=np.float32),
        }
        kwargs = dict(
            group=1,
            dilations=[2, 1],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
            auto_pad="NOTSET",
        )
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    @parameterized.parameterized.expand(
        [
            ("ReduceSum",),
            ("ReduceL1",),
            ("ReduceL2",),
            ("ReduceMin",),
            ("ReduceMax",),
            ("ReduceProd",),
            ("ReduceSumSquare",),
        ]
    )
    def test_reduce_op_no_axis(self, op):
        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, None)
        data = np.arange(6).reshape((1, 3, 2)).astype(np.float32)
        nodes = [make_node(op, ["X"], ["Y"], keepdims=0)]
        model = make_model(make_graph(nodes, "g", [X], [Y]))
        ref = ReferenceEvaluator(model)
        got = ref.run(None, {"X": data})
        r = got[0]
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, ())

    @parameterized.parameterized.expand([(1,), (2,), (3,), (4,), (5,), (6,)])
    def test_pad(self, dim):
        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        P = make_tensor_value_info("P", TensorProto.INT64, None)
        V = make_tensor_value_info("V", TensorProto.FLOAT, None)
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, None)
        value = np.array([-5], dtype=np.float32)

        node = make_node("Pad", inputs=["X", "P", "V"], outputs=["Y"], mode="constant")
        model = make_model(make_graph([node], "g", [X, P, V], [Y]))
        ref = ReferenceEvaluator(model)
        x = np.array([1], dtype=np.float32).reshape((1,) * dim)

        p = np.array([1, 1] * dim, dtype=np.int64)
        got = ref.run(None, {"X": x, "P": p, "V": value})[0]
        self.assertEqual(got.shape, (3,) * dim)
        self.assertEqual(got.dtype, np.float32)

        p = np.repeat([7, 3], dim).astype(np.int64)
        got = ref.run(None, {"X": x, "P": p, "V": value})[0]
        self.assertEqual(got.shape, (11,) * dim)
        self.assertEqual(got.dtype, np.float32)

    def test_constant_of_shape(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, None)

        nodes = [
            make_node("Shape", inputs=["X"], outputs=["shape"]),
            make_node(
                "ConstantOfShape",
                inputs=["shape"],
                outputs=["Y"],
                value=make_tensor("value", TensorProto.UINT16, [1], [1]),
            ),
        ]
        model = make_model(make_graph(nodes, "g", [X], [Y]))
        ref = ReferenceEvaluator(model)
        x = np.array(1, dtype=np.float32)
        got = ref.run(None, {"X": x})[0]
        self.assertEqual(got.shape, tuple())
        self.assertEqual(got.dtype, np.uint16)
        assert_allclose(np.array(1, dtype=np.uint16), got)

    def test_constant_of_shape_castlike(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, None)

        nodes = [
            make_node(
                "Constant",
                [],
                ["like"],
                value=make_tensor("c", TensorProto.UINT16, [1], [2]),
            ),
            make_node("Shape", inputs=["X"], outputs=["shape"]),
            make_node(
                "ConstantOfShape",
                inputs=["shape"],
                outputs=["cst"],
                value=make_tensor("value", TensorProto.INT64, [1], [1]),
            ),
            make_node("CastLike", ["cst", "like"], ["Y"]),
        ]
        model = make_model(make_graph(nodes, "g", [X], [Y]))
        ref = ReferenceEvaluator(model)
        x = np.array(1, dtype=np.float32)
        got = ref.run(None, {"X": x})[0]
        self.assertEqual(got.shape, tuple())
        self.assertEqual(got.dtype, np.uint16)
        assert_allclose(np.array(1, dtype=np.uint16), got)

    def test_dynamic_quantize_linear(self):
        feeds = {
            "X": np.array(
                [
                    [
                        -7.80749545e-02,
                        -3.80597055e-01,
                        1.33831516e-01,
                        -8.20474699e-02,
                        7.56645501e-02,
                        5.65112457e-02,
                        2.56818235e-01,
                        9.42316353e-02,
                        1.88027292e-01,
                        1.44878656e-01,
                        1.34825557e-01,
                        -2.04576910e-01,
                        1.68852255e-01,
                        6.23253360e-02,
                        4.30482924e-01,
                        -5.50433956e-02,
                        9.10681635e-02,
                        1.55332625e-01,
                        -4.53630984e-02,
                        3.99910688e-01,
                        -1.28678545e-01,
                        3.77916731e-02,
                        1.29872710e-01,
                        -1.12420328e-01,
                        -2.97306702e-02,
                        2.20508516e-01,
                        -5.88933006e-03,
                        4.81076002e-01,
                        -1.18835129e-01,
                        -4.45004404e-02,
                        -7.53675848e-02,
                        1.41112670e-01,
                        1.97793499e-01,
                        -7.71476865e-01,
                        8.64694864e-02,
                        1.73293594e-02,
                        1.28247693e-01,
                        7.58144110e-02,
                        -2.71435380e-01,
                        1.75212905e-01,
                        -2.47283235e-01,
                        -3.02810557e-02,
                        8.45039487e-02,
                        6.02229357e-01,
                        -1.04913145e-01,
                        -2.46705681e-01,
                        2.92073280e-01,
                        -3.88464853e-02,
                        4.26557302e-01,
                        -3.71325493e-01,
                        -3.11283618e-01,
                        7.85303488e-02,
                        3.18069518e-01,
                        -1.51467413e-01,
                        -1.02828763e-01,
                        9.29131880e-02,
                        2.55233884e-01,
                        5.00160515e-01,
                        -1.49993747e-01,
                        4.29408073e-01,
                        -1.91787735e-01,
                        3.16187665e-02,
                        -1.84284449e-02,
                        -1.62873864e-01,
                        -2.73632705e-01,
                        2.84725696e-01,
                        -2.87029266e-01,
                        -7.15534389e-02,
                        2.24836454e-01,
                        -1.70527741e-01,
                        -2.65601039e-01,
                        -2.68008932e-03,
                        1.44260898e-01,
                        7.80707747e-02,
                        2.73875445e-02,
                        -1.18391573e-01,
                        -6.44972250e-02,
                        -5.22445887e-03,
                        -2.96754301e-01,
                        1.05800219e-01,
                        2.62558222e-01,
                        3.62841524e-02,
                        9.44730639e-03,
                        1.75837606e-01,
                        2.69956529e-01,
                        3.02758247e-01,
                        -1.13724738e-01,
                        2.98936248e-01,
                        8.54668319e-02,
                        -6.74555600e-01,
                        4.38643873e-01,
                        1.27896462e-02,
                        9.20789093e-02,
                        1.93946883e-01,
                        1.97548166e-01,
                        2.82558739e-01,
                        -2.48879120e-01,
                        -3.93428057e-01,
                        6.45540953e-02,
                        -9.66283306e-03,
                    ],
                    [
                        -2.42438495e-01,
                        -3.58334243e-01,
                        1.22619808e-01,
                        -1.21529415e-01,
                        -5.23974374e-02,
                        -6.74922541e-02,
                        1.09727375e-01,
                        -3.56835872e-03,
                        1.51029706e-01,
                        1.18043356e-01,
                        8.16475376e-02,
                        -2.36466587e-01,
                        9.44180191e-02,
                        5.61679937e-02,
                        3.67988586e-01,
                        1.29441261e-01,
                        1.15772486e-01,
                        5.30351102e-02,
                        -1.04345076e-01,
                        1.29062623e-01,
                        -2.17205048e-01,
                        2.58089490e-02,
                        2.18848974e-01,
                        -8.36039707e-02,
                        5.43577969e-03,
                        7.87076280e-02,
                        1.19966723e-01,
                        2.81631678e-01,
                        -3.58020402e-02,
                        -9.65647772e-02,
                        3.17915753e-02,
                        2.49396205e-01,
                        2.23600790e-01,
                        -3.82718384e-01,
                        1.16506606e-01,
                        -3.19400802e-02,
                        1.60812005e-01,
                        9.81735960e-02,
                        -1.99046463e-01,
                        8.81427377e-02,
                        -1.65171087e-01,
                        -1.10251531e-01,
                        1.15387037e-01,
                        3.19312930e-01,
                        -1.14400804e-01,
                        -2.02447772e-01,
                        2.33669549e-01,
                        9.20853689e-02,
                        3.91551465e-01,
                        -3.58036369e-01,
                        -2.35071778e-01,
                        -5.00670634e-02,
                        1.65313914e-01,
                        -1.60922945e-01,
                        -1.95520848e-01,
                        1.82456985e-01,
                        3.80704433e-01,
                        4.19890225e-01,
                        -2.98131555e-02,
                        3.66110623e-01,
                        -2.81170905e-01,
                        -1.23942450e-01,
                        9.58625227e-02,
                        -5.90450205e-02,
                        -1.56236291e-01,
                        2.11865619e-01,
                        -1.75286919e-01,
                        -8.36539492e-02,
                        1.29381597e-01,
                        -1.66115880e-01,
                        -1.66922957e-01,
                        1.65688396e-01,
                        8.41224194e-02,
                        1.67839468e-01,
                        -4.03967649e-02,
                        -8.34071711e-02,
                        -5.65301552e-02,
                        1.67074010e-01,
                        -3.19734544e-01,
                        7.71123618e-02,
                        5.57036921e-02,
                        1.20709330e-01,
                        -6.63790107e-02,
                        1.36002287e-01,
                        3.42324018e-01,
                        3.42968374e-01,
                        -1.42380476e-01,
                        2.89662749e-01,
                        1.82179566e-02,
                        -4.96056974e-01,
                        2.64364302e-01,
                        7.38918930e-02,
                        1.11150607e-01,
                        -5.95749579e-02,
                        1.18562862e-01,
                        6.90007359e-02,
                        -2.08283514e-01,
                        -1.70682222e-01,
                        7.82715827e-02,
                        1.35489792e-01,
                    ],
                    [
                        -6.35757595e-02,
                        -3.20629895e-01,
                        9.38569903e-02,
                        -1.15190029e-01,
                        1.03646070e-02,
                        -4.31734361e-02,
                        2.31717676e-01,
                        4.01005745e-02,
                        1.18915148e-01,
                        2.10071519e-01,
                        -2.99234912e-02,
                        -2.93135524e-01,
                        -2.39588290e-01,
                        6.71441257e-02,
                        3.15238893e-01,
                        -5.08778207e-02,
                        1.16147280e-01,
                        -6.72954097e-02,
                        -1.63787514e-01,
                        1.60288364e-01,
                        -2.30847582e-01,
                        -2.22037435e-01,
                        4.50191796e-02,
                        -1.09636143e-01,
                        -6.00997508e-02,
                        2.14693844e-01,
                        -8.51289369e-03,
                        3.88416052e-01,
                        -1.18085533e-01,
                        -8.57385695e-02,
                        -1.18666515e-02,
                        3.29741478e-01,
                        2.03779504e-01,
                        -1.69492334e-01,
                        1.94324687e-01,
                        4.16374728e-02,
                        6.83876276e-02,
                        -1.85160581e-02,
                        -3.73600274e-02,
                        7.14804307e-02,
                        -3.01446438e-01,
                        1.74035393e-02,
                        3.35123807e-01,
                        4.17102218e-01,
                        -1.58562332e-01,
                        -2.54483074e-02,
                        1.99573949e-01,
                        7.95029774e-02,
                        4.82958555e-01,
                        -4.58213627e-01,
                        -2.67229170e-01,
                        1.27542794e-01,
                        4.47799414e-02,
                        -1.68686539e-01,
                        -8.92183557e-02,
                        9.79782715e-02,
                        2.77656261e-02,
                        2.96871901e-01,
                        6.29860088e-02,
                        1.77246213e-01,
                        -3.15523535e-01,
                        -1.74582571e-01,
                        1.25724282e-02,
                        -4.54988703e-02,
                        -1.29154682e-01,
                        2.13568255e-01,
                        -1.40891463e-01,
                        -2.66211092e-01,
                        2.62144595e-01,
                        -1.42889306e-01,
                        -6.67349845e-02,
                        1.63380653e-02,
                        -1.92995071e-02,
                        1.14811368e-01,
                        1.43584609e-01,
                        -2.65347548e-02,
                        9.32592154e-02,
                        2.23325342e-01,
                        -1.87100068e-01,
                        5.71197420e-02,
                        2.71253467e-01,
                        1.02890521e-01,
                        -3.04941833e-03,
                        1.10537663e-01,
                        2.75728375e-01,
                        2.11693868e-01,
                        -1.80009842e-01,
                        2.99300496e-02,
                        1.77923322e-01,
                        -4.53491032e-01,
                        1.94149211e-01,
                        2.47100577e-01,
                        -9.95091349e-03,
                        1.99301094e-01,
                        2.06564650e-01,
                        -1.99648179e-02,
                        -1.89629450e-01,
                        1.61689930e-02,
                        1.04817756e-01,
                        2.06400946e-01,
                    ],
                    [
                        -3.86251360e-02,
                        -3.07115853e-01,
                        3.74236181e-02,
                        -1.71237886e-01,
                        -2.77359486e-02,
                        -4.08068746e-02,
                        6.91853091e-02,
                        2.65322998e-03,
                        1.23958819e-01,
                        2.20951259e-01,
                        8.36500078e-02,
                        -3.14413190e-01,
                        1.34745941e-01,
                        -8.25274512e-02,
                        3.36270213e-01,
                        -2.49634441e-02,
                        -1.06189884e-02,
                        7.18819201e-02,
                        -1.73392966e-01,
                        2.98084319e-01,
                        -1.25626653e-01,
                        -2.17043106e-02,
                        2.87982523e-01,
                        -3.41932476e-03,
                        -6.89984411e-02,
                        -7.14176893e-03,
                        4.49542440e-02,
                        5.16424477e-01,
                        -1.02622584e-01,
                        2.02640779e-02,
                        2.30106711e-02,
                        1.93037808e-01,
                        2.03393996e-01,
                        -4.34632808e-01,
                        5.74068353e-02,
                        9.66466218e-02,
                        -7.19890296e-02,
                        4.23505977e-02,
                        -1.60067812e-01,
                        3.88947129e-02,
                        -3.32329512e-01,
                        1.91072702e-01,
                        3.79437394e-02,
                        4.33839470e-01,
                        -1.29231960e-01,
                        -1.46881789e-01,
                        2.47269243e-01,
                        2.86379829e-02,
                        2.92926908e-01,
                        -2.97341049e-01,
                        -3.40239167e-01,
                        1.52589783e-01,
                        8.81168991e-02,
                        1.40633471e-02,
                        -8.83188471e-02,
                        2.48367310e-01,
                        3.41982782e-01,
                        3.18781316e-01,
                        -1.15148552e-01,
                        2.54325420e-01,
                        -1.82771236e-01,
                        5.33889830e-02,
                        2.12034155e-02,
                        -9.78844613e-02,
                        -1.61611915e-01,
                        3.54084134e-01,
                        -1.25332132e-01,
                        -2.07989410e-01,
                        2.35610008e-02,
                        -1.35993093e-01,
                        -1.97377697e-01,
                        -1.21356212e-02,
                        7.86775351e-03,
                        4.71337497e-01,
                        9.49376822e-03,
                        4.25345525e-02,
                        1.14162050e-01,
                        6.27847165e-02,
                        -2.31957644e-01,
                        -8.33211765e-02,
                        2.02719584e-01,
                        -4.64919358e-02,
                        7.57966787e-02,
                        1.01521172e-01,
                        3.16580981e-01,
                        1.49488643e-01,
                        -1.20770879e-01,
                        2.56563038e-01,
                        1.66572407e-01,
                        -6.11343801e-01,
                        2.09183827e-01,
                        6.66101649e-02,
                        1.77328646e-01,
                        1.77777156e-01,
                        2.03266457e-01,
                        1.37545317e-01,
                        -1.38004154e-01,
                        -2.57656008e-01,
                        1.83920860e-01,
                        2.87696868e-02,
                    ],
                    [
                        5.14136627e-04,
                        -2.88997203e-01,
                        -3.34554128e-02,
                        -6.80408552e-02,
                        -4.61972654e-02,
                        1.75428241e-01,
                        1.86710209e-01,
                        -1.51355267e-01,
                        1.28381401e-01,
                        2.87129283e-01,
                        5.22154570e-03,
                        -3.53413224e-01,
                        -3.87947261e-02,
                        5.81918843e-02,
                        3.17016482e-01,
                        -2.51671404e-01,
                        7.01867491e-02,
                        -4.92945537e-02,
                        -2.73323953e-01,
                        1.27938241e-01,
                        -4.11552131e-01,
                        -2.23789401e-02,
                        1.95418939e-01,
                        -3.25946212e-01,
                        4.60528135e-02,
                        1.86884090e-01,
                        6.98191971e-02,
                        2.95638293e-01,
                        -1.80322871e-01,
                        -2.98620313e-02,
                        2.11789399e-01,
                        3.15145910e-01,
                        3.11763227e-01,
                        -5.78147054e-01,
                        6.43244758e-02,
                        -3.14367823e-02,
                        1.86190963e-01,
                        1.71108633e-01,
                        -6.29745722e-02,
                        7.48428777e-02,
                        -4.58003521e-01,
                        -3.01471800e-01,
                        2.17973694e-01,
                        5.73273778e-01,
                        -1.01379365e-01,
                        -2.46951967e-01,
                        1.58989042e-01,
                        -1.79126799e-01,
                        5.24153829e-01,
                        -4.64852840e-01,
                        -2.94867605e-01,
                        1.83558539e-01,
                        2.50552952e-01,
                        -8.56962949e-02,
                        -2.57554710e-01,
                        2.30709136e-01,
                        3.53280157e-01,
                        3.20184112e-01,
                        2.99184099e-02,
                        3.09098989e-01,
                        -2.02217728e-01,
                        -9.29201543e-02,
                        -1.20569356e-02,
                        -1.37087986e-01,
                        -2.16524690e-01,
                        1.39787242e-01,
                        -1.27902150e-01,
                        -2.64347821e-01,
                        9.29943919e-02,
                        -1.57217175e-01,
                        -3.86638314e-01,
                        7.90465698e-02,
                        4.07211930e-02,
                        -3.07695866e-02,
                        1.27393469e-01,
                        -1.18648581e-01,
                        -7.21216127e-02,
                        -3.71141285e-02,
                        -3.37082207e-01,
                        -6.23112544e-02,
                        3.52166295e-01,
                        1.51260465e-01,
                        5.03427610e-02,
                        1.90433189e-01,
                        5.21304548e-01,
                        3.85341585e-01,
                        -1.26457050e-01,
                        1.54961571e-01,
                        5.29025272e-02,
                        -5.06486952e-01,
                        2.28533953e-01,
                        1.78438187e-01,
                        -4.14765030e-02,
                        2.01903239e-01,
                        -3.89365852e-02,
                        8.84043872e-02,
                        -1.55351788e-01,
                        -9.06582028e-02,
                        9.95599255e-02,
                        -4.79989760e-02,
                    ],
                ],
                dtype=np.float32,
            )
        }

        expected = [
            np.array(
                [
                    [
                        129,
                        72,
                        168,
                        128,
                        157,
                        153,
                        191,
                        160,
                        178,
                        170,
                        168,
                        105,
                        174,
                        155,
                        223,
                        133,
                        160,
                        172,
                        135,
                        217,
                        119,
                        150,
                        167,
                        122,
                        137,
                        184,
                        142,
                        232,
                        121,
                        135,
                        129,
                        169,
                        180,
                        0,
                        159,
                        146,
                        167,
                        157,
                        93,
                        176,
                        97,
                        137,
                        159,
                        255,
                        124,
                        97,
                        197,
                        136,
                        222,
                        74,
                        85,
                        158,
                        202,
                        115,
                        124,
                        160,
                        190,
                        236,
                        115,
                        223,
                        107,
                        149,
                        140,
                        113,
                        92,
                        196,
                        90,
                        130,
                        185,
                        111,
                        94,
                        143,
                        170,
                        157,
                        148,
                        121,
                        131,
                        142,
                        88,
                        163,
                        192,
                        150,
                        145,
                        176,
                        193,
                        199,
                        122,
                        198,
                        159,
                        18,
                        224,
                        145,
                        160,
                        179,
                        180,
                        195,
                        97,
                        70,
                        155,
                        141,
                    ],
                    [
                        98,
                        76,
                        166,
                        120,
                        133,
                        130,
                        163,
                        142,
                        171,
                        165,
                        158,
                        99,
                        161,
                        153,
                        211,
                        167,
                        164,
                        153,
                        124,
                        167,
                        103,
                        148,
                        184,
                        127,
                        144,
                        158,
                        165,
                        195,
                        136,
                        125,
                        149,
                        189,
                        185,
                        72,
                        165,
                        137,
                        173,
                        161,
                        106,
                        159,
                        112,
                        123,
                        164,
                        202,
                        122,
                        105,
                        186,
                        160,
                        216,
                        77,
                        99,
                        134,
                        174,
                        113,
                        107,
                        177,
                        214,
                        221,
                        137,
                        211,
                        91,
                        120,
                        161,
                        132,
                        114,
                        182,
                        110,
                        127,
                        167,
                        112,
                        112,
                        174,
                        159,
                        174,
                        136,
                        128,
                        133,
                        174,
                        84,
                        157,
                        153,
                        165,
                        131,
                        168,
                        207,
                        207,
                        117,
                        197,
                        146,
                        51,
                        192,
                        157,
                        164,
                        132,
                        165,
                        156,
                        104,
                        111,
                        158,
                        168,
                    ],
                    [
                        131,
                        83,
                        160,
                        122,
                        145,
                        135,
                        186,
                        150,
                        165,
                        182,
                        137,
                        89,
                        99,
                        155,
                        202,
                        134,
                        165,
                        131,
                        113,
                        173,
                        100,
                        102,
                        151,
                        123,
                        132,
                        183,
                        141,
                        215,
                        121,
                        127,
                        141,
                        204,
                        181,
                        112,
                        179,
                        151,
                        156,
                        140,
                        136,
                        156,
                        87,
                        146,
                        205,
                        220,
                        114,
                        138,
                        180,
                        158,
                        233,
                        58,
                        93,
                        167,
                        151,
                        112,
                        126,
                        161,
                        148,
                        198,
                        155,
                        176,
                        84,
                        111,
                        145,
                        135,
                        119,
                        183,
                        117,
                        94,
                        192,
                        116,
                        131,
                        146,
                        139,
                        164,
                        170,
                        138,
                        160,
                        184,
                        108,
                        154,
                        193,
                        162,
                        142,
                        164,
                        194,
                        182,
                        110,
                        149,
                        176,
                        59,
                        179,
                        189,
                        141,
                        180,
                        181,
                        139,
                        108,
                        146,
                        162,
                        181,
                    ],
                    [
                        136,
                        86,
                        150,
                        111,
                        138,
                        135,
                        156,
                        143,
                        166,
                        184,
                        159,
                        85,
                        168,
                        128,
                        205,
                        138,
                        141,
                        156,
                        111,
                        198,
                        120,
                        139,
                        196,
                        142,
                        130,
                        142,
                        151,
                        239,
                        124,
                        147,
                        147,
                        179,
                        181,
                        62,
                        154,
                        161,
                        130,
                        151,
                        113,
                        150,
                        81,
                        178,
                        150,
                        224,
                        119,
                        116,
                        189,
                        148,
                        197,
                        88,
                        80,
                        171,
                        159,
                        146,
                        127,
                        189,
                        206,
                        202,
                        122,
                        190,
                        109,
                        153,
                        147,
                        125,
                        113,
                        209,
                        120,
                        104,
                        147,
                        118,
                        106,
                        141,
                        144,
                        230,
                        145,
                        151,
                        164,
                        155,
                        100,
                        128,
                        181,
                        134,
                        157,
                        162,
                        202,
                        171,
                        121,
                        191,
                        174,
                        30,
                        182,
                        155,
                        176,
                        176,
                        181,
                        169,
                        117,
                        95,
                        177,
                        148,
                    ],
                    [
                        143,
                        89,
                        137,
                        130,
                        134,
                        176,
                        178,
                        115,
                        167,
                        196,
                        144,
                        77,
                        136,
                        154,
                        202,
                        96,
                        156,
                        134,
                        92,
                        167,
                        67,
                        139,
                        179,
                        82,
                        152,
                        178,
                        156,
                        198,
                        110,
                        137,
                        182,
                        202,
                        201,
                        36,
                        155,
                        137,
                        178,
                        175,
                        131,
                        157,
                        58,
                        87,
                        183,
                        249,
                        124,
                        97,
                        173,
                        110,
                        240,
                        57,
                        88,
                        177,
                        190,
                        127,
                        95,
                        186,
                        209,
                        202,
                        149,
                        200,
                        105,
                        126,
                        141,
                        118,
                        103,
                        169,
                        119,
                        94,
                        160,
                        114,
                        71,
                        158,
                        151,
                        137,
                        167,
                        121,
                        130,
                        136,
                        80,
                        131,
                        208,
                        171,
                        152,
                        178,
                        240,
                        215,
                        120,
                        172,
                        153,
                        49,
                        185,
                        176,
                        135,
                        180,
                        136,
                        159,
                        114,
                        126,
                        161,
                        134,
                    ],
                ],
                dtype=np.uint8,
            ),
            np.array(0.005387083161622286, dtype=np.float32),
            np.array(143, dtype=np.uint8),
        ]

        X = make_tensor_value_info("X", TensorProto.FLOAT, None)
        Y = make_tensor_value_info("Y", TensorProto.UINT8, None)
        Scale = make_tensor_value_info("scale", TensorProto.FLOAT, None)
        Zp = make_tensor_value_info("zp", TensorProto.UINT8, None)

        nodes = [
            make_node(
                "DynamicQuantizeLinear",
                ["X"],
                ["Y", "scale", "zp"],
            ),
        ]
        model = make_model(
            make_graph(nodes, "g", [X], [Y, Scale, Zp]),
            opset_imports=[make_opsetid("", onnx_opset_version() - 1)],
        )
        ref = ReferenceEvaluator(model)
        got = ref.run(None, feeds)
        self.assertEqual(len(got), 3)
        for i in range(2, -1, -1):
            assert_allclose(expected[i], got[i])

    @parameterized.parameterized.expand(
        [
            (["abc", "def"], [".com", ".net"], ["abc.com", "def.net"], (2,)),
            (["cat", "dog", "snake"], ["s"], ["cats", "dogs", "snakes"], (3,)),
            ("cat", "s", "cats", ()),
            (["a", "ß", "y"], ["a", "ß", "y"], ["aa", "ßß", "yy"], (3,)),
        ]
    )
    def test_string_concat(self, a, b, expected, expected_shape):
        A = make_tensor_value_info("A", TensorProto.STRING, None)
        B = make_tensor_value_info("B", TensorProto.STRING, None)
        Y = make_tensor_value_info("Y", TensorProto.STRING, None)
        node = make_node("StringConcat", inputs=["A", "B"], outputs=["Y"])
        model = make_model(make_graph([node], "g", [A, B], [Y]))
        ref = ReferenceEvaluator(model)
        result, *_ = ref.run(None, {"A": np.array(a), "B": np.array(b)})
        np.testing.assert_array_equal(result, expected)
        self.assertIn(result.dtype.kind, {"O", "U"})
        self.assertEqual(result.shape, expected_shape)

    @parameterized.parameterized.expand(
        [
            (
                ["1,2,3", "4,5,6"],
                ",",
                None,
                [["1", "2", "3"], ["4", "5", "6"]],
                [3, 3],
            ),
            (
                ["1,", "4,6", ""],
                ",",
                None,
                [["1", ""], ["4", "6"], ["", ""]],
                [2, 2, 1],
            ),
            (
                ["1", "4,6", "4,5,6"],
                ",",
                1,
                [["1", ""], ["4", "6"], ["4", "5,6"]],
                [1, 2, 2],
            ),
            (
                [["1,", "4,6", "4,5,6"], ["1,", "4,6", "4,5,6"]],
                ",",
                None,
                [
                    [["1", "", ""], ["4", "6", ""], ["4", "5", "6"]],
                    [["1", "", ""], ["4", "6", ""], ["4", "5", "6"]],
                ],
                [[2, 2, 3], [2, 2, 3]],
            ),
            (
                ["hello world !", "  hello   world !", " hello world   ! "],
                None,
                None,
                [
                    ["hello", "world", "!"],
                    ["hello", "world", "!"],
                    ["hello", "world", "!"],
                ],
                [3, 3, 3],
            ),
            (
                ["hello world !", "  hello   world !", " hello world   ! "],
                "",
                None,
                [
                    ["hello", "world", "!"],
                    ["hello", "world", "!"],
                    ["hello", "world", "!"],
                ],
                [3, 3, 3],
            ),
            (
                ["o-n-n--x-", "o-n----nx"],
                "-",
                None,
                [["o", "n", "n", "", "x", ""], ["o", "n", "", "", "", "nx"]],
                [6, 6],
            ),
            (
                [],
                " ",
                2,
                np.array([]).reshape((0, 0)),
                [],
            ),
        ]
    )
    def test_string_split(
        self,
        x,
        delimiter,
        maxsplit,
        expected_split,
        expected_num_splits,
    ):
        X = make_tensor_value_info("X", TensorProto.STRING, (None))
        Splits = make_tensor_value_info("Splits", TensorProto.STRING, (None))
        MaxSplits = make_tensor_value_info("MaxSplits", TensorProto.INT32, (None))
        node = make_node(
            "StringSplit",
            inputs=["X"],
            outputs=["Splits", "MaxSplits"],
            delimiter=delimiter,
            maxsplit=maxsplit,
        )
        model = make_model(make_graph([node], "g", [X], [Splits, MaxSplits]))
        ref = ReferenceEvaluator(model)
        x = np.array(x, dtype=object)
        result, num_splits, *_ = ref.run(None, {"X": x})
        np.testing.assert_array_equal(result, np.array(expected_split, dtype=object))
        np.testing.assert_array_equal(
            num_splits, np.array(expected_num_splits, dtype=np.int64)
        )

    def test_qlinearconv_int8(self):
        node = make_node(
            "QLinearMatMul",
            inputs=[
                "a",
                "a_scale",
                "a_zero_point",
                "b",
                "b_scale",
                "b_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            outputs=["y"],
        )
        graph = make_graph(
            [node],
            "g",
            [
                make_tensor_value_info("a", TensorProto.FLOAT, [None, None]),
                make_tensor_value_info("a_scale", TensorProto.FLOAT, [1]),
                make_tensor_value_info("a_zero_point", TensorProto.INT8, [1]),
                make_tensor_value_info("b", TensorProto.FLOAT, [None, None]),
                make_tensor_value_info("b_scale", TensorProto.FLOAT, [1]),
                make_tensor_value_info("b_zero_point", TensorProto.INT8, [1]),
                make_tensor_value_info("y_scale", TensorProto.FLOAT, [1]),
                make_tensor_value_info("y_zero_point", TensorProto.INT8, [1]),
            ],
            [make_tensor_value_info("y", TensorProto.FLOAT, [None, None])],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 20)], ir_version=9
        )
        sess = ReferenceEvaluator(onnx_model)

        a = np.array([[208, 236, 0, 238], [3, 214, 255, 29]])
        a -= 127
        a = a.astype(np.int8)

        a_scale = np.array([0.0066], dtype=np.float32)
        a_zero_point = np.array([113 - 127], dtype=np.int8)

        b = np.array([[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]])
        b -= 127
        b = b.astype(np.int8)

        b_scale = np.array([0.00705], dtype=np.float32)
        b_zero_point = np.array([114 - 127], dtype=np.int8)

        y_scale = np.array([0.0107], dtype=np.float32)
        y_zero_point = np.array([118 - 127], dtype=np.int8)

        got = sess.run(
            None,
            dict(
                a=a,
                a_scale=a_scale,
                a_zero_point=a_zero_point,
                b=b,
                b_scale=b_scale,
                b_zero_point=b_zero_point,
                y_scale=y_scale,
                y_zero_point=y_zero_point,
            ),
        )

        np.testing.assert_array_equal(
            np.array([[41, -12, -9], [1, -75, 20]], dtype=np.int8), got[0]
        )

    @parameterized.parameterized.expand(
        [
            (
                ["www.google.com", "www.facebook.com", "www.bbc.co.uk"],
                r"www\.[\w.-]+\.\bcom\b",
                [True, True, False],
                (3,),
            ),
            (
                [["Onnx", "tensorflow", "Numpy"], ["Pytorch", "Cython", "numba"]],
                r"^[A-Z][a-z]*$",
                [[True, False, True], [True, True, False]],
                (2, 3),
            ),
            (
                [
                    "account@gmail.com",
                    "account@hotmail.com",
                    "not email",
                    "account2@yahoo.com",
                ],
                r"(\W|^)[\w.\-]{0,25}@(yahoo|gmail)\.com(\W|$)",
                [True, False, False, True],
                (4,),
            ),
        ]
    )
    @unittest.skipIf(
        sys.platform == "win32", "google-re2 package is not built for win32"
    )
    def test_regex_full_match(self, x, pattern, expected, expected_shape):
        X = make_tensor_value_info("X", TensorProto.STRING, None)
        Y = make_tensor_value_info("Y", TensorProto.BOOL, None)
        node = make_node("RegexFullMatch", inputs=["X"], outputs=["Y"], pattern=pattern)
        model = make_model(make_graph([node], "g", [X], [Y]))
        ref = ReferenceEvaluator(model)
        result, *_ = ref.run(None, {"X": np.array(x)})
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype.kind, "b")
        self.assertEqual(result.shape, expected_shape)

    @unittest.skipIf(
        sys.platform == "win32", "google-re2 package is not built for win32"
    )
    def test_regex_invalid_pattern(self):
        X = make_tensor_value_info("X", TensorProto.STRING, None)
        Y = make_tensor_value_info("Y", TensorProto.BOOL, None)
        node = make_node("RegexFullMatch", inputs=["X"], outputs=["Y"], pattern="x)")
        model = make_model(make_graph([node], "g", [X], [Y]))
        ref = ReferenceEvaluator(model)
        with self.assertRaises(ValueError):
            ref.run(None, {"X": np.array(["x"])})

    @parameterized.parameterized.expand(
        [
            (
                TensorProto.UINT4,
                [-1, 0, 1.5, 2, 3.3, 10, 20, 40],
                [0, 0, 2, 2, 4, 10, 20, 30],
            ),
            (TensorProto.UINT4, [-1, 0, 1.5, 2, 3.3, 10, 40], [0, 0, 2, 2, 4, 10, 30]),
            (TensorProto.UINT4, [0], [0]),
            (
                TensorProto.INT4,
                [-20, -14.5, 0, 1.5, 2, 3.3, 10, 20],
                [-16, -14, 0, 2, 2, 4, 10, 14],
            ),
            (
                TensorProto.INT4,
                [-20, -14.5, 0, 1.5, 2, 3.3, 10],
                [-16, -14, 0, 2, 2, 4, 10],
            ),
            (TensorProto.INT4, [0], [0]),
        ]
    )
    @unittest.skipIf(
        version_utils.numpy_older_than("1.22.0"),
        "The test requires numpy 1.22.0 or later",
    )
    def test_quantize_linear_int4(self, qtype, data, expected):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        model = make_model(
            make_graph(
                [
                    make_node(
                        "Constant",
                        [],
                        ["scale"],
                        value=make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                    ),
                    make_node(
                        "Constant",
                        [],
                        ["zero"],
                        value=make_tensor("zero", qtype, [1], [0]),
                    ),
                    make_node("QuantizeLinear", ["X", "scale", "zero"], ["T"]),
                    make_node("DequantizeLinear", ["T", "scale"], ["Y"], axis=0),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        got = ref.run(None, {"X": np.asarray(data)})
        assert_allclose(expected, got[0])

    @parameterized.parameterized.expand(
        itertools.product(
            (TensorProto.FLOAT, TensorProto.FLOAT16),
            (TensorProto.UINT4, TensorProto.INT4),
        )
    )
    def test_cast_int4_output(self, cast_from, cast_to):
        X = make_tensor_value_info("X", cast_from, [None])
        Y = make_tensor_value_info("Y", cast_to, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["Y"], to=cast_to),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2.4, 2.6, 4, 10], dtype=np.float32)
        signed = cast_to == TensorProto.INT4
        expected1 = np.array(
            [subbyte.float32_to_4bit_unpacked(x, signed=signed) for x in data]
        )
        got = ref.run(None, {"X": data})
        self.assertEqual(expected1.tolist(), got[0].tolist())

    @parameterized.parameterized.expand(
        itertools.product(
            (TensorProto.UINT4, TensorProto.INT4),
            (TensorProto.FLOAT, TensorProto.FLOAT16),
        )
    )
    def test_cast_int4_input(self, cast_from, cast_to):
        X = make_tensor_value_info("X", cast_from, [None])
        Y = make_tensor_value_info("Y", cast_to, [None])
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT),
                ],
                "g",
                [X],
                [Y],
            )
        )
        ref = ReferenceEvaluator(model)
        data = np.array(range(7), dtype=np.float32)
        cast_from_np = custom.uint4 if cast_from == TensorProto.UINT4 else custom.int4
        expected1 = np.array(
            [subbyte.float32_to_4bit_unpacked(x, cast_from_np) for x in data]
        )
        got = ref.run(None, {"X": data})
        self.assertEqual(expected1.tolist(), got[0].tolist())

    def test_a_function_calling_a_function_once(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        output = make_tensor_value_info("output", TensorProto.FLOAT, ["N"])
        Z = make_tensor_value_info("output", TensorProto.FLOAT, ["N"])

        func_def_add = make_function(
            "this",
            "fctadd",
            ["input2"],
            ["output"],
            [
                make_node("Constant", [], ["one"], value_floats=[1.0], name="CC0"),
                make_node("Add", ["input2", "one"], ["output"], name="A1"),
            ],
            opset_imports=[make_operatorsetid("", 15)],
        )

        func_def = make_function(
            "this",
            "fct",
            ["input"],
            ["output"],
            [
                make_node("Constant", [], ["one"], value_floats=[1.0], name="CC"),
                make_node("Greater", ["input", "one"], ["cond"]),
                make_node(
                    "If",
                    ["cond"],
                    ["output"],
                    then_branch=make_graph(
                        [make_node("fctadd", ["input"], ["output"], domain="this")],
                        "gthen",
                        [],
                        [output],
                    ),
                    else_branch=make_graph(
                        [make_node("Add", ["input", "one"], ["output"], domain="")],
                        "gelse",
                        [],
                        [output],
                    ),
                    name=":IF",
                ),
            ],
            opset_imports=[
                make_operatorsetid("", 15),
                make_operatorsetid("this", 1),
            ],
        )

        model_def = make_model(
            make_graph(
                [
                    make_node("fct", ["X"], ["output"], domain="this"),
                ],
                "test",
                [X],
                [Z],
            ),
            ir_version=7,
            opset_imports=[
                make_operatorsetid("", 15),
                make_operatorsetid("this", 1),
            ],
            functions=[func_def_add, func_def],
        )

        feeds = {"X": np.array([-5], dtype=np.float32)}
        oinf = ReferenceEvaluator(model_def)
        expected = oinf.run(None, feeds)

        # inlining does not work here
        # inlined = inline_local_functions(model_def)
        # oinf = ReferenceEvaluator(inlined)
        # goti = oinf.run(None, feeds)
        # self.assertEqual(expected[0].tolist(), goti[0].tolist())
        self.assertEqual(expected[0], np.array([-4], dtype=np.float32))

    def test_a_function_calling_a_function_double(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        output = make_tensor_value_info("output", TensorProto.FLOAT, ["N"])
        Z = make_tensor_value_info("output", TensorProto.FLOAT, ["N"])

        func_def_add = make_function(
            "this",
            "fctadd",
            ["input2"],
            ["output"],
            [
                make_node("Constant", [], ["one"], value_floats=[1.0], name="CC0"),
                make_node("Add", ["input2", "one"], ["output"], name="A1"),
            ],
            opset_imports=[make_operatorsetid("", 15)],
        )

        func_def = make_function(
            "this",
            "fct",
            ["input"],
            ["output"],
            [
                make_node("Constant", [], ["one"], value_floats=[1.0], name="CC"),
                make_node("Greater", ["input", "one"], ["cond"]),
                make_node(
                    "If",
                    ["cond"],
                    ["output"],
                    then_branch=make_graph(
                        [make_node("fctadd", ["input"], ["output"], domain="this")],
                        "gthen",
                        [],
                        [output],
                    ),
                    else_branch=make_graph(
                        [make_node("Add", ["input", "one"], ["output"], domain="")],
                        "gelse",
                        [],
                        [output],
                    ),
                    name=":IF",
                ),
            ],
            opset_imports=[
                make_operatorsetid("", 15),
                make_operatorsetid("this", 1),
            ],
        )

        model_def = make_model(
            make_graph(
                [
                    make_node("fct", ["X"], ["ztmp"], domain="this"),
                    make_node("fct", ["ztmp"], ["output"], domain="this"),
                ],
                "test",
                [X],
                [Z],
            ),
            ir_version=7,
            opset_imports=[
                make_operatorsetid("", 15),
                make_operatorsetid("this", 1),
            ],
            functions=[func_def_add, func_def],
        )

        feeds = {"X": np.array([-5], dtype=np.float32)}
        oinf = ReferenceEvaluator(model_def)
        expected = oinf.run(None, feeds)

        # inlining does not work here
        # inlined = inline_local_functions(model_def)
        # oinf = ReferenceEvaluator(inlined)
        # goti = oinf.run(None, feeds)
        # self.assertEqual(expected[0].tolist(), goti[0].tolist())
        self.assertEqual(expected[0], np.array([-3], dtype=np.float32))

    def test_overload_reference_implementation(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, ["N"])
        output = make_tensor_value_info("output", TensorProto.FLOAT, ["N"])
        Z = make_tensor_value_info("output", TensorProto.FLOAT, ["N"])

        func_def_add = make_function(
            "this",
            "fctadd",
            ["input2"],
            ["output"],
            [
                make_node("Constant", [], ["one"], value_floats=[1.0], name="CC0"),
                make_node("Add", ["input2", "one"], ["output"], name="A1"),
            ],
            opset_imports=[make_operatorsetid("", 15)],
        )

        func_def = make_function(
            "this",
            "fct",
            ["input"],
            ["output"],
            [
                make_node("Constant", [], ["one"], value_floats=[1.0], name="CC"),
                make_node("Greater", ["input", "one"], ["cond"]),
                make_node(
                    "If",
                    ["cond"],
                    ["output"],
                    then_branch=make_graph(
                        [make_node("fctadd", ["input"], ["output"], domain="this")],
                        "gthen",
                        [],
                        [output],
                    ),
                    else_branch=make_graph(
                        [make_node("Add", ["input", "one"], ["output"], domain="")],
                        "gelse",
                        [],
                        [output],
                    ),
                    name=":IF",
                ),
            ],
            opset_imports=[
                make_operatorsetid("", 15),
                make_operatorsetid("this", 1),
            ],
        )

        model_def = make_model(
            make_graph(
                [
                    make_node("fct", ["X"], ["ztmp"], domain="this"),
                    make_node("fct", ["ztmp"], ["output"], domain="this"),
                ],
                "test",
                [X],
                [Z],
            ),
            ir_version=7,
            opset_imports=[
                make_operatorsetid("", 15),
                make_operatorsetid("this", 1),
            ],
            functions=[func_def_add, func_def],
        )

        class MyReferenceEvaluator(ReferenceEvaluator):
            pass

        oinf = MyReferenceEvaluator(model_def)
        for v in oinf.functions_.values():
            self.assertIsInstance(v, MyReferenceEvaluator)

    @parameterized.parameterized.expand(
        [
            ("UINT4", 0.84),
            ("INT4", 0.84),
            ("FLOAT8E4M3FN", 0.23),
            ("FLOAT8E4M3FNUZ", 0.23),
            ("FLOAT8E5M2", 0.85),
            ("FLOAT8E5M2FNUZ", 0.85),
            ("DOUBLE", 0),
            ("FLOAT", 0),
            ("FLOAT16", 2e-3),
            ("BFLOAT16", 2e-2),
        ]
    )
    @skip_if_no_ml_dtypes
    def test_add_custom_dtype(self, stype, atol):
        itype = getattr(TensorProto, stype)
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["Xc"], to=itype),
                    make_node("Cast", ["Y"], ["Yc"], to=itype),
                    make_node("Add", ["Xc", "Yc"], ["Zc"]),
                    make_node("Cast", ["Zc"], ["Z"], to=TensorProto.FLOAT),
                ],
                "nd",
                [
                    make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None]),
                    make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None]),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, [None, None, None])],
            ),
            opset_imports=[make_opsetid("", 18)],
            ir_version=9,
        )

        ref = ReferenceEvaluator(model, verbose=0)

        x = (np.arange(18) / 6).reshape((2, 3, 3)).astype(np.float32)
        y = (np.arange(18) / 9).reshape((2, 3, 3)).astype(np.float32)
        feeds = dict(X=x, Y=y)
        expected = x + y
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got, atol=atol)

    @parameterized.parameterized.expand(
        [
            ("DOUBLE",),
            ("FLOAT",),
            ("FLOAT16",),
            ("BFLOAT16",),
            # Comparison fails with ml_dtypes
            # ("FLOAT8E4M3FN", ),
            # ("FLOAT8E4M3FNUZ", ),
            # ("FLOAT8E5M2", ),
            # ("FLOAT8E5M2FNUZ", ),
            # ("INT4", ),
            # ("UINT4", ),
        ]
    )
    @skip_if_no_ml_dtypes
    def test_cmp_custom_dtype(self, stype):
        itype = getattr(TensorProto, stype)
        model = make_model(
            make_graph(
                [
                    make_node("Cast", ["X"], ["Xc"], to=itype),
                    make_node("Cast", ["Y"], ["Yc"], to=itype),
                    make_node("Greater", ["Xc", "Yc"], ["Zc"]),
                    make_node("Cast", ["Zc"], ["Z"], to=TensorProto.BOOL),
                ],
                "nd",
                [
                    make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None]),
                    make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None]),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, [None, None, None])],
            ),
            opset_imports=[make_opsetid("", 18)],
            ir_version=9,
        )

        ref = ReferenceEvaluator(model)

        x = (np.arange(18) / 18).reshape((2, 3, 3)).astype(np.float32)
        y = ((np.arange(18) - 9) / 18).reshape((2, 3, 3)).astype(np.float32)
        feeds = dict(X=x, Y=y)
        expected = x >= y
        got = ref.run(None, feeds)[0]
        assert_almost_equal(expected, got)

    def test_scatter_elements_4d(self):
        model = make_model(
            make_graph(
                [
                    make_node(
                        "ScatterElements",
                        ["data", "indices", "updates"],
                        ["Z"],
                        axis=3,
                        reduction="add",
                    )
                ],
                "name",
                [
                    make_tensor_value_info("data", TensorProto.FLOAT, None),
                    make_tensor_value_info("indices", TensorProto.INT64, None),
                    make_tensor_value_info("updates", TensorProto.FLOAT, None),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18)],
        )
        data = np.zeros(2**4, dtype=np.float32).reshape((2, 2, 2, 2))
        indices = np.array([[[[0]]]], dtype=np.int64)
        updates = np.array([[[[1]]]], dtype=np.float32)
        y = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
        ).reshape((2, 2, 2, 2))
        ref = ReferenceEvaluator(model)
        got = ref.run(None, {"data": data, "indices": indices, "updates": updates})
        assert_allclose(y, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
