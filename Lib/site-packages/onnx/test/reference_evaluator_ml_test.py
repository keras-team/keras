# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# type: ignore
from __future__ import annotations

import itertools
import unittest
from functools import wraps
from os import getenv

import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore
from parameterized import parameterized

import onnx
from onnx import ONNX_ML, TensorProto, TypeProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_ml_opset_version, onnx_opset_version
from onnx.helper import (
    make_graph,
    make_model_gen_version,
    make_node,
    make_opsetid,
    make_tensor,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.aionnxml.op_tree_ensemble import (
    AggregationFunction,
    Mode,
    PostTransform,
)

# TODO (https://github.com/microsoft/onnxruntime/issues/14932): Get max supported version from onnxruntime directly
# For now, bump the version in CIs whenever there is a new onnxruntime release
ORT_MAX_IR_SUPPORTED_VERSION = int(getenv("ORT_MAX_IR_SUPPORTED_VERSION", "8"))
ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION = int(
    getenv("ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION", "18")
)
ORT_MAX_ML_OPSET_SUPPORTED_VERSION = int(
    getenv("ORT_MAX_ML_OPSET_SUPPORTED_VERSION", "3")
)

TARGET_OPSET = onnx_opset_version() - 2
TARGET_OPSET_ML = onnx_ml_opset_version()
OPSETS = [make_opsetid("", TARGET_OPSET), make_opsetid("ai.onnx.ml", TARGET_OPSET_ML)]


def has_onnxruntime():
    try:
        import onnxruntime

        del onnxruntime
    except ImportError:
        return False
    return True


def skip_if_no_onnxruntime(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not has_onnxruntime():
            raise unittest.SkipTest("onnxruntime not installed")
        fn(*args, **kwargs)

    return wrapper


class TestReferenceEvaluatorAiOnnxMl(unittest.TestCase):
    @staticmethod
    def _check_ort(onx, feeds, atol=0, rtol=0, equal=False, rev=False):
        if not has_onnxruntime():
            return
        from onnxruntime import InferenceSession

        onnx_domain_opset = ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION
        ml_domain_opset = ORT_MAX_ML_OPSET_SUPPORTED_VERSION
        for opset in onx.opset_import:
            if opset.domain in ("", "ai.onnx"):
                onnx_domain_opset = opset.version
                break
        for opset in onx.opset_import:
            if opset.domain == "ai.onnx.ml":
                ml_domain_opset = opset.version
                break
        # The new IR or opset version is not supported by onnxruntime yet
        if (
            onx.ir_version > ORT_MAX_IR_SUPPORTED_VERSION
            or onnx_domain_opset > ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION
            or ml_domain_opset > ORT_MAX_ML_OPSET_SUPPORTED_VERSION
        ):
            return

        ort = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        sess = ReferenceEvaluator(onx)
        expected = ort.run(None, feeds)
        got = sess.run(None, feeds)
        if len(expected) != len(got):
            raise AssertionError(
                f"onnxruntime returns a different number of output "
                f"{len(expected)} != {len(sess)} (ReferenceEvaluator)."
            )
        look = zip(reversed(expected), reversed(got)) if rev else zip(expected, got)
        for i, (e, g) in enumerate(look):
            if e.shape != g.shape:
                raise AssertionError(
                    f"Unexpected shape {g.shape} for output {i} "
                    f"(expecting {e.shape})\n{e!r}\n---\n{g!r}."
                )
            if equal:
                if e.tolist() != g.tolist():
                    raise AssertionError(
                        f"Discrepancies for output {i}"
                        f"\nexpected=\n{e}\n!=\nresults=\n{g}"
                    )
            else:
                assert_allclose(
                    actual=g,
                    desired=e,
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"Discrepancies for output {i} expected[0]={e.ravel()[0]}.",
                )

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_binarizer(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node("Binarizer", ["X"], ["Y"], threshold=5.5, domain="ai.onnx.ml")
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(12).reshape((3, 4)).astype(np.float32)
        expected = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.float32
        )
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_scaler(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "Scaler", ["X"], ["Y"], scale=[0.5], offset=[-4.5], domain="ai.onnx.ml"
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(12).reshape((3, 4)).astype(np.float32)
        expected = np.array(
            [
                [2.25, 2.75, 3.25, 3.75],
                [4.25, 4.75, 5.25, 5.75],
                [6.25, 6.75, 7.25, 7.75],
            ],
            dtype=np.float32,
        )
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_array_feature_extractor(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "ArrayFeatureExtractor", ["X", "A"], ["Y"], domain="ai.onnx.ml"
        )
        graph = make_graph([node1], "ml", [X, A], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(12).reshape((3, 4)).astype(np.float32)

        expected = np.array([[0, 4, 8]], dtype=np.float32).T
        feeds = {"X": x, "A": np.array([0], dtype=np.int64)}
        self._check_ort(onx, feeds)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, feeds)[0]
        assert_allclose(expected, got)

        expected = np.array([[0, 4, 8], [1, 5, 9]], dtype=np.float32).T
        feeds = {"X": x, "A": np.array([0, 1], dtype=np.int64)}
        self._check_ort(onx, feeds)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, feeds)[0]
        assert_allclose(expected, got)

        expected = np.array(
            [[0, 4, 8], [1, 5, 9], [0, 4, 8], [1, 5, 9], [0, 4, 8], [1, 5, 9]],
            dtype=np.float32,
        ).T
        feeds = {"X": x, "A": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)}
        self._check_ort(onx, feeds)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, feeds)[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_normalizer(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        x = np.arange(12).reshape((3, 4)).astype(np.float32)
        expected = {
            "MAX": x / x.max(axis=1, keepdims=1),
            "L1": x / np.abs(x).sum(axis=1, keepdims=1),
            "L2": x / (x**2).sum(axis=1, keepdims=1) ** 0.5,
        }
        for norm, value in expected.items():
            with self.subTest(norm=norm):
                node1 = make_node(
                    "Normalizer", ["X"], ["Y"], norm=norm, domain="ai.onnx.ml"
                )
                graph = make_graph([node1], "ml", [X], [Y])
                onx = make_model_gen_version(graph, opset_imports=OPSETS)
                check_model(onx)

                feeds = {"X": x}
                self._check_ort(onx, feeds, atol=1e-6)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, feeds)[0]
                assert_allclose(value, got, atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_feature_vectorizer(self):
        X = [
            make_tensor_value_info("X0", TensorProto.FLOAT, [None, None]),
            make_tensor_value_info("X1", TensorProto.FLOAT, [None, None]),
        ]
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        x = [
            np.arange(9).reshape((3, 3)).astype(np.float32),
            np.arange(9).reshape((3, 3)).astype(np.float32) + 0.5,
        ]
        expected = {
            (1,): np.array([[0], [3], [6]], dtype=np.float32),
            (2,): np.array([[0, 1], [3, 4], [6, 7]], dtype=np.float32),
            (4,): np.array(
                [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]], dtype=np.float32
            ),
            (1, 1): np.array([[0, 0.5], [3, 3.5], [6, 6.5]], dtype=np.float32),
            (0, 1): np.array([[0.5], [3.5], [6.5]], dtype=np.float32),
        }
        for inputdimensions, value in expected.items():
            att = (
                list(inputdimensions)
                if isinstance(inputdimensions, tuple)
                else inputdimensions
            )
            with self.subTest(inputdimensions=att):
                node1 = make_node(
                    "FeatureVectorizer",
                    [f"X{i}" for i in range(len(att))],
                    ["Y"],
                    inputdimensions=att,
                    domain="ai.onnx.ml",
                )
                graph = make_graph([node1], "ml", X[: len(att)], [Y])
                onx = make_model_gen_version(graph, opset_imports=OPSETS)
                check_model(onx)

                feeds = {f"X{i}": v for i, v in enumerate(x[: len(att)])}
                self._check_ort(onx, feeds, atol=1e-6)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, feeds)[0]
                assert_allclose(value, got, atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_imputer_float(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "Imputer",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            imputed_value_floats=np.array([0], dtype=np.float32),
            replaced_value_float=np.nan,
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([[0, 1, np.nan, 3]], dtype=np.float32).T
        expected = np.array([[0, 1, 0, 3]], dtype=np.float32).T
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_imputer_float_2d(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "Imputer",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            imputed_value_floats=np.array([0, 0.1], dtype=np.float32),
            replaced_value_float=np.nan,
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([[0, 1, np.nan, 3], [0, 1, np.nan, 3]], dtype=np.float32).T
        expected = np.array([[0, 1, 0, 3], [0, 1, 0.1, 3]], dtype=np.float32).T
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_imputer_int(self):
        X = make_tensor_value_info("X", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.INT64, [None, None])
        node1 = make_node(
            "Imputer",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            imputed_value_int64s=np.array([0], dtype=np.int64),
            replaced_value_int64=-1,
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([[0, 1, -1, 3]], dtype=np.int64).T
        expected = np.array([[0, 1, 0, 3]], dtype=np.int64).T
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_label_encoder_float_int(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.INT64, [None, None])
        node1 = make_node(
            "LabelEncoder",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            default_int64=-5,
            keys_floats=[4.0, 1.0, 2.0, 3.0],
            values_int64s=[0, 1, 2, 3],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([[0, 1, np.nan, 3, 4]], dtype=np.float32).T
        expected = np.array([[-5, 1, -5, 3, 0]], dtype=np.int64).T
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_label_encoder_int_string(self):
        X = make_tensor_value_info("X", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.STRING, [None, None])
        node1 = make_node(
            "LabelEncoder",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            default_string="NONE",
            keys_int64s=[1, 2, 3, 4],
            values_strings=["a", "b", "cc", "ddd"],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([[0, 1, 3, 4]], dtype=np.int64).T
        expected = np.array([["NONE"], ["a"], ["cc"], ["ddd"]])
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_label_encoder_int_string_tensor_attributes(self):
        X = make_tensor_value_info("X", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.STRING, [None, None])
        node = make_node(
            "LabelEncoder",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            keys_tensor=make_tensor(
                "keys_tensor", TensorProto.INT64, [4], [1, 2, 3, 4]
            ),
            values_tensor=make_tensor(
                "values_tensor", TensorProto.STRING, [4], ["a", "b", "cc", "ddd"]
            ),
            default_tensor=make_tensor(
                "default_tensor", TensorProto.STRING, [], ["NONE"]
            ),
        )
        graph = make_graph([node], "ml", [X], [Y])
        model = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(model)
        x = np.array([[0, 1, 3, 4]], dtype=np.int64).T
        expected = np.array([["NONE"], ["a"], ["cc"], ["ddd"]])
        self._check_ort(model, {"X": x}, equal=True)
        sess = ReferenceEvaluator(model)
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_dict_vectorizer(self):
        value_type = TypeProto()
        value_type.tensor_type.elem_type = TensorProto.INT64
        onnx_type = TypeProto()
        onnx_type.map_type.key_type = TensorProto.STRING
        onnx_type.map_type.value_type.CopyFrom(value_type)
        value_info = ValueInfoProto()
        value_info.name = "X"
        value_info.type.CopyFrom(onnx_type)

        X = value_info
        Y = make_tensor_value_info("Y", TensorProto.INT64, [None, None])
        node1 = make_node(
            "DictVectorizer",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            string_vocabulary=["a", "c", "b", "z"],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = {"a": np.array(4, dtype=np.int64), "c": np.array(8, dtype=np.int64)}
        expected = np.array([4, 8, 0, 0], dtype=np.int64)
        # Unexpected input data type. Actual: ((map(string,tensor(float)))) , expected: ((map(string,tensor(int64))))
        # self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_one_hot_encoder_int(self):
        X = make_tensor_value_info("X", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        node1 = make_node(
            "OneHotEncoder",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            zeros=1,
            cats_int64s=[1, 2, 3],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([[5, 1, 3], [2, 1, 3]], dtype=np.int64)
        expected = np.array(
            [[[0, 0, 0], [1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=np.float32,
        )
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_one_hot_encoder_string(self):
        X = make_tensor_value_info("X", TensorProto.STRING, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        node1 = make_node(
            "OneHotEncoder",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            zeros=1,
            cats_strings=["c1", "c2", "c3"],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([["c5", "c1", "c3"], ["c2", "c1", "c3"]])
        expected = np.array(
            [[[0, 0, 0], [1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=np.float32,
        )
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_one_hot_encoder_zeros(self):
        X = make_tensor_value_info("X", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        node1 = make_node(
            "OneHotEncoder",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            zeros=0,
            cats_int64s=[1, 2, 3],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.array([[2, 1, 3], [2, 1, 3]], dtype=np.int64)
        expected = np.array(
            [[[0, 1, 0], [1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=np.float32,
        )
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_linear_regressor(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "LinearRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            coefficients=[0.3, -0.77],
            intercepts=[0.5],
            post_transform="NONE",
            targets=1,
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(6).reshape((-1, 2)).astype(np.float32)
        expected = np.array([[-0.27], [-1.21], [-2.15]], dtype=np.float32)
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})
        assert_allclose(expected, got[0], atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_linear_regressor_2(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "LinearRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            coefficients=[0.3, -0.77, 0.3, -0.77],
            intercepts=[0.5, 0.7],
            post_transform="NONE",
            targets=2,
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(6).reshape((-1, 2)).astype(np.float32)
        expected = np.array(
            [[-0.27, -0.07], [-1.21, -1.01], [-2.15, -1.95]], dtype=np.float32
        )
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})
        assert_allclose(expected, got[0], atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_linear_classifier_multi(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        In = make_tensor_value_info("I", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        expected_post = {
            "NONE": [
                np.array([0, 2, 2], dtype=np.int64),
                np.array(
                    [[2.41, -2.12, 0.59], [0.67, -1.14, 1.35], [-1.07, -0.16, 2.11]],
                    dtype=np.float32,
                ),
            ],
            "LOGISTIC": [
                np.array([0, 2, 2], dtype=np.int64),
                np.array(
                    [
                        [0.917587, 0.107168, 0.643365],
                        [0.661503, 0.24232, 0.79413],
                        [0.255403, 0.460085, 0.891871],
                    ],
                    dtype=np.float32,
                ),
            ],
            "SOFTMAX": [
                np.array([0, 2, 2], dtype=np.int64),
                np.array(
                    [
                        [0.852656, 0.009192, 0.138152],
                        [0.318722, 0.05216, 0.629118],
                        [0.036323, 0.090237, 0.87344],
                    ],
                    dtype=np.float32,
                ),
            ],
            "SOFTMAX_ZERO": [
                np.array([0, 2, 2], dtype=np.int64),
                np.array(
                    [
                        [0.852656, 0.009192, 0.138152],
                        [0.318722, 0.05216, 0.629118],
                        [0.036323, 0.090237, 0.87344],
                    ],
                    dtype=np.float32,
                ),
            ],
            "PROBIT": [
                np.array([1, 1, 1], dtype=np.int64),
                np.array(
                    [
                        [-0.527324, -0.445471, -1.080504],
                        [-0.067731, 0.316014, -0.310748],
                        [0.377252, 1.405167, 0.295001],
                    ],
                    dtype=np.float32,
                ),
            ],
        }
        for post in ("SOFTMAX", "NONE", "LOGISTIC", "SOFTMAX_ZERO", "PROBIT"):
            if post == "PROBIT":
                coefficients = [0.058, 0.029, 0.09, 0.058, 0.029, 0.09]
                intercepts = [0.27, 0.27, 0.05]
            else:
                coefficients = [-0.58, -0.29, -0.09, 0.58, 0.29, 0.09]
                intercepts = [2.7, -2.7, 0.5]
            with self.subTest(post_transform=post):
                node1 = make_node(
                    "LinearClassifier",
                    ["X"],
                    ["I", "Y"],
                    domain="ai.onnx.ml",
                    classlabels_ints=[0, 1, 2],
                    coefficients=coefficients,
                    intercepts=intercepts,
                    multi_class=0,
                    post_transform=post,
                )
                graph = make_graph([node1], "ml", [X], [In, Y])
                onx = make_model_gen_version(graph, opset_imports=OPSETS)
                check_model(onx)
                x = np.arange(6).reshape((-1, 2)).astype(np.float32)
                self._check_ort(onx, {"X": x}, rev=True, atol=1e-4)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                expected = expected_post[post]
                assert_allclose(expected[1], got[1], atol=1e-4)
                assert_allclose(expected[0], got[0])

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_linear_classifier_binary(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        In = make_tensor_value_info("I", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        expected_post = {
            "NONE": [
                np.array([1, 1], dtype=np.int64),
                np.array([[-9.53, 9.53], [-6.65, 6.65]], dtype=np.float32),
            ],
            "LOGISTIC": [
                np.array([1, 1], dtype=np.int64),
                np.array(
                    [[7.263436e-05, 9.999274e-01], [1.292350e-03, 9.987077e-01]],
                    dtype=np.float32,
                ),
            ],
            "SOFTMAX": [
                np.array([1, 1], dtype=np.int64),
                np.array(
                    [[5.276517e-09, 1.000000e00], [1.674492e-06, 9.999983e-01]],
                    dtype=np.float32,
                ),
            ],
            "SOFTMAX_ZERO": [
                np.array([1, 1], dtype=np.int64),
                np.array(
                    [[5.276517e-09, 1.000000e00], [1.674492e-06, 9.999983e-01]],
                    dtype=np.float32,
                ),
            ],
        }
        x = np.arange(6).reshape((-1, 3)).astype(np.float32)
        for post in ("SOFTMAX", "NONE", "LOGISTIC", "SOFTMAX_ZERO"):
            expected = expected_post[post]
            with self.subTest(post_transform=post):
                node1 = make_node(
                    "LinearClassifier",
                    ["X"],
                    ["I", "Y"],
                    domain="ai.onnx.ml",
                    classlabels_ints=[0, 1],
                    coefficients=[-0.58, -0.29, -0.09],
                    intercepts=[10.0],
                    multi_class=0,
                    post_transform=post,
                )
                graph = make_graph([node1], "ml", [X], [In, Y])
                onx = make_model_gen_version(graph, opset_imports=OPSETS)
                check_model(onx)
                # onnxruntime answer seems odd.
                # self._check_ort(onx, {"X": x}, rev=True)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_linear_classifier_unary(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        In = make_tensor_value_info("I", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        expected_post = {
            "NONE": [
                np.array([1, 0], dtype=np.int64),
                np.array([[2.23], [-0.65]], dtype=np.float32),
            ],
            "LOGISTIC": [
                np.array([1, 0], dtype=np.int64),
                np.array([[0.902911], [0.34299]], dtype=np.float32),
            ],
            "SOFTMAX": [
                np.array([1, 1], dtype=np.int64),
                np.array([[1.0], [1.0]], dtype=np.float32),
            ],
            "SOFTMAX_ZERO": [
                np.array([1, 1], dtype=np.int64),
                np.array([[1.0], [1.0]], dtype=np.float32),
            ],
        }
        x = np.arange(6).reshape((-1, 3)).astype(np.float32)
        for post in ("NONE", "LOGISTIC", "SOFTMAX_ZERO", "SOFTMAX"):
            expected = expected_post[post]
            with self.subTest(post_transform=post):
                node1 = make_node(
                    "LinearClassifier",
                    ["X"],
                    ["I", "Y"],
                    domain="ai.onnx.ml",
                    classlabels_ints=[1],
                    coefficients=[-0.58, -0.29, -0.09],
                    intercepts=[2.7],
                    multi_class=0,
                    post_transform=post,
                )
                graph = make_graph([node1], "ml", [X], [In, Y])
                onx = make_model_gen_version(graph, opset_imports=OPSETS)
                check_model(onx)
                # onnxruntime answer seems odd.
                # self._check_ort(onx, {"X": x}, rev=True)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @staticmethod
    def _get_test_tree_ensemble_opset_latest(
        aggregate_function,
        rule=Mode.LEQ,
        unique_targets=False,
        input_type=TensorProto.FLOAT,
    ):
        X = make_tensor_value_info("X", input_type, [None, None])
        Y = make_tensor_value_info("Y", input_type, [None, None])
        if unique_targets:
            weights = [
                1.0,
                10.0,
                100.0,
                1000.0,
                10000.0,
                100000.0,
            ]
        else:
            weights = [
                0.07692307978868484,
                0.5,
                0.5,
                0.0,
                0.2857142984867096,
                0.5,
            ]
        node = make_node(
            "TreeEnsemble",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=1,
            aggregate_function=aggregate_function,
            membership_values=None,
            nodes_missing_value_tracks_true=None,
            nodes_hitrates=None,
            post_transform=0,
            tree_roots=[0, 2],
            nodes_splits=make_tensor(
                "node_splits",
                input_type,
                (4,),
                [
                    0.26645058393478394,
                    0.6214364767074585,
                    -0.5592705607414246,
                    -0.7208403944969177,
                ],
            ),
            nodes_featureids=[0, 2, 0, 0],
            nodes_modes=make_tensor(
                "nodes_modes",
                TensorProto.UINT8,
                (4,),
                [rule] * 4,
            ),
            nodes_truenodeids=[1, 0, 3, 4],
            nodes_trueleafs=[0, 1, 1, 1],
            nodes_falsenodeids=[2, 1, 3, 5],
            nodes_falseleafs=[1, 1, 0, 1],
            leaf_targetids=[0, 0, 0, 0, 0, 0],
            leaf_weights=make_tensor(
                "leaf_weights", input_type, (len(weights),), weights
            ),
        )
        graph = make_graph([node], "ml", [X], [Y])
        model = make_model_gen_version(graph, opset_imports=OPSETS)
        return model

    @staticmethod
    def _get_test_tree_ensemble_regressor(
        aggregate_function, rule="BRANCH_LEQ", unique_targets=False, base_values=None
    ):
        opsets = [make_opsetid("", TARGET_OPSET), make_opsetid("ai.onnx.ml", 3)]
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        if unique_targets:
            targets = [
                1.0,
                10.0,
                100.0,
                1000.0,
                10000.0,
                100000.0,
            ]
        else:
            targets = [
                0.07692307978868484,
                0.5,
                0.5,
                0.0,
                0.2857142984867096,
                0.5,
            ]
        node1 = make_node(
            "TreeEnsembleRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=1,
            aggregate_function=aggregate_function,
            base_values=base_values,
            nodes_falsenodeids=[4, 3, 0, 0, 0, 2, 0, 4, 0, 0],
            nodes_featureids=[0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            nodes_modes=[
                rule,
                rule,
                "LEAF",
                "LEAF",
                "LEAF",
                rule,
                "LEAF",
                rule,
                "LEAF",
                "LEAF",
            ],
            nodes_nodeids=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            nodes_treeids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            nodes_truenodeids=[1, 2, 0, 0, 0, 1, 0, 3, 0, 0],
            nodes_values=[
                0.26645058393478394,
                0.6214364767074585,
                0.0,
                0.0,
                0.0,
                -0.7208403944969177,
                0.0,
                -0.5592705607414246,
                0.0,
                0.0,
            ],
            post_transform="NONE",
            target_ids=[0, 0, 0, 0, 0, 0],
            target_nodeids=[2, 3, 4, 1, 3, 4],
            target_treeids=[0, 0, 0, 1, 1, 1],
            target_weights=targets,
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=opsets)
        check_model(onx)
        return onx

    @parameterized.expand(
        tuple(
            itertools.chain.from_iterable(
                (
                    (
                        AggregationFunction.SUM if opset5 else "SUM",
                        np.array(
                            [[0.576923], [0.576923], [0.576923]], dtype=np.float32
                        ),
                        opset5,
                    ),
                    (
                        AggregationFunction.AVERAGE if opset5 else "AVERAGE",
                        np.array(
                            [[0.288462], [0.288462], [0.288462]], dtype=np.float32
                        ),
                        opset5,
                    ),
                    (
                        AggregationFunction.MIN if opset5 else "MIN",
                        np.array(
                            [[0.076923], [0.076923], [0.076923]], dtype=np.float32
                        ),
                        opset5,
                    ),
                    (
                        AggregationFunction.MAX if opset5 else "MAX",
                        np.array([[0.5], [0.5], [0.5]], dtype=np.float32),
                        opset5,
                    ),
                )
                for opset5 in [True, False]
            )
        )
    )
    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_regressor_aggregation_functions(
        self, aggregate_function, expected_result, opset5
    ):
        x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
        model_factory = (
            self._get_test_tree_ensemble_opset_latest
            if opset5
            else self._get_test_tree_ensemble_regressor
        )
        model_proto = model_factory(
            aggregate_function,
        )
        sess = ReferenceEvaluator(model_proto)
        (actual,) = sess.run(None, {"X": x})
        assert_allclose(expected_result, actual, atol=1e-6)

    @parameterized.expand(
        tuple(
            itertools.chain.from_iterable(
                (
                    (
                        Mode.LEQ if opset5 else "BRANCH_LEQ",
                        np.array(
                            [[0.576923], [0.576923], [0.576923]], dtype=np.float32
                        ),
                        opset5,
                    ),
                    (
                        Mode.GT if opset5 else "BRANCH_GT",
                        np.array([[0.5], [0.5], [0.5]], dtype=np.float32),
                        opset5,
                    ),
                    (
                        Mode.LT if opset5 else "BRANCH_LT",
                        np.array(
                            [[0.576923], [0.576923], [0.576923]], dtype=np.float32
                        ),
                        opset5,
                    ),
                    (
                        Mode.GTE if opset5 else "BRANCH_GTE",
                        np.array([[0.5], [0.5], [0.5]], dtype=np.float32),
                        opset5,
                    ),
                    (
                        Mode.EQ if opset5 else "BRANCH_EQ",
                        np.array([[1.0], [1.0], [1.0]], dtype=np.float32),
                        opset5,
                    ),
                    (
                        Mode.NEQ if opset5 else "BRANCH_NEQ",
                        np.array(
                            [[0.076923], [0.076923], [0.076923]], dtype=np.float32
                        ),
                        opset5,
                    ),
                )
                for opset5 in [True, False]
            )
        )
    )
    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_regressor_rule(self, rule, expected, opset5):
        x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
        model_factory = (
            self._get_test_tree_ensemble_opset_latest
            if opset5
            else self._get_test_tree_ensemble_regressor
        )
        aggregate_function = AggregationFunction.SUM if opset5 else "SUM"

        model_proto = model_factory(aggregate_function, rule)
        sess = ReferenceEvaluator(model_proto)
        (actual,) = sess.run(None, {"X": x})
        assert_allclose(expected, actual, atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_regressor_2_targets_opset3(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        opsets = [make_opsetid("", TARGET_OPSET), make_opsetid("ai.onnx.ml", 3)]
        node1 = make_node(
            "TreeEnsembleRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=2,
            nodes_falsenodeids=[4, 3, 0, 0, 6, 0, 0, 4, 3, 0, 0, 6, 0, 0],
            nodes_featureids=[0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 0, 2, 0, 0],
            nodes_hitrates=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            nodes_modes=[
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
            ],
            nodes_nodeids=[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            nodes_treeids=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            nodes_truenodeids=[1, 2, 0, 0, 5, 0, 0, 1, 2, 0, 0, 5, 0, 0],
            nodes_values=[
                -0.3367232382297516,
                1.5326381921768188,
                0.0,
                0.0,
                -0.24646544456481934,
                0.0,
                0.0,
                -0.3367232382297516,
                0.6671845316886902,
                0.0,
                0.0,
                -0.24646544456481934,
                0.0,
                0.0,
            ],
            post_transform="NONE",
            target_ids=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            target_nodeids=[2, 2, 3, 3, 5, 5, 6, 6, 2, 2, 3, 3, 5, 5, 6, 6],
            target_treeids=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            target_weights=[
                0.0,
                2.5,
                0.5,
                3.0,
                0.15000000596046448,
                2.6500000953674316,
                0.5,
                3.0,
                0.02777777798473835,
                2.527777671813965,
                0.5,
                3.0,
                0.20000000298023224,
                2.700000047683716,
                0.5,
                3.0,
            ],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=opsets)
        check_model(onx)
        x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
        expected = np.array(
            [[0.027778, 5.027778], [1.0, 6.0], [1.0, 6.0]], dtype=np.float32
        )
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})
        assert_allclose(expected, got[0], atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_regressor_missing_opset3(self):
        x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
        x[2, 0] = 5
        x[1, :] = np.nan
        expected = np.array([[100001.0], [100100.0], [100100.0]], dtype=np.float32)
        onx = self._get_test_tree_ensemble_regressor("SUM", unique_targets=True)
        self._check_ort(onx, {"X": x}, equal=True)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})
        assert_allclose(expected, got[0], atol=1e-6)
        self.assertIn("op_type=TreeEnsembleRegressor", str(sess.rt_nodes_[0]))

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    @parameterized.expand(
        [(input_type,) for input_type in [TensorProto.FLOAT, TensorProto.DOUBLE]]
    )
    def test_tree_ensemble_missing_opset5(self, input_type):
        model = self._get_test_tree_ensemble_opset_latest(
            AggregationFunction.SUM, Mode.LEQ, True, input_type
        )
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(input_type)
        x = np.arange(9).reshape((-1, 3)).astype(np_dtype) / 10 - 0.5
        x[2, 0] = 5
        x[1, :] = np.nan
        expected = np.array([[100001.0], [100100.0], [100100.0]], dtype=np_dtype)
        session = ReferenceEvaluator(model)
        (actual,) = session.run(None, {"X": x})
        assert_allclose(expected, actual, atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_regressor_missing_opset5_float16(self):
        model = self._get_test_tree_ensemble_opset_latest(
            AggregationFunction.SUM, Mode.LEQ, False, TensorProto.FLOAT16
        )
        np_dtype = np.float16
        x = np.arange(9).reshape((-1, 3)).astype(np_dtype) / 10 - 0.5
        x[2, 0] = 5
        x[1, :] = np.nan
        expected = np.array([[0.577], [1.0], [1.0]], dtype=np_dtype)
        session = ReferenceEvaluator(model)
        (actual,) = session.run(None, {"X": x})
        assert_allclose(expected, actual, atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_single_tree_ensemble(self):
        X = make_tensor_value_info("X", TensorProto.DOUBLE, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.DOUBLE, [None, None])
        node = make_node(
            "TreeEnsemble",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=2,
            membership_values=None,
            nodes_missing_value_tracks_true=None,
            nodes_hitrates=None,
            aggregate_function=1,
            post_transform=PostTransform.NONE,
            tree_roots=[0],
            nodes_modes=make_tensor(
                "nodes_modes",
                TensorProto.UINT8,
                (3,),
                [Mode.LEQ] * 3,
            ),
            nodes_featureids=[0, 0, 0],
            nodes_splits=make_tensor(
                "nodes_splits",
                TensorProto.DOUBLE,
                (3,),
                np.array([3.14, 1.2, 4.2], dtype=np.float64),
            ),
            nodes_truenodeids=[1, 0, 1],
            nodes_trueleafs=[0, 1, 1],
            nodes_falsenodeids=[2, 2, 3],
            nodes_falseleafs=[0, 1, 1],
            leaf_targetids=[0, 1, 0, 1],
            leaf_weights=make_tensor(
                "leaf_weights",
                TensorProto.DOUBLE,
                (4,),
                np.array([5.23, 12.12, -12.23, 7.21], dtype=np.float64),
            ),
        )
        graph = make_graph([node], "ml", [X], [Y])
        model = make_model_gen_version(
            graph,
            opset_imports=[
                make_opsetid("", TARGET_OPSET),
                make_opsetid("ai.onnx.ml", 5),
            ],
        )
        check_model(model)
        session = ReferenceEvaluator(model)
        (output,) = session.run(
            None,
            {
                "X": np.array([1.2, 3.4, -0.12, 1.66, 4.14, 1.77], np.float64).reshape(
                    3, 2
                )
            },
        )
        np.testing.assert_equal(
            output, np.array([[5.23, 0], [5.23, 0], [0, 12.12]], dtype=np.float64)
        )

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_regressor_set_membership_opset5(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node = make_node(
            "TreeEnsemble",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=4,
            aggregate_function=AggregationFunction.SUM,
            membership_values=make_tensor(
                "membership_values",
                TensorProto.FLOAT,
                (8,),
                [1.2, 3.7, 8, 9, np.nan, 12, 7, np.nan],
            ),
            nodes_missing_value_tracks_true=None,
            nodes_hitrates=None,
            post_transform=PostTransform.NONE,
            tree_roots=[0],
            nodes_modes=make_tensor(
                "nodes_modes",
                TensorProto.UINT8,
                (3,),
                [Mode.LEQ, Mode.MEMBER, Mode.MEMBER],
            ),
            nodes_featureids=[0, 0, 0],
            nodes_splits=make_tensor(
                "nodes_splits",
                TensorProto.FLOAT,
                (3,),
                np.array([11, 232344.0, np.nan], dtype=np.float32),
            ),
            nodes_trueleafs=[0, 1, 1],
            nodes_truenodeids=[1, 0, 1],
            nodes_falseleafs=[1, 0, 1],
            nodes_falsenodeids=[2, 2, 3],
            leaf_targetids=[0, 1, 2, 3],
            leaf_weights=make_tensor(
                "leaf_weights", TensorProto.FLOAT, (4,), [1, 10, 1000, 100]
            ),
        )
        graph = make_graph([node], "ml", [X], [Y])
        model = make_model_gen_version(
            graph,
            opset_imports=OPSETS,
        )
        check_model(model)
        session = ReferenceEvaluator(model)
        X = np.array([1.2, 3.4, -0.12, np.nan, 12, 7], np.float32).reshape(-1, 1)
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 100],
                [0, 0, 0, 100],
                [0, 0, 1000, 0],
                [0, 0, 1000, 0],
                [0, 10, 0, 0],
            ],
            dtype=np.float32,
        )
        (output,) = session.run(None, {"X": X})
        np.testing.assert_equal(output, expected)

    @staticmethod
    def _get_test_svm_regressor(kernel_type, kernel_params):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "SVMRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            coefficients=[
                1.0,
                -1.0,
                0.8386201858520508,
                -0.8386201858520508,
                0.4470679759979248,
                -1.0,
                0.5529320240020752,
            ],
            kernel_params=kernel_params,
            kernel_type=kernel_type,
            n_supports=7,
            post_transform="NONE",
            rho=[0.5460880398750305],
            support_vectors=[
                -0.12850627303123474,
                0.08915442228317261,
                0.06881910562515259,
                -0.07938569784164429,
                -0.22557435929775238,
                -0.26520243287086487,
                0.9246066212654114,
                -0.025557516142725945,
                -0.5900523662567139,
                0.9735698699951172,
                -1.3385062217712402,
                0.3393094539642334,
                0.9432410001754761,
                -0.5228781700134277,
                0.5557093620300293,
                0.4191802740097046,
                0.43368014693260193,
                -1.0569839477539062,
                2.3318440914154053,
                0.06202844902873039,
                -0.9502395987510681,
            ],
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        return onx

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_regressor(self):
        x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
        expected_kernel = {
            "LINEAR": (
                [0.42438405752182007, 0.0, 3.0],
                np.array([[-0.468206], [0.227487], [0.92318]], dtype=np.float32),
            ),
            "POLY": (
                [0.3426632285118103, 0.0, 3.0],
                np.array([[0.527084], [0.543578], [0.546506]], dtype=np.float32),
            ),
            "RBF": (
                [0.30286383628845215, 0.0, 3.0],
                np.array([[0.295655], [0.477876], [0.695292]], dtype=np.float32),
            ),
            "SIGMOID": (
                [0.30682486295700073, 0.0, 3.0],
                np.array([[0.239304], [0.448929], [0.661689]], dtype=np.float32),
            ),
        }
        for kernel, (params, expected) in expected_kernel.items():
            with self.subTest(kernel=kernel):
                onx = self._get_test_svm_regressor(kernel, params)
                self._check_ort(onx, {"X": x}, atol=1e-6)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected, got[0], atol=1e-6)

    @staticmethod
    def _get_test_tree_ensemble_classifier_binary(post_transform):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        In = make_tensor_value_info("I", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "TreeEnsembleClassifier",
            ["X"],
            ["I", "Y"],
            domain="ai.onnx.ml",
            class_ids=[0, 0, 0, 0, 0, 0, 0],
            class_nodeids=[2, 3, 5, 6, 1, 3, 4],
            class_treeids=[0, 0, 0, 0, 1, 1, 1],
            class_weights=[
                0.0,
                0.1764705926179886,
                0.0,
                0.5,
                0.0,
                0.0,
                0.4285714328289032,
            ],
            classlabels_int64s=[0, 1],
            nodes_falsenodeids=[4, 3, 0, 0, 6, 0, 0, 2, 0, 4, 0, 0],
            nodes_featureids=[2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
            nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            nodes_modes=[
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
            ],
            nodes_nodeids=[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4],
            nodes_treeids=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            nodes_truenodeids=[1, 2, 0, 0, 5, 0, 0, 1, 0, 3, 0, 0],
            nodes_values=[
                0.6874135732650757,
                -0.3654803931713104,
                0.0,
                0.0,
                -1.926770806312561,
                0.0,
                0.0,
                -0.3654803931713104,
                0.0,
                -2.0783839225769043,
                0.0,
                0.0,
            ],
            post_transform=post_transform,
        )
        graph = make_graph([node1], "ml", [X], [In, Y])
        onx = make_model_gen_version(
            graph,
            opset_imports=[
                make_opsetid("", TARGET_OPSET),
                make_opsetid("ai.onnx.ml", 3),
            ],
        )
        check_model(onx)
        return onx

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_classifier_binary(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        expected_post = {
            "NONE": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[1.0, 0.0], [0.394958, 0.605042], [0.394958, 0.605042]],
                    dtype=np.float32,
                ),
            ),
            "LOGISTIC": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.5, 0.5], [0.353191, 0.646809], [0.353191, 0.646809]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.5, 0.5], [0.229686, 0.770314], [0.229686, 0.770314]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX_ZERO": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.5, 0.5], [0.229686, 0.770314], [0.229686, 0.770314]],
                    dtype=np.float32,
                ),
            ),
            "PROBIT": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.0, 0.0], [-0.266426, 0.266426], [-0.266426, 0.266426]],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_tree_ensemble_classifier_binary(post)
                if post in ("NONE",):
                    self._check_ort(onx, {"X": x})
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @staticmethod
    def _get_test_tree_ensemble_classifier_multi(post_transform):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        In = make_tensor_value_info("I", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "TreeEnsembleClassifier",
            ["X"],
            ["I", "Y"],
            domain="ai.onnx.ml",
            class_ids=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            class_nodeids=[2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1, 3, 3, 3, 4, 4, 4],
            class_treeids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            class_weights=[
                0.46666666865348816,
                0.0,
                0.03333333507180214,
                0.20000000298023224,
                0.23999999463558197,
                0.05999999865889549,
                0.0,
                0.5,
                0.0,
                0.5,
                0.0,
                0.0,
                0.44999998807907104,
                0.0,
                0.05000000074505806,
                0.10294117778539658,
                0.19117647409439087,
                0.20588235557079315,
            ],
            classlabels_int64s=[0, 1, 2],
            nodes_falsenodeids=[4, 3, 0, 0, 0, 2, 0, 4, 0, 0],
            nodes_featureids=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            nodes_modes=[
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
            ],
            nodes_nodeids=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            nodes_treeids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            nodes_truenodeids=[1, 2, 0, 0, 0, 1, 0, 3, 0, 0],
            nodes_values=[
                1.2495747804641724,
                -0.3050493597984314,
                0.0,
                0.0,
                0.0,
                -1.6830512285232544,
                0.0,
                -0.6751254796981812,
                0.0,
                0.0,
            ],
            post_transform=post_transform,
        )
        graph = make_graph([node1], "ml", [X], [In, Y])
        onx = make_model_gen_version(
            graph,
            opset_imports=[
                make_opsetid("", TARGET_OPSET),
                make_opsetid("ai.onnx.ml", 3),
            ],
        )
        check_model(onx)
        return onx

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_tree_ensemble_classifier_multi(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        expected_post = {
            "NONE": (
                np.array([0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        [0.916667, 0.0, 0.083333],
                        [0.569608, 0.191176, 0.239216],
                        [0.302941, 0.431176, 0.265882],
                    ],
                    dtype=np.float32,
                ),
            ),
            "LOGISTIC": (
                np.array([0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        [0.714362, 0.5, 0.520821],
                        [0.638673, 0.547649, 0.55952],
                        [0.575161, 0.606155, 0.566082],
                    ],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX": (
                np.array([0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        [0.545123, 0.217967, 0.23691],
                        [0.416047, 0.284965, 0.298988],
                        [0.322535, 0.366664, 0.310801],
                    ],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX_ZERO": (
                np.array([0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        [0.697059, 0.0, 0.302941],
                        [0.416047, 0.284965, 0.298988],
                        [0.322535, 0.366664, 0.310801],
                    ],
                    dtype=np.float32,
                ),
            ),
            "PROBIT": (
                np.array([0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        [1.383104, 0, -1.383105],
                        [0.175378, -0.873713, -0.708922],
                        [-0.516003, -0.173382, -0.625385],
                    ],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_tree_ensemble_classifier_multi(post)
                if post != "PROBIT":
                    self._check_ort(onx, {"X": x}, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @staticmethod
    def _get_test_svm_classifier_binary(post_transform, probability=True, linear=False):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        In = make_tensor_value_info("I", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        if linear:
            kwargs = {
                "classlabels_ints": [0, 1, 2, 3],
                "coefficients": [
                    -1.55181212e-01,
                    2.42698956e-01,
                    7.01893432e-03,
                    4.07614474e-01,
                    -3.24927823e-02,
                    2.79897536e-04,
                    -1.95771302e-01,
                    -3.52437368e-01,
                    -2.15973096e-02,
                    -4.38190277e-01,
                    4.56869105e-02,
                    -1.29375499e-02,
                ],
                "kernel_params": [0.001, 0.0, 3.0],
                "kernel_type": "LINEAR",
                "prob_a": [-5.139118194580078],
                "prob_b": [0.06399919837713242],
                "rho": [-0.07489691, -0.1764396, -0.21167431, -0.51619097],
                "post_transform": post_transform,
            }
        else:
            kwargs = {
                "classlabels_ints": [0, 1],
                "coefficients": [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                "kernel_params": [0.3824487328529358, 0.0, 3.0],
                "kernel_type": "RBF",
                "prob_a": [-5.139118194580078],
                "prob_b": [0.06399919837713242],
                "rho": [0.16708599030971527],
                "support_vectors": [
                    0.19125767052173615,
                    -1.062204122543335,
                    0.5006636381149292,
                    -0.5892484784126282,
                    -0.3196830451488495,
                    0.0984845906496048,
                    0.24746321141719818,
                    -1.1535362005233765,
                    0.4109955430030823,
                    -0.5937694907188416,
                    -1.3183348178863525,
                    -1.6423596143722534,
                    0.558641254901886,
                    -0.9218668341636658,
                    0.6264089345932007,
                    -0.16060839593410492,
                    -0.6365169882774353,
                    0.8335472345352173,
                    0.7539799213409424,
                    -0.3970031440258026,
                    -0.1780400276184082,
                    -0.616622805595398,
                    0.49261474609375,
                    0.4470972716808319,
                ],
                "vectors_per_class": [4, 4],
                "post_transform": post_transform,
            }

        if not probability:
            del kwargs["prob_a"]
            del kwargs["prob_b"]
        node1 = make_node(
            "SVMClassifier", ["X"], ["I", "Y"], domain="ai.onnx.ml", **kwargs
        )
        graph = make_graph([node1], "ml", [X], [In, Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        return onx

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_classifier_binary(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        expected_post = {
            "NONE": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.993287, 0.006713], [0.469401, 0.530599], [0.014997, 0.985003]],
                    dtype=np.float32,
                ),
            ),
            "LOGISTIC": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.729737, 0.501678], [0.615242, 0.629623], [0.503749, 0.7281]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.728411, 0.271589], [0.484705, 0.515295], [0.274879, 0.725121]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX_ZERO": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.728411, 0.271589], [0.484705, 0.515295], [0.274879, 0.725121]],
                    dtype=np.float32,
                ),
            ),
            "PROBIT": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[2.469393, -2.469391], [-0.076776, 0.076776], [-2.16853, 2.16853]],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_svm_classifier_binary(post)
                self._check_ort(onx, {"X": x}, rev=True, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-5)
                assert_allclose(expected[0], got[0])

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_classifier_binary_noprob(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        expected_post = {
            "NONE": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [
                        [-0.986073, 0.986073],
                        [0.011387, -0.011387],
                        [0.801808, -0.801808],
                    ],
                    dtype=np.float32,
                ),
            ),
            "LOGISTIC": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.271688, 0.728312], [0.502847, 0.497153], [0.690361, 0.309639]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.122158, 0.877842], [0.505693, 0.494307], [0.832523, 0.167477]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX_ZERO": (
                np.array([0, 1, 1], dtype=np.int64),
                np.array(
                    [[0.122158, 0.877842], [0.505693, 0.494307], [0.832523, 0.167477]],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_svm_classifier_binary(post, probability=False)
                if post not in {"LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT"}:
                    self._check_ort(onx, {"X": x}, rev=True, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_classifier_noprob_linear(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        nan = np.nan
        expected_post = {
            "NONE": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [-0.118086, -0.456685, 0.415783, 0.334506],
                        [-0.061364, -0.231444, 0.073899, 0.091242],
                        [-0.004642, -0.006203, -0.267985, -0.152023],
                    ],
                    dtype=np.float32,
                ),
            ),
            "LOGISTIC": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [0.470513, 0.387773, 0.602474, 0.582855],
                        [0.484664, 0.442396, 0.518466, 0.522795],
                        [0.498839, 0.498449, 0.433402, 0.462067],
                    ],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [0.200374, 0.14282, 0.341741, 0.315065],
                        [0.240772, 0.203115, 0.275645, 0.280467],
                        [0.275491, 0.275061, 0.211709, 0.237739],
                    ],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX_ZERO": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [0.200374, 0.14282, 0.341741, 0.315065],
                        [0.240772, 0.203115, 0.275645, 0.280467],
                        [0.275491, 0.275061, 0.211709, 0.237739],
                    ],
                    dtype=np.float32,
                ),
            ),
            "PROBIT": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [nan, nan, -0.212698, -0.427529],
                        [nan, nan, -1.447414, -1.333286],
                        [nan, nan, nan, nan],
                    ],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_svm_classifier_binary(
                    post, probability=False, linear=True
                )
                self._check_ort(onx, {"X": x}, rev=True, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_classifier_linear(self):
        # prob_a, prob_b are not used in this case.
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        nan = np.nan
        expected_post = {
            "NONE": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [-0.118086, -0.456685, 0.415783, 0.334506],
                        [-0.061364, -0.231444, 0.073899, 0.091242],
                        [-0.004642, -0.006203, -0.267985, -0.152023],
                    ],
                    dtype=np.float32,
                ),
            ),
            "LOGISTIC": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [0.470513, 0.387773, 0.602474, 0.582855],
                        [0.484664, 0.442396, 0.518466, 0.522795],
                        [0.498839, 0.498449, 0.433402, 0.462067],
                    ],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [0.200374, 0.14282, 0.341741, 0.315065],
                        [0.240772, 0.203115, 0.275645, 0.280467],
                        [0.275491, 0.275061, 0.211709, 0.237739],
                    ],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX_ZERO": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [0.200374, 0.14282, 0.341741, 0.315065],
                        [0.240772, 0.203115, 0.275645, 0.280467],
                        [0.275491, 0.275061, 0.211709, 0.237739],
                    ],
                    dtype=np.float32,
                ),
            ),
            "PROBIT": (
                np.array([2, 3, 0], dtype=np.int64),
                np.array(
                    [
                        [nan, nan, -0.212698, -0.427529],
                        [nan, nan, -1.447414, -1.333286],
                        [nan, nan, nan, nan],
                    ],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_svm_classifier_binary(
                    post, probability=True, linear=True
                )
                self._check_ort(onx, {"X": x}, rev=True, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @staticmethod
    def _get_test_svm_classifier_linear_sv(post_transform, probability=True):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        In = make_tensor_value_info("I", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        kwargs = {
            "classlabels_ints": [0, 1],
            "coefficients": [
                0.766398549079895,
                0.0871576070785522,
                0.110420741140842,
                -0.963976919651031,
            ],
            "support_vectors": [
                4.80000019073486,
                3.40000009536743,
                1.89999997615814,
                5.0,
                3.0,
                1.60000002384186,
                4.5,
                2.29999995231628,
                1.29999995231628,
                5.09999990463257,
                2.5,
                3.0,
            ],
            "kernel_params": [0.122462183237076, 0.0, 3.0],
            "kernel_type": "LINEAR",
            "prob_a": [-5.139118194580078],
            "prob_b": [0.06399919837713242],
            "rho": [2.23510527610779],
            "post_transform": post_transform,
            "vectors_per_class": [3, 1],
        }

        if not probability:
            del kwargs["prob_a"]
            del kwargs["prob_b"]
        node1 = make_node(
            "SVMClassifier", ["X"], ["I", "Y"], domain="ai.onnx.ml", **kwargs
        )
        graph = make_graph([node1], "ml", [X], [In, Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        return onx

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_classifier_binary_noprob_linear_sv(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        expected_post = {
            "NONE": (
                np.array([0, 0, 0], dtype=np.int64),
                np.array(
                    [[-2.662655, 2.662655], [-2.21481, 2.21481], [-1.766964, 1.766964]],
                    dtype=np.float32,
                ),
            ),
            "LOGISTIC": (
                np.array([0, 0, 0], dtype=np.int64),
                np.array(
                    [[0.065213, 0.934787], [0.098428, 0.901572], [0.14592, 0.85408]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX": (
                np.array([0, 0, 0], dtype=np.int64),
                np.array(
                    [[0.004843, 0.995157], [0.011779, 0.988221], [0.028362, 0.971638]],
                    dtype=np.float32,
                ),
            ),
            "SOFTMAX_ZERO": (
                np.array([0, 0, 0], dtype=np.int64),
                np.array(
                    [[0.004843, 0.995157], [0.011779, 0.988221], [0.028362, 0.971638]],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_svm_classifier_linear_sv(post, probability=False)
                if post not in {"LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"}:
                    self._check_ort(onx, {"X": x}, rev=True, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[1], got[1], atol=1e-6)
                assert_allclose(expected[0], got[0])

    @staticmethod
    def _get_test_svm_regressor_linear(post_transform, one_class=0):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        kwargs = {
            "coefficients": [0.28290501, -0.0266512, 0.01674867],
            "kernel_params": [0.001, 0.0, 3.0],
            "kernel_type": "LINEAR",
            "rho": [1.24032312],
            "post_transform": post_transform,
            "n_supports": 0,
            "one_class": one_class,
        }

        node1 = make_node("SVMRegressor", ["X"], ["Y"], domain="ai.onnx.ml", **kwargs)
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model_gen_version(graph, opset_imports=OPSETS)
        check_model(onx)
        return onx

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_regressor_linear(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        expected_post = {
            "NONE": (
                np.array(
                    [[0.96869], [1.132491], [1.296293]],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_svm_regressor_linear(post)
                self._check_ort(onx, {"X": x}, rev=True, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[0], got[0], atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_svm_regressor_linear_one_class(self):
        x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
        expected_post = {
            "NONE": (
                np.array(
                    [[1.0], [1.0], [1.0]],
                    dtype=np.float32,
                ),
            ),
        }
        for post, expected in expected_post.items():
            with self.subTest(post_transform=post):
                onx = self._get_test_svm_regressor_linear(post, one_class=1)
                self._check_ort(onx, {"X": x}, rev=True, atol=1e-5)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, {"X": x})
                assert_allclose(expected[0], got[0], atol=1e-6)

    def test_onnxrt_tfidf_vectorizer_ints(self):
        inputi = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int64)
        output = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
        ).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            np.int64
        )  # bigrams

        model = make_model_gen_version(
            make_graph(
                [
                    make_node(
                        "TfIdfVectorizer",
                        ["tokens"],
                        ["out"],
                        mode="TF",
                        min_gram_length=2,
                        max_gram_length=2,
                        max_skip_count=0,
                        ngram_counts=ngram_counts,
                        ngram_indexes=ngram_indexes,
                        pool_int64s=pool_int64s,
                    )
                ],
                "tfidf",
                [make_tensor_value_info("tokens", TensorProto.INT64, [None, None])],
                [make_tensor_value_info("out", TensorProto.FLOAT, [None, None])],
            ),
            opset_imports=OPSETS,
        )

        oinf = ReferenceEvaluator(model)
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())

    def test_onnxrt_tfidf_vectorizer_strings(self):
        inputi = np.array(
            [["i1", "i1", "i3", "i3", "i3", "i7"], ["i8", "i6", "i7", "i5", "i6", "i8"]]
        )
        output = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
        ).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_strings = np.array(
            ["i2", "i3", "i5", "i4", "i5", "i6", "i7", "i8", "i6", "i7"]
        )

        model = make_model_gen_version(
            make_graph(
                [
                    make_node(
                        "TfIdfVectorizer",
                        ["tokens"],
                        ["out"],
                        mode="TF",
                        min_gram_length=2,
                        max_gram_length=2,
                        max_skip_count=0,
                        ngram_counts=ngram_counts,
                        ngram_indexes=ngram_indexes,
                        pool_strings=pool_strings,
                    )
                ],
                "tfidf",
                [make_tensor_value_info("tokens", TensorProto.INT64, [None, None])],
                [make_tensor_value_info("out", TensorProto.FLOAT, [None, None])],
            ),
            opset_imports=OPSETS,
        )

        oinf = ReferenceEvaluator(model)
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
