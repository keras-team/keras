# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor


class TreeEnsemble(Base):
    @staticmethod
    def export_tree_ensemble_single_tree() -> None:
        node = onnx.helper.make_node(
            "TreeEnsemble",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=2,
            membership_values=None,
            nodes_missing_value_tracks_true=None,
            nodes_hitrates=None,
            aggregate_function=1,
            post_transform=0,
            tree_roots=[0],
            nodes_modes=make_tensor(
                "nodes_modes",
                onnx.TensorProto.UINT8,
                (3,),
                np.array([0, 0, 0], dtype=np.uint8),
            ),
            nodes_featureids=[0, 0, 0],
            nodes_splits=make_tensor(
                "nodes_splits",
                onnx.TensorProto.DOUBLE,
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
                onnx.TensorProto.DOUBLE,
                (4,),
                np.array([5.23, 12.12, -12.23, 7.21], dtype=np.float64),
            ),
        )

        x = np.array([1.2, 3.4, -0.12, 1.66, 4.14, 1.77], np.float64).reshape(3, 2)
        y = np.array([[5.23, 0], [5.23, 0], [0, 12.12]], dtype=np.float64)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_ai_onnx_ml_tree_ensemble_single_tree",
        )

    @staticmethod
    def export_tree_ensemble_set_membership() -> None:
        node = onnx.helper.make_node(
            "TreeEnsemble",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            n_targets=4,
            aggregate_function=1,
            membership_values=make_tensor(
                "membership_values",
                onnx.TensorProto.FLOAT,
                (8,),
                [1.2, 3.7, 8, 9, np.nan, 12, 7, np.nan],
            ),
            nodes_missing_value_tracks_true=None,
            nodes_hitrates=None,
            post_transform=0,
            tree_roots=[0],
            nodes_modes=make_tensor(
                "nodes_modes",
                onnx.TensorProto.UINT8,
                (3,),
                np.array([0, 6, 6], dtype=np.uint8),
            ),
            nodes_featureids=[0, 0, 0],
            nodes_splits=make_tensor(
                "nodes_splits",
                onnx.TensorProto.FLOAT,
                (3,),
                np.array([11, 232344.0, np.nan], dtype=np.float32),
            ),
            nodes_trueleafs=[0, 1, 1],
            nodes_truenodeids=[1, 0, 1],
            nodes_falseleafs=[1, 0, 1],
            nodes_falsenodeids=[2, 2, 3],
            leaf_targetids=[0, 1, 2, 3],
            leaf_weights=make_tensor(
                "leaf_weights", onnx.TensorProto.FLOAT, (4,), [1, 10, 1000, 100]
            ),
        )

        x = np.array([1.2, 3.4, -0.12, np.nan, 12, 7], np.float32).reshape(-1, 1)
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
        expect(
            node,
            inputs=[x],
            outputs=[expected],
            name="test_ai_onnx_ml_tree_ensemble_set_membership",
        )
