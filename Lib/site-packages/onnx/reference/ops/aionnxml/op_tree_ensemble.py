from __future__ import annotations

from enum import IntEnum
from typing import Callable

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class AggregationFunction(IntEnum):
    AVERAGE = 0
    SUM = 1
    MIN = 2
    MAX = 3


class PostTransform(IntEnum):
    NONE = 0
    SOFTMAX = 1
    LOGISTIC = 2
    SOFTMAX_ZERO = 3
    PROBIT = 4


class Mode(IntEnum):
    LEQ = 0
    LT = 1
    GTE = 2
    GT = 3
    EQ = 4
    NEQ = 5
    MEMBER = 6


class Leaf:
    def __init__(self, weight: float, target_id: int) -> None:
        self.weight = weight
        self.target_id = target_id

    # Produce the weight and target index
    def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.array([self.weight, self.target_id])

    def _print(self, prefix: list, indent: int = 0) -> None:
        prefix.append(
            " " * indent + f"Leaf WEIGHT: {self.weight}, TARGET: {self.target_id}\n"
        )

    def __repr__(self) -> str:
        prefix = []
        self._print(prefix)
        return "".join(prefix)


class Node:
    compare: Callable[[float, float | set[float]], bool]
    true_branch: Node | Leaf
    false_branch: Node | Leaf
    feature: int

    def __init__(
        self,
        mode: Mode,
        value: float | set[float],
        feature: int,
        missing_tracks_true: bool,
    ) -> None:
        if mode == Mode.LEQ:
            self.compare = lambda x: x[feature].item() <= value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.LT:
            self.compare = lambda x: x[feature].item() < value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.GTE:
            self.compare = lambda x: x[feature].item() >= value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.GT:
            self.compare = lambda x: x[feature].item() > value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.EQ:
            self.compare = lambda x: x[feature].item() == value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.NEQ:
            self.compare = lambda x: x[feature].item() != value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.MEMBER:
            self.compare = lambda x: x[feature].item() in value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        self.mode = mode
        self.value = value
        self.feature = feature

    def predict(self, x: np.ndarray) -> float:
        if self.compare(x):
            return self.true_branch.predict(x)
        else:
            return self.false_branch.predict(x)

    def _print(self, prefix: list, indent: int = 0) -> None:
        prefix.append(
            " " * indent
            + f"Node CMP: {self.mode}, SPLIT: {self.value}, FEATURE: {self.feature}\n"
        )
        self.true_branch._print(prefix, indent + 1)
        self.false_branch._print(prefix, indent + 1)

    def __repr__(self) -> str:
        prefix = []
        self._print(prefix)
        return "".join(prefix)


class TreeEnsemble(OpRunAiOnnxMl):
    def _run(
        self,
        X,
        nodes_splits,
        nodes_featureids,
        nodes_modes,
        nodes_truenodeids,
        nodes_falsenodeids,
        nodes_trueleafs,
        nodes_falseleafs,
        leaf_targetids,
        leaf_weights,
        tree_roots,
        post_transform=PostTransform.NONE,  # noqa: ARG002
        aggregate_function=AggregationFunction.SUM,
        nodes_hitrates=None,  # noqa: ARG002
        nodes_missing_value_tracks_true=None,
        membership_values=None,
        n_targets=None,
    ):
        if membership_values is None:
            # assert that no set membership ever appears
            if any(mode == Mode.MEMBER for mode in nodes_modes):
                raise ValueError(
                    "Cannot have set membership node without specifying set members"
                )
        elif np.isnan(membership_values).sum() != sum(
            int(mode == Mode.MEMBER) for mode in nodes_modes
        ):
            raise ValueError(
                "Must specify membership values for all set membership nodes"
            )

        # Build each tree in the ensemble. Note that the tree structure is implicitly defined by following the true and false indices in
        # `nodes_truenodeids` and `nodes_falsenodeids` to the leaves of each tree.
        set_membership_iter = (
            iter(membership_values) if membership_values is not None else None
        )

        def build_node(current_node_index, is_leaf) -> Node | Leaf:
            if is_leaf:
                return Leaf(
                    leaf_weights[current_node_index], leaf_targetids[current_node_index]
                )

            if nodes_modes[current_node_index] == Mode.MEMBER:
                # parse next sequence of set members
                set_members = set()
                while (set_member := next(set_membership_iter)) and not np.isnan(
                    set_member
                ):
                    set_members.add(set_member)
                node = Node(
                    nodes_modes[current_node_index],
                    set_members,
                    nodes_featureids[current_node_index],
                    (
                        nodes_missing_value_tracks_true[current_node_index]
                        if nodes_missing_value_tracks_true is not None
                        else False
                    ),
                )
            else:
                node = Node(
                    nodes_modes[current_node_index],
                    nodes_splits[current_node_index],
                    nodes_featureids[current_node_index],
                    (
                        nodes_missing_value_tracks_true[current_node_index]
                        if nodes_missing_value_tracks_true is not None
                        else False
                    ),
                )

            # recurse true and false branches
            node.true_branch = build_node(
                nodes_truenodeids[current_node_index],
                nodes_trueleafs[current_node_index],
            )
            node.false_branch = build_node(
                nodes_falsenodeids[current_node_index],
                nodes_falseleafs[current_node_index],
            )
            return node

        trees = []
        for root_index in tree_roots:
            # degenerate case (tree == leaf)
            is_leaf = (
                nodes_trueleafs[root_index]
                and nodes_falseleafs[root_index]
                and nodes_truenodeids[root_index] == nodes_falsenodeids[root_index]
            )
            trees.append(build_node(root_index, is_leaf))

        # predict each sample through tree
        raw_values = [
            np.apply_along_axis(tree.predict, axis=1, arr=X) for tree in trees
        ]
        weights, target_ids = zip(*[np.split(x, 2, axis=1) for x in raw_values])
        weights = np.concatenate(weights, axis=1)
        target_ids = np.concatenate(target_ids, axis=1).astype(np.int64)
        if aggregate_function in (
            AggregationFunction.SUM,
            AggregationFunction.AVERAGE,
        ):
            result = np.zeros((len(X), n_targets), dtype=X.dtype)
        elif aggregate_function == AggregationFunction.MIN:
            result = np.full((len(X), n_targets), np.finfo(X.dtype).max)
        elif aggregate_function == AggregationFunction.MAX:
            result = np.full((len(X), n_targets), np.finfo(X.dtype).min)
        else:
            raise NotImplementedError(
                f"aggregate_transform={aggregate_function!r} not supported yet."
            )
        for batch_num, (w, t) in enumerate(zip(weights, target_ids)):
            weight = w.reshape(-1)
            target_id = t.reshape(-1)
            if aggregate_function == AggregationFunction.SUM:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] += value
            elif aggregate_function == AggregationFunction.AVERAGE:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] += value / len(trees)
            elif aggregate_function == AggregationFunction.MIN:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] = min(result[batch_num, tid], value)
            elif aggregate_function == AggregationFunction.MAX:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] = max(result[batch_num, tid], value)
            else:
                raise NotImplementedError(
                    f"aggregate_transform={aggregate_function!r} not supported yet."
                )

        return (result,)
