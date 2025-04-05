"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar, List, Any
from typing_extensions import Annotated

def boosted_trees_aggregate_stats(node_ids: Annotated[Any, _atypes.Int32], gradients: Annotated[Any, _atypes.Float32], hessians: Annotated[Any, _atypes.Float32], feature: Annotated[Any, _atypes.Int32], max_splits: int, num_buckets: int, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Aggregates the summary of accumulated stats for the batch.

  The summary stats contains gradients and hessians accumulated for each node, feature dimension id and bucket.

  Args:
    node_ids: A `Tensor` of type `int32`.
      int32; Rank 1 Tensor containing node ids for each example, shape [batch_size].
    gradients: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, logits_dimension]) with gradients for each example.
    hessians: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, hessian_dimension]) with hessians for each example.
    feature: A `Tensor` of type `int32`.
      int32; Rank 2 feature Tensors (shape=[batch_size, feature_dimension]).
    max_splits: An `int` that is `>= 1`.
      int; the maximum number of splits possible in the whole tree.
    num_buckets: An `int` that is `>= 1`.
      int; equals to the maximum possible value of bucketized feature.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesAggregateStats", name, node_ids, gradients,
        hessians, feature, "max_splits", max_splits, "num_buckets",
        num_buckets)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_aggregate_stats_eager_fallback(
          node_ids, gradients, hessians, feature, max_splits=max_splits,
          num_buckets=num_buckets, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  max_splits = _execute.make_int(max_splits, "max_splits")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesAggregateStats", node_ids=node_ids, gradients=gradients,
                                      hessians=hessians, feature=feature,
                                      max_splits=max_splits,
                                      num_buckets=num_buckets, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_splits", _op._get_attr_int("max_splits"), "num_buckets",
              _op._get_attr_int("num_buckets"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesAggregateStats", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BoostedTreesAggregateStats = tf_export("raw_ops.BoostedTreesAggregateStats")(_ops.to_raw_op(boosted_trees_aggregate_stats))


def boosted_trees_aggregate_stats_eager_fallback(node_ids: Annotated[Any, _atypes.Int32], gradients: Annotated[Any, _atypes.Float32], hessians: Annotated[Any, _atypes.Float32], feature: Annotated[Any, _atypes.Int32], max_splits: int, num_buckets: int, name, ctx) -> Annotated[Any, _atypes.Float32]:
  max_splits = _execute.make_int(max_splits, "max_splits")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  node_ids = _ops.convert_to_tensor(node_ids, _dtypes.int32)
  gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
  hessians = _ops.convert_to_tensor(hessians, _dtypes.float32)
  feature = _ops.convert_to_tensor(feature, _dtypes.int32)
  _inputs_flat = [node_ids, gradients, hessians, feature]
  _attrs = ("max_splits", max_splits, "num_buckets", num_buckets)
  _result = _execute.execute(b"BoostedTreesAggregateStats", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesAggregateStats", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def boosted_trees_bucketize(float_values: Annotated[List[Any], _atypes.Float32], bucket_boundaries: Annotated[List[Any], _atypes.Float32], name=None):
  r"""Bucketize each feature based on bucket boundaries.

  An op that returns a list of float tensors, where each tensor represents the
  bucketized values for a single feature.

  Args:
    float_values: A list of `Tensor` objects with type `float32`.
      float; List of Rank 1 Tensor each containing float values for a single feature.
    bucket_boundaries: A list with the same length as `float_values` of `Tensor` objects with type `float32`.
      float; List of Rank 1 Tensors each containing the bucket boundaries for a single
      feature.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `float_values` of `Tensor` objects with type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesBucketize", name, float_values, bucket_boundaries)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_bucketize_eager_fallback(
          float_values, bucket_boundaries, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(float_values, (list, tuple)):
    raise TypeError(
        "Expected list for 'float_values' argument to "
        "'boosted_trees_bucketize' Op, not %r." % float_values)
  _attr_num_features = len(float_values)
  if not isinstance(bucket_boundaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucket_boundaries' argument to "
        "'boosted_trees_bucketize' Op, not %r." % bucket_boundaries)
  if len(bucket_boundaries) != _attr_num_features:
    raise ValueError(
        "List argument 'bucket_boundaries' to 'boosted_trees_bucketize' Op with length %d "
        "must match length %d of argument 'float_values'." %
        (len(bucket_boundaries), _attr_num_features))
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesBucketize", float_values=float_values,
                                 bucket_boundaries=bucket_boundaries,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_features", _op._get_attr_int("num_features"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesBucketize", _inputs_flat, _attrs, _result)
  return _result

BoostedTreesBucketize = tf_export("raw_ops.BoostedTreesBucketize")(_ops.to_raw_op(boosted_trees_bucketize))


def boosted_trees_bucketize_eager_fallback(float_values: Annotated[List[Any], _atypes.Float32], bucket_boundaries: Annotated[List[Any], _atypes.Float32], name, ctx):
  if not isinstance(float_values, (list, tuple)):
    raise TypeError(
        "Expected list for 'float_values' argument to "
        "'boosted_trees_bucketize' Op, not %r." % float_values)
  _attr_num_features = len(float_values)
  if not isinstance(bucket_boundaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucket_boundaries' argument to "
        "'boosted_trees_bucketize' Op, not %r." % bucket_boundaries)
  if len(bucket_boundaries) != _attr_num_features:
    raise ValueError(
        "List argument 'bucket_boundaries' to 'boosted_trees_bucketize' Op with length %d "
        "must match length %d of argument 'float_values'." %
        (len(bucket_boundaries), _attr_num_features))
  float_values = _ops.convert_n_to_tensor(float_values, _dtypes.float32)
  bucket_boundaries = _ops.convert_n_to_tensor(bucket_boundaries, _dtypes.float32)
  _inputs_flat = list(float_values) + list(bucket_boundaries)
  _attrs = ("num_features", _attr_num_features)
  _result = _execute.execute(b"BoostedTreesBucketize", _attr_num_features,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesBucketize", _inputs_flat, _attrs, _result)
  return _result

_BoostedTreesCalculateBestFeatureSplitOutput = collections.namedtuple(
    "BoostedTreesCalculateBestFeatureSplit",
    ["node_ids", "gains", "feature_dimensions", "thresholds", "left_node_contribs", "right_node_contribs", "split_with_default_directions"])


def boosted_trees_calculate_best_feature_split(node_id_range: Annotated[Any, _atypes.Int32], stats_summary: Annotated[Any, _atypes.Float32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], logits_dimension: int, split_type:str="inequality", name=None):
  r"""Calculates gains for each feature and returns the best possible split information for the feature.

  The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

  It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

  In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

  The output shapes are compatible in a way that the first dimension of all tensors are the same and equal to the number of possible split nodes for each feature.

  Args:
    node_id_range: A `Tensor` of type `int32`.
      A Rank 1 tensor (shape=[2]) to specify the range [first, last) of node ids to process within `stats_summary_list`. The nodes are iterated between the two nodes specified by the tensor, as like `for node_id in range(node_id_range[0], node_id_range[1])` (Note that the last index node_id_range[1] is exclusive).
    stats_summary: A `Tensor` of type `float32`.
      A Rank 4 tensor (#shape=[max_splits, feature_dims, bucket, stats_dims]) for accumulated stats summary (gradient/hessian) per node, per dimension, per buckets for each feature.
      The first dimension of the tensor is the maximum number of splits, and thus not all elements of it will be used, but only the indexes specified by node_ids will be used.
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    tree_complexity: A `Tensor` of type `float32`.
      adjustment to the gain, per leaf based.
    min_node_weight: A `Tensor` of type `float32`.
      minimum avg of hessians in a node before required for the node to be considered for splitting.
    logits_dimension: An `int` that is `>= 1`.
      The dimension of logit, i.e., number of classes.
    split_type: An optional `string` from: `"inequality", "equality"`. Defaults to `"inequality"`.
      A string indicating if this Op should perform inequality split or equality split.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (node_ids, gains, feature_dimensions, thresholds, left_node_contribs, right_node_contribs, split_with_default_directions).

    node_ids: A `Tensor` of type `int32`.
    gains: A `Tensor` of type `float32`.
    feature_dimensions: A `Tensor` of type `int32`.
    thresholds: A `Tensor` of type `int32`.
    left_node_contribs: A `Tensor` of type `float32`.
    right_node_contribs: A `Tensor` of type `float32`.
    split_with_default_directions: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesCalculateBestFeatureSplit", name, node_id_range,
        stats_summary, l1, l2, tree_complexity, min_node_weight,
        "logits_dimension", logits_dimension, "split_type", split_type)
      _result = _BoostedTreesCalculateBestFeatureSplitOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_calculate_best_feature_split_eager_fallback(
          node_id_range, stats_summary, l1, l2, tree_complexity,
          min_node_weight, logits_dimension=logits_dimension,
          split_type=split_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  if split_type is None:
    split_type = "inequality"
  split_type = _execute.make_str(split_type, "split_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesCalculateBestFeatureSplit", node_id_range=node_id_range,
                                                 stats_summary=stats_summary,
                                                 l1=l1, l2=l2,
                                                 tree_complexity=tree_complexity,
                                                 min_node_weight=min_node_weight,
                                                 logits_dimension=logits_dimension,
                                                 split_type=split_type,
                                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("logits_dimension", _op._get_attr_int("logits_dimension"),
              "split_type", _op.get_attr("split_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesCalculateBestFeatureSplit", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesCalculateBestFeatureSplitOutput._make(_result)
  return _result

BoostedTreesCalculateBestFeatureSplit = tf_export("raw_ops.BoostedTreesCalculateBestFeatureSplit")(_ops.to_raw_op(boosted_trees_calculate_best_feature_split))


def boosted_trees_calculate_best_feature_split_eager_fallback(node_id_range: Annotated[Any, _atypes.Int32], stats_summary: Annotated[Any, _atypes.Float32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], logits_dimension: int, split_type: str, name, ctx):
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  if split_type is None:
    split_type = "inequality"
  split_type = _execute.make_str(split_type, "split_type")
  node_id_range = _ops.convert_to_tensor(node_id_range, _dtypes.int32)
  stats_summary = _ops.convert_to_tensor(stats_summary, _dtypes.float32)
  l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
  l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
  tree_complexity = _ops.convert_to_tensor(tree_complexity, _dtypes.float32)
  min_node_weight = _ops.convert_to_tensor(min_node_weight, _dtypes.float32)
  _inputs_flat = [node_id_range, stats_summary, l1, l2, tree_complexity, min_node_weight]
  _attrs = ("logits_dimension", logits_dimension, "split_type", split_type)
  _result = _execute.execute(b"BoostedTreesCalculateBestFeatureSplit", 7,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesCalculateBestFeatureSplit", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesCalculateBestFeatureSplitOutput._make(_result)
  return _result

_BoostedTreesCalculateBestFeatureSplitV2Output = collections.namedtuple(
    "BoostedTreesCalculateBestFeatureSplitV2",
    ["node_ids", "gains", "feature_ids", "feature_dimensions", "thresholds", "left_node_contribs", "right_node_contribs", "split_with_default_directions"])


def boosted_trees_calculate_best_feature_split_v2(node_id_range: Annotated[Any, _atypes.Int32], stats_summaries_list: Annotated[List[Any], _atypes.Float32], split_types: Annotated[Any, _atypes.String], candidate_feature_ids: Annotated[Any, _atypes.Int32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], logits_dimension: int, name=None):
  r"""Calculates gains for each feature and returns the best possible split information for each node. However, if no split is found, then no split information is returned for that node.

  The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

  It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

  In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

  The output shapes are compatible in a way that the first dimension of all tensors are the same and equal to the number of possible split nodes for each feature.

  Args:
    node_id_range: A `Tensor` of type `int32`.
      A Rank 1 tensor (shape=[2]) to specify the range [first, last) of node ids to process within `stats_summary_list`. The nodes are iterated between the two nodes specified by the tensor, as like `for node_id in range(node_id_range[0], node_id_range[1])` (Note that the last index node_id_range[1] is exclusive).
    stats_summaries_list: A list of at least 1 `Tensor` objects with type `float32`.
      A list of Rank 4 tensor (#shape=[max_splits, feature_dims, bucket, stats_dims]) for accumulated stats summary (gradient/hessian) per node, per dimension, per buckets for each feature.
      The first dimension of the tensor is the maximum number of splits, and thus not all elements of it will be used, but only the indexes specified by node_ids will be used.
    split_types: A `Tensor` of type `string`.
      A Rank 1 tensor indicating if this Op should perform inequality split or equality split per feature.
    candidate_feature_ids: A `Tensor` of type `int32`.
      Rank 1 tensor with ids for each feature. This is the real id of the feature.
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    tree_complexity: A `Tensor` of type `float32`.
      adjustment to the gain, per leaf based.
    min_node_weight: A `Tensor` of type `float32`.
      minimum avg of hessians in a node before required for the node to be considered for splitting.
    logits_dimension: An `int` that is `>= 1`.
      The dimension of logit, i.e., number of classes.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (node_ids, gains, feature_ids, feature_dimensions, thresholds, left_node_contribs, right_node_contribs, split_with_default_directions).

    node_ids: A `Tensor` of type `int32`.
    gains: A `Tensor` of type `float32`.
    feature_ids: A `Tensor` of type `int32`.
    feature_dimensions: A `Tensor` of type `int32`.
    thresholds: A `Tensor` of type `int32`.
    left_node_contribs: A `Tensor` of type `float32`.
    right_node_contribs: A `Tensor` of type `float32`.
    split_with_default_directions: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesCalculateBestFeatureSplitV2", name, node_id_range,
        stats_summaries_list, split_types, candidate_feature_ids, l1, l2,
        tree_complexity, min_node_weight, "logits_dimension",
        logits_dimension)
      _result = _BoostedTreesCalculateBestFeatureSplitV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_calculate_best_feature_split_v2_eager_fallback(
          node_id_range, stats_summaries_list, split_types,
          candidate_feature_ids, l1, l2, tree_complexity, min_node_weight,
          logits_dimension=logits_dimension, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(stats_summaries_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'stats_summaries_list' argument to "
        "'boosted_trees_calculate_best_feature_split_v2' Op, not %r." % stats_summaries_list)
  _attr_num_features = len(stats_summaries_list)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesCalculateBestFeatureSplitV2", node_id_range=node_id_range,
                                                   stats_summaries_list=stats_summaries_list,
                                                   split_types=split_types,
                                                   candidate_feature_ids=candidate_feature_ids,
                                                   l1=l1, l2=l2,
                                                   tree_complexity=tree_complexity,
                                                   min_node_weight=min_node_weight,
                                                   logits_dimension=logits_dimension,
                                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_features", _op._get_attr_int("num_features"),
              "logits_dimension", _op._get_attr_int("logits_dimension"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesCalculateBestFeatureSplitV2", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesCalculateBestFeatureSplitV2Output._make(_result)
  return _result

BoostedTreesCalculateBestFeatureSplitV2 = tf_export("raw_ops.BoostedTreesCalculateBestFeatureSplitV2")(_ops.to_raw_op(boosted_trees_calculate_best_feature_split_v2))


def boosted_trees_calculate_best_feature_split_v2_eager_fallback(node_id_range: Annotated[Any, _atypes.Int32], stats_summaries_list: Annotated[List[Any], _atypes.Float32], split_types: Annotated[Any, _atypes.String], candidate_feature_ids: Annotated[Any, _atypes.Int32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], logits_dimension: int, name, ctx):
  if not isinstance(stats_summaries_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'stats_summaries_list' argument to "
        "'boosted_trees_calculate_best_feature_split_v2' Op, not %r." % stats_summaries_list)
  _attr_num_features = len(stats_summaries_list)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  node_id_range = _ops.convert_to_tensor(node_id_range, _dtypes.int32)
  stats_summaries_list = _ops.convert_n_to_tensor(stats_summaries_list, _dtypes.float32)
  split_types = _ops.convert_to_tensor(split_types, _dtypes.string)
  candidate_feature_ids = _ops.convert_to_tensor(candidate_feature_ids, _dtypes.int32)
  l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
  l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
  tree_complexity = _ops.convert_to_tensor(tree_complexity, _dtypes.float32)
  min_node_weight = _ops.convert_to_tensor(min_node_weight, _dtypes.float32)
  _inputs_flat = [node_id_range] + list(stats_summaries_list) + [split_types, candidate_feature_ids, l1, l2, tree_complexity, min_node_weight]
  _attrs = ("num_features", _attr_num_features, "logits_dimension",
  logits_dimension)
  _result = _execute.execute(b"BoostedTreesCalculateBestFeatureSplitV2", 8,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesCalculateBestFeatureSplitV2", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesCalculateBestFeatureSplitV2Output._make(_result)
  return _result

_BoostedTreesCalculateBestGainsPerFeatureOutput = collections.namedtuple(
    "BoostedTreesCalculateBestGainsPerFeature",
    ["node_ids_list", "gains_list", "thresholds_list", "left_node_contribs_list", "right_node_contribs_list"])


def boosted_trees_calculate_best_gains_per_feature(node_id_range: Annotated[Any, _atypes.Int32], stats_summary_list: Annotated[List[Any], _atypes.Float32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], max_splits: int, name=None):
  r"""Calculates gains for each feature and returns the best possible split information for the feature.

  The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

  It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

  In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

  The length of output lists are all of the same length, `num_features`.
  The output shapes are compatible in a way that the first dimension of all tensors of all lists are the same and equal to the number of possible split nodes for each feature.

  Args:
    node_id_range: A `Tensor` of type `int32`.
      A Rank 1 tensor (shape=[2]) to specify the range [first, last) of node ids to process within `stats_summary_list`. The nodes are iterated between the two nodes specified by the tensor, as like `for node_id in range(node_id_range[0], node_id_range[1])` (Note that the last index node_id_range[1] is exclusive).
    stats_summary_list: A list of at least 1 `Tensor` objects with type `float32`.
      A list of Rank 3 tensor (#shape=[max_splits, bucket, 2]) for accumulated stats summary (gradient/hessian) per node per buckets for each feature. The first dimension of the tensor is the maximum number of splits, and thus not all elements of it will be used, but only the indexes specified by node_ids will be used.
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    tree_complexity: A `Tensor` of type `float32`.
      adjustment to the gain, per leaf based.
    min_node_weight: A `Tensor` of type `float32`.
      minimum avg of hessians in a node before required for the node to be considered for splitting.
    max_splits: An `int` that is `>= 1`.
      the number of nodes that can be split in the whole tree. Used as a dimension of output tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (node_ids_list, gains_list, thresholds_list, left_node_contribs_list, right_node_contribs_list).

    node_ids_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `int32`.
    gains_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `float32`.
    thresholds_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `int32`.
    left_node_contribs_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `float32`.
    right_node_contribs_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesCalculateBestGainsPerFeature", name, node_id_range,
        stats_summary_list, l1, l2, tree_complexity, min_node_weight,
        "max_splits", max_splits)
      _result = _BoostedTreesCalculateBestGainsPerFeatureOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_calculate_best_gains_per_feature_eager_fallback(
          node_id_range, stats_summary_list, l1, l2, tree_complexity,
          min_node_weight, max_splits=max_splits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(stats_summary_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'stats_summary_list' argument to "
        "'boosted_trees_calculate_best_gains_per_feature' Op, not %r." % stats_summary_list)
  _attr_num_features = len(stats_summary_list)
  max_splits = _execute.make_int(max_splits, "max_splits")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesCalculateBestGainsPerFeature", node_id_range=node_id_range,
                                                    stats_summary_list=stats_summary_list,
                                                    l1=l1, l2=l2,
                                                    tree_complexity=tree_complexity,
                                                    min_node_weight=min_node_weight,
                                                    max_splits=max_splits,
                                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_splits", _op._get_attr_int("max_splits"), "num_features",
              _op._get_attr_int("num_features"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesCalculateBestGainsPerFeature", _inputs_flat, _attrs, _result)
  _result = [_result[:_attr_num_features]] + _result[_attr_num_features:]
  _result = _result[:1] + [_result[1:1 + _attr_num_features]] + _result[1 + _attr_num_features:]
  _result = _result[:2] + [_result[2:2 + _attr_num_features]] + _result[2 + _attr_num_features:]
  _result = _result[:3] + [_result[3:3 + _attr_num_features]] + _result[3 + _attr_num_features:]
  _result = _result[:4] + [_result[4:]]
  _result = _BoostedTreesCalculateBestGainsPerFeatureOutput._make(_result)
  return _result

BoostedTreesCalculateBestGainsPerFeature = tf_export("raw_ops.BoostedTreesCalculateBestGainsPerFeature")(_ops.to_raw_op(boosted_trees_calculate_best_gains_per_feature))


def boosted_trees_calculate_best_gains_per_feature_eager_fallback(node_id_range: Annotated[Any, _atypes.Int32], stats_summary_list: Annotated[List[Any], _atypes.Float32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], max_splits: int, name, ctx):
  if not isinstance(stats_summary_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'stats_summary_list' argument to "
        "'boosted_trees_calculate_best_gains_per_feature' Op, not %r." % stats_summary_list)
  _attr_num_features = len(stats_summary_list)
  max_splits = _execute.make_int(max_splits, "max_splits")
  node_id_range = _ops.convert_to_tensor(node_id_range, _dtypes.int32)
  stats_summary_list = _ops.convert_n_to_tensor(stats_summary_list, _dtypes.float32)
  l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
  l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
  tree_complexity = _ops.convert_to_tensor(tree_complexity, _dtypes.float32)
  min_node_weight = _ops.convert_to_tensor(min_node_weight, _dtypes.float32)
  _inputs_flat = [node_id_range] + list(stats_summary_list) + [l1, l2, tree_complexity, min_node_weight]
  _attrs = ("max_splits", max_splits, "num_features", _attr_num_features)
  _result = _execute.execute(b"BoostedTreesCalculateBestGainsPerFeature",
                             _attr_num_features + _attr_num_features +
                             _attr_num_features + _attr_num_features +
                             _attr_num_features, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesCalculateBestGainsPerFeature", _inputs_flat, _attrs, _result)
  _result = [_result[:_attr_num_features]] + _result[_attr_num_features:]
  _result = _result[:1] + [_result[1:1 + _attr_num_features]] + _result[1 + _attr_num_features:]
  _result = _result[:2] + [_result[2:2 + _attr_num_features]] + _result[2 + _attr_num_features:]
  _result = _result[:3] + [_result[3:3 + _attr_num_features]] + _result[3 + _attr_num_features:]
  _result = _result[:4] + [_result[4:]]
  _result = _BoostedTreesCalculateBestGainsPerFeatureOutput._make(_result)
  return _result


def boosted_trees_center_bias(tree_ensemble_handle: Annotated[Any, _atypes.Resource], mean_gradients: Annotated[Any, _atypes.Float32], mean_hessians: Annotated[Any, _atypes.Float32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Calculates the prior from the training data (the bias) and fills in the first node with the logits' prior. Returns a boolean indicating whether to continue centering.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble.
    mean_gradients: A `Tensor` of type `float32`.
      A tensor with shape=[logits_dimension] with mean of gradients for a first node.
    mean_hessians: A `Tensor` of type `float32`.
      A tensor with shape=[logits_dimension] mean of hessians for a first node.
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesCenterBias", name, tree_ensemble_handle,
        mean_gradients, mean_hessians, l1, l2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_center_bias_eager_fallback(
          tree_ensemble_handle, mean_gradients, mean_hessians, l1, l2,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesCenterBias", tree_ensemble_handle=tree_ensemble_handle,
                                  mean_gradients=mean_gradients,
                                  mean_hessians=mean_hessians, l1=l1, l2=l2,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesCenterBias", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BoostedTreesCenterBias = tf_export("raw_ops.BoostedTreesCenterBias")(_ops.to_raw_op(boosted_trees_center_bias))


def boosted_trees_center_bias_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], mean_gradients: Annotated[Any, _atypes.Float32], mean_hessians: Annotated[Any, _atypes.Float32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Bool]:
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  mean_gradients = _ops.convert_to_tensor(mean_gradients, _dtypes.float32)
  mean_hessians = _ops.convert_to_tensor(mean_hessians, _dtypes.float32)
  l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
  l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
  _inputs_flat = [tree_ensemble_handle, mean_gradients, mean_hessians, l1, l2]
  _attrs = None
  _result = _execute.execute(b"BoostedTreesCenterBias", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesCenterBias", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def boosted_trees_create_ensemble(tree_ensemble_handle: Annotated[Any, _atypes.Resource], stamp_token: Annotated[Any, _atypes.Int64], tree_ensemble_serialized: Annotated[Any, _atypes.String], name=None):
  r"""Creates a tree ensemble model and returns a handle to it.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble resource to be created.
    stamp_token: A `Tensor` of type `int64`.
      Token to use as the initial value of the resource stamp.
    tree_ensemble_serialized: A `Tensor` of type `string`.
      Serialized proto of the tree ensemble.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesCreateEnsemble", name, tree_ensemble_handle,
        stamp_token, tree_ensemble_serialized)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_create_ensemble_eager_fallback(
          tree_ensemble_handle, stamp_token, tree_ensemble_serialized,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesCreateEnsemble", tree_ensemble_handle=tree_ensemble_handle,
                                      stamp_token=stamp_token,
                                      tree_ensemble_serialized=tree_ensemble_serialized,
                                      name=name)
  return _op
BoostedTreesCreateEnsemble = tf_export("raw_ops.BoostedTreesCreateEnsemble")(_ops.to_raw_op(boosted_trees_create_ensemble))


def boosted_trees_create_ensemble_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], stamp_token: Annotated[Any, _atypes.Int64], tree_ensemble_serialized: Annotated[Any, _atypes.String], name, ctx):
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  stamp_token = _ops.convert_to_tensor(stamp_token, _dtypes.int64)
  tree_ensemble_serialized = _ops.convert_to_tensor(tree_ensemble_serialized, _dtypes.string)
  _inputs_flat = [tree_ensemble_handle, stamp_token, tree_ensemble_serialized]
  _attrs = None
  _result = _execute.execute(b"BoostedTreesCreateEnsemble", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def boosted_trees_create_quantile_stream_resource(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], epsilon: Annotated[Any, _atypes.Float32], num_streams: Annotated[Any, _atypes.Int64], max_elements:int=1099511627776, name=None):
  r"""Create the Resource for Quantile Streams.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource; Handle to quantile stream resource.
    epsilon: A `Tensor` of type `float32`.
      float; The required approximation error of the stream resource.
    num_streams: A `Tensor` of type `int64`.
      int; The number of streams managed by the resource that shares the same epsilon.
    max_elements: An optional `int`. Defaults to `1099511627776`.
      int; The maximum number of data points that can be fed to the stream.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesCreateQuantileStreamResource", name,
        quantile_stream_resource_handle, epsilon, num_streams, "max_elements",
        max_elements)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_create_quantile_stream_resource_eager_fallback(
          quantile_stream_resource_handle, epsilon, num_streams,
          max_elements=max_elements, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if max_elements is None:
    max_elements = 1099511627776
  max_elements = _execute.make_int(max_elements, "max_elements")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesCreateQuantileStreamResource", quantile_stream_resource_handle=quantile_stream_resource_handle,
                                                    epsilon=epsilon,
                                                    num_streams=num_streams,
                                                    max_elements=max_elements,
                                                    name=name)
  return _op
BoostedTreesCreateQuantileStreamResource = tf_export("raw_ops.BoostedTreesCreateQuantileStreamResource")(_ops.to_raw_op(boosted_trees_create_quantile_stream_resource))


def boosted_trees_create_quantile_stream_resource_eager_fallback(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], epsilon: Annotated[Any, _atypes.Float32], num_streams: Annotated[Any, _atypes.Int64], max_elements: int, name, ctx):
  if max_elements is None:
    max_elements = 1099511627776
  max_elements = _execute.make_int(max_elements, "max_elements")
  quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
  epsilon = _ops.convert_to_tensor(epsilon, _dtypes.float32)
  num_streams = _ops.convert_to_tensor(num_streams, _dtypes.int64)
  _inputs_flat = [quantile_stream_resource_handle, epsilon, num_streams]
  _attrs = ("max_elements", max_elements)
  _result = _execute.execute(b"BoostedTreesCreateQuantileStreamResource", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def boosted_trees_deserialize_ensemble(tree_ensemble_handle: Annotated[Any, _atypes.Resource], stamp_token: Annotated[Any, _atypes.Int64], tree_ensemble_serialized: Annotated[Any, _atypes.String], name=None):
  r"""Deserializes a serialized tree ensemble config and replaces current tree

  ensemble.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble.
    stamp_token: A `Tensor` of type `int64`.
      Token to use as the new value of the resource stamp.
    tree_ensemble_serialized: A `Tensor` of type `string`.
      Serialized proto of the ensemble.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesDeserializeEnsemble", name, tree_ensemble_handle,
        stamp_token, tree_ensemble_serialized)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_deserialize_ensemble_eager_fallback(
          tree_ensemble_handle, stamp_token, tree_ensemble_serialized,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesDeserializeEnsemble", tree_ensemble_handle=tree_ensemble_handle,
                                           stamp_token=stamp_token,
                                           tree_ensemble_serialized=tree_ensemble_serialized,
                                           name=name)
  return _op
BoostedTreesDeserializeEnsemble = tf_export("raw_ops.BoostedTreesDeserializeEnsemble")(_ops.to_raw_op(boosted_trees_deserialize_ensemble))


def boosted_trees_deserialize_ensemble_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], stamp_token: Annotated[Any, _atypes.Int64], tree_ensemble_serialized: Annotated[Any, _atypes.String], name, ctx):
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  stamp_token = _ops.convert_to_tensor(stamp_token, _dtypes.int64)
  tree_ensemble_serialized = _ops.convert_to_tensor(tree_ensemble_serialized, _dtypes.string)
  _inputs_flat = [tree_ensemble_handle, stamp_token, tree_ensemble_serialized]
  _attrs = None
  _result = _execute.execute(b"BoostedTreesDeserializeEnsemble", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def boosted_trees_ensemble_resource_handle_op(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a handle to a BoostedTreesEnsembleResource

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesEnsembleResourceHandleOp", name, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_ensemble_resource_handle_op_eager_fallback(
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesEnsembleResourceHandleOp", container=container,
                                                shared_name=shared_name,
                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesEnsembleResourceHandleOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BoostedTreesEnsembleResourceHandleOp = tf_export("raw_ops.BoostedTreesEnsembleResourceHandleOp")(_ops.to_raw_op(boosted_trees_ensemble_resource_handle_op))


def boosted_trees_ensemble_resource_handle_op_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"BoostedTreesEnsembleResourceHandleOp", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesEnsembleResourceHandleOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def boosted_trees_example_debug_outputs(tree_ensemble_handle: Annotated[Any, _atypes.Resource], bucketized_features: Annotated[List[Any], _atypes.Int32], logits_dimension: int, name=None) -> Annotated[Any, _atypes.String]:
  r"""Debugging/model interpretability outputs for each example.

  It traverses all the trees and computes debug metrics for individual examples,
  such as getting split feature ids and logits after each split along the decision
  path used to compute directional feature contributions.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
    bucketized_features: A list of at least 1 `Tensor` objects with type `int32`.
      A list of rank 1 Tensors containing bucket id for each
      feature.
    logits_dimension: An `int`.
      scalar, dimension of the logits, to be used for constructing the protos in
      examples_debug_outputs_serialized.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesExampleDebugOutputs", name, tree_ensemble_handle,
        bucketized_features, "logits_dimension", logits_dimension)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_example_debug_outputs_eager_fallback(
          tree_ensemble_handle, bucketized_features,
          logits_dimension=logits_dimension, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(bucketized_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features' argument to "
        "'boosted_trees_example_debug_outputs' Op, not %r." % bucketized_features)
  _attr_num_bucketized_features = len(bucketized_features)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesExampleDebugOutputs", tree_ensemble_handle=tree_ensemble_handle,
                                           bucketized_features=bucketized_features,
                                           logits_dimension=logits_dimension,
                                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_bucketized_features",
              _op._get_attr_int("num_bucketized_features"),
              "logits_dimension", _op._get_attr_int("logits_dimension"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesExampleDebugOutputs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BoostedTreesExampleDebugOutputs = tf_export("raw_ops.BoostedTreesExampleDebugOutputs")(_ops.to_raw_op(boosted_trees_example_debug_outputs))


def boosted_trees_example_debug_outputs_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], bucketized_features: Annotated[List[Any], _atypes.Int32], logits_dimension: int, name, ctx) -> Annotated[Any, _atypes.String]:
  if not isinstance(bucketized_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features' argument to "
        "'boosted_trees_example_debug_outputs' Op, not %r." % bucketized_features)
  _attr_num_bucketized_features = len(bucketized_features)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  bucketized_features = _ops.convert_n_to_tensor(bucketized_features, _dtypes.int32)
  _inputs_flat = [tree_ensemble_handle] + list(bucketized_features)
  _attrs = ("num_bucketized_features", _attr_num_bucketized_features,
  "logits_dimension", logits_dimension)
  _result = _execute.execute(b"BoostedTreesExampleDebugOutputs", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesExampleDebugOutputs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def boosted_trees_flush_quantile_summaries(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], num_features: int, name=None):
  r"""Flush the quantile summaries from each quantile stream resource.

  An op that outputs a list of quantile summaries of a quantile stream resource.
  Each summary Tensor is rank 2, containing summaries (value, weight, min_rank,
  max_rank) for a single feature.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource handle referring to a QuantileStreamResource.
    num_features: An `int` that is `>= 0`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_features` `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesFlushQuantileSummaries", name,
        quantile_stream_resource_handle, "num_features", num_features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_flush_quantile_summaries_eager_fallback(
          quantile_stream_resource_handle, num_features=num_features,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_features = _execute.make_int(num_features, "num_features")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesFlushQuantileSummaries", quantile_stream_resource_handle=quantile_stream_resource_handle,
                                              num_features=num_features,
                                              name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("num_features", _op._get_attr_int("num_features"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesFlushQuantileSummaries", _inputs_flat, _attrs, _result)
  return _result

BoostedTreesFlushQuantileSummaries = tf_export("raw_ops.BoostedTreesFlushQuantileSummaries")(_ops.to_raw_op(boosted_trees_flush_quantile_summaries))


def boosted_trees_flush_quantile_summaries_eager_fallback(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], num_features: int, name, ctx):
  num_features = _execute.make_int(num_features, "num_features")
  quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
  _inputs_flat = [quantile_stream_resource_handle]
  _attrs = ("num_features", num_features)
  _result = _execute.execute(b"BoostedTreesFlushQuantileSummaries",
                             num_features, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesFlushQuantileSummaries", _inputs_flat, _attrs, _result)
  return _result

_BoostedTreesGetEnsembleStatesOutput = collections.namedtuple(
    "BoostedTreesGetEnsembleStates",
    ["stamp_token", "num_trees", "num_finalized_trees", "num_attempted_layers", "last_layer_nodes_range"])


def boosted_trees_get_ensemble_states(tree_ensemble_handle: Annotated[Any, _atypes.Resource], name=None):
  r"""Retrieves the tree ensemble resource stamp token, number of trees and growing statistics.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (stamp_token, num_trees, num_finalized_trees, num_attempted_layers, last_layer_nodes_range).

    stamp_token: A `Tensor` of type `int64`.
    num_trees: A `Tensor` of type `int32`.
    num_finalized_trees: A `Tensor` of type `int32`.
    num_attempted_layers: A `Tensor` of type `int32`.
    last_layer_nodes_range: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesGetEnsembleStates", name, tree_ensemble_handle)
      _result = _BoostedTreesGetEnsembleStatesOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_get_ensemble_states_eager_fallback(
          tree_ensemble_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesGetEnsembleStates", tree_ensemble_handle=tree_ensemble_handle,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesGetEnsembleStates", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesGetEnsembleStatesOutput._make(_result)
  return _result

BoostedTreesGetEnsembleStates = tf_export("raw_ops.BoostedTreesGetEnsembleStates")(_ops.to_raw_op(boosted_trees_get_ensemble_states))


def boosted_trees_get_ensemble_states_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], name, ctx):
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  _inputs_flat = [tree_ensemble_handle]
  _attrs = None
  _result = _execute.execute(b"BoostedTreesGetEnsembleStates", 5,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesGetEnsembleStates", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesGetEnsembleStatesOutput._make(_result)
  return _result


def boosted_trees_make_quantile_summaries(float_values: Annotated[List[Any], _atypes.Float32], example_weights: Annotated[Any, _atypes.Float32], epsilon: Annotated[Any, _atypes.Float32], name=None):
  r"""Makes the summary of quantiles for the batch.

  An op that takes a list of tensors (one tensor per feature) and outputs the
  quantile summaries for each tensor.

  Args:
    float_values: A list of `Tensor` objects with type `float32`.
      float; List of Rank 1 Tensors each containing values for a single feature.
    example_weights: A `Tensor` of type `float32`.
      float; Rank 1 Tensor with weights per instance.
    epsilon: A `Tensor` of type `float32`.
      float; The required maximum approximation error.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `float_values` of `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesMakeQuantileSummaries", name, float_values,
        example_weights, epsilon)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_make_quantile_summaries_eager_fallback(
          float_values, example_weights, epsilon, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(float_values, (list, tuple)):
    raise TypeError(
        "Expected list for 'float_values' argument to "
        "'boosted_trees_make_quantile_summaries' Op, not %r." % float_values)
  _attr_num_features = len(float_values)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesMakeQuantileSummaries", float_values=float_values,
                                             example_weights=example_weights,
                                             epsilon=epsilon, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_features", _op._get_attr_int("num_features"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesMakeQuantileSummaries", _inputs_flat, _attrs, _result)
  return _result

BoostedTreesMakeQuantileSummaries = tf_export("raw_ops.BoostedTreesMakeQuantileSummaries")(_ops.to_raw_op(boosted_trees_make_quantile_summaries))


def boosted_trees_make_quantile_summaries_eager_fallback(float_values: Annotated[List[Any], _atypes.Float32], example_weights: Annotated[Any, _atypes.Float32], epsilon: Annotated[Any, _atypes.Float32], name, ctx):
  if not isinstance(float_values, (list, tuple)):
    raise TypeError(
        "Expected list for 'float_values' argument to "
        "'boosted_trees_make_quantile_summaries' Op, not %r." % float_values)
  _attr_num_features = len(float_values)
  float_values = _ops.convert_n_to_tensor(float_values, _dtypes.float32)
  example_weights = _ops.convert_to_tensor(example_weights, _dtypes.float32)
  epsilon = _ops.convert_to_tensor(epsilon, _dtypes.float32)
  _inputs_flat = list(float_values) + [example_weights, epsilon]
  _attrs = ("num_features", _attr_num_features)
  _result = _execute.execute(b"BoostedTreesMakeQuantileSummaries",
                             _attr_num_features, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesMakeQuantileSummaries", _inputs_flat, _attrs, _result)
  return _result


def boosted_trees_make_stats_summary(node_ids: Annotated[Any, _atypes.Int32], gradients: Annotated[Any, _atypes.Float32], hessians: Annotated[Any, _atypes.Float32], bucketized_features_list: Annotated[List[Any], _atypes.Int32], max_splits: int, num_buckets: int, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Makes the summary of accumulated stats for the batch.

  The summary stats contains gradients and hessians accumulated into the corresponding node and bucket for each example.

  Args:
    node_ids: A `Tensor` of type `int32`.
      int32 Rank 1 Tensor containing node ids, which each example falls into for the requested layer.
    gradients: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[#examples, 1]) for gradients.
    hessians: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[#examples, 1]) for hessians.
    bucketized_features_list: A list of at least 1 `Tensor` objects with type `int32`.
      int32 list of Rank 1 Tensors, each containing the bucketized feature (for each feature column).
    max_splits: An `int` that is `>= 1`.
      int; the maximum number of splits possible in the whole tree.
    num_buckets: An `int` that is `>= 1`.
      int; equals to the maximum possible value of bucketized feature.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesMakeStatsSummary", name, node_ids, gradients,
        hessians, bucketized_features_list, "max_splits", max_splits,
        "num_buckets", num_buckets)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_make_stats_summary_eager_fallback(
          node_ids, gradients, hessians, bucketized_features_list,
          max_splits=max_splits, num_buckets=num_buckets, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(bucketized_features_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features_list' argument to "
        "'boosted_trees_make_stats_summary' Op, not %r." % bucketized_features_list)
  _attr_num_features = len(bucketized_features_list)
  max_splits = _execute.make_int(max_splits, "max_splits")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesMakeStatsSummary", node_ids=node_ids,
                                        gradients=gradients,
                                        hessians=hessians,
                                        bucketized_features_list=bucketized_features_list,
                                        max_splits=max_splits,
                                        num_buckets=num_buckets, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_splits", _op._get_attr_int("max_splits"), "num_buckets",
              _op._get_attr_int("num_buckets"), "num_features",
              _op._get_attr_int("num_features"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesMakeStatsSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BoostedTreesMakeStatsSummary = tf_export("raw_ops.BoostedTreesMakeStatsSummary")(_ops.to_raw_op(boosted_trees_make_stats_summary))


def boosted_trees_make_stats_summary_eager_fallback(node_ids: Annotated[Any, _atypes.Int32], gradients: Annotated[Any, _atypes.Float32], hessians: Annotated[Any, _atypes.Float32], bucketized_features_list: Annotated[List[Any], _atypes.Int32], max_splits: int, num_buckets: int, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if not isinstance(bucketized_features_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features_list' argument to "
        "'boosted_trees_make_stats_summary' Op, not %r." % bucketized_features_list)
  _attr_num_features = len(bucketized_features_list)
  max_splits = _execute.make_int(max_splits, "max_splits")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  node_ids = _ops.convert_to_tensor(node_ids, _dtypes.int32)
  gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
  hessians = _ops.convert_to_tensor(hessians, _dtypes.float32)
  bucketized_features_list = _ops.convert_n_to_tensor(bucketized_features_list, _dtypes.int32)
  _inputs_flat = [node_ids, gradients, hessians] + list(bucketized_features_list)
  _attrs = ("max_splits", max_splits, "num_buckets", num_buckets,
  "num_features", _attr_num_features)
  _result = _execute.execute(b"BoostedTreesMakeStatsSummary", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesMakeStatsSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def boosted_trees_predict(tree_ensemble_handle: Annotated[Any, _atypes.Resource], bucketized_features: Annotated[List[Any], _atypes.Int32], logits_dimension: int, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Runs multiple additive regression ensemble predictors on input instances and

  computes the logits. It is designed to be used during prediction.
  It traverses all the trees and calculates the final score for each instance.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
    bucketized_features: A list of at least 1 `Tensor` objects with type `int32`.
      A list of rank 1 Tensors containing bucket id for each
      feature.
    logits_dimension: An `int`.
      scalar, dimension of the logits, to be used for partial logits
      shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesPredict", name, tree_ensemble_handle,
        bucketized_features, "logits_dimension", logits_dimension)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_predict_eager_fallback(
          tree_ensemble_handle, bucketized_features,
          logits_dimension=logits_dimension, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(bucketized_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features' argument to "
        "'boosted_trees_predict' Op, not %r." % bucketized_features)
  _attr_num_bucketized_features = len(bucketized_features)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesPredict", tree_ensemble_handle=tree_ensemble_handle,
                               bucketized_features=bucketized_features,
                               logits_dimension=logits_dimension, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_bucketized_features",
              _op._get_attr_int("num_bucketized_features"),
              "logits_dimension", _op._get_attr_int("logits_dimension"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesPredict", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BoostedTreesPredict = tf_export("raw_ops.BoostedTreesPredict")(_ops.to_raw_op(boosted_trees_predict))


def boosted_trees_predict_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], bucketized_features: Annotated[List[Any], _atypes.Int32], logits_dimension: int, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if not isinstance(bucketized_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features' argument to "
        "'boosted_trees_predict' Op, not %r." % bucketized_features)
  _attr_num_bucketized_features = len(bucketized_features)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  bucketized_features = _ops.convert_n_to_tensor(bucketized_features, _dtypes.int32)
  _inputs_flat = [tree_ensemble_handle] + list(bucketized_features)
  _attrs = ("num_bucketized_features", _attr_num_bucketized_features,
  "logits_dimension", logits_dimension)
  _result = _execute.execute(b"BoostedTreesPredict", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesPredict", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def boosted_trees_quantile_stream_resource_add_summaries(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], summaries: Annotated[List[Any], _atypes.Float32], name=None):
  r"""Add the quantile summaries to each quantile stream resource.

  An op that adds a list of quantile summaries to a quantile stream resource. Each
  summary Tensor is rank 2, containing summaries (value, weight, min_rank, max_rank)
  for a single feature.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource handle referring to a QuantileStreamResource.
    summaries: A list of `Tensor` objects with type `float32`.
      string; List of Rank 2 Tensor each containing the summaries for a single feature.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesQuantileStreamResourceAddSummaries", name,
        quantile_stream_resource_handle, summaries)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_quantile_stream_resource_add_summaries_eager_fallback(
          quantile_stream_resource_handle, summaries, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(summaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'summaries' argument to "
        "'boosted_trees_quantile_stream_resource_add_summaries' Op, not %r." % summaries)
  _attr_num_features = len(summaries)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesQuantileStreamResourceAddSummaries", quantile_stream_resource_handle=quantile_stream_resource_handle,
                                                          summaries=summaries,
                                                          name=name)
  return _op
BoostedTreesQuantileStreamResourceAddSummaries = tf_export("raw_ops.BoostedTreesQuantileStreamResourceAddSummaries")(_ops.to_raw_op(boosted_trees_quantile_stream_resource_add_summaries))


def boosted_trees_quantile_stream_resource_add_summaries_eager_fallback(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], summaries: Annotated[List[Any], _atypes.Float32], name, ctx):
  if not isinstance(summaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'summaries' argument to "
        "'boosted_trees_quantile_stream_resource_add_summaries' Op, not %r." % summaries)
  _attr_num_features = len(summaries)
  quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
  summaries = _ops.convert_n_to_tensor(summaries, _dtypes.float32)
  _inputs_flat = [quantile_stream_resource_handle] + list(summaries)
  _attrs = ("num_features", _attr_num_features)
  _result = _execute.execute(b"BoostedTreesQuantileStreamResourceAddSummaries",
                             0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def boosted_trees_quantile_stream_resource_deserialize(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], bucket_boundaries: Annotated[List[Any], _atypes.Float32], name=None):
  r"""Deserialize bucket boundaries and ready flag into current QuantileAccumulator.

  An op that deserializes bucket boundaries and are boundaries ready flag into current QuantileAccumulator.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource handle referring to a QuantileStreamResource.
    bucket_boundaries: A list of at least 1 `Tensor` objects with type `float32`.
      float; List of Rank 1 Tensors each containing the bucket boundaries for a feature.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesQuantileStreamResourceDeserialize", name,
        quantile_stream_resource_handle, bucket_boundaries)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_quantile_stream_resource_deserialize_eager_fallback(
          quantile_stream_resource_handle, bucket_boundaries, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(bucket_boundaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucket_boundaries' argument to "
        "'boosted_trees_quantile_stream_resource_deserialize' Op, not %r." % bucket_boundaries)
  _attr_num_streams = len(bucket_boundaries)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesQuantileStreamResourceDeserialize", quantile_stream_resource_handle=quantile_stream_resource_handle,
                                                         bucket_boundaries=bucket_boundaries,
                                                         name=name)
  return _op
BoostedTreesQuantileStreamResourceDeserialize = tf_export("raw_ops.BoostedTreesQuantileStreamResourceDeserialize")(_ops.to_raw_op(boosted_trees_quantile_stream_resource_deserialize))


def boosted_trees_quantile_stream_resource_deserialize_eager_fallback(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], bucket_boundaries: Annotated[List[Any], _atypes.Float32], name, ctx):
  if not isinstance(bucket_boundaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucket_boundaries' argument to "
        "'boosted_trees_quantile_stream_resource_deserialize' Op, not %r." % bucket_boundaries)
  _attr_num_streams = len(bucket_boundaries)
  quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
  bucket_boundaries = _ops.convert_n_to_tensor(bucket_boundaries, _dtypes.float32)
  _inputs_flat = [quantile_stream_resource_handle] + list(bucket_boundaries)
  _attrs = ("num_streams", _attr_num_streams)
  _result = _execute.execute(b"BoostedTreesQuantileStreamResourceDeserialize",
                             0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def boosted_trees_quantile_stream_resource_flush(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], num_buckets: Annotated[Any, _atypes.Int64], generate_quantiles:bool=False, name=None):
  r"""Flush the summaries for a quantile stream resource.

  An op that flushes the summaries for a quantile stream resource.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource handle referring to a QuantileStreamResource.
    num_buckets: A `Tensor` of type `int64`.
      int; approximate number of buckets unless using generate_quantiles.
    generate_quantiles: An optional `bool`. Defaults to `False`.
      bool; If True, the output will be the num_quantiles for each stream where the ith
      entry is the ith quantile of the input with an approximation error of epsilon.
      Duplicate values may be present.
      If False, the output will be the points in the histogram that we got which roughly
      translates to 1/epsilon boundaries and without any duplicates.
      Default to False.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesQuantileStreamResourceFlush", name,
        quantile_stream_resource_handle, num_buckets, "generate_quantiles",
        generate_quantiles)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_quantile_stream_resource_flush_eager_fallback(
          quantile_stream_resource_handle, num_buckets,
          generate_quantiles=generate_quantiles, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if generate_quantiles is None:
    generate_quantiles = False
  generate_quantiles = _execute.make_bool(generate_quantiles, "generate_quantiles")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesQuantileStreamResourceFlush", quantile_stream_resource_handle=quantile_stream_resource_handle,
                                                   num_buckets=num_buckets,
                                                   generate_quantiles=generate_quantiles,
                                                   name=name)
  return _op
BoostedTreesQuantileStreamResourceFlush = tf_export("raw_ops.BoostedTreesQuantileStreamResourceFlush")(_ops.to_raw_op(boosted_trees_quantile_stream_resource_flush))


def boosted_trees_quantile_stream_resource_flush_eager_fallback(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], num_buckets: Annotated[Any, _atypes.Int64], generate_quantiles: bool, name, ctx):
  if generate_quantiles is None:
    generate_quantiles = False
  generate_quantiles = _execute.make_bool(generate_quantiles, "generate_quantiles")
  quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
  num_buckets = _ops.convert_to_tensor(num_buckets, _dtypes.int64)
  _inputs_flat = [quantile_stream_resource_handle, num_buckets]
  _attrs = ("generate_quantiles", generate_quantiles)
  _result = _execute.execute(b"BoostedTreesQuantileStreamResourceFlush", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def boosted_trees_quantile_stream_resource_get_bucket_boundaries(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], num_features: int, name=None):
  r"""Generate the bucket boundaries for each feature based on accumulated summaries.

  An op that returns a list of float tensors for a quantile stream resource. Each
  tensor is Rank 1 containing bucket boundaries for a single feature.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource handle referring to a QuantileStreamResource.
    num_features: An `int` that is `>= 0`.
      inferred int; number of features to get bucket boundaries for.
    name: A name for the operation (optional).

  Returns:
    A list of `num_features` `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesQuantileStreamResourceGetBucketBoundaries", name,
        quantile_stream_resource_handle, "num_features", num_features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_quantile_stream_resource_get_bucket_boundaries_eager_fallback(
          quantile_stream_resource_handle, num_features=num_features,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_features = _execute.make_int(num_features, "num_features")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesQuantileStreamResourceGetBucketBoundaries", quantile_stream_resource_handle=quantile_stream_resource_handle,
                                                                 num_features=num_features,
                                                                 name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("num_features", _op._get_attr_int("num_features"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesQuantileStreamResourceGetBucketBoundaries", _inputs_flat, _attrs, _result)
  return _result

BoostedTreesQuantileStreamResourceGetBucketBoundaries = tf_export("raw_ops.BoostedTreesQuantileStreamResourceGetBucketBoundaries")(_ops.to_raw_op(boosted_trees_quantile_stream_resource_get_bucket_boundaries))


def boosted_trees_quantile_stream_resource_get_bucket_boundaries_eager_fallback(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], num_features: int, name, ctx):
  num_features = _execute.make_int(num_features, "num_features")
  quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
  _inputs_flat = [quantile_stream_resource_handle]
  _attrs = ("num_features", num_features)
  _result = _execute.execute(b"BoostedTreesQuantileStreamResourceGetBucketBoundaries",
                             num_features, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesQuantileStreamResourceGetBucketBoundaries", _inputs_flat, _attrs, _result)
  return _result


def boosted_trees_quantile_stream_resource_handle_op(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a handle to a BoostedTreesQuantileStreamResource.

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesQuantileStreamResourceHandleOp", name, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_quantile_stream_resource_handle_op_eager_fallback(
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesQuantileStreamResourceHandleOp", container=container,
                                                      shared_name=shared_name,
                                                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesQuantileStreamResourceHandleOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BoostedTreesQuantileStreamResourceHandleOp = tf_export("raw_ops.BoostedTreesQuantileStreamResourceHandleOp")(_ops.to_raw_op(boosted_trees_quantile_stream_resource_handle_op))


def boosted_trees_quantile_stream_resource_handle_op_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"BoostedTreesQuantileStreamResourceHandleOp", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesQuantileStreamResourceHandleOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_BoostedTreesSerializeEnsembleOutput = collections.namedtuple(
    "BoostedTreesSerializeEnsemble",
    ["stamp_token", "tree_ensemble_serialized"])


def boosted_trees_serialize_ensemble(tree_ensemble_handle: Annotated[Any, _atypes.Resource], name=None):
  r"""Serializes the tree ensemble to a proto.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (stamp_token, tree_ensemble_serialized).

    stamp_token: A `Tensor` of type `int64`.
    tree_ensemble_serialized: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesSerializeEnsemble", name, tree_ensemble_handle)
      _result = _BoostedTreesSerializeEnsembleOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_serialize_ensemble_eager_fallback(
          tree_ensemble_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesSerializeEnsemble", tree_ensemble_handle=tree_ensemble_handle,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesSerializeEnsemble", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesSerializeEnsembleOutput._make(_result)
  return _result

BoostedTreesSerializeEnsemble = tf_export("raw_ops.BoostedTreesSerializeEnsemble")(_ops.to_raw_op(boosted_trees_serialize_ensemble))


def boosted_trees_serialize_ensemble_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], name, ctx):
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  _inputs_flat = [tree_ensemble_handle]
  _attrs = None
  _result = _execute.execute(b"BoostedTreesSerializeEnsemble", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesSerializeEnsemble", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesSerializeEnsembleOutput._make(_result)
  return _result

_BoostedTreesSparseAggregateStatsOutput = collections.namedtuple(
    "BoostedTreesSparseAggregateStats",
    ["stats_summary_indices", "stats_summary_values", "stats_summary_shape"])


def boosted_trees_sparse_aggregate_stats(node_ids: Annotated[Any, _atypes.Int32], gradients: Annotated[Any, _atypes.Float32], hessians: Annotated[Any, _atypes.Float32], feature_indices: Annotated[Any, _atypes.Int32], feature_values: Annotated[Any, _atypes.Int32], feature_shape: Annotated[Any, _atypes.Int32], max_splits: int, num_buckets: int, name=None):
  r"""Aggregates the summary of accumulated stats for the batch.

  The summary stats contains gradients and hessians accumulated for each node, bucket and dimension id.

  Args:
    node_ids: A `Tensor` of type `int32`.
      int32; Rank 1 Tensor containing node ids for each example, shape [batch_size].
    gradients: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, logits_dimension]) with gradients for each example.
    hessians: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, hessian_dimension]) with hessians for each example.
    feature_indices: A `Tensor` of type `int32`.
      int32; Rank 2 indices of feature sparse Tensors (shape=[number of sparse entries, 2]).
      Number of sparse entries across all instances from the batch. The first value is
      the index of the instance, the second is dimension of the feature. The second axis
      can only have 2 values, i.e., the input dense version of Tensor can only be matrix.
    feature_values: A `Tensor` of type `int32`.
      int32; Rank 1 values of feature sparse Tensors (shape=[number of sparse entries]).
      Number of sparse entries across all instances from the batch. The first value is
      the index of the instance, the second is dimension of the feature.
    feature_shape: A `Tensor` of type `int32`.
      int32; Rank 1 dense shape of feature sparse Tensors (shape=[2]).
      The first axis can only have 2 values, [batch_size, feature_dimension].
    max_splits: An `int` that is `>= 1`.
      int; the maximum number of splits possible in the whole tree.
    num_buckets: An `int` that is `>= 1`.
      int; equals to the maximum possible value of bucketized feature + 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (stats_summary_indices, stats_summary_values, stats_summary_shape).

    stats_summary_indices: A `Tensor` of type `int32`.
    stats_summary_values: A `Tensor` of type `float32`.
    stats_summary_shape: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesSparseAggregateStats", name, node_ids, gradients,
        hessians, feature_indices, feature_values, feature_shape,
        "max_splits", max_splits, "num_buckets", num_buckets)
      _result = _BoostedTreesSparseAggregateStatsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_sparse_aggregate_stats_eager_fallback(
          node_ids, gradients, hessians, feature_indices, feature_values,
          feature_shape, max_splits=max_splits, num_buckets=num_buckets,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  max_splits = _execute.make_int(max_splits, "max_splits")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesSparseAggregateStats", node_ids=node_ids,
                                            gradients=gradients,
                                            hessians=hessians,
                                            feature_indices=feature_indices,
                                            feature_values=feature_values,
                                            feature_shape=feature_shape,
                                            max_splits=max_splits,
                                            num_buckets=num_buckets,
                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_splits", _op._get_attr_int("max_splits"), "num_buckets",
              _op._get_attr_int("num_buckets"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesSparseAggregateStats", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesSparseAggregateStatsOutput._make(_result)
  return _result

BoostedTreesSparseAggregateStats = tf_export("raw_ops.BoostedTreesSparseAggregateStats")(_ops.to_raw_op(boosted_trees_sparse_aggregate_stats))


def boosted_trees_sparse_aggregate_stats_eager_fallback(node_ids: Annotated[Any, _atypes.Int32], gradients: Annotated[Any, _atypes.Float32], hessians: Annotated[Any, _atypes.Float32], feature_indices: Annotated[Any, _atypes.Int32], feature_values: Annotated[Any, _atypes.Int32], feature_shape: Annotated[Any, _atypes.Int32], max_splits: int, num_buckets: int, name, ctx):
  max_splits = _execute.make_int(max_splits, "max_splits")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  node_ids = _ops.convert_to_tensor(node_ids, _dtypes.int32)
  gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
  hessians = _ops.convert_to_tensor(hessians, _dtypes.float32)
  feature_indices = _ops.convert_to_tensor(feature_indices, _dtypes.int32)
  feature_values = _ops.convert_to_tensor(feature_values, _dtypes.int32)
  feature_shape = _ops.convert_to_tensor(feature_shape, _dtypes.int32)
  _inputs_flat = [node_ids, gradients, hessians, feature_indices, feature_values, feature_shape]
  _attrs = ("max_splits", max_splits, "num_buckets", num_buckets)
  _result = _execute.execute(b"BoostedTreesSparseAggregateStats", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesSparseAggregateStats", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesSparseAggregateStatsOutput._make(_result)
  return _result

_BoostedTreesSparseCalculateBestFeatureSplitOutput = collections.namedtuple(
    "BoostedTreesSparseCalculateBestFeatureSplit",
    ["node_ids", "gains", "feature_dimensions", "thresholds", "left_node_contribs", "right_node_contribs", "split_with_default_directions"])


def boosted_trees_sparse_calculate_best_feature_split(node_id_range: Annotated[Any, _atypes.Int32], stats_summary_indices: Annotated[Any, _atypes.Int32], stats_summary_values: Annotated[Any, _atypes.Float32], stats_summary_shape: Annotated[Any, _atypes.Int32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], logits_dimension: int, split_type:str="inequality", name=None):
  r"""Calculates gains for each feature and returns the best possible split information for the feature.

  The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

  It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

  In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

  The output shapes are compatible in a way that the first dimension of all tensors are the same and equal to the number of possible split nodes for each feature.

  Args:
    node_id_range: A `Tensor` of type `int32`.
      A Rank 1 tensor (shape=[2]) to specify the range [first, last) of node ids to process within `stats_summary_list`. The nodes are iterated between the two nodes specified by the tensor, as like `for node_id in range(node_id_range[0], node_id_range[1])` (Note that the last index node_id_range[1] is exclusive).
    stats_summary_indices: A `Tensor` of type `int32`.
      A Rank 2 int64 tensor of dense shape [N, 4] (N specifies the number of non-zero values) for accumulated stats summary (gradient/hessian) per node per bucket for each feature. The second dimension contains node id, feature dimension, bucket id, and stats dim.
      stats dim is the sum of logits dimension and hessian dimension, hessian dimension can either be logits dimension if diagonal hessian is used, or logits dimension^2 if full hessian is used.
    stats_summary_values: A `Tensor` of type `float32`.
      A Rank 1 float tensor of dense shape [N] (N specifies the number of non-zero values), which supplies the values for each element in summary_indices.
    stats_summary_shape: A `Tensor` of type `int32`.
      A Rank 1 float tensor of dense shape [4], which specifies the dense shape of the sparse tensor, which is [num tree nodes, feature dimensions, num buckets, stats dim].
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    tree_complexity: A `Tensor` of type `float32`.
      adjustment to the gain, per leaf based.
    min_node_weight: A `Tensor` of type `float32`.
      minimum avg of hessians in a node before required for the node to be considered for splitting.
    logits_dimension: An `int` that is `>= 1`.
      The dimension of logit, i.e., number of classes.
    split_type: An optional `string` from: `"inequality"`. Defaults to `"inequality"`.
      A string indicating if this Op should perform inequality split or equality split.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (node_ids, gains, feature_dimensions, thresholds, left_node_contribs, right_node_contribs, split_with_default_directions).

    node_ids: A `Tensor` of type `int32`.
    gains: A `Tensor` of type `float32`.
    feature_dimensions: A `Tensor` of type `int32`.
    thresholds: A `Tensor` of type `int32`.
    left_node_contribs: A `Tensor` of type `float32`.
    right_node_contribs: A `Tensor` of type `float32`.
    split_with_default_directions: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesSparseCalculateBestFeatureSplit", name,
        node_id_range, stats_summary_indices, stats_summary_values,
        stats_summary_shape, l1, l2, tree_complexity, min_node_weight,
        "logits_dimension", logits_dimension, "split_type", split_type)
      _result = _BoostedTreesSparseCalculateBestFeatureSplitOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_sparse_calculate_best_feature_split_eager_fallback(
          node_id_range, stats_summary_indices, stats_summary_values,
          stats_summary_shape, l1, l2, tree_complexity, min_node_weight,
          logits_dimension=logits_dimension, split_type=split_type, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  if split_type is None:
    split_type = "inequality"
  split_type = _execute.make_str(split_type, "split_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesSparseCalculateBestFeatureSplit", node_id_range=node_id_range,
                                                       stats_summary_indices=stats_summary_indices,
                                                       stats_summary_values=stats_summary_values,
                                                       stats_summary_shape=stats_summary_shape,
                                                       l1=l1, l2=l2,
                                                       tree_complexity=tree_complexity,
                                                       min_node_weight=min_node_weight,
                                                       logits_dimension=logits_dimension,
                                                       split_type=split_type,
                                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("logits_dimension", _op._get_attr_int("logits_dimension"),
              "split_type", _op.get_attr("split_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesSparseCalculateBestFeatureSplit", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesSparseCalculateBestFeatureSplitOutput._make(_result)
  return _result

BoostedTreesSparseCalculateBestFeatureSplit = tf_export("raw_ops.BoostedTreesSparseCalculateBestFeatureSplit")(_ops.to_raw_op(boosted_trees_sparse_calculate_best_feature_split))


def boosted_trees_sparse_calculate_best_feature_split_eager_fallback(node_id_range: Annotated[Any, _atypes.Int32], stats_summary_indices: Annotated[Any, _atypes.Int32], stats_summary_values: Annotated[Any, _atypes.Float32], stats_summary_shape: Annotated[Any, _atypes.Int32], l1: Annotated[Any, _atypes.Float32], l2: Annotated[Any, _atypes.Float32], tree_complexity: Annotated[Any, _atypes.Float32], min_node_weight: Annotated[Any, _atypes.Float32], logits_dimension: int, split_type: str, name, ctx):
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  if split_type is None:
    split_type = "inequality"
  split_type = _execute.make_str(split_type, "split_type")
  node_id_range = _ops.convert_to_tensor(node_id_range, _dtypes.int32)
  stats_summary_indices = _ops.convert_to_tensor(stats_summary_indices, _dtypes.int32)
  stats_summary_values = _ops.convert_to_tensor(stats_summary_values, _dtypes.float32)
  stats_summary_shape = _ops.convert_to_tensor(stats_summary_shape, _dtypes.int32)
  l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
  l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
  tree_complexity = _ops.convert_to_tensor(tree_complexity, _dtypes.float32)
  min_node_weight = _ops.convert_to_tensor(min_node_weight, _dtypes.float32)
  _inputs_flat = [node_id_range, stats_summary_indices, stats_summary_values, stats_summary_shape, l1, l2, tree_complexity, min_node_weight]
  _attrs = ("logits_dimension", logits_dimension, "split_type", split_type)
  _result = _execute.execute(b"BoostedTreesSparseCalculateBestFeatureSplit",
                             7, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesSparseCalculateBestFeatureSplit", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesSparseCalculateBestFeatureSplitOutput._make(_result)
  return _result

_BoostedTreesTrainingPredictOutput = collections.namedtuple(
    "BoostedTreesTrainingPredict",
    ["partial_logits", "tree_ids", "node_ids"])


def boosted_trees_training_predict(tree_ensemble_handle: Annotated[Any, _atypes.Resource], cached_tree_ids: Annotated[Any, _atypes.Int32], cached_node_ids: Annotated[Any, _atypes.Int32], bucketized_features: Annotated[List[Any], _atypes.Int32], logits_dimension: int, name=None):
  r"""Runs multiple additive regression ensemble predictors on input instances and

  computes the update to cached logits. It is designed to be used during training.
  It traverses the trees starting from cached tree id and cached node id and
  calculates the updates to be pushed to the cache.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
    cached_tree_ids: A `Tensor` of type `int32`.
      Rank 1 Tensor containing cached tree ids which is the starting
      tree of prediction.
    cached_node_ids: A `Tensor` of type `int32`.
      Rank 1 Tensor containing cached node id which is the starting
      node of prediction.
    bucketized_features: A list of at least 1 `Tensor` objects with type `int32`.
      A list of rank 1 Tensors containing bucket id for each
      feature.
    logits_dimension: An `int`.
      scalar, dimension of the logits, to be used for partial logits
      shape.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (partial_logits, tree_ids, node_ids).

    partial_logits: A `Tensor` of type `float32`.
    tree_ids: A `Tensor` of type `int32`.
    node_ids: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesTrainingPredict", name, tree_ensemble_handle,
        cached_tree_ids, cached_node_ids, bucketized_features,
        "logits_dimension", logits_dimension)
      _result = _BoostedTreesTrainingPredictOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_training_predict_eager_fallback(
          tree_ensemble_handle, cached_tree_ids, cached_node_ids,
          bucketized_features, logits_dimension=logits_dimension, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(bucketized_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features' argument to "
        "'boosted_trees_training_predict' Op, not %r." % bucketized_features)
  _attr_num_bucketized_features = len(bucketized_features)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesTrainingPredict", tree_ensemble_handle=tree_ensemble_handle,
                                       cached_tree_ids=cached_tree_ids,
                                       cached_node_ids=cached_node_ids,
                                       bucketized_features=bucketized_features,
                                       logits_dimension=logits_dimension,
                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_bucketized_features",
              _op._get_attr_int("num_bucketized_features"),
              "logits_dimension", _op._get_attr_int("logits_dimension"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BoostedTreesTrainingPredict", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesTrainingPredictOutput._make(_result)
  return _result

BoostedTreesTrainingPredict = tf_export("raw_ops.BoostedTreesTrainingPredict")(_ops.to_raw_op(boosted_trees_training_predict))


def boosted_trees_training_predict_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], cached_tree_ids: Annotated[Any, _atypes.Int32], cached_node_ids: Annotated[Any, _atypes.Int32], bucketized_features: Annotated[List[Any], _atypes.Int32], logits_dimension: int, name, ctx):
  if not isinstance(bucketized_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'bucketized_features' argument to "
        "'boosted_trees_training_predict' Op, not %r." % bucketized_features)
  _attr_num_bucketized_features = len(bucketized_features)
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  cached_tree_ids = _ops.convert_to_tensor(cached_tree_ids, _dtypes.int32)
  cached_node_ids = _ops.convert_to_tensor(cached_node_ids, _dtypes.int32)
  bucketized_features = _ops.convert_n_to_tensor(bucketized_features, _dtypes.int32)
  _inputs_flat = [tree_ensemble_handle, cached_tree_ids, cached_node_ids] + list(bucketized_features)
  _attrs = ("num_bucketized_features", _attr_num_bucketized_features,
  "logits_dimension", logits_dimension)
  _result = _execute.execute(b"BoostedTreesTrainingPredict", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BoostedTreesTrainingPredict", _inputs_flat, _attrs, _result)
  _result = _BoostedTreesTrainingPredictOutput._make(_result)
  return _result


def boosted_trees_update_ensemble(tree_ensemble_handle: Annotated[Any, _atypes.Resource], feature_ids: Annotated[Any, _atypes.Int32], node_ids: Annotated[List[Any], _atypes.Int32], gains: Annotated[List[Any], _atypes.Float32], thresholds: Annotated[List[Any], _atypes.Int32], left_node_contribs: Annotated[List[Any], _atypes.Float32], right_node_contribs: Annotated[List[Any], _atypes.Float32], max_depth: Annotated[Any, _atypes.Int32], learning_rate: Annotated[Any, _atypes.Float32], pruning_mode: int, name=None):
  r"""Updates the tree ensemble by either adding a layer to the last tree being grown

  or by starting a new tree.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the ensemble variable.
    feature_ids: A `Tensor` of type `int32`.
      Rank 1 tensor with ids for each feature. This is the real id of
      the feature that will be used in the split.
    node_ids: A list of `Tensor` objects with type `int32`.
      List of rank 1 tensors representing the nodes for which this feature
      has a split.
    gains: A list with the same length as `node_ids` of `Tensor` objects with type `float32`.
      List of rank 1 tensors representing the gains for each of the feature's
      split.
    thresholds: A list with the same length as `node_ids` of `Tensor` objects with type `int32`.
      List of rank 1 tensors representing the thesholds for each of the
      feature's split.
    left_node_contribs: A list with the same length as `node_ids` of `Tensor` objects with type `float32`.
      List of rank 2 tensors with left leaf contribs for each of
      the feature's splits. Will be added to the previous node values to constitute
      the values of the left nodes.
    right_node_contribs: A list with the same length as `node_ids` of `Tensor` objects with type `float32`.
      List of rank 2 tensors with right leaf contribs for each
      of the feature's splits. Will be added to the previous node values to constitute
      the values of the right nodes.
    max_depth: A `Tensor` of type `int32`. Max depth of the tree to build.
    learning_rate: A `Tensor` of type `float32`.
      shrinkage const for each new tree.
    pruning_mode: An `int` that is `>= 0`.
      0-No pruning, 1-Pre-pruning, 2-Post-pruning.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesUpdateEnsemble", name, tree_ensemble_handle,
        feature_ids, node_ids, gains, thresholds, left_node_contribs,
        right_node_contribs, max_depth, learning_rate, "pruning_mode",
        pruning_mode)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_update_ensemble_eager_fallback(
          tree_ensemble_handle, feature_ids, node_ids, gains, thresholds,
          left_node_contribs, right_node_contribs, max_depth, learning_rate,
          pruning_mode=pruning_mode, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(node_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'node_ids' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % node_ids)
  _attr_num_features = len(node_ids)
  if not isinstance(gains, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % gains)
  if len(gains) != _attr_num_features:
    raise ValueError(
        "List argument 'gains' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(gains), _attr_num_features))
  if not isinstance(thresholds, (list, tuple)):
    raise TypeError(
        "Expected list for 'thresholds' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % thresholds)
  if len(thresholds) != _attr_num_features:
    raise ValueError(
        "List argument 'thresholds' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(thresholds), _attr_num_features))
  if not isinstance(left_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'left_node_contribs' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % left_node_contribs)
  if len(left_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'left_node_contribs' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(left_node_contribs), _attr_num_features))
  if not isinstance(right_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'right_node_contribs' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % right_node_contribs)
  if len(right_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'right_node_contribs' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(right_node_contribs), _attr_num_features))
  pruning_mode = _execute.make_int(pruning_mode, "pruning_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesUpdateEnsemble", tree_ensemble_handle=tree_ensemble_handle,
                                      feature_ids=feature_ids,
                                      node_ids=node_ids, gains=gains,
                                      thresholds=thresholds,
                                      left_node_contribs=left_node_contribs,
                                      right_node_contribs=right_node_contribs,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      pruning_mode=pruning_mode, name=name)
  return _op
BoostedTreesUpdateEnsemble = tf_export("raw_ops.BoostedTreesUpdateEnsemble")(_ops.to_raw_op(boosted_trees_update_ensemble))


def boosted_trees_update_ensemble_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], feature_ids: Annotated[Any, _atypes.Int32], node_ids: Annotated[List[Any], _atypes.Int32], gains: Annotated[List[Any], _atypes.Float32], thresholds: Annotated[List[Any], _atypes.Int32], left_node_contribs: Annotated[List[Any], _atypes.Float32], right_node_contribs: Annotated[List[Any], _atypes.Float32], max_depth: Annotated[Any, _atypes.Int32], learning_rate: Annotated[Any, _atypes.Float32], pruning_mode: int, name, ctx):
  if not isinstance(node_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'node_ids' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % node_ids)
  _attr_num_features = len(node_ids)
  if not isinstance(gains, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % gains)
  if len(gains) != _attr_num_features:
    raise ValueError(
        "List argument 'gains' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(gains), _attr_num_features))
  if not isinstance(thresholds, (list, tuple)):
    raise TypeError(
        "Expected list for 'thresholds' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % thresholds)
  if len(thresholds) != _attr_num_features:
    raise ValueError(
        "List argument 'thresholds' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(thresholds), _attr_num_features))
  if not isinstance(left_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'left_node_contribs' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % left_node_contribs)
  if len(left_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'left_node_contribs' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(left_node_contribs), _attr_num_features))
  if not isinstance(right_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'right_node_contribs' argument to "
        "'boosted_trees_update_ensemble' Op, not %r." % right_node_contribs)
  if len(right_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'right_node_contribs' to 'boosted_trees_update_ensemble' Op with length %d "
        "must match length %d of argument 'node_ids'." %
        (len(right_node_contribs), _attr_num_features))
  pruning_mode = _execute.make_int(pruning_mode, "pruning_mode")
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  feature_ids = _ops.convert_to_tensor(feature_ids, _dtypes.int32)
  node_ids = _ops.convert_n_to_tensor(node_ids, _dtypes.int32)
  gains = _ops.convert_n_to_tensor(gains, _dtypes.float32)
  thresholds = _ops.convert_n_to_tensor(thresholds, _dtypes.int32)
  left_node_contribs = _ops.convert_n_to_tensor(left_node_contribs, _dtypes.float32)
  right_node_contribs = _ops.convert_n_to_tensor(right_node_contribs, _dtypes.float32)
  max_depth = _ops.convert_to_tensor(max_depth, _dtypes.int32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  _inputs_flat = [tree_ensemble_handle, feature_ids] + list(node_ids) + list(gains) + list(thresholds) + list(left_node_contribs) + list(right_node_contribs) + [max_depth, learning_rate]
  _attrs = ("pruning_mode", pruning_mode, "num_features", _attr_num_features)
  _result = _execute.execute(b"BoostedTreesUpdateEnsemble", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def boosted_trees_update_ensemble_v2(tree_ensemble_handle: Annotated[Any, _atypes.Resource], feature_ids: Annotated[List[Any], _atypes.Int32], dimension_ids: Annotated[List[Any], _atypes.Int32], node_ids: Annotated[List[Any], _atypes.Int32], gains: Annotated[List[Any], _atypes.Float32], thresholds: Annotated[List[Any], _atypes.Int32], left_node_contribs: Annotated[List[Any], _atypes.Float32], right_node_contribs: Annotated[List[Any], _atypes.Float32], split_types: Annotated[List[Any], _atypes.String], max_depth: Annotated[Any, _atypes.Int32], learning_rate: Annotated[Any, _atypes.Float32], pruning_mode: Annotated[Any, _atypes.Int32], logits_dimension:int=1, name=None):
  r"""Updates the tree ensemble by adding a layer to the last tree being grown

  or by starting a new tree.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the ensemble variable.
    feature_ids: A list of at least 1 `Tensor` objects with type `int32`.
      Rank 1 tensor with ids for each feature. This is the real id of
      the feature that will be used in the split.
    dimension_ids: A list of `Tensor` objects with type `int32`.
      List of rank 1 tensors representing the dimension in each feature.
    node_ids: A list with the same length as `dimension_ids` of `Tensor` objects with type `int32`.
      List of rank 1 tensors representing the nodes for which this feature
      has a split.
    gains: A list with the same length as `dimension_ids` of `Tensor` objects with type `float32`.
      List of rank 1 tensors representing the gains for each of the feature's
      split.
    thresholds: A list with the same length as `dimension_ids` of `Tensor` objects with type `int32`.
      List of rank 1 tensors representing the thesholds for each of the
      feature's split.
    left_node_contribs: A list with the same length as `dimension_ids` of `Tensor` objects with type `float32`.
      List of rank 2 tensors with left leaf contribs for each of
      the feature's splits. Will be added to the previous node values to constitute
      the values of the left nodes.
    right_node_contribs: A list with the same length as `dimension_ids` of `Tensor` objects with type `float32`.
      List of rank 2 tensors with right leaf contribs for each
      of the feature's splits. Will be added to the previous node values to constitute
      the values of the right nodes.
    split_types: A list with the same length as `dimension_ids` of `Tensor` objects with type `string`.
      List of rank 1 tensors representing the split type for each feature.
    max_depth: A `Tensor` of type `int32`. Max depth of the tree to build.
    learning_rate: A `Tensor` of type `float32`.
      shrinkage const for each new tree.
    pruning_mode: A `Tensor` of type `int32`.
      0-No pruning, 1-Pre-pruning, 2-Post-pruning.
    logits_dimension: An optional `int`. Defaults to `1`.
      scalar, dimension of the logits
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BoostedTreesUpdateEnsembleV2", name, tree_ensemble_handle,
        feature_ids, dimension_ids, node_ids, gains, thresholds,
        left_node_contribs, right_node_contribs, split_types, max_depth,
        learning_rate, pruning_mode, "logits_dimension", logits_dimension)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return boosted_trees_update_ensemble_v2_eager_fallback(
          tree_ensemble_handle, feature_ids, dimension_ids, node_ids, gains,
          thresholds, left_node_contribs, right_node_contribs, split_types,
          max_depth, learning_rate, pruning_mode,
          logits_dimension=logits_dimension, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dimension_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimension_ids' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % dimension_ids)
  _attr_num_features = len(dimension_ids)
  if not isinstance(node_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'node_ids' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % node_ids)
  if len(node_ids) != _attr_num_features:
    raise ValueError(
        "List argument 'node_ids' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(node_ids), _attr_num_features))
  if not isinstance(gains, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % gains)
  if len(gains) != _attr_num_features:
    raise ValueError(
        "List argument 'gains' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(gains), _attr_num_features))
  if not isinstance(thresholds, (list, tuple)):
    raise TypeError(
        "Expected list for 'thresholds' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % thresholds)
  if len(thresholds) != _attr_num_features:
    raise ValueError(
        "List argument 'thresholds' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(thresholds), _attr_num_features))
  if not isinstance(left_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'left_node_contribs' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % left_node_contribs)
  if len(left_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'left_node_contribs' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(left_node_contribs), _attr_num_features))
  if not isinstance(right_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'right_node_contribs' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % right_node_contribs)
  if len(right_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'right_node_contribs' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(right_node_contribs), _attr_num_features))
  if not isinstance(split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'split_types' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % split_types)
  if len(split_types) != _attr_num_features:
    raise ValueError(
        "List argument 'split_types' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(split_types), _attr_num_features))
  if not isinstance(feature_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_ids' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % feature_ids)
  _attr_num_groups = len(feature_ids)
  if logits_dimension is None:
    logits_dimension = 1
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BoostedTreesUpdateEnsembleV2", tree_ensemble_handle=tree_ensemble_handle,
                                        feature_ids=feature_ids,
                                        dimension_ids=dimension_ids,
                                        node_ids=node_ids, gains=gains,
                                        thresholds=thresholds,
                                        left_node_contribs=left_node_contribs,
                                        right_node_contribs=right_node_contribs,
                                        split_types=split_types,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        pruning_mode=pruning_mode,
                                        logits_dimension=logits_dimension,
                                        name=name)
  return _op
BoostedTreesUpdateEnsembleV2 = tf_export("raw_ops.BoostedTreesUpdateEnsembleV2")(_ops.to_raw_op(boosted_trees_update_ensemble_v2))


def boosted_trees_update_ensemble_v2_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], feature_ids: Annotated[List[Any], _atypes.Int32], dimension_ids: Annotated[List[Any], _atypes.Int32], node_ids: Annotated[List[Any], _atypes.Int32], gains: Annotated[List[Any], _atypes.Float32], thresholds: Annotated[List[Any], _atypes.Int32], left_node_contribs: Annotated[List[Any], _atypes.Float32], right_node_contribs: Annotated[List[Any], _atypes.Float32], split_types: Annotated[List[Any], _atypes.String], max_depth: Annotated[Any, _atypes.Int32], learning_rate: Annotated[Any, _atypes.Float32], pruning_mode: Annotated[Any, _atypes.Int32], logits_dimension: int, name, ctx):
  if not isinstance(dimension_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimension_ids' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % dimension_ids)
  _attr_num_features = len(dimension_ids)
  if not isinstance(node_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'node_ids' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % node_ids)
  if len(node_ids) != _attr_num_features:
    raise ValueError(
        "List argument 'node_ids' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(node_ids), _attr_num_features))
  if not isinstance(gains, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % gains)
  if len(gains) != _attr_num_features:
    raise ValueError(
        "List argument 'gains' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(gains), _attr_num_features))
  if not isinstance(thresholds, (list, tuple)):
    raise TypeError(
        "Expected list for 'thresholds' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % thresholds)
  if len(thresholds) != _attr_num_features:
    raise ValueError(
        "List argument 'thresholds' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(thresholds), _attr_num_features))
  if not isinstance(left_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'left_node_contribs' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % left_node_contribs)
  if len(left_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'left_node_contribs' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(left_node_contribs), _attr_num_features))
  if not isinstance(right_node_contribs, (list, tuple)):
    raise TypeError(
        "Expected list for 'right_node_contribs' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % right_node_contribs)
  if len(right_node_contribs) != _attr_num_features:
    raise ValueError(
        "List argument 'right_node_contribs' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(right_node_contribs), _attr_num_features))
  if not isinstance(split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'split_types' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % split_types)
  if len(split_types) != _attr_num_features:
    raise ValueError(
        "List argument 'split_types' to 'boosted_trees_update_ensemble_v2' Op with length %d "
        "must match length %d of argument 'dimension_ids'." %
        (len(split_types), _attr_num_features))
  if not isinstance(feature_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_ids' argument to "
        "'boosted_trees_update_ensemble_v2' Op, not %r." % feature_ids)
  _attr_num_groups = len(feature_ids)
  if logits_dimension is None:
    logits_dimension = 1
  logits_dimension = _execute.make_int(logits_dimension, "logits_dimension")
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  feature_ids = _ops.convert_n_to_tensor(feature_ids, _dtypes.int32)
  dimension_ids = _ops.convert_n_to_tensor(dimension_ids, _dtypes.int32)
  node_ids = _ops.convert_n_to_tensor(node_ids, _dtypes.int32)
  gains = _ops.convert_n_to_tensor(gains, _dtypes.float32)
  thresholds = _ops.convert_n_to_tensor(thresholds, _dtypes.int32)
  left_node_contribs = _ops.convert_n_to_tensor(left_node_contribs, _dtypes.float32)
  right_node_contribs = _ops.convert_n_to_tensor(right_node_contribs, _dtypes.float32)
  split_types = _ops.convert_n_to_tensor(split_types, _dtypes.string)
  max_depth = _ops.convert_to_tensor(max_depth, _dtypes.int32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  pruning_mode = _ops.convert_to_tensor(pruning_mode, _dtypes.int32)
  _inputs_flat = [tree_ensemble_handle] + list(feature_ids) + list(dimension_ids) + list(node_ids) + list(gains) + list(thresholds) + list(left_node_contribs) + list(right_node_contribs) + list(split_types) + [max_depth, learning_rate, pruning_mode]
  _attrs = ("num_features", _attr_num_features, "logits_dimension",
  logits_dimension, "num_groups", _attr_num_groups)
  _result = _execute.execute(b"BoostedTreesUpdateEnsembleV2", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def is_boosted_trees_ensemble_initialized(tree_ensemble_handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Checks whether a tree ensemble has been initialized.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble resource.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsBoostedTreesEnsembleInitialized", name, tree_ensemble_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return is_boosted_trees_ensemble_initialized_eager_fallback(
          tree_ensemble_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsBoostedTreesEnsembleInitialized", tree_ensemble_handle=tree_ensemble_handle,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsBoostedTreesEnsembleInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsBoostedTreesEnsembleInitialized = tf_export("raw_ops.IsBoostedTreesEnsembleInitialized")(_ops.to_raw_op(is_boosted_trees_ensemble_initialized))


def is_boosted_trees_ensemble_initialized_eager_fallback(tree_ensemble_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Bool]:
  tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
  _inputs_flat = [tree_ensemble_handle]
  _attrs = None
  _result = _execute.execute(b"IsBoostedTreesEnsembleInitialized", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsBoostedTreesEnsembleInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def is_boosted_trees_quantile_stream_resource_initialized(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Checks whether a quantile stream has been initialized.

  An Op that checks if quantile stream resource is initialized.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource; The reference to quantile stream resource handle.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsBoostedTreesQuantileStreamResourceInitialized", name,
        quantile_stream_resource_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return is_boosted_trees_quantile_stream_resource_initialized_eager_fallback(
          quantile_stream_resource_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsBoostedTreesQuantileStreamResourceInitialized", quantile_stream_resource_handle=quantile_stream_resource_handle,
                                                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsBoostedTreesQuantileStreamResourceInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsBoostedTreesQuantileStreamResourceInitialized = tf_export("raw_ops.IsBoostedTreesQuantileStreamResourceInitialized")(_ops.to_raw_op(is_boosted_trees_quantile_stream_resource_initialized))


def is_boosted_trees_quantile_stream_resource_initialized_eager_fallback(quantile_stream_resource_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Bool]:
  quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
  _inputs_flat = [quantile_stream_resource_handle]
  _attrs = None
  _result = _execute.execute(b"IsBoostedTreesQuantileStreamResourceInitialized",
                             1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsBoostedTreesQuantileStreamResourceInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

