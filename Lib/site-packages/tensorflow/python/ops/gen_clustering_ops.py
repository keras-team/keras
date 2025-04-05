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

def kmc2_chain_initialization(distances: Annotated[Any, _atypes.Float32], seed: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the index of a data point that should be added to the seed set.

  Entries in distances are assumed to be squared distances of candidate points to
  the already sampled centers in the seed set. The op constructs one Markov chain
  of the k-MC^2 algorithm and returns the index of one candidate point to be added
  as an additional cluster center.

  Args:
    distances: A `Tensor` of type `float32`.
      Vector with squared distances to the closest previously sampled cluster center
      for each candidate point.
    seed: A `Tensor` of type `int64`.
      Scalar. Seed for initializing the random number generator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "KMC2ChainInitialization", name, distances, seed)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return kmc2_chain_initialization_eager_fallback(
          distances, seed, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "KMC2ChainInitialization", distances=distances, seed=seed, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "KMC2ChainInitialization", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

KMC2ChainInitialization = tf_export("raw_ops.KMC2ChainInitialization")(_ops.to_raw_op(kmc2_chain_initialization))


def kmc2_chain_initialization_eager_fallback(distances: Annotated[Any, _atypes.Float32], seed: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, _atypes.Int64]:
  distances = _ops.convert_to_tensor(distances, _dtypes.float32)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  _inputs_flat = [distances, seed]
  _attrs = None
  _result = _execute.execute(b"KMC2ChainInitialization", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "KMC2ChainInitialization", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def kmeans_plus_plus_initialization(points: Annotated[Any, _atypes.Float32], num_to_sample: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], num_retries_per_sample: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Selects num_to_sample rows of input using the KMeans++ criterion.

  Rows of points are assumed to be input points. One row is selected at random.
  Subsequent rows are sampled with probability proportional to the squared L2
  distance from the nearest row selected thus far till num_to_sample rows have
  been sampled.

  Args:
    points: A `Tensor` of type `float32`.
      Matrix of shape (n, d). Rows are assumed to be input points.
    num_to_sample: A `Tensor` of type `int64`.
      Scalar. The number of rows to sample. This value must not be larger than n.
    seed: A `Tensor` of type `int64`.
      Scalar. Seed for initializing the random number generator.
    num_retries_per_sample: A `Tensor` of type `int64`.
      Scalar. For each row that is sampled, this parameter
      specifies the number of additional points to draw from the current
      distribution before selecting the best. If a negative value is specified, a
      heuristic is used to sample O(log(num_to_sample)) additional points.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "KmeansPlusPlusInitialization", name, points, num_to_sample,
        seed, num_retries_per_sample)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return kmeans_plus_plus_initialization_eager_fallback(
          points, num_to_sample, seed, num_retries_per_sample, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "KmeansPlusPlusInitialization", points=points,
                                        num_to_sample=num_to_sample,
                                        seed=seed,
                                        num_retries_per_sample=num_retries_per_sample,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "KmeansPlusPlusInitialization", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

KmeansPlusPlusInitialization = tf_export("raw_ops.KmeansPlusPlusInitialization")(_ops.to_raw_op(kmeans_plus_plus_initialization))


def kmeans_plus_plus_initialization_eager_fallback(points: Annotated[Any, _atypes.Float32], num_to_sample: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], num_retries_per_sample: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, _atypes.Float32]:
  points = _ops.convert_to_tensor(points, _dtypes.float32)
  num_to_sample = _ops.convert_to_tensor(num_to_sample, _dtypes.int64)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  num_retries_per_sample = _ops.convert_to_tensor(num_retries_per_sample, _dtypes.int64)
  _inputs_flat = [points, num_to_sample, seed, num_retries_per_sample]
  _attrs = None
  _result = _execute.execute(b"KmeansPlusPlusInitialization", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "KmeansPlusPlusInitialization", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_NearestNeighborsOutput = collections.namedtuple(
    "NearestNeighbors",
    ["nearest_center_indices", "nearest_center_distances"])


def nearest_neighbors(points: Annotated[Any, _atypes.Float32], centers: Annotated[Any, _atypes.Float32], k: Annotated[Any, _atypes.Int64], name=None):
  r"""Selects the k nearest centers for each point.

  Rows of points are assumed to be input points. Rows of centers are assumed to be
  the list of candidate centers. For each point, the k centers that have least L2
  distance to it are computed.

  Args:
    points: A `Tensor` of type `float32`.
      Matrix of shape (n, d). Rows are assumed to be input points.
    centers: A `Tensor` of type `float32`.
      Matrix of shape (m, d). Rows are assumed to be centers.
    k: A `Tensor` of type `int64`.
      Number of nearest centers to return for each point. If k is larger than m, then
      only m centers are returned.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (nearest_center_indices, nearest_center_distances).

    nearest_center_indices: A `Tensor` of type `int64`.
    nearest_center_distances: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NearestNeighbors", name, points, centers, k)
      _result = _NearestNeighborsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return nearest_neighbors_eager_fallback(
          points, centers, k, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NearestNeighbors", points=points, centers=centers, k=k, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NearestNeighbors", _inputs_flat, _attrs, _result)
  _result = _NearestNeighborsOutput._make(_result)
  return _result

NearestNeighbors = tf_export("raw_ops.NearestNeighbors")(_ops.to_raw_op(nearest_neighbors))


def nearest_neighbors_eager_fallback(points: Annotated[Any, _atypes.Float32], centers: Annotated[Any, _atypes.Float32], k: Annotated[Any, _atypes.Int64], name, ctx):
  points = _ops.convert_to_tensor(points, _dtypes.float32)
  centers = _ops.convert_to_tensor(centers, _dtypes.float32)
  k = _ops.convert_to_tensor(k, _dtypes.int64)
  _inputs_flat = [points, centers, k]
  _attrs = None
  _result = _execute.execute(b"NearestNeighbors", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NearestNeighbors", _inputs_flat, _attrs, _result)
  _result = _NearestNeighborsOutput._make(_result)
  return _result

