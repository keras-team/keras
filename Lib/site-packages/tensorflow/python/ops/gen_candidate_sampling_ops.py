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
_AllCandidateSamplerOutput = collections.namedtuple(
    "AllCandidateSampler",
    ["sampled_candidates", "true_expected_count", "sampled_expected_count"])


def all_candidate_sampler(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, seed:int=0, seed2:int=0, name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`. Number of candidates to produce.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AllCandidateSampler", name, true_classes, "num_true", num_true,
        "num_sampled", num_sampled, "unique", unique, "seed", seed, "seed2",
        seed2)
      _result = _AllCandidateSamplerOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return all_candidate_sampler_eager_fallback(
          true_classes, num_true=num_true, num_sampled=num_sampled,
          unique=unique, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AllCandidateSampler", true_classes=true_classes, num_true=num_true,
                               num_sampled=num_sampled, unique=unique,
                               seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_true", _op._get_attr_int("num_true"), "num_sampled",
              _op._get_attr_int("num_sampled"), "unique",
              _op._get_attr_bool("unique"), "seed", _op._get_attr_int("seed"),
              "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AllCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _AllCandidateSamplerOutput._make(_result)
  return _result

AllCandidateSampler = tf_export("raw_ops.AllCandidateSampler")(_ops.to_raw_op(all_candidate_sampler))


def all_candidate_sampler_eager_fallback(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, seed: int, seed2: int, name, ctx):
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
  _inputs_flat = [true_classes]
  _attrs = ("num_true", num_true, "num_sampled", num_sampled, "unique",
  unique, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"AllCandidateSampler", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AllCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _AllCandidateSamplerOutput._make(_result)
  return _result

_ComputeAccidentalHitsOutput = collections.namedtuple(
    "ComputeAccidentalHits",
    ["indices", "ids", "weights"])


def compute_accidental_hits(true_classes: Annotated[Any, _atypes.Int64], sampled_candidates: Annotated[Any, _atypes.Int64], num_true: int, seed:int=0, seed2:int=0, name=None):
  r"""Computes the ids of the positions in sampled_candidates that match true_labels.

  When doing log-odds NCE, the result of this op should be passed through a
  SparseToDense op, then added to the logits of the sampled candidates. This has
  the effect of 'removing' the sampled labels that match the true labels by
  making the classifier sure that they are sampled labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      The true_classes output of UnpackSparseLabels.
    sampled_candidates: A `Tensor` of type `int64`.
      The sampled_candidates output of CandidateSampler.
    num_true: An `int`. Number of true labels per context.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, ids, weights).

    indices: A `Tensor` of type `int32`.
    ids: A `Tensor` of type `int64`.
    weights: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ComputeAccidentalHits", name, true_classes, sampled_candidates,
        "num_true", num_true, "seed", seed, "seed2", seed2)
      _result = _ComputeAccidentalHitsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return compute_accidental_hits_eager_fallback(
          true_classes, sampled_candidates, num_true=num_true, seed=seed,
          seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_true = _execute.make_int(num_true, "num_true")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ComputeAccidentalHits", true_classes=true_classes,
                                 sampled_candidates=sampled_candidates,
                                 num_true=num_true, seed=seed, seed2=seed2,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_true", _op._get_attr_int("num_true"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ComputeAccidentalHits", _inputs_flat, _attrs, _result)
  _result = _ComputeAccidentalHitsOutput._make(_result)
  return _result

ComputeAccidentalHits = tf_export("raw_ops.ComputeAccidentalHits")(_ops.to_raw_op(compute_accidental_hits))


def compute_accidental_hits_eager_fallback(true_classes: Annotated[Any, _atypes.Int64], sampled_candidates: Annotated[Any, _atypes.Int64], num_true: int, seed: int, seed2: int, name, ctx):
  num_true = _execute.make_int(num_true, "num_true")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
  sampled_candidates = _ops.convert_to_tensor(sampled_candidates, _dtypes.int64)
  _inputs_flat = [true_classes, sampled_candidates]
  _attrs = ("num_true", num_true, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"ComputeAccidentalHits", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ComputeAccidentalHits", _inputs_flat, _attrs, _result)
  _result = _ComputeAccidentalHitsOutput._make(_result)
  return _result

_FixedUnigramCandidateSamplerOutput = collections.namedtuple(
    "FixedUnigramCandidateSampler",
    ["sampled_candidates", "true_expected_count", "sampled_expected_count"])


def fixed_unigram_candidate_sampler(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, vocab_file:str="", distortion:float=1, num_reserved_ids:int=0, num_shards:int=1, shard:int=0, unigrams=[], seed:int=0, seed2:int=0, name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  A unigram sampler could use a fixed unigram distribution read from a
  file or passed in as an in-memory array instead of building up the distribution
  from data on the fly. There is also an option to skew the distribution by
  applying a distortion power to the weights.

  The vocabulary file should be in CSV-like format, with the last field
  being the weight associated with the word.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    vocab_file: An optional `string`. Defaults to `""`.
      Each valid line in this file (which should have a CSV-like format)
      corresponds to a valid word ID. IDs are in sequential order, starting from
      num_reserved_ids. The last entry in each line is expected to be a value
      corresponding to the count or relative probability. Exactly one of vocab_file
      and unigrams needs to be passed to this op.
    distortion: An optional `float`. Defaults to `1`.
      The distortion is used to skew the unigram probability distribution.
      Each weight is first raised to the distortion's power before adding to the
      internal unigram distribution. As a result, distortion = 1.0 gives regular
      unigram sampling (as defined by the vocab file), and distortion = 0.0 gives
      a uniform distribution.
    num_reserved_ids: An optional `int`. Defaults to `0`.
      Optionally some reserved IDs can be added in the range [0,
      ..., num_reserved_ids) by the users. One use case is that a special unknown
      word token is used as ID 0. These IDs will have a sampling probability of 0.
    num_shards: An optional `int` that is `>= 1`. Defaults to `1`.
      A sampler can be used to sample from a subset of the original range
      in order to speed up the whole computation through parallelism. This parameter
      (together with 'shard') indicates the number of partitions that are being
      used in the overall computation.
    shard: An optional `int` that is `>= 0`. Defaults to `0`.
      A sampler can be used to sample from a subset of the original range
      in order to speed up the whole computation through parallelism. This parameter
      (together with 'num_shards') indicates the particular partition number of a
      sampler op, when partitioning is being used.
    unigrams: An optional list of `floats`. Defaults to `[]`.
      A list of unigram counts or probabilities, one per ID in sequential
      order. Exactly one of vocab_file and unigrams should be passed to this op.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FixedUnigramCandidateSampler", name, true_classes, "num_true",
        num_true, "num_sampled", num_sampled, "unique", unique, "range_max",
        range_max, "vocab_file", vocab_file, "distortion", distortion,
        "num_reserved_ids", num_reserved_ids, "num_shards", num_shards,
        "shard", shard, "unigrams", unigrams, "seed", seed, "seed2", seed2)
      _result = _FixedUnigramCandidateSamplerOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fixed_unigram_candidate_sampler_eager_fallback(
          true_classes, num_true=num_true, num_sampled=num_sampled,
          unique=unique, range_max=range_max, vocab_file=vocab_file,
          distortion=distortion, num_reserved_ids=num_reserved_ids,
          num_shards=num_shards, shard=shard, unigrams=unigrams, seed=seed,
          seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if vocab_file is None:
    vocab_file = ""
  vocab_file = _execute.make_str(vocab_file, "vocab_file")
  if distortion is None:
    distortion = 1
  distortion = _execute.make_float(distortion, "distortion")
  if num_reserved_ids is None:
    num_reserved_ids = 0
  num_reserved_ids = _execute.make_int(num_reserved_ids, "num_reserved_ids")
  if num_shards is None:
    num_shards = 1
  num_shards = _execute.make_int(num_shards, "num_shards")
  if shard is None:
    shard = 0
  shard = _execute.make_int(shard, "shard")
  if unigrams is None:
    unigrams = []
  if not isinstance(unigrams, (list, tuple)):
    raise TypeError(
        "Expected list for 'unigrams' argument to "
        "'fixed_unigram_candidate_sampler' Op, not %r." % unigrams)
  unigrams = [_execute.make_float(_f, "unigrams") for _f in unigrams]
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FixedUnigramCandidateSampler", true_classes=true_classes,
                                        num_true=num_true,
                                        num_sampled=num_sampled,
                                        unique=unique, range_max=range_max,
                                        vocab_file=vocab_file,
                                        distortion=distortion,
                                        num_reserved_ids=num_reserved_ids,
                                        num_shards=num_shards, shard=shard,
                                        unigrams=unigrams, seed=seed,
                                        seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_true", _op._get_attr_int("num_true"), "num_sampled",
              _op._get_attr_int("num_sampled"), "unique",
              _op._get_attr_bool("unique"), "range_max",
              _op._get_attr_int("range_max"), "vocab_file",
              _op.get_attr("vocab_file"), "distortion",
              _op.get_attr("distortion"), "num_reserved_ids",
              _op._get_attr_int("num_reserved_ids"), "num_shards",
              _op._get_attr_int("num_shards"), "shard",
              _op._get_attr_int("shard"), "unigrams",
              _op.get_attr("unigrams"), "seed", _op._get_attr_int("seed"),
              "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FixedUnigramCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _FixedUnigramCandidateSamplerOutput._make(_result)
  return _result

FixedUnigramCandidateSampler = tf_export("raw_ops.FixedUnigramCandidateSampler")(_ops.to_raw_op(fixed_unigram_candidate_sampler))


def fixed_unigram_candidate_sampler_eager_fallback(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, vocab_file: str, distortion: float, num_reserved_ids: int, num_shards: int, shard: int, unigrams, seed: int, seed2: int, name, ctx):
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if vocab_file is None:
    vocab_file = ""
  vocab_file = _execute.make_str(vocab_file, "vocab_file")
  if distortion is None:
    distortion = 1
  distortion = _execute.make_float(distortion, "distortion")
  if num_reserved_ids is None:
    num_reserved_ids = 0
  num_reserved_ids = _execute.make_int(num_reserved_ids, "num_reserved_ids")
  if num_shards is None:
    num_shards = 1
  num_shards = _execute.make_int(num_shards, "num_shards")
  if shard is None:
    shard = 0
  shard = _execute.make_int(shard, "shard")
  if unigrams is None:
    unigrams = []
  if not isinstance(unigrams, (list, tuple)):
    raise TypeError(
        "Expected list for 'unigrams' argument to "
        "'fixed_unigram_candidate_sampler' Op, not %r." % unigrams)
  unigrams = [_execute.make_float(_f, "unigrams") for _f in unigrams]
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
  _inputs_flat = [true_classes]
  _attrs = ("num_true", num_true, "num_sampled", num_sampled, "unique",
  unique, "range_max", range_max, "vocab_file", vocab_file, "distortion",
  distortion, "num_reserved_ids", num_reserved_ids, "num_shards", num_shards,
  "shard", shard, "unigrams", unigrams, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"FixedUnigramCandidateSampler", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FixedUnigramCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _FixedUnigramCandidateSamplerOutput._make(_result)
  return _result

_LearnedUnigramCandidateSamplerOutput = collections.namedtuple(
    "LearnedUnigramCandidateSampler",
    ["sampled_candidates", "true_expected_count", "sampled_expected_count"])


def learned_unigram_candidate_sampler(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed:int=0, seed2:int=0, name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LearnedUnigramCandidateSampler", name, true_classes,
        "num_true", num_true, "num_sampled", num_sampled, "unique", unique,
        "range_max", range_max, "seed", seed, "seed2", seed2)
      _result = _LearnedUnigramCandidateSamplerOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return learned_unigram_candidate_sampler_eager_fallback(
          true_classes, num_true=num_true, num_sampled=num_sampled,
          unique=unique, range_max=range_max, seed=seed, seed2=seed2,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LearnedUnigramCandidateSampler", true_classes=true_classes,
                                          num_true=num_true,
                                          num_sampled=num_sampled,
                                          unique=unique, range_max=range_max,
                                          seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_true", _op._get_attr_int("num_true"), "num_sampled",
              _op._get_attr_int("num_sampled"), "unique",
              _op._get_attr_bool("unique"), "range_max",
              _op._get_attr_int("range_max"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LearnedUnigramCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _LearnedUnigramCandidateSamplerOutput._make(_result)
  return _result

LearnedUnigramCandidateSampler = tf_export("raw_ops.LearnedUnigramCandidateSampler")(_ops.to_raw_op(learned_unigram_candidate_sampler))


def learned_unigram_candidate_sampler_eager_fallback(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed: int, seed2: int, name, ctx):
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
  _inputs_flat = [true_classes]
  _attrs = ("num_true", num_true, "num_sampled", num_sampled, "unique",
  unique, "range_max", range_max, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"LearnedUnigramCandidateSampler", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LearnedUnigramCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _LearnedUnigramCandidateSamplerOutput._make(_result)
  return _result

_LogUniformCandidateSamplerOutput = collections.namedtuple(
    "LogUniformCandidateSampler",
    ["sampled_candidates", "true_expected_count", "sampled_expected_count"])


def log_uniform_candidate_sampler(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed:int=0, seed2:int=0, name=None):
  r"""Generates labels for candidate sampling with a log-uniform distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LogUniformCandidateSampler", name, true_classes, "num_true",
        num_true, "num_sampled", num_sampled, "unique", unique, "range_max",
        range_max, "seed", seed, "seed2", seed2)
      _result = _LogUniformCandidateSamplerOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return log_uniform_candidate_sampler_eager_fallback(
          true_classes, num_true=num_true, num_sampled=num_sampled,
          unique=unique, range_max=range_max, seed=seed, seed2=seed2,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LogUniformCandidateSampler", true_classes=true_classes,
                                      num_true=num_true,
                                      num_sampled=num_sampled, unique=unique,
                                      range_max=range_max, seed=seed,
                                      seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_true", _op._get_attr_int("num_true"), "num_sampled",
              _op._get_attr_int("num_sampled"), "unique",
              _op._get_attr_bool("unique"), "range_max",
              _op._get_attr_int("range_max"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LogUniformCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _LogUniformCandidateSamplerOutput._make(_result)
  return _result

LogUniformCandidateSampler = tf_export("raw_ops.LogUniformCandidateSampler")(_ops.to_raw_op(log_uniform_candidate_sampler))


def log_uniform_candidate_sampler_eager_fallback(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed: int, seed2: int, name, ctx):
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
  _inputs_flat = [true_classes]
  _attrs = ("num_true", num_true, "num_sampled", num_sampled, "unique",
  unique, "range_max", range_max, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"LogUniformCandidateSampler", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LogUniformCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _LogUniformCandidateSamplerOutput._make(_result)
  return _result

_ThreadUnsafeUnigramCandidateSamplerOutput = collections.namedtuple(
    "ThreadUnsafeUnigramCandidateSampler",
    ["sampled_candidates", "true_expected_count", "sampled_expected_count"])


def thread_unsafe_unigram_candidate_sampler(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed:int=0, seed2:int=0, name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ThreadUnsafeUnigramCandidateSampler", name, true_classes,
        "num_true", num_true, "num_sampled", num_sampled, "unique", unique,
        "range_max", range_max, "seed", seed, "seed2", seed2)
      _result = _ThreadUnsafeUnigramCandidateSamplerOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return thread_unsafe_unigram_candidate_sampler_eager_fallback(
          true_classes, num_true=num_true, num_sampled=num_sampled,
          unique=unique, range_max=range_max, seed=seed, seed2=seed2,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ThreadUnsafeUnigramCandidateSampler", true_classes=true_classes,
                                               num_true=num_true,
                                               num_sampled=num_sampled,
                                               unique=unique,
                                               range_max=range_max, seed=seed,
                                               seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_true", _op._get_attr_int("num_true"), "num_sampled",
              _op._get_attr_int("num_sampled"), "unique",
              _op._get_attr_bool("unique"), "range_max",
              _op._get_attr_int("range_max"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ThreadUnsafeUnigramCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _ThreadUnsafeUnigramCandidateSamplerOutput._make(_result)
  return _result

ThreadUnsafeUnigramCandidateSampler = tf_export("raw_ops.ThreadUnsafeUnigramCandidateSampler")(_ops.to_raw_op(thread_unsafe_unigram_candidate_sampler))


def thread_unsafe_unigram_candidate_sampler_eager_fallback(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed: int, seed2: int, name, ctx):
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
  _inputs_flat = [true_classes]
  _attrs = ("num_true", num_true, "num_sampled", num_sampled, "unique",
  unique, "range_max", range_max, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"ThreadUnsafeUnigramCandidateSampler", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ThreadUnsafeUnigramCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _ThreadUnsafeUnigramCandidateSamplerOutput._make(_result)
  return _result

_UniformCandidateSamplerOutput = collections.namedtuple(
    "UniformCandidateSampler",
    ["sampled_candidates", "true_expected_count", "sampled_expected_count"])


def uniform_candidate_sampler(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed:int=0, seed2:int=0, name=None):
  r"""Generates labels for candidate sampling with a uniform distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformCandidateSampler", name, true_classes, "num_true",
        num_true, "num_sampled", num_sampled, "unique", unique, "range_max",
        range_max, "seed", seed, "seed2", seed2)
      _result = _UniformCandidateSamplerOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_candidate_sampler_eager_fallback(
          true_classes, num_true=num_true, num_sampled=num_sampled,
          unique=unique, range_max=range_max, seed=seed, seed2=seed2,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformCandidateSampler", true_classes=true_classes,
                                   num_true=num_true, num_sampled=num_sampled,
                                   unique=unique, range_max=range_max,
                                   seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_true", _op._get_attr_int("num_true"), "num_sampled",
              _op._get_attr_int("num_sampled"), "unique",
              _op._get_attr_bool("unique"), "range_max",
              _op._get_attr_int("range_max"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _UniformCandidateSamplerOutput._make(_result)
  return _result

UniformCandidateSampler = tf_export("raw_ops.UniformCandidateSampler")(_ops.to_raw_op(uniform_candidate_sampler))


def uniform_candidate_sampler_eager_fallback(true_classes: Annotated[Any, _atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed: int, seed2: int, name, ctx):
  num_true = _execute.make_int(num_true, "num_true")
  num_sampled = _execute.make_int(num_sampled, "num_sampled")
  unique = _execute.make_bool(unique, "unique")
  range_max = _execute.make_int(range_max, "range_max")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
  _inputs_flat = [true_classes]
  _attrs = ("num_true", num_true, "num_sampled", num_sampled, "unique",
  unique, "range_max", range_max, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"UniformCandidateSampler", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformCandidateSampler", _inputs_flat, _attrs, _result)
  _result = _UniformCandidateSamplerOutput._make(_result)
  return _result

