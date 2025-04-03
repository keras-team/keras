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
_GenerateVocabRemappingOutput = collections.namedtuple(
    "GenerateVocabRemapping",
    ["remapping", "num_present"])


def generate_vocab_remapping(new_vocab_file: Annotated[Any, _atypes.String], old_vocab_file: Annotated[Any, _atypes.String], new_vocab_offset: int, num_new_vocab: int, old_vocab_size:int=-1, name=None):
  r"""Given a path to new and old vocabulary files, returns a remapping Tensor of

  length `num_new_vocab`, where `remapping[i]` contains the row number in the old
  vocabulary that corresponds to row `i` in the new vocabulary (starting at line
  `new_vocab_offset` and up to `num_new_vocab` entities), or `-1` if entry `i`
  in the new vocabulary is not in the old vocabulary.  The old vocabulary is
  constrained to the first `old_vocab_size` entries if `old_vocab_size` is not the
  default value of -1.

  `num_vocab_offset` enables
  use in the partitioned variable case, and should generally be set through
  examining partitioning info.  The format of the files should be a text file,
  with each line containing a single entity within the vocabulary.

  For example, with `new_vocab_file` a text file containing each of the following
  elements on a single line: `[f0, f1, f2, f3]`, old_vocab_file = [f1, f0, f3],
  `num_new_vocab = 3, new_vocab_offset = 1`, the returned remapping would be
  `[0, -1, 2]`.

  The op also returns a count of how many entries in the new vocabulary
  were present in the old vocabulary, which is used to calculate the number of
  values to initialize in a weight matrix remapping

  This functionality can be used to remap both row vocabularies (typically,
  features) and column vocabularies (typically, classes) from TensorFlow
  checkpoints.  Note that the partitioning logic relies on contiguous vocabularies
  corresponding to div-partitioned variables.  Moreover, the underlying remapping
  uses an IndexTable (as opposed to an inexact CuckooTable), so client code should
  use the corresponding index_table_from_file() as the FeatureColumn framework
  does (as opposed to tf.feature_to_id(), which uses a CuckooTable).

  Args:
    new_vocab_file: A `Tensor` of type `string`. Path to the new vocab file.
    old_vocab_file: A `Tensor` of type `string`. Path to the old vocab file.
    new_vocab_offset: An `int` that is `>= 0`.
      How many entries into the new vocab file to start reading.
    num_new_vocab: An `int` that is `>= 0`.
      Number of entries in the new vocab file to remap.
    old_vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of entries in the old vocab file to consider.  If -1,
      use the entire old vocabulary.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (remapping, num_present).

    remapping: A `Tensor` of type `int64`.
    num_present: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GenerateVocabRemapping", name, new_vocab_file, old_vocab_file,
        "new_vocab_offset", new_vocab_offset, "num_new_vocab", num_new_vocab,
        "old_vocab_size", old_vocab_size)
      _result = _GenerateVocabRemappingOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return generate_vocab_remapping_eager_fallback(
          new_vocab_file, old_vocab_file, new_vocab_offset=new_vocab_offset,
          num_new_vocab=num_new_vocab, old_vocab_size=old_vocab_size,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  new_vocab_offset = _execute.make_int(new_vocab_offset, "new_vocab_offset")
  num_new_vocab = _execute.make_int(num_new_vocab, "num_new_vocab")
  if old_vocab_size is None:
    old_vocab_size = -1
  old_vocab_size = _execute.make_int(old_vocab_size, "old_vocab_size")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GenerateVocabRemapping", new_vocab_file=new_vocab_file,
                                  old_vocab_file=old_vocab_file,
                                  new_vocab_offset=new_vocab_offset,
                                  num_new_vocab=num_new_vocab,
                                  old_vocab_size=old_vocab_size, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("new_vocab_offset", _op._get_attr_int("new_vocab_offset"),
              "num_new_vocab", _op._get_attr_int("num_new_vocab"),
              "old_vocab_size", _op._get_attr_int("old_vocab_size"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GenerateVocabRemapping", _inputs_flat, _attrs, _result)
  _result = _GenerateVocabRemappingOutput._make(_result)
  return _result

GenerateVocabRemapping = tf_export("raw_ops.GenerateVocabRemapping")(_ops.to_raw_op(generate_vocab_remapping))


def generate_vocab_remapping_eager_fallback(new_vocab_file: Annotated[Any, _atypes.String], old_vocab_file: Annotated[Any, _atypes.String], new_vocab_offset: int, num_new_vocab: int, old_vocab_size: int, name, ctx):
  new_vocab_offset = _execute.make_int(new_vocab_offset, "new_vocab_offset")
  num_new_vocab = _execute.make_int(num_new_vocab, "num_new_vocab")
  if old_vocab_size is None:
    old_vocab_size = -1
  old_vocab_size = _execute.make_int(old_vocab_size, "old_vocab_size")
  new_vocab_file = _ops.convert_to_tensor(new_vocab_file, _dtypes.string)
  old_vocab_file = _ops.convert_to_tensor(old_vocab_file, _dtypes.string)
  _inputs_flat = [new_vocab_file, old_vocab_file]
  _attrs = ("new_vocab_offset", new_vocab_offset, "num_new_vocab",
  num_new_vocab, "old_vocab_size", old_vocab_size)
  _result = _execute.execute(b"GenerateVocabRemapping", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GenerateVocabRemapping", _inputs_flat, _attrs, _result)
  _result = _GenerateVocabRemappingOutput._make(_result)
  return _result


def load_and_remap_matrix(ckpt_path: Annotated[Any, _atypes.String], old_tensor_name: Annotated[Any, _atypes.String], row_remapping: Annotated[Any, _atypes.Int64], col_remapping: Annotated[Any, _atypes.Int64], initializing_values: Annotated[Any, _atypes.Float32], num_rows: int, num_cols: int, max_rows_in_memory:int=-1, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Loads a 2-D (matrix) `Tensor` with name `old_tensor_name` from the checkpoint

  at `ckpt_path` and potentially reorders its rows and columns using the
  specified remappings.

  Most users should use one of the wrapper initializers (such as
  `tf.contrib.framework.load_and_remap_matrix_initializer`) instead of this
  function directly.

  The remappings are 1-D tensors with the following properties:

  * `row_remapping` must have exactly `num_rows` entries. Row `i` of the output
    matrix will be initialized from the row corresponding to index
    `row_remapping[i]` in the old `Tensor` from the checkpoint.
  * `col_remapping` must have either 0 entries (indicating that no column
    reordering is needed) or `num_cols` entries. If specified, column `j` of the
    output matrix will be initialized from the column corresponding to index
    `col_remapping[j]` in the old `Tensor` from the checkpoint.
  * A value of -1 in either of the remappings signifies a "missing" entry. In that
    case, values from the `initializing_values` tensor will be used to fill that
    missing row or column. If `row_remapping` has `r` missing entries and
    `col_remapping` has `c` missing entries, then the following condition must be
    true:

  `(r * num_cols) + (c * num_rows) - (r * c) == len(initializing_values)`

  The remapping tensors can be generated using the GenerateVocabRemapping op.

  As an example, with row_remapping = [1, 0, -1], col_remapping = [0, 2, -1],
  initializing_values = [0.5, -0.5, 0.25, -0.25, 42], and w(i, j) representing
  the value from row i, column j of the old tensor in the checkpoint, the output
  matrix will look like the following:

  [[w(1, 0),  w(1, 2),  0.5],
   [w(0, 0),  w(0, 2), -0.5],
   [0.25,    -0.25,      42]]

  Args:
    ckpt_path: A `Tensor` of type `string`.
      Path to the TensorFlow checkpoint (version 2, `TensorBundle`) from
      which the old matrix `Tensor` will be loaded.
    old_tensor_name: A `Tensor` of type `string`.
      Name of the 2-D `Tensor` to load from checkpoint.
    row_remapping: A `Tensor` of type `int64`.
      An int `Tensor` of row remappings (generally created by
      `generate_vocab_remapping`).  Even if no row remapping is needed, this must
      still be an index-valued Tensor (e.g. [0, 1, 2, ...]), or a shifted
      index-valued `Tensor` (e.g. [8, 9, 10, ...], for partitioned `Variables`).
    col_remapping: A `Tensor` of type `int64`.
      An int `Tensor` of column remappings (generally created by
      `generate_vocab_remapping`).  May be a size-0 `Tensor` if only row remapping
      is to be done (e.g. column ordering is the same).
    initializing_values: A `Tensor` of type `float32`.
      A float `Tensor` containing  values to fill in for cells
      in the output matrix that are not loaded from the checkpoint. Length must be
      exactly the same as the number of missing / new cells.
    num_rows: An `int` that is `>= 0`.
      Number of rows (length of the 1st dimension) in the output matrix.
    num_cols: An `int` that is `>= 1`.
      Number of columns (length of the 2nd dimension) in the output matrix.
    max_rows_in_memory: An optional `int`. Defaults to `-1`.
      The maximum number of rows to load from the checkpoint at
      once. If less than or equal to 0, the entire matrix will be loaded into
      memory. Setting this arg trades increased disk reads for lower memory usage.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadAndRemapMatrix", name, ckpt_path, old_tensor_name,
        row_remapping, col_remapping, initializing_values, "num_rows",
        num_rows, "num_cols", num_cols, "max_rows_in_memory",
        max_rows_in_memory)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_and_remap_matrix_eager_fallback(
          ckpt_path, old_tensor_name, row_remapping, col_remapping,
          initializing_values, num_rows=num_rows, num_cols=num_cols,
          max_rows_in_memory=max_rows_in_memory, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_rows = _execute.make_int(num_rows, "num_rows")
  num_cols = _execute.make_int(num_cols, "num_cols")
  if max_rows_in_memory is None:
    max_rows_in_memory = -1
  max_rows_in_memory = _execute.make_int(max_rows_in_memory, "max_rows_in_memory")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadAndRemapMatrix", ckpt_path=ckpt_path,
                              old_tensor_name=old_tensor_name,
                              row_remapping=row_remapping,
                              col_remapping=col_remapping,
                              initializing_values=initializing_values,
                              num_rows=num_rows, num_cols=num_cols,
                              max_rows_in_memory=max_rows_in_memory,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_rows", _op._get_attr_int("num_rows"), "num_cols",
              _op._get_attr_int("num_cols"), "max_rows_in_memory",
              _op._get_attr_int("max_rows_in_memory"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LoadAndRemapMatrix", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LoadAndRemapMatrix = tf_export("raw_ops.LoadAndRemapMatrix")(_ops.to_raw_op(load_and_remap_matrix))


def load_and_remap_matrix_eager_fallback(ckpt_path: Annotated[Any, _atypes.String], old_tensor_name: Annotated[Any, _atypes.String], row_remapping: Annotated[Any, _atypes.Int64], col_remapping: Annotated[Any, _atypes.Int64], initializing_values: Annotated[Any, _atypes.Float32], num_rows: int, num_cols: int, max_rows_in_memory: int, name, ctx) -> Annotated[Any, _atypes.Float32]:
  num_rows = _execute.make_int(num_rows, "num_rows")
  num_cols = _execute.make_int(num_cols, "num_cols")
  if max_rows_in_memory is None:
    max_rows_in_memory = -1
  max_rows_in_memory = _execute.make_int(max_rows_in_memory, "max_rows_in_memory")
  ckpt_path = _ops.convert_to_tensor(ckpt_path, _dtypes.string)
  old_tensor_name = _ops.convert_to_tensor(old_tensor_name, _dtypes.string)
  row_remapping = _ops.convert_to_tensor(row_remapping, _dtypes.int64)
  col_remapping = _ops.convert_to_tensor(col_remapping, _dtypes.int64)
  initializing_values = _ops.convert_to_tensor(initializing_values, _dtypes.float32)
  _inputs_flat = [ckpt_path, old_tensor_name, row_remapping, col_remapping, initializing_values]
  _attrs = ("num_rows", num_rows, "num_cols", num_cols, "max_rows_in_memory",
  max_rows_in_memory)
  _result = _execute.execute(b"LoadAndRemapMatrix", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LoadAndRemapMatrix", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

