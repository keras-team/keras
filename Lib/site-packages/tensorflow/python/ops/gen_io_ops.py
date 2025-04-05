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

def fixed_length_record_reader(record_bytes: int, header_bytes:int=0, footer_bytes:int=0, hop_bytes:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs fixed-length records from a file.

  Args:
    record_bytes: An `int`. Number of bytes in the record.
    header_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the header, defaults to 0.
    footer_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the footer, defaults to 0.
    hop_bytes: An optional `int`. Defaults to `0`.
      Number of bytes to hop before each read. Default of 0 means using
      record_bytes.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("fixed_length_record_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  record_bytes = _execute.make_int(record_bytes, "record_bytes")
  if header_bytes is None:
    header_bytes = 0
  header_bytes = _execute.make_int(header_bytes, "header_bytes")
  if footer_bytes is None:
    footer_bytes = 0
  footer_bytes = _execute.make_int(footer_bytes, "footer_bytes")
  if hop_bytes is None:
    hop_bytes = 0
  hop_bytes = _execute.make_int(hop_bytes, "hop_bytes")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FixedLengthRecordReader", record_bytes=record_bytes,
                                   header_bytes=header_bytes,
                                   footer_bytes=footer_bytes,
                                   hop_bytes=hop_bytes, container=container,
                                   shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("header_bytes", _op._get_attr_int("header_bytes"),
              "record_bytes", _op._get_attr_int("record_bytes"),
              "footer_bytes", _op._get_attr_int("footer_bytes"), "hop_bytes",
              _op._get_attr_int("hop_bytes"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FixedLengthRecordReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FixedLengthRecordReader = tf_export("raw_ops.FixedLengthRecordReader")(_ops.to_raw_op(fixed_length_record_reader))


def fixed_length_record_reader_eager_fallback(record_bytes: int, header_bytes: int, footer_bytes: int, hop_bytes: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("fixed_length_record_reader op does not support eager execution. Arg 'reader_handle' is a ref.")

def fixed_length_record_reader_v2(record_bytes: int, header_bytes:int=0, footer_bytes:int=0, hop_bytes:int=0, container:str="", shared_name:str="", encoding:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs fixed-length records from a file.

  Args:
    record_bytes: An `int`. Number of bytes in the record.
    header_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the header, defaults to 0.
    footer_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the footer, defaults to 0.
    hop_bytes: An optional `int`. Defaults to `0`.
      Number of bytes to hop before each read. Default of 0 means using
      record_bytes.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    encoding: An optional `string`. Defaults to `""`.
      The type of encoding for the file. Currently ZLIB and GZIP
      are supported. Defaults to none.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FixedLengthRecordReaderV2", name, "header_bytes", header_bytes,
        "record_bytes", record_bytes, "footer_bytes", footer_bytes,
        "hop_bytes", hop_bytes, "container", container, "shared_name",
        shared_name, "encoding", encoding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fixed_length_record_reader_v2_eager_fallback(
          header_bytes=header_bytes, record_bytes=record_bytes,
          footer_bytes=footer_bytes, hop_bytes=hop_bytes, container=container,
          shared_name=shared_name, encoding=encoding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  record_bytes = _execute.make_int(record_bytes, "record_bytes")
  if header_bytes is None:
    header_bytes = 0
  header_bytes = _execute.make_int(header_bytes, "header_bytes")
  if footer_bytes is None:
    footer_bytes = 0
  footer_bytes = _execute.make_int(footer_bytes, "footer_bytes")
  if hop_bytes is None:
    hop_bytes = 0
  hop_bytes = _execute.make_int(hop_bytes, "hop_bytes")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if encoding is None:
    encoding = ""
  encoding = _execute.make_str(encoding, "encoding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FixedLengthRecordReaderV2", record_bytes=record_bytes,
                                     header_bytes=header_bytes,
                                     footer_bytes=footer_bytes,
                                     hop_bytes=hop_bytes, container=container,
                                     shared_name=shared_name,
                                     encoding=encoding, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("header_bytes", _op._get_attr_int("header_bytes"),
              "record_bytes", _op._get_attr_int("record_bytes"),
              "footer_bytes", _op._get_attr_int("footer_bytes"), "hop_bytes",
              _op._get_attr_int("hop_bytes"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "encoding",
              _op.get_attr("encoding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FixedLengthRecordReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FixedLengthRecordReaderV2 = tf_export("raw_ops.FixedLengthRecordReaderV2")(_ops.to_raw_op(fixed_length_record_reader_v2))


def fixed_length_record_reader_v2_eager_fallback(record_bytes: int, header_bytes: int, footer_bytes: int, hop_bytes: int, container: str, shared_name: str, encoding: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  record_bytes = _execute.make_int(record_bytes, "record_bytes")
  if header_bytes is None:
    header_bytes = 0
  header_bytes = _execute.make_int(header_bytes, "header_bytes")
  if footer_bytes is None:
    footer_bytes = 0
  footer_bytes = _execute.make_int(footer_bytes, "footer_bytes")
  if hop_bytes is None:
    hop_bytes = 0
  hop_bytes = _execute.make_int(hop_bytes, "hop_bytes")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if encoding is None:
    encoding = ""
  encoding = _execute.make_str(encoding, "encoding")
  _inputs_flat = []
  _attrs = ("header_bytes", header_bytes, "record_bytes", record_bytes,
  "footer_bytes", footer_bytes, "hop_bytes", hop_bytes, "container",
  container, "shared_name", shared_name, "encoding", encoding)
  _result = _execute.execute(b"FixedLengthRecordReaderV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FixedLengthRecordReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def identity_reader(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  ReaderRead will take the front
  work string and output (work, work).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("identity_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IdentityReader", container=container, shared_name=shared_name,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IdentityReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IdentityReader = tf_export("raw_ops.IdentityReader")(_ops.to_raw_op(identity_reader))


def identity_reader_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("identity_reader op does not support eager execution. Arg 'reader_handle' is a ref.")

def identity_reader_v2(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  ReaderRead will take the front
  work string and output (work, work).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IdentityReaderV2", name, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return identity_reader_v2_eager_fallback(
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
        "IdentityReaderV2", container=container, shared_name=shared_name,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IdentityReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IdentityReaderV2 = tf_export("raw_ops.IdentityReaderV2")(_ops.to_raw_op(identity_reader_v2))


def identity_reader_v2_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"IdentityReaderV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IdentityReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def lmdb_reader(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the records from a LMDB file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("lmdb_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LMDBReader", container=container, shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LMDBReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LMDBReader = tf_export("raw_ops.LMDBReader")(_ops.to_raw_op(lmdb_reader))


def lmdb_reader_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("lmdb_reader op does not support eager execution. Arg 'reader_handle' is a ref.")

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.matching_files', v1=['io.matching_files', 'matching_files'])
@deprecated_endpoints('matching_files')
def matching_files(pattern: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns the set of files matching one or more glob patterns.

  Note that this routine only supports wildcard characters in the
  basename portion of the pattern, not in the directory portion.
  Note also that the order of filenames returned is deterministic.

  Args:
    pattern: A `Tensor` of type `string`.
      Shell wildcard pattern(s). Scalar or vector of type string.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatchingFiles", name, pattern)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_matching_files(
          (pattern, name,), None)
      if _result is not NotImplemented:
        return _result
      return matching_files_eager_fallback(
          pattern, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            matching_files, (), dict(pattern=pattern, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_matching_files(
        (pattern, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatchingFiles", pattern=pattern, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          matching_files, (), dict(pattern=pattern, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatchingFiles", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatchingFiles = tf_export("raw_ops.MatchingFiles")(_ops.to_raw_op(matching_files))
_dispatcher_for_matching_files = matching_files._tf_type_based_dispatcher.Dispatch


def matching_files_eager_fallback(pattern: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  pattern = _ops.convert_to_tensor(pattern, _dtypes.string)
  _inputs_flat = [pattern]
  _attrs = None
  _result = _execute.execute(b"MatchingFiles", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatchingFiles", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def merge_v2_checkpoints(checkpoint_prefixes: Annotated[Any, _atypes.String], destination_prefix: Annotated[Any, _atypes.String], delete_old_dirs:bool=True, allow_missing_files:bool=False, name=None):
  r"""V2 format specific: merges the metadata files of sharded checkpoints.  The

  result is one logical checkpoint, with one physical metadata file and renamed
  data files.

  Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.

  If delete_old_dirs is true, attempts to delete recursively the dirname of each
  path in the input checkpoint_prefixes.  This is useful when those paths are non
  user-facing temporary locations.

  If allow_missing_files is true, merges the checkpoint prefixes as long as
  at least one file exists. Otherwise, if no files exist, an error will be thrown.
  The default value for allow_missing_files is false.

  Args:
    checkpoint_prefixes: A `Tensor` of type `string`.
      prefixes of V2 checkpoints to merge.
    destination_prefix: A `Tensor` of type `string`.
      scalar.  The desired final prefix.  Allowed to be the same
      as one of the checkpoint_prefixes.
    delete_old_dirs: An optional `bool`. Defaults to `True`. see above.
    allow_missing_files: An optional `bool`. Defaults to `False`. see above.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MergeV2Checkpoints", name, checkpoint_prefixes,
        destination_prefix, "delete_old_dirs", delete_old_dirs,
        "allow_missing_files", allow_missing_files)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return merge_v2_checkpoints_eager_fallback(
          checkpoint_prefixes, destination_prefix,
          delete_old_dirs=delete_old_dirs,
          allow_missing_files=allow_missing_files, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if delete_old_dirs is None:
    delete_old_dirs = True
  delete_old_dirs = _execute.make_bool(delete_old_dirs, "delete_old_dirs")
  if allow_missing_files is None:
    allow_missing_files = False
  allow_missing_files = _execute.make_bool(allow_missing_files, "allow_missing_files")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MergeV2Checkpoints", checkpoint_prefixes=checkpoint_prefixes,
                              destination_prefix=destination_prefix,
                              delete_old_dirs=delete_old_dirs,
                              allow_missing_files=allow_missing_files,
                              name=name)
  return _op
MergeV2Checkpoints = tf_export("raw_ops.MergeV2Checkpoints")(_ops.to_raw_op(merge_v2_checkpoints))


def merge_v2_checkpoints_eager_fallback(checkpoint_prefixes: Annotated[Any, _atypes.String], destination_prefix: Annotated[Any, _atypes.String], delete_old_dirs: bool, allow_missing_files: bool, name, ctx):
  if delete_old_dirs is None:
    delete_old_dirs = True
  delete_old_dirs = _execute.make_bool(delete_old_dirs, "delete_old_dirs")
  if allow_missing_files is None:
    allow_missing_files = False
  allow_missing_files = _execute.make_bool(allow_missing_files, "allow_missing_files")
  checkpoint_prefixes = _ops.convert_to_tensor(checkpoint_prefixes, _dtypes.string)
  destination_prefix = _ops.convert_to_tensor(destination_prefix, _dtypes.string)
  _inputs_flat = [checkpoint_prefixes, destination_prefix]
  _attrs = ("delete_old_dirs", delete_old_dirs, "allow_missing_files",
  allow_missing_files)
  _result = _execute.execute(b"MergeV2Checkpoints", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def read_file(filename: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Reads and outputs the entire contents of the input filename.

  Args:
    filename: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReadFile", name, filename)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return read_file_eager_fallback(
          filename, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReadFile", filename=filename, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReadFile", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReadFile = tf_export("raw_ops.ReadFile")(_ops.to_raw_op(read_file))


def read_file_eager_fallback(filename: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  _inputs_flat = [filename]
  _attrs = None
  _result = _execute.execute(b"ReadFile", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReadFile", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def reader_num_records_produced(reader_handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of records this Reader has produced.

  This is the same as the number of ReaderRead executions that have
  succeeded.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("reader_num_records_produced op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderNumRecordsProduced", reader_handle=reader_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderNumRecordsProduced", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReaderNumRecordsProduced = tf_export("raw_ops.ReaderNumRecordsProduced")(_ops.to_raw_op(reader_num_records_produced))


def reader_num_records_produced_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int64]:
  raise RuntimeError("reader_num_records_produced op does not support eager execution. Arg 'reader_handle' is a ref.")

def reader_num_records_produced_v2(reader_handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of records this Reader has produced.

  This is the same as the number of ReaderRead executions that have
  succeeded.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReaderNumRecordsProducedV2", name, reader_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reader_num_records_produced_v2_eager_fallback(
          reader_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderNumRecordsProducedV2", reader_handle=reader_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderNumRecordsProducedV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReaderNumRecordsProducedV2 = tf_export("raw_ops.ReaderNumRecordsProducedV2")(_ops.to_raw_op(reader_num_records_produced_v2))


def reader_num_records_produced_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Int64]:
  reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
  _inputs_flat = [reader_handle]
  _attrs = None
  _result = _execute.execute(b"ReaderNumRecordsProducedV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReaderNumRecordsProducedV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def reader_num_work_units_completed(reader_handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of work units this Reader has finished processing.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("reader_num_work_units_completed op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderNumWorkUnitsCompleted", reader_handle=reader_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderNumWorkUnitsCompleted", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReaderNumWorkUnitsCompleted = tf_export("raw_ops.ReaderNumWorkUnitsCompleted")(_ops.to_raw_op(reader_num_work_units_completed))


def reader_num_work_units_completed_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int64]:
  raise RuntimeError("reader_num_work_units_completed op does not support eager execution. Arg 'reader_handle' is a ref.")

def reader_num_work_units_completed_v2(reader_handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of work units this Reader has finished processing.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReaderNumWorkUnitsCompletedV2", name, reader_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reader_num_work_units_completed_v2_eager_fallback(
          reader_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderNumWorkUnitsCompletedV2", reader_handle=reader_handle,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderNumWorkUnitsCompletedV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReaderNumWorkUnitsCompletedV2 = tf_export("raw_ops.ReaderNumWorkUnitsCompletedV2")(_ops.to_raw_op(reader_num_work_units_completed_v2))


def reader_num_work_units_completed_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Int64]:
  reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
  _inputs_flat = [reader_handle]
  _attrs = None
  _result = _execute.execute(b"ReaderNumWorkUnitsCompletedV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReaderNumWorkUnitsCompletedV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_ReaderReadOutput = collections.namedtuple(
    "ReaderRead",
    ["key", "value"])


def reader_read(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], name=None):
  r"""Returns the next record (key, value pair) produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    queue_handle: A `Tensor` of type mutable `string`.
      Handle to a Queue, with string work items.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, value).

    key: A `Tensor` of type `string`.
    value: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("reader_read op does not support eager execution. Arg 'queue_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderRead", reader_handle=reader_handle, queue_handle=queue_handle,
                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderRead", _inputs_flat, _attrs, _result)
  _result = _ReaderReadOutput._make(_result)
  return _result

ReaderRead = tf_export("raw_ops.ReaderRead")(_ops.to_raw_op(reader_read))


def reader_read_eager_fallback(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], name, ctx):
  raise RuntimeError("reader_read op does not support eager execution. Arg 'queue_handle' is a ref.")
_ReaderReadUpToOutput = collections.namedtuple(
    "ReaderReadUpTo",
    ["keys", "values"])


def reader_read_up_to(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], num_records: Annotated[Any, _atypes.Int64], name=None):
  r"""Returns up to `num_records` (key, value) pairs produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).
  It may return less than `num_records` even before the last batch.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a `Reader`.
    queue_handle: A `Tensor` of type mutable `string`.
      Handle to a `Queue`, with string work items.
    num_records: A `Tensor` of type `int64`.
      number of records to read from `Reader`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `string`.
    values: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("reader_read_up_to op does not support eager execution. Arg 'queue_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderReadUpTo", reader_handle=reader_handle,
                          queue_handle=queue_handle, num_records=num_records,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderReadUpTo", _inputs_flat, _attrs, _result)
  _result = _ReaderReadUpToOutput._make(_result)
  return _result

ReaderReadUpTo = tf_export("raw_ops.ReaderReadUpTo")(_ops.to_raw_op(reader_read_up_to))


def reader_read_up_to_eager_fallback(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], num_records: Annotated[Any, _atypes.Int64], name, ctx):
  raise RuntimeError("reader_read_up_to op does not support eager execution. Arg 'queue_handle' is a ref.")
_ReaderReadUpToV2Output = collections.namedtuple(
    "ReaderReadUpToV2",
    ["keys", "values"])


def reader_read_up_to_v2(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], num_records: Annotated[Any, _atypes.Int64], name=None):
  r"""Returns up to `num_records` (key, value) pairs produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).
  It may return less than `num_records` even before the last batch.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a `Reader`.
    queue_handle: A `Tensor` of type `resource`.
      Handle to a `Queue`, with string work items.
    num_records: A `Tensor` of type `int64`.
      number of records to read from `Reader`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `string`.
    values: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReaderReadUpToV2", name, reader_handle, queue_handle,
        num_records)
      _result = _ReaderReadUpToV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reader_read_up_to_v2_eager_fallback(
          reader_handle, queue_handle, num_records, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderReadUpToV2", reader_handle=reader_handle,
                            queue_handle=queue_handle,
                            num_records=num_records, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderReadUpToV2", _inputs_flat, _attrs, _result)
  _result = _ReaderReadUpToV2Output._make(_result)
  return _result

ReaderReadUpToV2 = tf_export("raw_ops.ReaderReadUpToV2")(_ops.to_raw_op(reader_read_up_to_v2))


def reader_read_up_to_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], num_records: Annotated[Any, _atypes.Int64], name, ctx):
  reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
  queue_handle = _ops.convert_to_tensor(queue_handle, _dtypes.resource)
  num_records = _ops.convert_to_tensor(num_records, _dtypes.int64)
  _inputs_flat = [reader_handle, queue_handle, num_records]
  _attrs = None
  _result = _execute.execute(b"ReaderReadUpToV2", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReaderReadUpToV2", _inputs_flat, _attrs, _result)
  _result = _ReaderReadUpToV2Output._make(_result)
  return _result

_ReaderReadV2Output = collections.namedtuple(
    "ReaderReadV2",
    ["key", "value"])


def reader_read_v2(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], name=None):
  r"""Returns the next record (key, value pair) produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    queue_handle: A `Tensor` of type `resource`.
      Handle to a Queue, with string work items.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, value).

    key: A `Tensor` of type `string`.
    value: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReaderReadV2", name, reader_handle, queue_handle)
      _result = _ReaderReadV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reader_read_v2_eager_fallback(
          reader_handle, queue_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderReadV2", reader_handle=reader_handle,
                        queue_handle=queue_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderReadV2", _inputs_flat, _attrs, _result)
  _result = _ReaderReadV2Output._make(_result)
  return _result

ReaderReadV2 = tf_export("raw_ops.ReaderReadV2")(_ops.to_raw_op(reader_read_v2))


def reader_read_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], name, ctx):
  reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
  queue_handle = _ops.convert_to_tensor(queue_handle, _dtypes.resource)
  _inputs_flat = [reader_handle, queue_handle]
  _attrs = None
  _result = _execute.execute(b"ReaderReadV2", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReaderReadV2", _inputs_flat, _attrs, _result)
  _result = _ReaderReadV2Output._make(_result)
  return _result


def reader_reset(reader_handle: Annotated[Any, _atypes.String], name=None):
  r"""Restore a Reader to its initial clean state.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("reader_reset op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderReset", reader_handle=reader_handle, name=name)
  return _op
ReaderReset = tf_export("raw_ops.ReaderReset")(_ops.to_raw_op(reader_reset))


def reader_reset_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx):
  raise RuntimeError("reader_reset op does not support eager execution. Arg 'reader_handle' is a ref.")

def reader_reset_v2(reader_handle: Annotated[Any, _atypes.Resource], name=None):
  r"""Restore a Reader to its initial clean state.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReaderResetV2", name, reader_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reader_reset_v2_eager_fallback(
          reader_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderResetV2", reader_handle=reader_handle, name=name)
  return _op
ReaderResetV2 = tf_export("raw_ops.ReaderResetV2")(_ops.to_raw_op(reader_reset_v2))


def reader_reset_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx):
  reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
  _inputs_flat = [reader_handle]
  _attrs = None
  _result = _execute.execute(b"ReaderResetV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def reader_restore_state(reader_handle: Annotated[Any, _atypes.String], state: Annotated[Any, _atypes.String], name=None):
  r"""Restore a reader to a previously saved state.

  Not all Readers support being restored, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    state: A `Tensor` of type `string`.
      Result of a ReaderSerializeState of a Reader with type
      matching reader_handle.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("reader_restore_state op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderRestoreState", reader_handle=reader_handle, state=state,
                              name=name)
  return _op
ReaderRestoreState = tf_export("raw_ops.ReaderRestoreState")(_ops.to_raw_op(reader_restore_state))


def reader_restore_state_eager_fallback(reader_handle: Annotated[Any, _atypes.String], state: Annotated[Any, _atypes.String], name, ctx):
  raise RuntimeError("reader_restore_state op does not support eager execution. Arg 'reader_handle' is a ref.")

def reader_restore_state_v2(reader_handle: Annotated[Any, _atypes.Resource], state: Annotated[Any, _atypes.String], name=None):
  r"""Restore a reader to a previously saved state.

  Not all Readers support being restored, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    state: A `Tensor` of type `string`.
      Result of a ReaderSerializeState of a Reader with type
      matching reader_handle.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReaderRestoreStateV2", name, reader_handle, state)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reader_restore_state_v2_eager_fallback(
          reader_handle, state, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderRestoreStateV2", reader_handle=reader_handle, state=state,
                                name=name)
  return _op
ReaderRestoreStateV2 = tf_export("raw_ops.ReaderRestoreStateV2")(_ops.to_raw_op(reader_restore_state_v2))


def reader_restore_state_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], state: Annotated[Any, _atypes.String], name, ctx):
  reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
  state = _ops.convert_to_tensor(state, _dtypes.string)
  _inputs_flat = [reader_handle, state]
  _attrs = None
  _result = _execute.execute(b"ReaderRestoreStateV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def reader_serialize_state(reader_handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Produce a string tensor that encodes the state of a Reader.

  Not all Readers support being serialized, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("reader_serialize_state op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderSerializeState", reader_handle=reader_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderSerializeState", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReaderSerializeState = tf_export("raw_ops.ReaderSerializeState")(_ops.to_raw_op(reader_serialize_state))


def reader_serialize_state_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("reader_serialize_state op does not support eager execution. Arg 'reader_handle' is a ref.")

def reader_serialize_state_v2(reader_handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Produce a string tensor that encodes the state of a Reader.

  Not all Readers support being serialized, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReaderSerializeStateV2", name, reader_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reader_serialize_state_v2_eager_fallback(
          reader_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReaderSerializeStateV2", reader_handle=reader_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReaderSerializeStateV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReaderSerializeStateV2 = tf_export("raw_ops.ReaderSerializeStateV2")(_ops.to_raw_op(reader_serialize_state_v2))


def reader_serialize_state_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
  _inputs_flat = [reader_handle]
  _attrs = None
  _result = _execute.execute(b"ReaderSerializeStateV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReaderSerializeStateV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Restore_dt = TypeVar("TV_Restore_dt", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def restore(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], dt: TV_Restore_dt, preferred_shard:int=-1, name=None) -> Annotated[Any, TV_Restore_dt]:
  r"""Restores a tensor from checkpoint files.

  Reads a tensor stored in one or several files. If there are several files (for
  instance because a tensor was saved as slices), `file_pattern` may contain
  wildcard symbols (`*` and `?`) in the filename portion only, not in the
  directory portion.

  If a `file_pattern` matches several files, `preferred_shard` can be used to hint
  in which file the requested tensor is likely to be found. This op will first
  open the file at index `preferred_shard` in the list of matching files and try
  to restore tensors from that file.  Only if some tensors or tensor slices are
  not found in that first file, then the Op opens all the files. Setting
  `preferred_shard` to match the value passed as the `shard` input
  of a matching `Save` Op may speed up Restore.  This attribute only affects
  performance, not correctness.  The default value -1 means files are processed in
  order.

  See also `RestoreSlice`.

  Args:
    file_pattern: A `Tensor` of type `string`.
      Must have a single element. The pattern of the files from
      which we read the tensor.
    tensor_name: A `Tensor` of type `string`.
      Must have a single element. The name of the tensor to be
      restored.
    dt: A `tf.DType`. The type of the tensor to be restored.
    preferred_shard: An optional `int`. Defaults to `-1`.
      Index of file to open first if multiple files match
      `file_pattern`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dt`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Restore", name, file_pattern, tensor_name, "dt", dt,
        "preferred_shard", preferred_shard)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return restore_eager_fallback(
          file_pattern, tensor_name, dt=dt, preferred_shard=preferred_shard,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dt = _execute.make_type(dt, "dt")
  if preferred_shard is None:
    preferred_shard = -1
  preferred_shard = _execute.make_int(preferred_shard, "preferred_shard")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Restore", file_pattern=file_pattern, tensor_name=tensor_name, dt=dt,
                   preferred_shard=preferred_shard, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dt", _op._get_attr_type("dt"), "preferred_shard",
              _op._get_attr_int("preferred_shard"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Restore", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Restore = tf_export("raw_ops.Restore")(_ops.to_raw_op(restore))


def restore_eager_fallback(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], dt: TV_Restore_dt, preferred_shard: int, name, ctx) -> Annotated[Any, TV_Restore_dt]:
  dt = _execute.make_type(dt, "dt")
  if preferred_shard is None:
    preferred_shard = -1
  preferred_shard = _execute.make_int(preferred_shard, "preferred_shard")
  file_pattern = _ops.convert_to_tensor(file_pattern, _dtypes.string)
  tensor_name = _ops.convert_to_tensor(tensor_name, _dtypes.string)
  _inputs_flat = [file_pattern, tensor_name]
  _attrs = ("dt", dt, "preferred_shard", preferred_shard)
  _result = _execute.execute(b"Restore", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Restore", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RestoreSlice_dt = TypeVar("TV_RestoreSlice_dt", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def restore_slice(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], shape_and_slice: Annotated[Any, _atypes.String], dt: TV_RestoreSlice_dt, preferred_shard:int=-1, name=None) -> Annotated[Any, TV_RestoreSlice_dt]:
  r"""Restores a tensor from checkpoint files.

  This is like `Restore` except that restored tensor can be listed as filling
  only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
  larger tensor and the slice that the restored tensor covers.

  The `shape_and_slice` input has the same format as the
  elements of the `shapes_and_slices` input of the `SaveSlices` op.

  Args:
    file_pattern: A `Tensor` of type `string`.
      Must have a single element. The pattern of the files from
      which we read the tensor.
    tensor_name: A `Tensor` of type `string`.
      Must have a single element. The name of the tensor to be
      restored.
    shape_and_slice: A `Tensor` of type `string`.
      Scalar. The shapes and slice specifications to use when
      restoring a tensors.
    dt: A `tf.DType`. The type of the tensor to be restored.
    preferred_shard: An optional `int`. Defaults to `-1`.
      Index of file to open first if multiple files match
      `file_pattern`. See the documentation for `Restore`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dt`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RestoreSlice", name, file_pattern, tensor_name,
        shape_and_slice, "dt", dt, "preferred_shard", preferred_shard)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return restore_slice_eager_fallback(
          file_pattern, tensor_name, shape_and_slice, dt=dt,
          preferred_shard=preferred_shard, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dt = _execute.make_type(dt, "dt")
  if preferred_shard is None:
    preferred_shard = -1
  preferred_shard = _execute.make_int(preferred_shard, "preferred_shard")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RestoreSlice", file_pattern=file_pattern, tensor_name=tensor_name,
                        shape_and_slice=shape_and_slice, dt=dt,
                        preferred_shard=preferred_shard, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dt", _op._get_attr_type("dt"), "preferred_shard",
              _op._get_attr_int("preferred_shard"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RestoreSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RestoreSlice = tf_export("raw_ops.RestoreSlice")(_ops.to_raw_op(restore_slice))


def restore_slice_eager_fallback(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], shape_and_slice: Annotated[Any, _atypes.String], dt: TV_RestoreSlice_dt, preferred_shard: int, name, ctx) -> Annotated[Any, TV_RestoreSlice_dt]:
  dt = _execute.make_type(dt, "dt")
  if preferred_shard is None:
    preferred_shard = -1
  preferred_shard = _execute.make_int(preferred_shard, "preferred_shard")
  file_pattern = _ops.convert_to_tensor(file_pattern, _dtypes.string)
  tensor_name = _ops.convert_to_tensor(tensor_name, _dtypes.string)
  shape_and_slice = _ops.convert_to_tensor(shape_and_slice, _dtypes.string)
  _inputs_flat = [file_pattern, tensor_name, shape_and_slice]
  _attrs = ("dt", dt, "preferred_shard", preferred_shard)
  _result = _execute.execute(b"RestoreSlice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RestoreSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def restore_v2(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], dtypes, name=None):
  r"""Restores tensors from a V2 checkpoint.

  For backward compatibility with the V1 format, this Op currently allows
  restoring from a V1 checkpoint as well:
    - This Op first attempts to find the V2 index file pointed to by "prefix", and
      if found proceed to read it as a V2 checkpoint;
    - Otherwise the V1 read path is invoked.
  Relying on this behavior is not recommended, as the ability to fall back to read
  V1 might be deprecated and eventually removed.

  By default, restores the named tensors in full.  If the caller wishes to restore
  specific slices of stored tensors, "shape_and_slices" should be non-empty
  strings and correspondingly well-formed.

  Callers must ensure all the named tensors are indeed stored in the checkpoint.

  Args:
    prefix: A `Tensor` of type `string`.
      Must have a single element.  The prefix of a V2 checkpoint.
    tensor_names: A `Tensor` of type `string`.
      shape {N}.  The names of the tensors to be restored.
    shape_and_slices: A `Tensor` of type `string`.
      shape {N}.  The slice specs of the tensors to be restored.
      Empty strings indicate that they are non-partitioned tensors.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      shape {N}.  The list of expected dtype for the tensors.  Must match
      those stored in the checkpoint.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RestoreV2", name, prefix, tensor_names, shape_and_slices,
        "dtypes", dtypes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return restore_v2_eager_fallback(
          prefix, tensor_names, shape_and_slices, dtypes=dtypes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'restore_v2' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RestoreV2", prefix=prefix, tensor_names=tensor_names,
                     shape_and_slices=shape_and_slices, dtypes=dtypes,
                     name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("dtypes", _op.get_attr("dtypes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RestoreV2", _inputs_flat, _attrs, _result)
  return _result

RestoreV2 = tf_export("raw_ops.RestoreV2")(_ops.to_raw_op(restore_v2))


def restore_v2_eager_fallback(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], dtypes, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'restore_v2' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  prefix = _ops.convert_to_tensor(prefix, _dtypes.string)
  tensor_names = _ops.convert_to_tensor(tensor_names, _dtypes.string)
  shape_and_slices = _ops.convert_to_tensor(shape_and_slices, _dtypes.string)
  _inputs_flat = [prefix, tensor_names, shape_and_slices]
  _attrs = ("dtypes", dtypes)
  _result = _execute.execute(b"RestoreV2", len(dtypes), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RestoreV2", _inputs_flat, _attrs, _result)
  return _result


def save(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], data, name=None):
  r"""Saves the input tensors to disk.

  The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
  is written to `filename` with name `tensor_names[i]`.

  See also `SaveSlices`.

  Args:
    filename: A `Tensor` of type `string`.
      Must have a single element. The name of the file to which we write
      the tensor.
    tensor_names: A `Tensor` of type `string`.
      Shape `[N]`. The names of the tensors to be saved.
    data: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Save", name, filename, tensor_names, data)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return save_eager_fallback(
          filename, tensor_names, data, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Save", filename=filename, tensor_names=tensor_names, data=data,
                name=name)
  return _op
Save = tf_export("raw_ops.Save")(_ops.to_raw_op(save))


def save_eager_fallback(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], data, name, ctx):
  _attr_T, data = _execute.convert_to_mixed_eager_tensors(data, ctx)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  tensor_names = _ops.convert_to_tensor(tensor_names, _dtypes.string)
  _inputs_flat = [filename, tensor_names] + list(data)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Save", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


def save_slices(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shapes_and_slices: Annotated[Any, _atypes.String], data, name=None):
  r"""Saves input tensors slices to disk.

  This is like `Save` except that tensors can be listed in the saved file as being
  a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
  larger tensor and the slice that this tensor covers. `shapes_and_slices` must
  have as many elements as `tensor_names`.

  Elements of the `shapes_and_slices` input must either be:

  *  The empty string, in which case the corresponding tensor is
     saved normally.
  *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
     `dimI` are the dimensions of the larger tensor and `slice-spec`
     specifies what part is covered by the tensor to save.

  `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
  where each `sliceI` is either:

  *  The string `-` meaning that the slice covers all indices of this dimension
  *  `start,length` where `start` and `length` are integers.  In that
     case the slice covers `length` indices starting at `start`.

  See also `Save`.

  Args:
    filename: A `Tensor` of type `string`.
      Must have a single element. The name of the file to which we write the
      tensor.
    tensor_names: A `Tensor` of type `string`.
      Shape `[N]`. The names of the tensors to be saved.
    shapes_and_slices: A `Tensor` of type `string`.
      Shape `[N]`.  The shapes and slice specifications to use when
      saving the tensors.
    data: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SaveSlices", name, filename, tensor_names, shapes_and_slices,
        data)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return save_slices_eager_fallback(
          filename, tensor_names, shapes_and_slices, data, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SaveSlices", filename=filename, tensor_names=tensor_names,
                      shapes_and_slices=shapes_and_slices, data=data,
                      name=name)
  return _op
SaveSlices = tf_export("raw_ops.SaveSlices")(_ops.to_raw_op(save_slices))


def save_slices_eager_fallback(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shapes_and_slices: Annotated[Any, _atypes.String], data, name, ctx):
  _attr_T, data = _execute.convert_to_mixed_eager_tensors(data, ctx)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  tensor_names = _ops.convert_to_tensor(tensor_names, _dtypes.string)
  shapes_and_slices = _ops.convert_to_tensor(shapes_and_slices, _dtypes.string)
  _inputs_flat = [filename, tensor_names, shapes_and_slices] + list(data)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SaveSlices", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def save_v2(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], tensors, name=None):
  r"""Saves tensors in V2 checkpoint format.

  By default, saves the named tensors in full.  If the caller wishes to save
  specific slices of full tensors, "shape_and_slices" should be non-empty strings
  and correspondingly well-formed.

  Args:
    prefix: A `Tensor` of type `string`.
      Must have a single element. The prefix of the V2 checkpoint to which we
      write the tensors.
    tensor_names: A `Tensor` of type `string`.
      shape {N}. The names of the tensors to be saved.
    shape_and_slices: A `Tensor` of type `string`.
      shape {N}.  The slice specs of the tensors to be saved.
      Empty strings indicate that they are non-partitioned tensors.
    tensors: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SaveV2", name, prefix, tensor_names, shape_and_slices, tensors)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return save_v2_eager_fallback(
          prefix, tensor_names, shape_and_slices, tensors, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SaveV2", prefix=prefix, tensor_names=tensor_names,
                  shape_and_slices=shape_and_slices, tensors=tensors,
                  name=name)
  return _op
SaveV2 = tf_export("raw_ops.SaveV2")(_ops.to_raw_op(save_v2))


def save_v2_eager_fallback(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], tensors, name, ctx):
  _attr_dtypes, tensors = _execute.convert_to_mixed_eager_tensors(tensors, ctx)
  prefix = _ops.convert_to_tensor(prefix, _dtypes.string)
  tensor_names = _ops.convert_to_tensor(tensor_names, _dtypes.string)
  shape_and_slices = _ops.convert_to_tensor(shape_and_slices, _dtypes.string)
  _inputs_flat = [prefix, tensor_names, shape_and_slices] + list(tensors)
  _attrs = ("dtypes", _attr_dtypes)
  _result = _execute.execute(b"SaveV2", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


def sharded_filename(basename: Annotated[Any, _atypes.String], shard: Annotated[Any, _atypes.Int32], num_shards: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.String]:
  r"""Generate a sharded filename. The filename is printf formatted as

     %s-%05d-of-%05d, basename, shard, num_shards.

  Args:
    basename: A `Tensor` of type `string`.
    shard: A `Tensor` of type `int32`.
    num_shards: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShardedFilename", name, basename, shard, num_shards)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sharded_filename_eager_fallback(
          basename, shard, num_shards, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShardedFilename", basename=basename, shard=shard,
                           num_shards=num_shards, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShardedFilename", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShardedFilename = tf_export("raw_ops.ShardedFilename")(_ops.to_raw_op(sharded_filename))


def sharded_filename_eager_fallback(basename: Annotated[Any, _atypes.String], shard: Annotated[Any, _atypes.Int32], num_shards: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.String]:
  basename = _ops.convert_to_tensor(basename, _dtypes.string)
  shard = _ops.convert_to_tensor(shard, _dtypes.int32)
  num_shards = _ops.convert_to_tensor(num_shards, _dtypes.int32)
  _inputs_flat = [basename, shard, num_shards]
  _attrs = None
  _result = _execute.execute(b"ShardedFilename", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShardedFilename", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def sharded_filespec(basename: Annotated[Any, _atypes.String], num_shards: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.String]:
  r"""Generate a glob pattern matching all sharded file names.

  Args:
    basename: A `Tensor` of type `string`.
    num_shards: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShardedFilespec", name, basename, num_shards)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sharded_filespec_eager_fallback(
          basename, num_shards, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShardedFilespec", basename=basename, num_shards=num_shards,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShardedFilespec", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShardedFilespec = tf_export("raw_ops.ShardedFilespec")(_ops.to_raw_op(sharded_filespec))


def sharded_filespec_eager_fallback(basename: Annotated[Any, _atypes.String], num_shards: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.String]:
  basename = _ops.convert_to_tensor(basename, _dtypes.string)
  num_shards = _ops.convert_to_tensor(num_shards, _dtypes.int32)
  _inputs_flat = [basename, num_shards]
  _attrs = None
  _result = _execute.execute(b"ShardedFilespec", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShardedFilespec", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tf_record_reader(container:str="", shared_name:str="", compression_type:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the records from a TensorFlow Records file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    compression_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tf_record_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if compression_type is None:
    compression_type = ""
  compression_type = _execute.make_str(compression_type, "compression_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TFRecordReader", container=container, shared_name=shared_name,
                          compression_type=compression_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "compression_type",
              _op.get_attr("compression_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TFRecordReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TFRecordReader = tf_export("raw_ops.TFRecordReader")(_ops.to_raw_op(tf_record_reader))


def tf_record_reader_eager_fallback(container: str, shared_name: str, compression_type: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("tf_record_reader op does not support eager execution. Arg 'reader_handle' is a ref.")

def tf_record_reader_v2(container:str="", shared_name:str="", compression_type:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the records from a TensorFlow Records file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    compression_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TFRecordReaderV2", name, "container", container, "shared_name",
        shared_name, "compression_type", compression_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tf_record_reader_v2_eager_fallback(
          container=container, shared_name=shared_name,
          compression_type=compression_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if compression_type is None:
    compression_type = ""
  compression_type = _execute.make_str(compression_type, "compression_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TFRecordReaderV2", container=container, shared_name=shared_name,
                            compression_type=compression_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "compression_type",
              _op.get_attr("compression_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TFRecordReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TFRecordReaderV2 = tf_export("raw_ops.TFRecordReaderV2")(_ops.to_raw_op(tf_record_reader_v2))


def tf_record_reader_v2_eager_fallback(container: str, shared_name: str, compression_type: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if compression_type is None:
    compression_type = ""
  compression_type = _execute.make_str(compression_type, "compression_type")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name,
  "compression_type", compression_type)
  _result = _execute.execute(b"TFRecordReaderV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TFRecordReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def text_line_reader(skip_header_lines:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the lines of a file delimited by '\n'.

  Args:
    skip_header_lines: An optional `int`. Defaults to `0`.
      Number of lines to skip from the beginning of every file.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("text_line_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if skip_header_lines is None:
    skip_header_lines = 0
  skip_header_lines = _execute.make_int(skip_header_lines, "skip_header_lines")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TextLineReader", skip_header_lines=skip_header_lines,
                          container=container, shared_name=shared_name,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("skip_header_lines", _op._get_attr_int("skip_header_lines"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TextLineReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TextLineReader = tf_export("raw_ops.TextLineReader")(_ops.to_raw_op(text_line_reader))


def text_line_reader_eager_fallback(skip_header_lines: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("text_line_reader op does not support eager execution. Arg 'reader_handle' is a ref.")

def text_line_reader_v2(skip_header_lines:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the lines of a file delimited by '\n'.

  Args:
    skip_header_lines: An optional `int`. Defaults to `0`.
      Number of lines to skip from the beginning of every file.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TextLineReaderV2", name, "skip_header_lines",
        skip_header_lines, "container", container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return text_line_reader_v2_eager_fallback(
          skip_header_lines=skip_header_lines, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if skip_header_lines is None:
    skip_header_lines = 0
  skip_header_lines = _execute.make_int(skip_header_lines, "skip_header_lines")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TextLineReaderV2", skip_header_lines=skip_header_lines,
                            container=container, shared_name=shared_name,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("skip_header_lines", _op._get_attr_int("skip_header_lines"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TextLineReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TextLineReaderV2 = tf_export("raw_ops.TextLineReaderV2")(_ops.to_raw_op(text_line_reader_v2))


def text_line_reader_v2_eager_fallback(skip_header_lines: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if skip_header_lines is None:
    skip_header_lines = 0
  skip_header_lines = _execute.make_int(skip_header_lines, "skip_header_lines")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("skip_header_lines", skip_header_lines, "container", container,
  "shared_name", shared_name)
  _result = _execute.execute(b"TextLineReaderV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TextLineReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def whole_file_reader(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of ReaderRead will
  be a filename (key) and the contents of that file (value).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("whole_file_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WholeFileReader", container=container, shared_name=shared_name,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "WholeFileReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

WholeFileReader = tf_export("raw_ops.WholeFileReader")(_ops.to_raw_op(whole_file_reader))


def whole_file_reader_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("whole_file_reader op does not support eager execution. Arg 'reader_handle' is a ref.")

def whole_file_reader_v2(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of ReaderRead will
  be a filename (key) and the contents of that file (value).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "WholeFileReaderV2", name, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return whole_file_reader_v2_eager_fallback(
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
        "WholeFileReaderV2", container=container, shared_name=shared_name,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "WholeFileReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

WholeFileReaderV2 = tf_export("raw_ops.WholeFileReaderV2")(_ops.to_raw_op(whole_file_reader_v2))


def whole_file_reader_v2_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"WholeFileReaderV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "WholeFileReaderV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.write_file', v1=['io.write_file', 'write_file'])
@deprecated_endpoints('write_file')
def write_file(filename: Annotated[Any, _atypes.String], contents: Annotated[Any, _atypes.String], name=None):
  r"""Writes `contents` to the file at input `filename`.

  Creates the file and recursively creates directory if it does not exist.

  Args:
    filename: A `Tensor` of type `string`.
      scalar. The name of the file to which we write the contents.
    contents: A `Tensor` of type `string`.
      scalar. The content to be written to the output file.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "WriteFile", name, filename, contents)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_write_file(
          (filename, contents, name,), None)
      if _result is not NotImplemented:
        return _result
      return write_file_eager_fallback(
          filename, contents, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            write_file, (), dict(filename=filename, contents=contents,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_write_file(
        (filename, contents, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteFile", filename=filename, contents=contents, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          write_file, (), dict(filename=filename, contents=contents,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
WriteFile = tf_export("raw_ops.WriteFile")(_ops.to_raw_op(write_file))
_dispatcher_for_write_file = write_file._tf_type_based_dispatcher.Dispatch


def write_file_eager_fallback(filename: Annotated[Any, _atypes.String], contents: Annotated[Any, _atypes.String], name, ctx):
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [filename, contents]
  _attrs = None
  _result = _execute.execute(b"WriteFile", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

