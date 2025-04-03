// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_IO_OPS_H_
#define TENSORFLOW_CC_OPS_IO_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup io_ops Io Ops
/// @{

/// A Reader that outputs fixed-length records from a file.
///
/// Args:
/// * scope: A Scope object
/// * record_bytes: Number of bytes in the record.
///
/// Optional attributes (see `Attrs`):
/// * header_bytes: Number of bytes in the header, defaults to 0.
/// * footer_bytes: Number of bytes in the footer, defaults to 0.
/// * hop_bytes: Number of bytes to hop before each read. Default of 0 means using
/// record_bytes.
/// * container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// * encoding: The type of encoding for the file. Currently ZLIB and GZIP
/// are supported. Defaults to none.
///
/// Returns:
/// * `Output`: The handle to reference the Reader.
class FixedLengthRecordReader {
 public:
  /// Optional attribute setters for FixedLengthRecordReader
  struct Attrs {
    /// Number of bytes in the header, defaults to 0.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs HeaderBytes(int64 x) {
      Attrs ret = *this;
      ret.header_bytes_ = x;
      return ret;
    }

    /// Number of bytes in the footer, defaults to 0.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs FooterBytes(int64 x) {
      Attrs ret = *this;
      ret.footer_bytes_ = x;
      return ret;
    }

    /// Number of bytes to hop before each read. Default of 0 means using
    /// record_bytes.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs HopBytes(int64 x) {
      Attrs ret = *this;
      ret.hop_bytes_ = x;
      return ret;
    }

    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// The type of encoding for the file. Currently ZLIB and GZIP
    /// are supported. Defaults to none.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Encoding(StringPiece x) {
      Attrs ret = *this;
      ret.encoding_ = x;
      return ret;
    }

    int64 header_bytes_ = 0;
    int64 footer_bytes_ = 0;
    int64 hop_bytes_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    StringPiece encoding_ = "";
  };
  FixedLengthRecordReader(const ::tensorflow::Scope& scope, int64 record_bytes);
  FixedLengthRecordReader(const ::tensorflow::Scope& scope, int64 record_bytes,
                        const FixedLengthRecordReader::Attrs& attrs);
  operator ::tensorflow::Output() const { return reader_handle; }
  operator ::tensorflow::Input() const { return reader_handle; }
  ::tensorflow::Node* node() const { return reader_handle.node(); }

  static Attrs HeaderBytes(int64 x) {
    return Attrs().HeaderBytes(x);
  }
  static Attrs FooterBytes(int64 x) {
    return Attrs().FooterBytes(x);
  }
  static Attrs HopBytes(int64 x) {
    return Attrs().HopBytes(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs Encoding(StringPiece x) {
    return Attrs().Encoding(x);
  }

  Operation operation;
  ::tensorflow::Output reader_handle;
};

/// A Reader that outputs the queued work as both the key and value.
///
/// To use, enqueue strings in a Queue.  ReaderRead will take the front
/// work string and output (work, work).
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
///
/// Returns:
/// * `Output`: The handle to reference the Reader.
class IdentityReader {
 public:
  /// Optional attribute setters for IdentityReader
  struct Attrs {
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  IdentityReader(const ::tensorflow::Scope& scope);
  IdentityReader(const ::tensorflow::Scope& scope, const IdentityReader::Attrs&
               attrs);
  operator ::tensorflow::Output() const { return reader_handle; }
  operator ::tensorflow::Input() const { return reader_handle; }
  ::tensorflow::Node* node() const { return reader_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output reader_handle;
};

/// A Reader that outputs the records from a LMDB file.
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
///
/// Returns:
/// * `Output`: The handle to reference the Reader.
class LMDBReader {
 public:
  /// Optional attribute setters for LMDBReader
  struct Attrs {
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  LMDBReader(const ::tensorflow::Scope& scope);
  LMDBReader(const ::tensorflow::Scope& scope, const LMDBReader::Attrs& attrs);
  operator ::tensorflow::Output() const { return reader_handle; }
  operator ::tensorflow::Input() const { return reader_handle; }
  ::tensorflow::Node* node() const { return reader_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output reader_handle;
};

/// Returns the set of files matching one or more glob patterns.
///
/// Note that this routine only supports wildcard characters in the
/// basename portion of the pattern, not in the directory portion.
/// Note also that the order of filenames returned is deterministic.
///
/// Args:
/// * scope: A Scope object
/// * pattern: Shell wildcard pattern(s). Scalar or vector of type string.
///
/// Returns:
/// * `Output`: A vector of matching filenames.
class MatchingFiles {
 public:
  MatchingFiles(const ::tensorflow::Scope& scope, ::tensorflow::Input pattern);
  operator ::tensorflow::Output() const { return filenames; }
  operator ::tensorflow::Input() const { return filenames; }
  ::tensorflow::Node* node() const { return filenames.node(); }

  Operation operation;
  ::tensorflow::Output filenames;
};

/// V2 format specific: merges the metadata files of sharded checkpoints.  The
///
/// result is one logical checkpoint, with one physical metadata file and renamed
/// data files.
///
/// Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.
///
/// If delete_old_dirs is true, attempts to delete recursively the dirname of each
/// path in the input checkpoint_prefixes.  This is useful when those paths are non
/// user-facing temporary locations.
///
/// If allow_missing_files is true, merges the checkpoint prefixes as long as
/// at least one file exists. Otherwise, if no files exist, an error will be thrown.
/// The default value for allow_missing_files is false.
///
/// Args:
/// * scope: A Scope object
/// * checkpoint_prefixes: prefixes of V2 checkpoints to merge.
/// * destination_prefix: scalar.  The desired final prefix.  Allowed to be the same
/// as one of the checkpoint_prefixes.
///
/// Optional attributes (see `Attrs`):
/// * delete_old_dirs: see above.
/// * allow_missing_files: see above.
///
/// Returns:
/// * the created `Operation`
class MergeV2Checkpoints {
 public:
  /// Optional attribute setters for MergeV2Checkpoints
  struct Attrs {
    /// see above.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs DeleteOldDirs(bool x) {
      Attrs ret = *this;
      ret.delete_old_dirs_ = x;
      return ret;
    }

    /// see above.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AllowMissingFiles(bool x) {
      Attrs ret = *this;
      ret.allow_missing_files_ = x;
      return ret;
    }

    bool delete_old_dirs_ = true;
    bool allow_missing_files_ = false;
  };
  MergeV2Checkpoints(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   checkpoint_prefixes, ::tensorflow::Input destination_prefix);
  MergeV2Checkpoints(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   checkpoint_prefixes, ::tensorflow::Input destination_prefix,
                   const MergeV2Checkpoints::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs DeleteOldDirs(bool x) {
    return Attrs().DeleteOldDirs(x);
  }
  static Attrs AllowMissingFiles(bool x) {
    return Attrs().AllowMissingFiles(x);
  }

  Operation operation;
};

/// Reads and outputs the entire contents of the input filename.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The contents tensor.
class ReadFile {
 public:
  ReadFile(const ::tensorflow::Scope& scope, ::tensorflow::Input filename);
  operator ::tensorflow::Output() const { return contents; }
  operator ::tensorflow::Input() const { return contents; }
  ::tensorflow::Node* node() const { return contents.node(); }

  Operation operation;
  ::tensorflow::Output contents;
};

/// Returns the number of records this Reader has produced.
///
/// This is the same as the number of ReaderRead executions that have
/// succeeded.
///
/// Args:
/// * scope: A Scope object
/// * reader_handle: Handle to a Reader.
///
/// Returns:
/// * `Output`: The records_produced tensor.
class ReaderNumRecordsProduced {
 public:
  ReaderNumRecordsProduced(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         reader_handle);
  operator ::tensorflow::Output() const { return records_produced; }
  operator ::tensorflow::Input() const { return records_produced; }
  ::tensorflow::Node* node() const { return records_produced.node(); }

  Operation operation;
  ::tensorflow::Output records_produced;
};

/// Returns the number of work units this Reader has finished processing.
///
/// Args:
/// * scope: A Scope object
/// * reader_handle: Handle to a Reader.
///
/// Returns:
/// * `Output`: The units_completed tensor.
class ReaderNumWorkUnitsCompleted {
 public:
  ReaderNumWorkUnitsCompleted(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input reader_handle);
  operator ::tensorflow::Output() const { return units_completed; }
  operator ::tensorflow::Input() const { return units_completed; }
  ::tensorflow::Node* node() const { return units_completed.node(); }

  Operation operation;
  ::tensorflow::Output units_completed;
};

/// Returns up to `num_records` (key, value) pairs produced by a Reader.
///
/// Will dequeue from the input queue if necessary (e.g. when the
/// Reader needs to start reading from a new file since it has finished
/// with the previous file).
/// It may return less than `num_records` even before the last batch.
///
/// Args:
/// * scope: A Scope object
/// * reader_handle: Handle to a `Reader`.
/// * queue_handle: Handle to a `Queue`, with string work items.
/// * num_records: number of records to read from `Reader`.
///
/// Returns:
/// * `Output` keys: A 1-D tensor.
/// * `Output` values: A 1-D tensor.
class ReaderReadUpTo {
 public:
  ReaderReadUpTo(const ::tensorflow::Scope& scope, ::tensorflow::Input
               reader_handle, ::tensorflow::Input queue_handle,
               ::tensorflow::Input num_records);

  Operation operation;
  ::tensorflow::Output keys;
  ::tensorflow::Output values;
};

/// Returns the next record (key, value pair) produced by a Reader.
///
/// Will dequeue from the input queue if necessary (e.g. when the
/// Reader needs to start reading from a new file since it has finished
/// with the previous file).
///
/// Args:
/// * scope: A Scope object
/// * reader_handle: Handle to a Reader.
/// * queue_handle: Handle to a Queue, with string work items.
///
/// Returns:
/// * `Output` key: A scalar.
/// * `Output` value: A scalar.
class ReaderRead {
 public:
  ReaderRead(const ::tensorflow::Scope& scope, ::tensorflow::Input reader_handle,
           ::tensorflow::Input queue_handle);

  Operation operation;
  ::tensorflow::Output key;
  ::tensorflow::Output value;
};

/// Restore a Reader to its initial clean state.
///
/// Args:
/// * scope: A Scope object
/// * reader_handle: Handle to a Reader.
///
/// Returns:
/// * the created `Operation`
class ReaderReset {
 public:
  ReaderReset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            reader_handle);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Restore a reader to a previously saved state.
///
/// Not all Readers support being restored, so this can produce an
/// Unimplemented error.
///
/// Args:
/// * scope: A Scope object
/// * reader_handle: Handle to a Reader.
/// * state: Result of a ReaderSerializeState of a Reader with type
/// matching reader_handle.
///
/// Returns:
/// * the created `Operation`
class ReaderRestoreState {
 public:
  ReaderRestoreState(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   reader_handle, ::tensorflow::Input state);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Produce a string tensor that encodes the state of a Reader.
///
/// Not all Readers support being serialized, so this can produce an
/// Unimplemented error.
///
/// Args:
/// * scope: A Scope object
/// * reader_handle: Handle to a Reader.
///
/// Returns:
/// * `Output`: The state tensor.
class ReaderSerializeState {
 public:
  ReaderSerializeState(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     reader_handle);
  operator ::tensorflow::Output() const { return state; }
  operator ::tensorflow::Input() const { return state; }
  ::tensorflow::Node* node() const { return state.node(); }

  Operation operation;
  ::tensorflow::Output state;
};

/// Restores a tensor from checkpoint files.
///
/// Reads a tensor stored in one or several files. If there are several files (for
/// instance because a tensor was saved as slices), `file_pattern` may contain
/// wildcard symbols (`*` and `?`) in the filename portion only, not in the
/// directory portion.
///
/// If a `file_pattern` matches several files, `preferred_shard` can be used to hint
/// in which file the requested tensor is likely to be found. This op will first
/// open the file at index `preferred_shard` in the list of matching files and try
/// to restore tensors from that file.  Only if some tensors or tensor slices are
/// not found in that first file, then the Op opens all the files. Setting
/// `preferred_shard` to match the value passed as the `shard` input
/// of a matching `Save` Op may speed up Restore.  This attribute only affects
/// performance, not correctness.  The default value -1 means files are processed in
/// order.
///
/// See also `RestoreSlice`.
///
/// Args:
/// * scope: A Scope object
/// * file_pattern: Must have a single element. The pattern of the files from
/// which we read the tensor.
/// * tensor_name: Must have a single element. The name of the tensor to be
/// restored.
/// * dt: The type of the tensor to be restored.
///
/// Optional attributes (see `Attrs`):
/// * preferred_shard: Index of file to open first if multiple files match
/// `file_pattern`.
///
/// Returns:
/// * `Output`: The restored tensor.
class Restore {
 public:
  /// Optional attribute setters for Restore
  struct Attrs {
    /// Index of file to open first if multiple files match
    /// `file_pattern`.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs PreferredShard(int64 x) {
      Attrs ret = *this;
      ret.preferred_shard_ = x;
      return ret;
    }

    int64 preferred_shard_ = -1;
  };
  Restore(const ::tensorflow::Scope& scope, ::tensorflow::Input file_pattern,
        ::tensorflow::Input tensor_name, DataType dt);
  Restore(const ::tensorflow::Scope& scope, ::tensorflow::Input file_pattern,
        ::tensorflow::Input tensor_name, DataType dt, const Restore::Attrs&
        attrs);
  operator ::tensorflow::Output() const { return tensor; }
  operator ::tensorflow::Input() const { return tensor; }
  ::tensorflow::Node* node() const { return tensor.node(); }

  static Attrs PreferredShard(int64 x) {
    return Attrs().PreferredShard(x);
  }

  Operation operation;
  ::tensorflow::Output tensor;
};

/// Restores a tensor from checkpoint files.
///
/// This is like `Restore` except that restored tensor can be listed as filling
/// only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
/// larger tensor and the slice that the restored tensor covers.
///
/// The `shape_and_slice` input has the same format as the
/// elements of the `shapes_and_slices` input of the `SaveSlices` op.
///
/// Args:
/// * scope: A Scope object
/// * file_pattern: Must have a single element. The pattern of the files from
/// which we read the tensor.
/// * tensor_name: Must have a single element. The name of the tensor to be
/// restored.
/// * shape_and_slice: Scalar. The shapes and slice specifications to use when
/// restoring a tensors.
/// * dt: The type of the tensor to be restored.
///
/// Optional attributes (see `Attrs`):
/// * preferred_shard: Index of file to open first if multiple files match
/// `file_pattern`. See the documentation for `Restore`.
///
/// Returns:
/// * `Output`: The restored tensor.
class RestoreSlice {
 public:
  /// Optional attribute setters for RestoreSlice
  struct Attrs {
    /// Index of file to open first if multiple files match
    /// `file_pattern`. See the documentation for `Restore`.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs PreferredShard(int64 x) {
      Attrs ret = *this;
      ret.preferred_shard_ = x;
      return ret;
    }

    int64 preferred_shard_ = -1;
  };
  RestoreSlice(const ::tensorflow::Scope& scope, ::tensorflow::Input
             file_pattern, ::tensorflow::Input tensor_name, ::tensorflow::Input
             shape_and_slice, DataType dt);
  RestoreSlice(const ::tensorflow::Scope& scope, ::tensorflow::Input
             file_pattern, ::tensorflow::Input tensor_name, ::tensorflow::Input
             shape_and_slice, DataType dt, const RestoreSlice::Attrs& attrs);
  operator ::tensorflow::Output() const { return tensor; }
  operator ::tensorflow::Input() const { return tensor; }
  ::tensorflow::Node* node() const { return tensor.node(); }

  static Attrs PreferredShard(int64 x) {
    return Attrs().PreferredShard(x);
  }

  Operation operation;
  ::tensorflow::Output tensor;
};

/// Restores tensors from a V2 checkpoint.
///
/// For backward compatibility with the V1 format, this Op currently allows
/// restoring from a V1 checkpoint as well:
///   - This Op first attempts to find the V2 index file pointed to by "prefix", and
///     if found proceed to read it as a V2 checkpoint;
///   - Otherwise the V1 read path is invoked.
/// Relying on this behavior is not recommended, as the ability to fall back to read
/// V1 might be deprecated and eventually removed.
///
/// By default, restores the named tensors in full.  If the caller wishes to restore
/// specific slices of stored tensors, "shape_and_slices" should be non-empty
/// strings and correspondingly well-formed.
///
/// Callers must ensure all the named tensors are indeed stored in the checkpoint.
///
/// Args:
/// * scope: A Scope object
/// * prefix: Must have a single element.  The prefix of a V2 checkpoint.
/// * tensor_names: shape {N}.  The names of the tensors to be restored.
/// * shape_and_slices: shape {N}.  The slice specs of the tensors to be restored.
/// Empty strings indicate that they are non-partitioned tensors.
/// * dtypes: shape {N}.  The list of expected dtype for the tensors.  Must match
/// those stored in the checkpoint.
///
/// Returns:
/// * `OutputList`: shape {N}.  The restored tensors, whose shapes are read from the
/// checkpoint directly.
class RestoreV2 {
 public:
  RestoreV2(const ::tensorflow::Scope& scope, ::tensorflow::Input prefix,
          ::tensorflow::Input tensor_names, ::tensorflow::Input
          shape_and_slices, const DataTypeSlice& dtypes);
  ::tensorflow::Output operator[](size_t index) const { return tensors[index]; }


  Operation operation;
  ::tensorflow::OutputList tensors;
};

/// Saves the input tensors to disk.
///
/// The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
/// is written to `filename` with name `tensor_names[i]`.
///
/// See also `SaveSlices`.
///
/// Args:
/// * scope: A Scope object
/// * filename: Must have a single element. The name of the file to which we write
/// the tensor.
/// * tensor_names: Shape `[N]`. The names of the tensors to be saved.
/// * data: `N` tensors to save.
///
/// Returns:
/// * the created `Operation`
class Save {
 public:
  Save(const ::tensorflow::Scope& scope, ::tensorflow::Input filename,
     ::tensorflow::Input tensor_names, ::tensorflow::InputList data);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Saves input tensors slices to disk.
///
/// This is like `Save` except that tensors can be listed in the saved file as being
/// a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
/// larger tensor and the slice that this tensor covers. `shapes_and_slices` must
/// have as many elements as `tensor_names`.
///
/// Elements of the `shapes_and_slices` input must either be:
///
/// *  The empty string, in which case the corresponding tensor is
///    saved normally.
/// *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
///    `dimI` are the dimensions of the larger tensor and `slice-spec`
///    specifies what part is covered by the tensor to save.
///
/// `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
/// where each `sliceI` is either:
///
/// *  The string `-` meaning that the slice covers all indices of this dimension
/// *  `start,length` where `start` and `length` are integers.  In that
///    case the slice covers `length` indices starting at `start`.
///
/// See also `Save`.
///
/// Args:
/// * scope: A Scope object
/// * filename: Must have a single element. The name of the file to which we write the
/// tensor.
/// * tensor_names: Shape `[N]`. The names of the tensors to be saved.
/// * shapes_and_slices: Shape `[N]`.  The shapes and slice specifications to use when
/// saving the tensors.
/// * data: `N` tensors to save.
///
/// Returns:
/// * the created `Operation`
class SaveSlices {
 public:
  SaveSlices(const ::tensorflow::Scope& scope, ::tensorflow::Input filename,
           ::tensorflow::Input tensor_names, ::tensorflow::Input
           shapes_and_slices, ::tensorflow::InputList data);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Saves tensors in V2 checkpoint format.
///
/// By default, saves the named tensors in full.  If the caller wishes to save
/// specific slices of full tensors, "shape_and_slices" should be non-empty strings
/// and correspondingly well-formed.
///
/// Args:
/// * scope: A Scope object
/// * prefix: Must have a single element. The prefix of the V2 checkpoint to which we
/// write the tensors.
/// * tensor_names: shape {N}. The names of the tensors to be saved.
/// * shape_and_slices: shape {N}.  The slice specs of the tensors to be saved.
/// Empty strings indicate that they are non-partitioned tensors.
/// * tensors: `N` tensors to save.
///
/// Returns:
/// * the created `Operation`
class SaveV2 {
 public:
  SaveV2(const ::tensorflow::Scope& scope, ::tensorflow::Input prefix,
       ::tensorflow::Input tensor_names, ::tensorflow::Input shape_and_slices,
       ::tensorflow::InputList tensors);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Generate a sharded filename. The filename is printf formatted as
///
///    %s-%05d-of-%05d, basename, shard, num_shards.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The filename tensor.
class ShardedFilename {
 public:
  ShardedFilename(const ::tensorflow::Scope& scope, ::tensorflow::Input basename,
                ::tensorflow::Input shard, ::tensorflow::Input num_shards);
  operator ::tensorflow::Output() const { return filename; }
  operator ::tensorflow::Input() const { return filename; }
  ::tensorflow::Node* node() const { return filename.node(); }

  Operation operation;
  ::tensorflow::Output filename;
};

/// Generate a glob pattern matching all sharded file names.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The filename tensor.
class ShardedFilespec {
 public:
  ShardedFilespec(const ::tensorflow::Scope& scope, ::tensorflow::Input basename,
                ::tensorflow::Input num_shards);
  operator ::tensorflow::Output() const { return filename; }
  operator ::tensorflow::Input() const { return filename; }
  ::tensorflow::Node* node() const { return filename.node(); }

  Operation operation;
  ::tensorflow::Output filename;
};

/// A Reader that outputs the records from a TensorFlow Records file.
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
///
/// Returns:
/// * `Output`: The handle to reference the Reader.
class TFRecordReader {
 public:
  /// Optional attribute setters for TFRecordReader
  struct Attrs {
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CompressionType(StringPiece x) {
      Attrs ret = *this;
      ret.compression_type_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    StringPiece compression_type_ = "";
  };
  TFRecordReader(const ::tensorflow::Scope& scope);
  TFRecordReader(const ::tensorflow::Scope& scope, const TFRecordReader::Attrs&
               attrs);
  operator ::tensorflow::Output() const { return reader_handle; }
  operator ::tensorflow::Input() const { return reader_handle; }
  ::tensorflow::Node* node() const { return reader_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs CompressionType(StringPiece x) {
    return Attrs().CompressionType(x);
  }

  Operation operation;
  ::tensorflow::Output reader_handle;
};

/// A Reader that outputs the lines of a file delimited by '\n'.
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * skip_header_lines: Number of lines to skip from the beginning of every file.
/// * container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
///
/// Returns:
/// * `Output`: The handle to reference the Reader.
class TextLineReader {
 public:
  /// Optional attribute setters for TextLineReader
  struct Attrs {
    /// Number of lines to skip from the beginning of every file.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs SkipHeaderLines(int64 x) {
      Attrs ret = *this;
      ret.skip_header_lines_ = x;
      return ret;
    }

    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 skip_header_lines_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  TextLineReader(const ::tensorflow::Scope& scope);
  TextLineReader(const ::tensorflow::Scope& scope, const TextLineReader::Attrs&
               attrs);
  operator ::tensorflow::Output() const { return reader_handle; }
  operator ::tensorflow::Input() const { return reader_handle; }
  ::tensorflow::Node* node() const { return reader_handle.node(); }

  static Attrs SkipHeaderLines(int64 x) {
    return Attrs().SkipHeaderLines(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output reader_handle;
};

/// A Reader that outputs the entire contents of a file as a value.
///
/// To use, enqueue filenames in a Queue.  The output of ReaderRead will
/// be a filename (key) and the contents of that file (value).
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
///
/// Returns:
/// * `Output`: The handle to reference the Reader.
class WholeFileReader {
 public:
  /// Optional attribute setters for WholeFileReader
  struct Attrs {
    /// If non-empty, this reader is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this reader is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  WholeFileReader(const ::tensorflow::Scope& scope);
  WholeFileReader(const ::tensorflow::Scope& scope, const WholeFileReader::Attrs&
                attrs);
  operator ::tensorflow::Output() const { return reader_handle; }
  operator ::tensorflow::Input() const { return reader_handle; }
  ::tensorflow::Node* node() const { return reader_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output reader_handle;
};

/// Writes `contents` to the file at input `filename`.
///
/// Creates the file and recursively creates directory if it does not exist.
///
/// Args:
/// * scope: A Scope object
/// * filename: scalar. The name of the file to which we write the contents.
/// * contents: scalar. The content to be written to the output file.
///
/// Returns:
/// * the created `Operation`
class WriteFile {
 public:
  WriteFile(const ::tensorflow::Scope& scope, ::tensorflow::Input filename,
          ::tensorflow::Input contents);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_IO_OPS_H_
