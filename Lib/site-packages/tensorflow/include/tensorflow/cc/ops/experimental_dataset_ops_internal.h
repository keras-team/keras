// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_EXPERIMENTAL_DATASET_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_EXPERIMENTAL_DATASET_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup experimental_dataset_ops_internal Experimental Dataset Ops Internal
/// @{

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class AssertCardinalityDataset {
 public:
  AssertCardinalityDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input_dataset, ::tensorflow::Input cardinality, const
                         DataTypeSlice& output_types, const
                         gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// A transformation that asserts which transformations happen next.
///
/// This transformation checks whether the camel-case names (i.e. "FlatMap", not
/// "flat_map") of the transformations following this transformation match the list
/// of names in the `transformations` argument. If there is a mismatch, the
/// transformation raises an exception.
///
/// The check occurs when iterating over the contents of the dataset, which
/// means that the check happens *after* any static optimizations are applied
/// to the dataset graph.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// `AssertNextDataset` passes through the outputs of its input dataset.
/// * transformations: A `tf.string` vector `tf.Tensor` identifying the transformations that are
/// expected to happen next.
///
/// Returns:
/// * `Output`: The handle tensor.
class AssertNextDataset {
 public:
  AssertNextDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input transformations, const
                  DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// A transformation that asserts which transformations happened previously.
///
/// This transformation checks the names and, optionally, the attribute name-value
/// pairs in the `transformations` argument against those of the transformations
/// that preceded this transformation.  If there is a mismatch, the transformation
/// raises an exception.
///
/// The check occurs when iterating over the contents of the dataset, which
/// means that the check happens *after* any static optimizations are applied
/// to the dataset graph.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// `AssertPrevDataset` passes through the outputs of its input dataset.
/// * transformations: A `tf.string` vector `tf.Tensor` identifying the transformations, with optional
/// attribute name-value pairs, that are expected to have happened previously.
///
/// Returns:
/// * `Output`: The handle tensor.
class AssertPrevDataset {
 public:
  AssertPrevDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input transformations, const
                  DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that shards the input dataset.
///
/// Creates a dataset that shards the input dataset by num_workers, returning a
/// sharded dataset for the index-th worker. This attempts to automatically shard
/// a dataset by examining the Dataset graph and inserting a shard op before the
/// inputs to a reader Dataset (e.g. CSVDataset, TFRecordDataset).
///
/// This dataset will throw a NotFound error if we cannot shard the dataset
/// automatically.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * num_workers: A scalar representing the number of workers to distribute this dataset across.
/// * index: A scalar representing the index of the current worker out of num_workers.
///
/// Returns:
/// * `Output`: The handle tensor.
class AutoShardDataset {
 public:
  /// Optional attribute setters for AutoShardDataset
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs AutoShardPolicy(int64 x) {
      Attrs ret = *this;
      ret.auto_shard_policy_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NumReplicas(int64 x) {
      Attrs ret = *this;
      ret.num_replicas_ = x;
      return ret;
    }

    int64 auto_shard_policy_ = 0;
    int64 num_replicas_ = 0;
  };
  AutoShardDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::Input num_workers,
                 ::tensorflow::Input index, const DataTypeSlice& output_types,
                 const gtl::ArraySlice<PartialTensorShape>& output_shapes);
  AutoShardDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::Input num_workers,
                 ::tensorflow::Input index, const DataTypeSlice& output_types,
                 const gtl::ArraySlice<PartialTensorShape>& output_shapes,
                 const AutoShardDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs AutoShardPolicy(int64 x) {
    return Attrs().AutoShardPolicy(x);
  }
  static Attrs NumReplicas(int64 x) {
    return Attrs().NumReplicas(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Records the bytes size of each element of `input_dataset` in a StatsAggregator.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class BytesProducedStatsDataset {
 public:
  BytesProducedStatsDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, ::tensorflow::Input tag, const
                          DataTypeSlice& output_types, const
                          gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class CSVDataset {
 public:
  CSVDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input filenames,
           ::tensorflow::Input compression_type, ::tensorflow::Input
           buffer_size, ::tensorflow::Input header, ::tensorflow::Input
           field_delim, ::tensorflow::Input use_quote_delim,
           ::tensorflow::Input na_value, ::tensorflow::Input select_cols,
           ::tensorflow::InputList record_defaults, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class CSVDatasetV2 {
 public:
  CSVDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input filenames,
             ::tensorflow::Input compression_type, ::tensorflow::Input
             buffer_size, ::tensorflow::Input header, ::tensorflow::Input
             field_delim, ::tensorflow::Input use_quote_delim,
             ::tensorflow::Input na_value, ::tensorflow::Input select_cols,
             ::tensorflow::InputList record_defaults, ::tensorflow::Input
             exclude_cols, const gtl::ArraySlice<PartialTensorShape>&
             output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ChooseFastestBranchDataset {
 public:
  ChooseFastestBranchDataset(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input input_dataset,
                           ::tensorflow::Input ratio_numerator,
                           ::tensorflow::Input ratio_denominator,
                           ::tensorflow::InputList other_arguments, int64
                           num_elements_per_branch, const
                           gtl::ArraySlice<NameAttrList>& branches, const
                           gtl::ArraySlice<int>& other_arguments_lengths, const
                           DataTypeSlice& output_types, const
                           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ChooseFastestDataset {
 public:
  ChooseFastestDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                     input_datasets, int64 num_experiments, const
                     DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Compresses a dataset element.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The compressed tensor.
class CompressElement {
 public:
  CompressElement(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                components);
  operator ::tensorflow::Output() const { return compressed; }
  operator ::tensorflow::Input() const { return compressed; }
  ::tensorflow::Node* node() const { return compressed.node(); }

  Operation operation;
  ::tensorflow::Output compressed;
};

/// Computes the static batch size of a dataset sans partial batches.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The batch_size tensor.
class ComputeBatchSize {
 public:
  ComputeBatchSize(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset);
  operator ::tensorflow::Output() const { return batch_size; }
  operator ::tensorflow::Input() const { return batch_size; }
  ::tensorflow::Node* node() const { return batch_size.node(); }

  Operation operation;
  ::tensorflow::Output batch_size;
};

/// Creates a dataset that reads data from the tf.data service.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class DataServiceDataset {
 public:
  /// Optional attribute setters for DataServiceDataset
  struct Attrs {
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TaskRefreshIntervalHintMs(int64 x) {
      Attrs ret = *this;
      ret.task_refresh_interval_hint_ms_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs DataTransferProtocol(StringPiece x) {
      Attrs ret = *this;
      ret.data_transfer_protocol_ = x;
      return ret;
    }

    /// Defaults to "AUTO"
    TF_MUST_USE_RESULT Attrs TargetWorkers(StringPiece x) {
      Attrs ret = *this;
      ret.target_workers_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CrossTrainerCacheOptions(StringPiece x) {
      Attrs ret = *this;
      ret.cross_trainer_cache_options_ = x;
      return ret;
    }

    int64 task_refresh_interval_hint_ms_ = -1;
    StringPiece data_transfer_protocol_ = "";
    StringPiece target_workers_ = "AUTO";
    StringPiece cross_trainer_cache_options_ = "";
  };
  DataServiceDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   dataset_id, ::tensorflow::Input processing_mode,
                   ::tensorflow::Input address, ::tensorflow::Input protocol,
                   ::tensorflow::Input job_name, ::tensorflow::Input
                   max_outstanding_requests, ::tensorflow::Input
                   iteration_counter, const DataTypeSlice& output_types, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes);
  DataServiceDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   dataset_id, ::tensorflow::Input processing_mode,
                   ::tensorflow::Input address, ::tensorflow::Input protocol,
                   ::tensorflow::Input job_name, ::tensorflow::Input
                   max_outstanding_requests, ::tensorflow::Input
                   iteration_counter, const DataTypeSlice& output_types, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                   DataServiceDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs TaskRefreshIntervalHintMs(int64 x) {
    return Attrs().TaskRefreshIntervalHintMs(x);
  }
  static Attrs DataTransferProtocol(StringPiece x) {
    return Attrs().DataTransferProtocol(x);
  }
  static Attrs TargetWorkers(StringPiece x) {
    return Attrs().TargetWorkers(x);
  }
  static Attrs CrossTrainerCacheOptions(StringPiece x) {
    return Attrs().CrossTrainerCacheOptions(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that reads data from the tf.data service.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class DataServiceDatasetV2 {
 public:
  /// Optional attribute setters for DataServiceDatasetV2
  struct Attrs {
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TaskRefreshIntervalHintMs(int64 x) {
      Attrs ret = *this;
      ret.task_refresh_interval_hint_ms_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs DataTransferProtocol(StringPiece x) {
      Attrs ret = *this;
      ret.data_transfer_protocol_ = x;
      return ret;
    }

    /// Defaults to "AUTO"
    TF_MUST_USE_RESULT Attrs TargetWorkers(StringPiece x) {
      Attrs ret = *this;
      ret.target_workers_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CrossTrainerCacheOptions(StringPiece x) {
      Attrs ret = *this;
      ret.cross_trainer_cache_options_ = x;
      return ret;
    }

    int64 task_refresh_interval_hint_ms_ = -1;
    StringPiece data_transfer_protocol_ = "";
    StringPiece target_workers_ = "AUTO";
    StringPiece cross_trainer_cache_options_ = "";
  };
  DataServiceDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     dataset_id, ::tensorflow::Input processing_mode,
                     ::tensorflow::Input address, ::tensorflow::Input protocol,
                     ::tensorflow::Input job_name, ::tensorflow::Input
                     consumer_index, ::tensorflow::Input num_consumers,
                     ::tensorflow::Input max_outstanding_requests,
                     ::tensorflow::Input iteration_counter, const
                     DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  DataServiceDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     dataset_id, ::tensorflow::Input processing_mode,
                     ::tensorflow::Input address, ::tensorflow::Input protocol,
                     ::tensorflow::Input job_name, ::tensorflow::Input
                     consumer_index, ::tensorflow::Input num_consumers,
                     ::tensorflow::Input max_outstanding_requests,
                     ::tensorflow::Input iteration_counter, const
                     DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     DataServiceDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs TaskRefreshIntervalHintMs(int64 x) {
    return Attrs().TaskRefreshIntervalHintMs(x);
  }
  static Attrs DataTransferProtocol(StringPiece x) {
    return Attrs().DataTransferProtocol(x);
  }
  static Attrs TargetWorkers(StringPiece x) {
    return Attrs().TargetWorkers(x);
  }
  static Attrs CrossTrainerCacheOptions(StringPiece x) {
    return Attrs().CrossTrainerCacheOptions(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that reads data from the tf.data service.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class DataServiceDatasetV3 {
 public:
  /// Optional attribute setters for DataServiceDatasetV3
  struct Attrs {
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TaskRefreshIntervalHintMs(int64 x) {
      Attrs ret = *this;
      ret.task_refresh_interval_hint_ms_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs DataTransferProtocol(StringPiece x) {
      Attrs ret = *this;
      ret.data_transfer_protocol_ = x;
      return ret;
    }

    /// Defaults to "AUTO"
    TF_MUST_USE_RESULT Attrs TargetWorkers(StringPiece x) {
      Attrs ret = *this;
      ret.target_workers_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Uncompress(bool x) {
      Attrs ret = *this;
      ret.uncompress_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CrossTrainerCacheOptions(StringPiece x) {
      Attrs ret = *this;
      ret.cross_trainer_cache_options_ = x;
      return ret;
    }

    int64 task_refresh_interval_hint_ms_ = -1;
    StringPiece data_transfer_protocol_ = "";
    StringPiece target_workers_ = "AUTO";
    bool uncompress_ = false;
    StringPiece cross_trainer_cache_options_ = "";
  };
  DataServiceDatasetV3(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     dataset_id, ::tensorflow::Input processing_mode,
                     ::tensorflow::Input address, ::tensorflow::Input protocol,
                     ::tensorflow::Input job_name, ::tensorflow::Input
                     consumer_index, ::tensorflow::Input num_consumers,
                     ::tensorflow::Input max_outstanding_requests,
                     ::tensorflow::Input iteration_counter, const
                     DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     NameAttrList& uncompress_fn);
  DataServiceDatasetV3(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     dataset_id, ::tensorflow::Input processing_mode,
                     ::tensorflow::Input address, ::tensorflow::Input protocol,
                     ::tensorflow::Input job_name, ::tensorflow::Input
                     consumer_index, ::tensorflow::Input num_consumers,
                     ::tensorflow::Input max_outstanding_requests,
                     ::tensorflow::Input iteration_counter, const
                     DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     NameAttrList& uncompress_fn, const
                     DataServiceDatasetV3::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs TaskRefreshIntervalHintMs(int64 x) {
    return Attrs().TaskRefreshIntervalHintMs(x);
  }
  static Attrs DataTransferProtocol(StringPiece x) {
    return Attrs().DataTransferProtocol(x);
  }
  static Attrs TargetWorkers(StringPiece x) {
    return Attrs().TargetWorkers(x);
  }
  static Attrs Uncompress(bool x) {
    return Attrs().Uncompress(x);
  }
  static Attrs CrossTrainerCacheOptions(StringPiece x) {
    return Attrs().CrossTrainerCacheOptions(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that reads data from the tf.data service.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class DataServiceDatasetV4 {
 public:
  /// Optional attribute setters for DataServiceDatasetV4
  struct Attrs {
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TaskRefreshIntervalHintMs(int64 x) {
      Attrs ret = *this;
      ret.task_refresh_interval_hint_ms_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs DataTransferProtocol(StringPiece x) {
      Attrs ret = *this;
      ret.data_transfer_protocol_ = x;
      return ret;
    }

    /// Defaults to "AUTO"
    TF_MUST_USE_RESULT Attrs TargetWorkers(StringPiece x) {
      Attrs ret = *this;
      ret.target_workers_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Uncompress(bool x) {
      Attrs ret = *this;
      ret.uncompress_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CrossTrainerCacheOptions(StringPiece x) {
      Attrs ret = *this;
      ret.cross_trainer_cache_options_ = x;
      return ret;
    }

    int64 task_refresh_interval_hint_ms_ = -1;
    StringPiece data_transfer_protocol_ = "";
    StringPiece target_workers_ = "AUTO";
    bool uncompress_ = false;
    StringPiece cross_trainer_cache_options_ = "";
  };
  DataServiceDatasetV4(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     dataset_id, ::tensorflow::Input processing_mode,
                     ::tensorflow::Input address, ::tensorflow::Input protocol,
                     ::tensorflow::Input job_name, ::tensorflow::Input
                     consumer_index, ::tensorflow::Input num_consumers,
                     ::tensorflow::Input max_outstanding_requests,
                     ::tensorflow::Input iteration_counter, const
                     DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     NameAttrList& uncompress_fn);
  DataServiceDatasetV4(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     dataset_id, ::tensorflow::Input processing_mode,
                     ::tensorflow::Input address, ::tensorflow::Input protocol,
                     ::tensorflow::Input job_name, ::tensorflow::Input
                     consumer_index, ::tensorflow::Input num_consumers,
                     ::tensorflow::Input max_outstanding_requests,
                     ::tensorflow::Input iteration_counter, const
                     DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     NameAttrList& uncompress_fn, const
                     DataServiceDatasetV4::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs TaskRefreshIntervalHintMs(int64 x) {
    return Attrs().TaskRefreshIntervalHintMs(x);
  }
  static Attrs DataTransferProtocol(StringPiece x) {
    return Attrs().DataTransferProtocol(x);
  }
  static Attrs TargetWorkers(StringPiece x) {
    return Attrs().TargetWorkers(x);
  }
  static Attrs Uncompress(bool x) {
    return Attrs().Uncompress(x);
  }
  static Attrs CrossTrainerCacheOptions(StringPiece x) {
    return Attrs().CrossTrainerCacheOptions(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset from the given `graph_def`.
///
/// Creates a dataset from the provided `graph_def`.
///
/// Args:
/// * scope: A Scope object
/// * graph_def: The graph representation of the dataset (as serialized GraphDef).
///
/// Returns:
/// * `Output`: A variant tensor representing the dataset.
class DatasetFromGraph {
 public:
  DatasetFromGraph(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 graph_def);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Writes the given dataset to the given file using the TFRecord format.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to write.
/// * filename: A scalar string tensor representing the filename to use.
/// * compression_type: A scalar string tensor containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
///
/// Returns:
/// * the created `Operation`
class DatasetToTFRecord {
 public:
  DatasetToTFRecord(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input filename,
                  ::tensorflow::Input compression_type);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Creates a dataset that batches input elements into a SparseTensor.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A handle to an input dataset. Must have a single component.
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch.
/// * row_shape: A vector representing the dense shape of each row in the produced
/// SparseTensor. The shape may be partially specified, using `-1` to indicate
/// that a particular dimension should use the maximum size of all batch elements.
///
/// Returns:
/// * `Output`: The handle tensor.
class DenseToSparseBatchDataset {
 public:
  DenseToSparseBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, ::tensorflow::Input batch_size,
                          ::tensorflow::Input row_shape, const DataTypeSlice&
                          output_types, const
                          gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// A substitute for `InterleaveDataset` on a fixed list of `N` datasets.
///
/// Args:
/// * scope: A Scope object
/// * selector_input_dataset: A dataset of scalar `DT_INT64` elements that determines which of the
/// `N` data inputs should produce the next output element.
/// * data_input_datasets: `N` datasets with the same type that will be interleaved according to
/// the values of `selector_input_dataset`.
///
/// Returns:
/// * `Output`: The handle tensor.
class DirectedInterleaveDataset {
 public:
  /// Optional attribute setters for DirectedInterleaveDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs StopOnEmptyDataset(bool x) {
      Attrs ret = *this;
      ret.stop_on_empty_dataset_ = x;
      return ret;
    }

    bool stop_on_empty_dataset_ = false;
  };
  DirectedInterleaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          selector_input_dataset, ::tensorflow::InputList
                          data_input_datasets, const DataTypeSlice&
                          output_types, const
                          gtl::ArraySlice<PartialTensorShape>& output_shapes);
  DirectedInterleaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          selector_input_dataset, ::tensorflow::InputList
                          data_input_datasets, const DataTypeSlice&
                          output_types, const
                          gtl::ArraySlice<PartialTensorShape>& output_shapes,
                          const DirectedInterleaveDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs StopOnEmptyDataset(bool x) {
    return Attrs().StopOnEmptyDataset(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class DistributedSave {
 public:
  /// Optional attribute setters for DistributedSave
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  DistributedSave(const ::tensorflow::Scope& scope, ::tensorflow::Input dataset,
                ::tensorflow::Input directory, ::tensorflow::Input address);
  DistributedSave(const ::tensorflow::Scope& scope, ::tensorflow::Input dataset,
                ::tensorflow::Input directory, ::tensorflow::Input address,
                const DistributedSave::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class DummyIterationCounter {
 public:
  DummyIterationCounter(const ::tensorflow::Scope& scope);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalAssertNextDataset {
 public:
  ExperimentalAssertNextDataset(const ::tensorflow::Scope& scope,
                              ::tensorflow::Input input_dataset,
                              ::tensorflow::Input transformations, const
                              DataTypeSlice& output_types, const
                              gtl::ArraySlice<PartialTensorShape>&
                              output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that shards the input dataset.
///
/// Creates a dataset that shards the input dataset by num_workers, returning a
/// sharded dataset for the index-th worker. This attempts to automatically shard
/// a dataset by examining the Dataset graph and inserting a shard op before the
/// inputs to a reader Dataset (e.g. CSVDataset, TFRecordDataset).
///
/// This dataset will throw a NotFound error if we cannot shard the dataset
/// automatically.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * num_workers: A scalar representing the number of workers to distribute this dataset across.
/// * index: A scalar representing the index of the current worker out of num_workers.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalAutoShardDataset {
 public:
  /// Optional attribute setters for ExperimentalAutoShardDataset
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs AutoShardPolicy(int64 x) {
      Attrs ret = *this;
      ret.auto_shard_policy_ = x;
      return ret;
    }

    int64 auto_shard_policy_ = 0;
  };
  ExperimentalAutoShardDataset(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input_dataset,
                             ::tensorflow::Input num_workers,
                             ::tensorflow::Input index, const DataTypeSlice&
                             output_types, const
                             gtl::ArraySlice<PartialTensorShape>&
                             output_shapes);
  ExperimentalAutoShardDataset(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input_dataset,
                             ::tensorflow::Input num_workers,
                             ::tensorflow::Input index, const DataTypeSlice&
                             output_types, const
                             gtl::ArraySlice<PartialTensorShape>&
                             output_shapes, const
                             ExperimentalAutoShardDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs AutoShardPolicy(int64 x) {
    return Attrs().AutoShardPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Records the bytes size of each element of `input_dataset` in a StatsAggregator.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalBytesProducedStatsDataset {
 public:
  ExperimentalBytesProducedStatsDataset(const ::tensorflow::Scope& scope,
                                      ::tensorflow::Input input_dataset,
                                      ::tensorflow::Input tag, const
                                      DataTypeSlice& output_types, const
                                      gtl::ArraySlice<PartialTensorShape>&
                                      output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalCSVDataset {
 public:
  ExperimentalCSVDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       filenames, ::tensorflow::Input compression_type,
                       ::tensorflow::Input buffer_size, ::tensorflow::Input
                       header, ::tensorflow::Input field_delim,
                       ::tensorflow::Input use_quote_delim, ::tensorflow::Input
                       na_value, ::tensorflow::Input select_cols,
                       ::tensorflow::InputList record_defaults, const
                       gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalChooseFastestDataset {
 public:
  ExperimentalChooseFastestDataset(const ::tensorflow::Scope& scope,
                                 ::tensorflow::InputList input_datasets, int64
                                 num_experiments, const DataTypeSlice&
                                 output_types, const
                                 gtl::ArraySlice<PartialTensorShape>&
                                 output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Returns the cardinality of `input_dataset`.
///
/// Returns the cardinality of `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to return cardinality for.
///
/// Returns:
/// * `Output`: The cardinality of `input_dataset`. Named constants are used to represent
/// infinite and unknown cardinality.
class ExperimentalDatasetCardinality {
 public:
  ExperimentalDatasetCardinality(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input_dataset);
  operator ::tensorflow::Output() const { return cardinality; }
  operator ::tensorflow::Input() const { return cardinality; }
  ::tensorflow::Node* node() const { return cardinality.node(); }

  Operation operation;
  ::tensorflow::Output cardinality;
};

/// Writes the given dataset to the given file using the TFRecord format.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to write.
/// * filename: A scalar string tensor representing the filename to use.
/// * compression_type: A scalar string tensor containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
///
/// Returns:
/// * the created `Operation`
class ExperimentalDatasetToTFRecord {
 public:
  ExperimentalDatasetToTFRecord(const ::tensorflow::Scope& scope,
                              ::tensorflow::Input input_dataset,
                              ::tensorflow::Input filename, ::tensorflow::Input
                              compression_type);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Creates a dataset that batches input elements into a SparseTensor.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A handle to an input dataset. Must have a single component.
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch.
/// * row_shape: A vector representing the dense shape of each row in the produced
/// SparseTensor. The shape may be partially specified, using `-1` to indicate
/// that a particular dimension should use the maximum size of all batch elements.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalDenseToSparseBatchDataset {
 public:
  ExperimentalDenseToSparseBatchDataset(const ::tensorflow::Scope& scope,
                                      ::tensorflow::Input input_dataset,
                                      ::tensorflow::Input batch_size,
                                      ::tensorflow::Input row_shape, const
                                      DataTypeSlice& output_types, const
                                      gtl::ArraySlice<PartialTensorShape>&
                                      output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// A substitute for `InterleaveDataset` on a fixed list of `N` datasets.
///
/// Args:
/// * scope: A Scope object
/// * selector_input_dataset: A dataset of scalar `DT_INT64` elements that determines which of the
/// `N` data inputs should produce the next output element.
/// * data_input_datasets: `N` datasets with the same type that will be interleaved according to
/// the values of `selector_input_dataset`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalDirectedInterleaveDataset {
 public:
  ExperimentalDirectedInterleaveDataset(const ::tensorflow::Scope& scope,
                                      ::tensorflow::Input
                                      selector_input_dataset,
                                      ::tensorflow::InputList
                                      data_input_datasets, const DataTypeSlice&
                                      output_types, const
                                      gtl::ArraySlice<PartialTensorShape>&
                                      output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that computes a group-by on `input_dataset`.
///
/// Creates a dataset that computes a group-by on `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * key_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `key_func`.
/// * init_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `init_func`.
/// * reduce_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `reduce_func`.
/// * finalize_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `finalize_func`.
/// * key_func: A function mapping an element of `input_dataset`, concatenated
/// with `key_func_other_arguments` to a scalar value of type DT_INT64.
/// * init_func: A function mapping a key of type DT_INT64, concatenated with
/// `init_func_other_arguments` to the initial reducer state.
/// * reduce_func: A function mapping the current reducer state and an element of `input_dataset`,
/// concatenated with `reduce_func_other_arguments` to a new reducer state.
/// * finalize_func: A function mapping the final reducer state to an output element.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalGroupByReducerDataset {
 public:
  ExperimentalGroupByReducerDataset(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input input_dataset,
                                  ::tensorflow::InputList
                                  key_func_other_arguments,
                                  ::tensorflow::InputList
                                  init_func_other_arguments,
                                  ::tensorflow::InputList
                                  reduce_func_other_arguments,
                                  ::tensorflow::InputList
                                  finalize_func_other_arguments, const
                                  NameAttrList& key_func, const NameAttrList&
                                  init_func, const NameAttrList& reduce_func,
                                  const NameAttrList& finalize_func, const
                                  DataTypeSlice& output_types, const
                                  gtl::ArraySlice<PartialTensorShape>&
                                  output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that computes a windowed group-by on `input_dataset`.
///
/// // TODO(mrry): Support non-int64 keys.
///
/// Args:
/// * scope: A Scope object
/// * key_func: A function mapping an element of `input_dataset`, concatenated
/// with `key_func_other_arguments` to a scalar value of type DT_INT64.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalGroupByWindowDataset {
 public:
  ExperimentalGroupByWindowDataset(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input input_dataset,
                                 ::tensorflow::InputList
                                 key_func_other_arguments,
                                 ::tensorflow::InputList
                                 reduce_func_other_arguments,
                                 ::tensorflow::InputList
                                 window_size_func_other_arguments, const
                                 NameAttrList& key_func, const NameAttrList&
                                 reduce_func, const NameAttrList&
                                 window_size_func, const DataTypeSlice&
                                 output_types, const
                                 gtl::ArraySlice<PartialTensorShape>&
                                 output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that contains the elements of `input_dataset` ignoring errors.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalIgnoreErrorsDataset {
 public:
  /// Optional attribute setters for ExperimentalIgnoreErrorsDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs LogWarning(bool x) {
      Attrs ret = *this;
      ret.log_warning_ = x;
      return ret;
    }

    bool log_warning_ = false;
  };
  ExperimentalIgnoreErrorsDataset(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input input_dataset, const
                                DataTypeSlice& output_types, const
                                gtl::ArraySlice<PartialTensorShape>&
                                output_shapes);
  ExperimentalIgnoreErrorsDataset(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input input_dataset, const
                                DataTypeSlice& output_types, const
                                gtl::ArraySlice<PartialTensorShape>&
                                output_shapes, const
                                ExperimentalIgnoreErrorsDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs LogWarning(bool x) {
    return Attrs().LogWarning(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Returns the name of the device on which `resource` has been placed.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The device tensor.
class ExperimentalIteratorGetDevice {
 public:
  ExperimentalIteratorGetDevice(const ::tensorflow::Scope& scope,
                              ::tensorflow::Input resource);
  operator ::tensorflow::Output() const { return device; }
  operator ::tensorflow::Input() const { return device; }
  ::tensorflow::Node* node() const { return device.node(); }

  Operation operation;
  ::tensorflow::Output device;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalLMDBDataset {
 public:
  ExperimentalLMDBDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        filenames, const DataTypeSlice& output_types, const
                        gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Records the latency of producing `input_dataset` elements in a StatsAggregator.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalLatencyStatsDataset {
 public:
  ExperimentalLatencyStatsDataset(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input input_dataset,
                                ::tensorflow::Input tag, const DataTypeSlice&
                                output_types, const
                                gtl::ArraySlice<PartialTensorShape>&
                                output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that fuses mapping with batching.
///
/// Creates a dataset that applies `f` to the outputs of `input_dataset` and then
/// batches `batch_size` of them.
///
/// Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
/// to `batch_size * num_parallel_batches` copies of `f` in parallel.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * other_arguments: A list of tensors, typically values that were captured when building a closure
/// for `f`.
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch. It determines the number of concurrent invocations of `f` that process
/// elements from `input_dataset` in parallel.
/// * num_parallel_calls: A scalar representing the maximum number of parallel invocations of the `map_fn`
/// function. Applying the `map_fn` on consecutive input elements in parallel has
/// the potential to improve input pipeline throughput.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
/// * f: A function to apply to the outputs of `input_dataset`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalMapAndBatchDataset {
 public:
  /// Optional attribute setters for ExperimentalMapAndBatchDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs PreserveCardinality(bool x) {
      Attrs ret = *this;
      ret.preserve_cardinality_ = x;
      return ret;
    }

    bool preserve_cardinality_ = false;
  };
  ExperimentalMapAndBatchDataset(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input_dataset,
                               ::tensorflow::InputList other_arguments,
                               ::tensorflow::Input batch_size,
                               ::tensorflow::Input num_parallel_calls,
                               ::tensorflow::Input drop_remainder, const
                               NameAttrList& f, const DataTypeSlice&
                               output_types, const
                               gtl::ArraySlice<PartialTensorShape>&
                               output_shapes);
  ExperimentalMapAndBatchDataset(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input_dataset,
                               ::tensorflow::InputList other_arguments,
                               ::tensorflow::Input batch_size,
                               ::tensorflow::Input num_parallel_calls,
                               ::tensorflow::Input drop_remainder, const
                               NameAttrList& f, const DataTypeSlice&
                               output_types, const
                               gtl::ArraySlice<PartialTensorShape>&
                               output_shapes, const
                               ExperimentalMapAndBatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs PreserveCardinality(bool x) {
    return Attrs().PreserveCardinality(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalMapDataset {
 public:
  /// Optional attribute setters for ExperimentalMapDataset
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseInterOpParallelism(bool x) {
      Attrs ret = *this;
      ret.use_inter_op_parallelism_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs PreserveCardinality(bool x) {
      Attrs ret = *this;
      ret.preserve_cardinality_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ForceSynchronous(bool x) {
      Attrs ret = *this;
      ret.force_synchronous_ = x;
      return ret;
    }

    bool use_inter_op_parallelism_ = true;
    bool preserve_cardinality_ = false;
    bool force_synchronous_ = false;
  };
  ExperimentalMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       input_dataset, ::tensorflow::InputList other_arguments,
                       const NameAttrList& f, const DataTypeSlice&
                       output_types, const gtl::ArraySlice<PartialTensorShape>&
                       output_shapes);
  ExperimentalMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       input_dataset, ::tensorflow::InputList other_arguments,
                       const NameAttrList& f, const DataTypeSlice&
                       output_types, const gtl::ArraySlice<PartialTensorShape>&
                       output_shapes, const ExperimentalMapDataset::Attrs&
                       attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs UseInterOpParallelism(bool x) {
    return Attrs().UseInterOpParallelism(x);
  }
  static Attrs PreserveCardinality(bool x) {
    return Attrs().PreserveCardinality(x);
  }
  static Attrs ForceSynchronous(bool x) {
    return Attrs().ForceSynchronous(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalMatchingFilesDataset {
 public:
  ExperimentalMatchingFilesDataset(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input patterns);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that overrides the maximum intra-op parallelism.
///
/// Args:
/// * scope: A Scope object
/// * max_intra_op_parallelism: Identifies the maximum intra-op parallelism to use.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalMaxIntraOpParallelismDataset {
 public:
  ExperimentalMaxIntraOpParallelismDataset(const ::tensorflow::Scope& scope,
                                         ::tensorflow::Input input_dataset,
                                         ::tensorflow::Input
                                         max_intra_op_parallelism, const
                                         DataTypeSlice& output_types, const
                                         gtl::ArraySlice<PartialTensorShape>&
                                         output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalNonSerializableDataset {
 public:
  ExperimentalNonSerializableDataset(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input input_dataset, const
                                   DataTypeSlice& output_types, const
                                   gtl::ArraySlice<PartialTensorShape>&
                                   output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// The resulting dataset is similar to the `InterleaveDataset`, with the exception
/// that if retrieving the next value from a dataset would cause the requester to
/// block, it will skip that input dataset. This dataset is especially useful
/// when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
/// allows the training step to proceed so long as some data is available.
///
/// !! WARNING !! This dataset is not deterministic!
///
/// Args:
/// * scope: A Scope object
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalParallelInterleaveDataset {
 public:
  ExperimentalParallelInterleaveDataset(const ::tensorflow::Scope& scope,
                                      ::tensorflow::Input input_dataset,
                                      ::tensorflow::InputList other_arguments,
                                      ::tensorflow::Input cycle_length,
                                      ::tensorflow::Input block_length,
                                      ::tensorflow::Input sloppy,
                                      ::tensorflow::Input
                                      buffer_output_elements,
                                      ::tensorflow::Input
                                      prefetch_input_elements, const
                                      NameAttrList& f, const DataTypeSlice&
                                      output_types, const
                                      gtl::ArraySlice<PartialTensorShape>&
                                      output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Transforms `input_dataset` containing `Example` protos as vectors of DT_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.
///
/// Args:
/// * scope: A Scope object
/// * dense_defaults: A dict mapping string keys to `Tensor`s.
/// The keys of the dict must match the dense_keys of the feature.
/// * sparse_keys: A list of string keys in the examples features.
/// The results for these keys will be returned as `SparseTensor` objects.
/// * dense_keys: A list of Ndense string Tensors (scalars).
/// The keys expected in the Examples features associated with dense values.
/// * sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
/// Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
/// and `tf.string` (`BytesList`) are supported.
/// * dense_shapes: List of tuples with the same length as `dense_keys`.
/// The shape of the data for each dense feature referenced by `dense_keys`.
/// Required for any input tensors identified by `dense_keys`.  Must be
/// either fully defined, or may contain an unknown first dimension.
/// An unknown first dimension means the feature is treated as having
/// a variable number of blocks, and the output shape along this dimension
/// is considered unknown at graph build time.  Padding is applied for
/// minibatch elements smaller than the maximum number of blocks for the
/// given feature along this dimension.
/// * output_types: The type list for the return values.
/// * output_shapes: The list of shapes being produced.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalParseExampleDataset {
 public:
  /// Optional attribute setters for ExperimentalParseExampleDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Sloppy(bool x) {
      Attrs ret = *this;
      ret.sloppy_ = x;
      return ret;
    }

    bool sloppy_ = false;
  };
  ExperimentalParseExampleDataset(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input input_dataset,
                                ::tensorflow::Input num_parallel_calls,
                                ::tensorflow::InputList dense_defaults, const
                                gtl::ArraySlice<::tensorflow::tstring>&
                                sparse_keys, const
                                gtl::ArraySlice<::tensorflow::tstring>&
                                dense_keys, const DataTypeSlice& sparse_types,
                                const gtl::ArraySlice<PartialTensorShape>&
                                dense_shapes, const DataTypeSlice&
                                output_types, const
                                gtl::ArraySlice<PartialTensorShape>&
                                output_shapes);
  ExperimentalParseExampleDataset(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input input_dataset,
                                ::tensorflow::Input num_parallel_calls,
                                ::tensorflow::InputList dense_defaults, const
                                gtl::ArraySlice<::tensorflow::tstring>&
                                sparse_keys, const
                                gtl::ArraySlice<::tensorflow::tstring>&
                                dense_keys, const DataTypeSlice& sparse_types,
                                const gtl::ArraySlice<PartialTensorShape>&
                                dense_shapes, const DataTypeSlice&
                                output_types, const
                                gtl::ArraySlice<PartialTensorShape>&
                                output_shapes, const
                                ExperimentalParseExampleDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Sloppy(bool x) {
    return Attrs().Sloppy(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that uses a custom thread pool to compute `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * num_threads: Identifies the number of threads to use for the private threadpool.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalPrivateThreadPoolDataset {
 public:
  ExperimentalPrivateThreadPoolDataset(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input input_dataset,
                                     ::tensorflow::Input num_threads, const
                                     DataTypeSlice& output_types, const
                                     gtl::ArraySlice<PartialTensorShape>&
                                     output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a Dataset that returns pseudorandom numbers.
///
/// Args:
/// * scope: A Scope object
/// * seed: A scalar seed for the random number generator. If either seed or
/// seed2 is set to be non-zero, the random number generator is seeded
/// by the given seed.  Otherwise, a random seed is used.
/// * seed2: A second scalar seed to avoid seed collision.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalRandomDataset {
 public:
  ExperimentalRandomDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          seed, ::tensorflow::Input seed2, const DataTypeSlice&
                          output_types, const
                          gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that changes the batch size.
///
/// Creates a dataset that changes the batch size of the dataset to current batch
/// size // num_replicas.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * num_replicas: A scalar representing the number of replicas to distribute this batch across. As
/// a result of this transformation the current batch size would end up being
/// divided  by this parameter.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalRebatchDataset {
 public:
  /// Optional attribute setters for ExperimentalRebatchDataset
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseFallback(bool x) {
      Attrs ret = *this;
      ret.use_fallback_ = x;
      return ret;
    }

    bool use_fallback_ = true;
  };
  ExperimentalRebatchDataset(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input input_dataset,
                           ::tensorflow::Input num_replicas, const
                           DataTypeSlice& output_types, const
                           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ExperimentalRebatchDataset(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input input_dataset,
                           ::tensorflow::Input num_replicas, const
                           DataTypeSlice& output_types, const
                           gtl::ArraySlice<PartialTensorShape>& output_shapes,
                           const ExperimentalRebatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs UseFallback(bool x) {
    return Attrs().UseFallback(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset successively reduces `f` over the elements of `input_dataset`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalScanDataset {
 public:
  /// Optional attribute setters for ExperimentalScanDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs PreserveCardinality(bool x) {
      Attrs ret = *this;
      ret.preserve_cardinality_ = x;
      return ret;
    }

    bool preserve_cardinality_ = false;
  };
  ExperimentalScanDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input_dataset, ::tensorflow::InputList initial_state,
                        ::tensorflow::InputList other_arguments, const
                        NameAttrList& f, const DataTypeSlice& output_types,
                        const gtl::ArraySlice<PartialTensorShape>&
                        output_shapes);
  ExperimentalScanDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input_dataset, ::tensorflow::InputList initial_state,
                        ::tensorflow::InputList other_arguments, const
                        NameAttrList& f, const DataTypeSlice& output_types,
                        const gtl::ArraySlice<PartialTensorShape>&
                        output_shapes, const ExperimentalScanDataset::Attrs&
                        attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs PreserveCardinality(bool x) {
    return Attrs().PreserveCardinality(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalSetStatsAggregatorDataset {
 public:
  ExperimentalSetStatsAggregatorDataset(const ::tensorflow::Scope& scope,
                                      ::tensorflow::Input input_dataset,
                                      ::tensorflow::Input stats_aggregator,
                                      ::tensorflow::Input tag,
                                      ::tensorflow::Input counter_prefix, const
                                      DataTypeSlice& output_types, const
                                      gtl::ArraySlice<PartialTensorShape>&
                                      output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalSleepDataset {
 public:
  ExperimentalSleepDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input_dataset, ::tensorflow::Input sleep_microseconds,
                         const DataTypeSlice& output_types, const
                         gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that passes a sliding window over `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * window_size: A scalar representing the number of elements in the
/// sliding window.
/// * window_shift: A scalar representing the steps moving the sliding window
/// forward in one iteration. It must be positive.
/// * window_stride: A scalar representing the stride of the input elements of the sliding window.
/// It must be positive.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalSlidingWindowDataset {
 public:
  ExperimentalSlidingWindowDataset(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input input_dataset,
                                 ::tensorflow::Input window_size,
                                 ::tensorflow::Input window_shift,
                                 ::tensorflow::Input window_stride, const
                                 DataTypeSlice& output_types, const
                                 gtl::ArraySlice<PartialTensorShape>&
                                 output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that executes a SQL query and emits rows of the result set.
///
/// Args:
/// * scope: A Scope object
/// * driver_name: The database type. Currently, the only supported type is 'sqlite'.
/// * data_source_name: A connection string to connect to the database.
/// * query: A SQL query to execute.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalSqlDataset {
 public:
  ExperimentalSqlDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       driver_name, ::tensorflow::Input data_source_name,
                       ::tensorflow::Input query, const DataTypeSlice&
                       output_types, const gtl::ArraySlice<PartialTensorShape>&
                       output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a statistics manager resource.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalStatsAggregatorHandle {
 public:
  /// Optional attribute setters for ExperimentalStatsAggregatorHandle
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  ExperimentalStatsAggregatorHandle(const ::tensorflow::Scope& scope);
  ExperimentalStatsAggregatorHandle(const ::tensorflow::Scope& scope, const
                                  ExperimentalStatsAggregatorHandle::Attrs&
                                  attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Produces a summary of any statistics recorded by the given statistics manager.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The summary tensor.
class ExperimentalStatsAggregatorSummary {
 public:
  ExperimentalStatsAggregatorSummary(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input iterator);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Creates a dataset that stops iteration when predicate` is false.
///
/// The `predicate` function must return a scalar boolean and accept the
/// following arguments:
///
/// * One tensor for each component of an element of `input_dataset`.
/// * One tensor for each value in `other_arguments`.
///
/// Args:
/// * scope: A Scope object
/// * other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `predicate`.
/// * predicate: A function returning a scalar boolean.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalTakeWhileDataset {
 public:
  ExperimentalTakeWhileDataset(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input_dataset,
                             ::tensorflow::InputList other_arguments, const
                             NameAttrList& predicate, const DataTypeSlice&
                             output_types, const
                             gtl::ArraySlice<PartialTensorShape>&
                             output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that uses a custom thread pool to compute `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * thread_pool: A resource produced by the ThreadPoolHandle op.
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalThreadPoolDataset {
 public:
  ExperimentalThreadPoolDataset(const ::tensorflow::Scope& scope,
                              ::tensorflow::Input input_dataset,
                              ::tensorflow::Input thread_pool, const
                              DataTypeSlice& output_types, const
                              gtl::ArraySlice<PartialTensorShape>&
                              output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that uses a custom thread pool to compute `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * num_threads: The number of threads in the thread pool.
/// * display_name: A human-readable name for the threads that may be visible in some
/// visualizations.
/// threadpool.
///
/// Optional attributes (see `Attrs`):
/// * max_intra_op_parallelism: The maximum degree of parallelism to use within operations that execute on this
/// threadpool.
///
/// Returns:
/// * `Output`: A resource that can be consumed by one or more ExperimentalThreadPoolDataset
/// ops.
class ExperimentalThreadPoolHandle {
 public:
  /// Optional attribute setters for ExperimentalThreadPoolHandle
  struct Attrs {
    /// The maximum degree of parallelism to use within operations that execute on this
    /// threadpool.
    ///
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs MaxIntraOpParallelism(int64 x) {
      Attrs ret = *this;
      ret.max_intra_op_parallelism_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 max_intra_op_parallelism_ = 1;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  ExperimentalThreadPoolHandle(const ::tensorflow::Scope& scope, int64
                             num_threads, StringPiece display_name);
  ExperimentalThreadPoolHandle(const ::tensorflow::Scope& scope, int64
                             num_threads, StringPiece display_name, const
                             ExperimentalThreadPoolHandle::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs MaxIntraOpParallelism(int64 x) {
    return Attrs().MaxIntraOpParallelism(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// A dataset that splits the elements of its input into multiple elements.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalUnbatchDataset {
 public:
  ExperimentalUnbatchDataset(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input input_dataset, const
                           DataTypeSlice& output_types, const
                           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that contains the unique elements of `input_dataset`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ExperimentalUniqueDataset {
 public:
  ExperimentalUniqueDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, const DataTypeSlice& output_types,
                          const gtl::ArraySlice<PartialTensorShape>&
                          output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Gets the element at the specified index in a dataset.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The components tensor.
class GetElementAtIndex {
 public:
  GetElementAtIndex(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  dataset, ::tensorflow::Input index, const DataTypeSlice&
                  output_types, const gtl::ArraySlice<PartialTensorShape>&
                  output_shapes);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  Operation operation;
  ::tensorflow::OutputList components;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class GlobalShuffleDataset {
 public:
  /// Optional attribute setters for GlobalShuffleDataset
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ReshuffleEachIteration(bool x) {
      Attrs ret = *this;
      ret.reshuffle_each_iteration_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool reshuffle_each_iteration_ = true;
    StringPiece metadata_ = "";
  };
  GlobalShuffleDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input seed,
                     ::tensorflow::Input seed2, ::tensorflow::Input
                     seed_generator, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  GlobalShuffleDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input seed,
                     ::tensorflow::Input seed2, ::tensorflow::Input
                     seed_generator, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     GlobalShuffleDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs ReshuffleEachIteration(bool x) {
    return Attrs().ReshuffleEachIteration(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that computes a group-by on `input_dataset`.
///
/// Creates a dataset that computes a group-by on `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * key_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `key_func`.
/// * init_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `init_func`.
/// * reduce_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `reduce_func`.
/// * finalize_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `finalize_func`.
/// * key_func: A function mapping an element of `input_dataset`, concatenated
/// with `key_func_other_arguments` to a scalar value of type DT_INT64.
/// * init_func: A function mapping a key of type DT_INT64, concatenated with
/// `init_func_other_arguments` to the initial reducer state.
/// * reduce_func: A function mapping the current reducer state and an element of `input_dataset`,
/// concatenated with `reduce_func_other_arguments` to a new reducer state.
/// * finalize_func: A function mapping the final reducer state to an output element.
///
/// Returns:
/// * `Output`: The handle tensor.
class GroupByReducerDataset {
 public:
  GroupByReducerDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_dataset, ::tensorflow::InputList
                      key_func_other_arguments, ::tensorflow::InputList
                      init_func_other_arguments, ::tensorflow::InputList
                      reduce_func_other_arguments, ::tensorflow::InputList
                      finalize_func_other_arguments, const NameAttrList&
                      key_func, const NameAttrList& init_func, const
                      NameAttrList& reduce_func, const NameAttrList&
                      finalize_func, const DataTypeSlice& output_types, const
                      gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that computes a windowed group-by on `input_dataset`.
///
/// // TODO(mrry): Support non-int64 keys.
///
/// Args:
/// * scope: A Scope object
/// * key_func: A function mapping an element of `input_dataset`, concatenated
/// with `key_func_other_arguments` to a scalar value of type DT_INT64.
///
/// Returns:
/// * `Output`: The handle tensor.
class GroupByWindowDataset {
 public:
  /// Optional attribute setters for GroupByWindowDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  GroupByWindowDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::InputList
                     key_func_other_arguments, ::tensorflow::InputList
                     reduce_func_other_arguments, ::tensorflow::InputList
                     window_size_func_other_arguments, const NameAttrList&
                     key_func, const NameAttrList& reduce_func, const
                     NameAttrList& window_size_func, const DataTypeSlice&
                     output_types, const gtl::ArraySlice<PartialTensorShape>&
                     output_shapes);
  GroupByWindowDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::InputList
                     key_func_other_arguments, ::tensorflow::InputList
                     reduce_func_other_arguments, ::tensorflow::InputList
                     window_size_func_other_arguments, const NameAttrList&
                     key_func, const NameAttrList& reduce_func, const
                     NameAttrList& window_size_func, const DataTypeSlice&
                     output_types, const gtl::ArraySlice<PartialTensorShape>&
                     output_shapes, const GroupByWindowDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that contains the elements of `input_dataset` ignoring errors.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class IgnoreErrorsDataset {
 public:
  /// Optional attribute setters for IgnoreErrorsDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs LogWarning(bool x) {
      Attrs ret = *this;
      ret.log_warning_ = x;
      return ret;
    }

    bool log_warning_ = false;
  };
  IgnoreErrorsDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input_dataset, const DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes);
  IgnoreErrorsDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input_dataset, const DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                    IgnoreErrorsDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs LogWarning(bool x) {
    return Attrs().LogWarning(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class IndexFlatMapDataset {
 public:
  /// Optional attribute setters for IndexFlatMapDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  IndexFlatMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input_dataset, ::tensorflow::InputList map_func_other_args,
                    ::tensorflow::InputList index_map_func_other_args,
                    ::tensorflow::Input output_cardinality, const NameAttrList&
                    map_func, const NameAttrList& index_map_func, const
                    DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes);
  IndexFlatMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input_dataset, ::tensorflow::InputList map_func_other_args,
                    ::tensorflow::InputList index_map_func_other_args,
                    ::tensorflow::Input output_cardinality, const NameAttrList&
                    map_func, const NameAttrList& index_map_func, const
                    DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                    IndexFlatMapDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class InitializeTableFromDataset {
 public:
  InitializeTableFromDataset(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input table_handle,
                           ::tensorflow::Input dataset);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Returns the name of the device on which `resource` has been placed.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The device tensor.
class IteratorGetDevice {
 public:
  IteratorGetDevice(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  resource);
  operator ::tensorflow::Output() const { return device; }
  operator ::tensorflow::Input() const { return device; }
  ::tensorflow::Node* node() const { return device.node(); }

  Operation operation;
  ::tensorflow::Output device;
};

/// Returns the serialized model proto of an iterator resource.
///
/// Returns the serialized model proto of an iterator resource.
///
/// Args:
/// * scope: A Scope object
/// * iterator: An resource from an dataset iterator.
///
/// Returns:
/// * `Output`: A serialized model proto.
class IteratorGetModelProto {
 public:
  IteratorGetModelProto(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      iterator);
  operator ::tensorflow::Output() const { return model_proto; }
  operator ::tensorflow::Input() const { return model_proto; }
  ::tensorflow::Node* node() const { return model_proto.node(); }

  Operation operation;
  ::tensorflow::Output model_proto;
};

/// Creates a dataset that emits the key-value pairs in one or more LMDB files.
///
/// The Lightning Memory-Mapped Database Manager, or LMDB, is an embedded binary
/// key-value database. This dataset can read the contents of LMDB database files,
/// the names of which generally have the `.mdb` suffix.
///
/// Each output element consists of a key-value pair represented as a pair of
/// scalar string `Tensor`s, where the first `Tensor` contains the key and the
/// second `Tensor` contains the value.
///
/// LMDB uses different file formats on big- and little-endian machines.
/// `LMDBDataset` can only read files in the format of the host machine.
///
/// Args:
/// * scope: A Scope object
/// * filenames: A scalar or a vector containing the name(s) of the binary file(s) to be
/// read.
///
/// Returns:
/// * `Output`: The handle tensor.
class LMDBDataset {
 public:
  LMDBDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input filenames,
            const DataTypeSlice& output_types, const
            gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Records the latency of producing `input_dataset` elements in a StatsAggregator.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class LatencyStatsDataset {
 public:
  LatencyStatsDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input_dataset, ::tensorflow::Input tag, const
                    DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// The resulting dataset is similar to the `InterleaveDataset`, with the exception
/// that if retrieving the next value from a dataset would cause the requester to
/// block, it will skip that input dataset. This dataset is especially useful
/// when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
/// allows the training step to proceed so long as some data is available.
///
/// !! WARNING !! This dataset is not deterministic!
///
/// Args:
/// * scope: A Scope object
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Returns:
/// * `Output`: The handle tensor.
class LegacyParallelInterleaveDatasetV2 {
 public:
  /// Optional attribute setters for LegacyParallelInterleaveDatasetV2
  struct Attrs {
    /// Defaults to "default"
    TF_MUST_USE_RESULT Attrs Deterministic(StringPiece x) {
      Attrs ret = *this;
      ret.deterministic_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece deterministic_ = "default";
    StringPiece metadata_ = "";
  };
  LegacyParallelInterleaveDatasetV2(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input input_dataset,
                                  ::tensorflow::InputList other_arguments,
                                  ::tensorflow::Input cycle_length,
                                  ::tensorflow::Input block_length,
                                  ::tensorflow::Input buffer_output_elements,
                                  ::tensorflow::Input prefetch_input_elements,
                                  const NameAttrList& f, const DataTypeSlice&
                                  output_types, const
                                  gtl::ArraySlice<PartialTensorShape>&
                                  output_shapes);
  LegacyParallelInterleaveDatasetV2(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input input_dataset,
                                  ::tensorflow::InputList other_arguments,
                                  ::tensorflow::Input cycle_length,
                                  ::tensorflow::Input block_length,
                                  ::tensorflow::Input buffer_output_elements,
                                  ::tensorflow::Input prefetch_input_elements,
                                  const NameAttrList& f, const DataTypeSlice&
                                  output_types, const
                                  gtl::ArraySlice<PartialTensorShape>&
                                  output_shapes, const
                                  LegacyParallelInterleaveDatasetV2::Attrs&
                                  attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Deterministic(StringPiece x) {
    return Attrs().Deterministic(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that emits each of `tensors` once.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ListDataset {
 public:
  /// Optional attribute setters for ListDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  ListDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList tensors,
            const DataTypeSlice& output_types, const
            gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ListDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList tensors,
            const DataTypeSlice& output_types, const
            gtl::ArraySlice<PartialTensorShape>& output_shapes, const
            ListDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ListSnapshotChunksDataset {
 public:
  ListSnapshotChunksDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          snapshot_path, const DataTypeSlice& output_types,
                          const gtl::ArraySlice<PartialTensorShape>&
                          output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class LoadDataset {
 public:
  /// Optional attribute setters for LoadDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Compression(StringPiece x) {
      Attrs ret = *this;
      ret.compression_ = x;
      return ret;
    }

    StringPiece compression_ = "";
  };
  LoadDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input path,
            ::tensorflow::InputList reader_func_other_args, const
            DataTypeSlice& output_types, const
            gtl::ArraySlice<PartialTensorShape>& output_shapes, const
            NameAttrList& reader_func);
  LoadDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input path,
            ::tensorflow::InputList reader_func_other_args, const
            DataTypeSlice& output_types, const
            gtl::ArraySlice<PartialTensorShape>& output_shapes, const
            NameAttrList& reader_func, const LoadDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Compression(StringPiece x) {
    return Attrs().Compression(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that fuses mapping with batching.
///
/// Creates a dataset that applies `f` to the outputs of `input_dataset` and then
/// batches `batch_size` of them.
///
/// Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
/// to `batch_size * num_parallel_batches` copies of `f` in parallel.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * other_arguments: A list of tensors, typically values that were captured when building a closure
/// for `f`.
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch. It determines the number of concurrent invocations of `f` that process
/// elements from `input_dataset` in parallel.
/// * num_parallel_calls: A scalar representing the maximum number of parallel invocations of the `map_fn`
/// function. Applying the `map_fn` on consecutive input elements in parallel has
/// the potential to improve input pipeline throughput.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
/// * f: A function to apply to the outputs of `input_dataset`.
///
/// Returns:
/// * `Output`: The handle tensor.
class MapAndBatchDataset {
 public:
  /// Optional attribute setters for MapAndBatchDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs PreserveCardinality(bool x) {
      Attrs ret = *this;
      ret.preserve_cardinality_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool preserve_cardinality_ = false;
    StringPiece metadata_ = "";
  };
  MapAndBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::InputList other_arguments,
                   ::tensorflow::Input batch_size, ::tensorflow::Input
                   num_parallel_calls, ::tensorflow::Input drop_remainder,
                   const NameAttrList& f, const DataTypeSlice& output_types,
                   const gtl::ArraySlice<PartialTensorShape>& output_shapes);
  MapAndBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::InputList other_arguments,
                   ::tensorflow::Input batch_size, ::tensorflow::Input
                   num_parallel_calls, ::tensorflow::Input drop_remainder,
                   const NameAttrList& f, const DataTypeSlice& output_types,
                   const gtl::ArraySlice<PartialTensorShape>& output_shapes,
                   const MapAndBatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs PreserveCardinality(bool x) {
    return Attrs().PreserveCardinality(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class MatchingFilesDataset {
 public:
  MatchingFilesDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     patterns);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that overrides the maximum intra-op parallelism.
///
/// Args:
/// * scope: A Scope object
/// * max_intra_op_parallelism: Identifies the maximum intra-op parallelism to use.
///
/// Returns:
/// * `Output`: The handle tensor.
class MaxIntraOpParallelismDataset {
 public:
  MaxIntraOpParallelismDataset(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input_dataset,
                             ::tensorflow::Input max_intra_op_parallelism,
                             const DataTypeSlice& output_types, const
                             gtl::ArraySlice<PartialTensorShape>&
                             output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class NonSerializableDataset {
 public:
  NonSerializableDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       input_dataset, const DataTypeSlice& output_types, const
                       gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// The resulting dataset is similar to the `InterleaveDataset`, with the exception
/// that if retrieving the next value from a dataset would cause the requester to
/// block, it will skip that input dataset. This dataset is especially useful
/// when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
/// allows the training step to proceed so long as some data is available.
///
/// !! WARNING !! If the `sloppy` parameter is set to `True`, the operation of this
/// dataset will not be deterministic!
///
/// This dataset has been superseded by `ParallelInterleaveDatasetV2`.  New code
/// should use `ParallelInterleaveDatasetV2`.
///
/// The Python API `tf.data.experimental.parallel_interleave` creates instances of
/// this op. `tf.data.experimental.parallel_interleave` is a deprecated API.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: Dataset that produces a stream of arguments for the function `f`.
/// * other_arguments: Additional arguments to pass to `f` beyond those produced by `input_dataset`.
/// Evaluated once when the dataset is instantiated.
/// * cycle_length: Number of datasets (each created by applying `f` to the elements of
/// `input_dataset`) among which the `ParallelInterleaveDataset` will cycle in a
/// round-robin fashion.
/// * block_length: Number of elements at a time to produce from each interleaved invocation of a
/// dataset returned by `f`.
/// * sloppy: If `True`, return elements as they become available, even if that means returning
/// these elements in a non-deterministic order. Sloppy operation may result in better
/// performance in the presence of stragglers, but the dataset will still block if
/// all of its open streams are blocked.
/// If `False`, always return elements in a deterministic order.
/// * buffer_output_elements: The number of elements each iterator being interleaved should buffer (similar
/// to the `.prefetch()` transformation for each interleaved iterator).
/// * prefetch_input_elements: Determines the number of iterators to prefetch, allowing buffers to warm up and
/// data to be pre-fetched without blocking the main thread.
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParallelInterleaveDataset {
 public:
  /// Optional attribute setters for ParallelInterleaveDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  ParallelInterleaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, ::tensorflow::InputList
                          other_arguments, ::tensorflow::Input cycle_length,
                          ::tensorflow::Input block_length, ::tensorflow::Input
                          sloppy, ::tensorflow::Input buffer_output_elements,
                          ::tensorflow::Input prefetch_input_elements, const
                          NameAttrList& f, const DataTypeSlice& output_types,
                          const gtl::ArraySlice<PartialTensorShape>&
                          output_shapes);
  ParallelInterleaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, ::tensorflow::InputList
                          other_arguments, ::tensorflow::Input cycle_length,
                          ::tensorflow::Input block_length, ::tensorflow::Input
                          sloppy, ::tensorflow::Input buffer_output_elements,
                          ::tensorflow::Input prefetch_input_elements, const
                          NameAttrList& f, const DataTypeSlice& output_types,
                          const gtl::ArraySlice<PartialTensorShape>&
                          output_shapes, const
                          ParallelInterleaveDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Transforms `input_dataset` containing `Example` protos as vectors of DT_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.
///
/// Args:
/// * scope: A Scope object
/// * dense_defaults: A dict mapping string keys to `Tensor`s.
/// The keys of the dict must match the dense_keys of the feature.
/// * sparse_keys: A list of string keys in the examples features.
/// The results for these keys will be returned as `SparseTensor` objects.
/// * dense_keys: A list of Ndense string Tensors (scalars).
/// The keys expected in the Examples features associated with dense values.
/// * sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
/// Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
/// and `tf.string` (`BytesList`) are supported.
/// * dense_shapes: List of tuples with the same length as `dense_keys`.
/// The shape of the data for each dense feature referenced by `dense_keys`.
/// Required for any input tensors identified by `dense_keys`.  Must be
/// either fully defined, or may contain an unknown first dimension.
/// An unknown first dimension means the feature is treated as having
/// a variable number of blocks, and the output shape along this dimension
/// is considered unknown at graph build time.  Padding is applied for
/// minibatch elements smaller than the maximum number of blocks for the
/// given feature along this dimension.
/// * output_types: The type list for the return values.
/// * output_shapes: The list of shapes being produced.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParseExampleDataset {
 public:
  /// Optional attribute setters for ParseExampleDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Sloppy(bool x) {
      Attrs ret = *this;
      ret.sloppy_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs RaggedKeys(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.ragged_keys_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs RaggedValueTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.ragged_value_types_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs RaggedSplitTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.ragged_split_types_ = x;
      return ret;
    }

    bool sloppy_ = false;
    gtl::ArraySlice<::tensorflow::tstring> ragged_keys_ = {};
    DataTypeSlice ragged_value_types_ = {};
    DataTypeSlice ragged_split_types_ = {};
  };
  ParseExampleDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input_dataset, ::tensorflow::Input num_parallel_calls,
                    ::tensorflow::InputList dense_defaults, const
                    gtl::ArraySlice<::tensorflow::tstring>& sparse_keys, const
                    gtl::ArraySlice<::tensorflow::tstring>& dense_keys, const
                    DataTypeSlice& sparse_types, const
                    gtl::ArraySlice<PartialTensorShape>& dense_shapes, const
                    DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ParseExampleDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input_dataset, ::tensorflow::Input num_parallel_calls,
                    ::tensorflow::InputList dense_defaults, const
                    gtl::ArraySlice<::tensorflow::tstring>& sparse_keys, const
                    gtl::ArraySlice<::tensorflow::tstring>& dense_keys, const
                    DataTypeSlice& sparse_types, const
                    gtl::ArraySlice<PartialTensorShape>& dense_shapes, const
                    DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                    ParseExampleDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Sloppy(bool x) {
    return Attrs().Sloppy(x);
  }
  static Attrs RaggedKeys(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().RaggedKeys(x);
  }
  static Attrs RaggedValueTypes(const DataTypeSlice& x) {
    return Attrs().RaggedValueTypes(x);
  }
  static Attrs RaggedSplitTypes(const DataTypeSlice& x) {
    return Attrs().RaggedSplitTypes(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Transforms `input_dataset` containing `Example` protos as vectors of DT_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.
///
/// Args:
/// * scope: A Scope object
/// * dense_defaults: A dict mapping string keys to `Tensor`s.
/// The keys of the dict must match the dense_keys of the feature.
/// * sparse_keys: A list of string keys in the examples features.
/// The results for these keys will be returned as `SparseTensor` objects.
/// * dense_keys: A list of Ndense string Tensors (scalars).
/// The keys expected in the Examples features associated with dense values.
/// * sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
/// Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
/// and `tf.string` (`BytesList`) are supported.
/// * dense_shapes: List of tuples with the same length as `dense_keys`.
/// The shape of the data for each dense feature referenced by `dense_keys`.
/// Required for any input tensors identified by `dense_keys`.  Must be
/// either fully defined, or may contain an unknown first dimension.
/// An unknown first dimension means the feature is treated as having
/// a variable number of blocks, and the output shape along this dimension
/// is considered unknown at graph build time.  Padding is applied for
/// minibatch elements smaller than the maximum number of blocks for the
/// given feature along this dimension.
/// * output_types: The type list for the return values.
/// * output_shapes: The list of shapes being produced.
///
/// Optional attributes (see `Attrs`):
/// * deterministic: A string indicating the op-level determinism to use. Deterministic controls
/// whether the dataset is allowed to return elements out of order if the next
/// element to be returned isn't available, but a later element is. Options are
/// "true", "false", and "default". "default" indicates that determinism should be
/// decided by the `experimental_deterministic` parameter of `tf.data.Options`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParseExampleDatasetV2 {
 public:
  /// Optional attribute setters for ParseExampleDatasetV2
  struct Attrs {
    /// A string indicating the op-level determinism to use. Deterministic controls
    /// whether the dataset is allowed to return elements out of order if the next
    /// element to be returned isn't available, but a later element is. Options are
    /// "true", "false", and "default". "default" indicates that determinism should be
    /// decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    ///
    /// Defaults to "default"
    TF_MUST_USE_RESULT Attrs Deterministic(StringPiece x) {
      Attrs ret = *this;
      ret.deterministic_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs RaggedKeys(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.ragged_keys_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs RaggedValueTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.ragged_value_types_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs RaggedSplitTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.ragged_split_types_ = x;
      return ret;
    }

    StringPiece deterministic_ = "default";
    gtl::ArraySlice<::tensorflow::tstring> ragged_keys_ = {};
    DataTypeSlice ragged_value_types_ = {};
    DataTypeSlice ragged_split_types_ = {};
  };
  ParseExampleDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_dataset, ::tensorflow::Input num_parallel_calls,
                      ::tensorflow::InputList dense_defaults, const
                      gtl::ArraySlice<::tensorflow::tstring>& sparse_keys,
                      const gtl::ArraySlice<::tensorflow::tstring>& dense_keys,
                      const DataTypeSlice& sparse_types, const
                      gtl::ArraySlice<PartialTensorShape>& dense_shapes, const
                      DataTypeSlice& output_types, const
                      gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ParseExampleDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_dataset, ::tensorflow::Input num_parallel_calls,
                      ::tensorflow::InputList dense_defaults, const
                      gtl::ArraySlice<::tensorflow::tstring>& sparse_keys,
                      const gtl::ArraySlice<::tensorflow::tstring>& dense_keys,
                      const DataTypeSlice& sparse_types, const
                      gtl::ArraySlice<PartialTensorShape>& dense_shapes, const
                      DataTypeSlice& output_types, const
                      gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                      ParseExampleDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Deterministic(StringPiece x) {
    return Attrs().Deterministic(x);
  }
  static Attrs RaggedKeys(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().RaggedKeys(x);
  }
  static Attrs RaggedValueTypes(const DataTypeSlice& x) {
    return Attrs().RaggedValueTypes(x);
  }
  static Attrs RaggedSplitTypes(const DataTypeSlice& x) {
    return Attrs().RaggedSplitTypes(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that uses a custom thread pool to compute `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * num_threads: Identifies the number of threads to use for the private threadpool.
///
/// Returns:
/// * `Output`: The handle tensor.
class PrivateThreadPoolDataset {
 public:
  PrivateThreadPoolDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input_dataset, ::tensorflow::Input num_threads, const
                         DataTypeSlice& output_types, const
                         gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a Dataset that returns pseudorandom numbers.
///
/// Creates a Dataset that returns a stream of uniformly distributed
/// pseudorandom 64-bit signed integers.
///
/// In the TensorFlow Python API, you can instantiate this dataset via the
/// class `tf.data.experimental.RandomDataset`.
///
/// Instances of this dataset are also created as a result of the
/// `hoist_random_uniform` static optimization. Whether this optimization is
/// performed is determined by the `experimental_optimization.hoist_random_uniform`
/// option of `tf.data.Options`.
///
/// Args:
/// * scope: A Scope object
/// * seed: A scalar seed for the random number generator. If either seed or
/// seed2 is set to be non-zero, the random number generator is seeded
/// by the given seed.  Otherwise, a random seed is used.
/// * seed2: A second scalar seed to avoid seed collision.
///
/// Returns:
/// * `Output`: The handle tensor.
class RandomDataset {
 public:
  /// Optional attribute setters for RandomDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  RandomDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input seed,
              ::tensorflow::Input seed2, const DataTypeSlice& output_types,
              const gtl::ArraySlice<PartialTensorShape>& output_shapes);
  RandomDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input seed,
              ::tensorflow::Input seed2, const DataTypeSlice& output_types,
              const gtl::ArraySlice<PartialTensorShape>& output_shapes, const
              RandomDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a Dataset that returns pseudorandom numbers.
///
/// Creates a Dataset that returns a stream of uniformly distributed
/// pseudorandom 64-bit signed integers. It accepts a boolean attribute that
/// determines if the random number generators are re-applied at each epoch. The
/// default value is True which means that the seeds are applied and the same
/// sequence of random numbers are generated at each epoch. If set to False, the
/// seeds are not re-applied and a different sequence of random numbers are
/// generated at each epoch.
///
/// In the TensorFlow Python API, you can instantiate this dataset via the
/// class `tf.data.experimental.RandomDatasetV2`.
///
/// Args:
/// * scope: A Scope object
/// * seed: A scalar seed for the random number generator. If either seed or
/// seed2 is set to be non-zero, the random number generator is seeded
/// by the given seed.  Otherwise, a random seed is used.
/// * seed2: A second scalar seed to avoid seed collision.
/// * seed_generator: A resource for the random number seed generator.
///
/// Optional attributes (see `Attrs`):
/// * rerandomize_each_iteration: A boolean attribute to rerandomize the sequence of random numbers generated
/// at each epoch.
///
/// Returns:
/// * `Output`: The handle tensor.
class RandomDatasetV2 {
 public:
  /// Optional attribute setters for RandomDatasetV2
  struct Attrs {
    /// A boolean attribute to rerandomize the sequence of random numbers generated
    /// at each epoch.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs RerandomizeEachIteration(bool x) {
      Attrs ret = *this;
      ret.rerandomize_each_iteration_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool rerandomize_each_iteration_ = false;
    StringPiece metadata_ = "";
  };
  RandomDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input seed,
                ::tensorflow::Input seed2, ::tensorflow::Input seed_generator,
                const DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes);
  RandomDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input seed,
                ::tensorflow::Input seed2, ::tensorflow::Input seed_generator,
                const DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                RandomDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs RerandomizeEachIteration(bool x) {
    return Attrs().RerandomizeEachIteration(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that changes the batch size.
///
/// Creates a dataset that changes the batch size of the dataset to current batch
/// size // num_workers.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * num_replicas: A scalar representing the number of replicas to distribute this batch across. As
/// a result of this transformation the current batch size would end up being
/// divided  by this parameter.
///
/// Returns:
/// * `Output`: The handle tensor.
class RebatchDataset {
 public:
  /// Optional attribute setters for RebatchDataset
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseFallback(bool x) {
      Attrs ret = *this;
      ret.use_fallback_ = x;
      return ret;
    }

    bool use_fallback_ = true;
  };
  RebatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input num_replicas, const
               DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes);
  RebatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input num_replicas, const
               DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes, const
               RebatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs UseFallback(bool x) {
    return Attrs().UseFallback(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that changes the batch size.
///
/// Creates a dataset that rebatches elements from `input_dataset` into new batch
/// sizes.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * batch_sizes: A vector of integers representing the size of batches to produce. These values
/// are cycled through in order.
///
/// Returns:
/// * `Output`: The handle tensor.
class RebatchDatasetV2 {
 public:
  RebatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::Input batch_sizes,
                 ::tensorflow::Input drop_remainder, const DataTypeSlice&
                 output_types, const gtl::ArraySlice<PartialTensorShape>&
                 output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Registers a dataset with the tf.data service.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The dataset_id tensor.
class RegisterDataset {
 public:
  /// Optional attribute setters for RegisterDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ElementSpec(StringPiece x) {
      Attrs ret = *this;
      ret.element_spec_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece element_spec_ = "";
    StringPiece metadata_ = "";
  };
  RegisterDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input dataset,
                ::tensorflow::Input address, ::tensorflow::Input protocol,
                int64 external_state_policy);
  RegisterDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input dataset,
                ::tensorflow::Input address, ::tensorflow::Input protocol,
                int64 external_state_policy, const RegisterDataset::Attrs&
                attrs);
  operator ::tensorflow::Output() const { return dataset_id; }
  operator ::tensorflow::Input() const { return dataset_id; }
  ::tensorflow::Node* node() const { return dataset_id.node(); }

  static Attrs ElementSpec(StringPiece x) {
    return Attrs().ElementSpec(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output dataset_id;
};

/// Registers a dataset with the tf.data service.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The dataset_id tensor.
class RegisterDatasetV2 {
 public:
  /// Optional attribute setters for RegisterDatasetV2
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ElementSpec(StringPiece x) {
      Attrs ret = *this;
      ret.element_spec_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs RequestedDatasetId(StringPiece x) {
      Attrs ret = *this;
      ret.requested_dataset_id_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece element_spec_ = "";
    StringPiece requested_dataset_id_ = "";
    StringPiece metadata_ = "";
  };
  RegisterDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  dataset, ::tensorflow::Input address, ::tensorflow::Input
                  protocol, int64 external_state_policy);
  RegisterDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  dataset, ::tensorflow::Input address, ::tensorflow::Input
                  protocol, int64 external_state_policy, const
                  RegisterDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return dataset_id; }
  operator ::tensorflow::Input() const { return dataset_id; }
  ::tensorflow::Node* node() const { return dataset_id.node(); }

  static Attrs ElementSpec(StringPiece x) {
    return Attrs().ElementSpec(x);
  }
  static Attrs RequestedDatasetId(StringPiece x) {
    return Attrs().RequestedDatasetId(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output dataset_id;
};

/// Creates a dataset that takes a Bernoulli sample of the contents of another dataset.
///
/// There is no transformation in the `tf.data` Python API for creating this dataset.
/// Instead, it is created as a result of the `filter_with_random_uniform_fusion`
/// static optimization. Whether this optimization is performed is determined by the
/// `experimental_optimization.filter_with_random_uniform_fusion` option of
/// `tf.data.Options`.
///
/// Args:
/// * scope: A Scope object
/// * rate: A scalar representing the sample rate. Each element of `input_dataset` is
/// retained with this probability, independent of all other elements.
/// * seed: A scalar representing seed of random number generator.
/// * seed2: A scalar representing seed2 of random number generator.
///
/// Returns:
/// * `Output`: The handle tensor.
class SamplingDataset {
 public:
  SamplingDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input rate, ::tensorflow::Input
                seed, ::tensorflow::Input seed2, const DataTypeSlice&
                output_types, const gtl::ArraySlice<PartialTensorShape>&
                output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class SaveDataset {
 public:
  /// Optional attribute setters for SaveDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Compression(StringPiece x) {
      Attrs ret = *this;
      ret.compression_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseShardFunc(bool x) {
      Attrs ret = *this;
      ret.use_shard_func_ = x;
      return ret;
    }

    StringPiece compression_ = "";
    bool use_shard_func_ = true;
  };
  SaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::Input path, ::tensorflow::InputList
            shard_func_other_args, const NameAttrList& shard_func);
  SaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::Input path, ::tensorflow::InputList
            shard_func_other_args, const NameAttrList& shard_func, const
            SaveDataset::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Compression(StringPiece x) {
    return Attrs().Compression(x);
  }
  static Attrs UseShardFunc(bool x) {
    return Attrs().UseShardFunc(x);
  }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class SaveDatasetV2 {
 public:
  /// Optional attribute setters for SaveDatasetV2
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Compression(StringPiece x) {
      Attrs ret = *this;
      ret.compression_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseShardFunc(bool x) {
      Attrs ret = *this;
      ret.use_shard_func_ = x;
      return ret;
    }

    StringPiece compression_ = "";
    bool use_shard_func_ = true;
  };
  SaveDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::Input path, ::tensorflow::InputList
              shard_func_other_args, const NameAttrList& shard_func, const
              DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes);
  SaveDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::Input path, ::tensorflow::InputList
              shard_func_other_args, const NameAttrList& shard_func, const
              DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes, const
              SaveDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Compression(StringPiece x) {
    return Attrs().Compression(x);
  }
  static Attrs UseShardFunc(bool x) {
    return Attrs().UseShardFunc(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset successively reduces `f` over the elements of `input_dataset`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ScanDataset {
 public:
  /// Optional attribute setters for ScanDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs PreserveCardinality(bool x) {
      Attrs ret = *this;
      ret.preserve_cardinality_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseDefaultDevice(bool x) {
      Attrs ret = *this;
      ret.use_default_device_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool preserve_cardinality_ = false;
    bool use_default_device_ = true;
    StringPiece metadata_ = "";
  };
  ScanDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::InputList initial_state,
            ::tensorflow::InputList other_arguments, const NameAttrList& f,
            const DataTypeSlice& output_types, const
            gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ScanDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::InputList initial_state,
            ::tensorflow::InputList other_arguments, const NameAttrList& f,
            const DataTypeSlice& output_types, const
            gtl::ArraySlice<PartialTensorShape>& output_shapes, const
            ScanDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs PreserveCardinality(bool x) {
    return Attrs().PreserveCardinality(x);
  }
  static Attrs UseDefaultDevice(bool x) {
    return Attrs().UseDefaultDevice(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class SetStatsAggregatorDataset {
 public:
  SetStatsAggregatorDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, ::tensorflow::Input stats_aggregator,
                          ::tensorflow::Input tag, ::tensorflow::Input
                          counter_prefix, const DataTypeSlice& output_types,
                          const gtl::ArraySlice<PartialTensorShape>&
                          output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class SleepDataset {
 public:
  SleepDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, ::tensorflow::Input sleep_microseconds, const
             DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that passes a sliding window over `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * window_size: A scalar representing the number of elements in the
/// sliding window.
/// * window_shift: A scalar representing the steps moving the sliding window
/// forward in one iteration. It must be positive.
/// * window_stride: A scalar representing the stride of the input elements of the sliding window.
/// It must be positive.
///
/// Returns:
/// * `Output`: The handle tensor.
class SlidingWindowDataset {
 public:
  /// Optional attribute setters for SlidingWindowDataset
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs DropRemainder(bool x) {
      Attrs ret = *this;
      ret.drop_remainder_ = x;
      return ret;
    }

    bool drop_remainder_ = true;
  };
  SlidingWindowDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input window_size,
                     ::tensorflow::Input window_shift, ::tensorflow::Input
                     window_stride, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  SlidingWindowDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input window_size,
                     ::tensorflow::Input window_shift, ::tensorflow::Input
                     window_stride, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     SlidingWindowDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs DropRemainder(bool x) {
    return Attrs().DropRemainder(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class SnapshotChunkDataset {
 public:
  /// Optional attribute setters for SnapshotChunkDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Compression(StringPiece x) {
      Attrs ret = *this;
      ret.compression_ = x;
      return ret;
    }

    StringPiece compression_ = "";
  };
  SnapshotChunkDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     chunk_file, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  SnapshotChunkDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     chunk_file, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     SnapshotChunkDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Compression(StringPiece x) {
    return Attrs().Compression(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that will write to / read from a snapshot.
///
/// This dataset attempts to determine whether a valid snapshot exists at the
/// `snapshot_path`, and reads from the snapshot in lieu of using `input_dataset`.
/// If not, it will run the preprocessing pipeline as usual, and write out a
/// snapshot of the data processed for future use.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * path: The path we should write snapshots to / read snapshots from.
///
/// Returns:
/// * `Output`: The handle tensor.
class SnapshotDataset {
 public:
  /// Optional attribute setters for SnapshotDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Compression(StringPiece x) {
      Attrs ret = *this;
      ret.compression_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ReaderPathPrefix(StringPiece x) {
      Attrs ret = *this;
      ret.reader_path_prefix_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs WriterPathPrefix(StringPiece x) {
      Attrs ret = *this;
      ret.writer_path_prefix_ = x;
      return ret;
    }

    /// Defaults to 10737418240
    TF_MUST_USE_RESULT Attrs ShardSizeBytes(int64 x) {
      Attrs ret = *this;
      ret.shard_size_bytes_ = x;
      return ret;
    }

    /// Defaults to 86400
    TF_MUST_USE_RESULT Attrs PendingSnapshotExpirySeconds(int64 x) {
      Attrs ret = *this;
      ret.pending_snapshot_expiry_seconds_ = x;
      return ret;
    }

    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs NumReaderThreads(int64 x) {
      Attrs ret = *this;
      ret.num_reader_threads_ = x;
      return ret;
    }

    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs ReaderBufferSize(int64 x) {
      Attrs ret = *this;
      ret.reader_buffer_size_ = x;
      return ret;
    }

    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs NumWriterThreads(int64 x) {
      Attrs ret = *this;
      ret.num_writer_threads_ = x;
      return ret;
    }

    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs WriterBufferSize(int64 x) {
      Attrs ret = *this;
      ret.writer_buffer_size_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ShuffleOnRead(bool x) {
      Attrs ret = *this;
      ret.shuffle_on_read_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    /// Defaults to "auto"
    TF_MUST_USE_RESULT Attrs Mode(StringPiece x) {
      Attrs ret = *this;
      ret.mode_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SnapshotName(StringPiece x) {
      Attrs ret = *this;
      ret.snapshot_name_ = x;
      return ret;
    }

    StringPiece compression_ = "";
    StringPiece reader_path_prefix_ = "";
    StringPiece writer_path_prefix_ = "";
    int64 shard_size_bytes_ = 10737418240;
    int64 pending_snapshot_expiry_seconds_ = 86400;
    int64 num_reader_threads_ = 1;
    int64 reader_buffer_size_ = 1;
    int64 num_writer_threads_ = 1;
    int64 writer_buffer_size_ = 1;
    bool shuffle_on_read_ = false;
    int64 seed_ = 0;
    int64 seed2_ = 0;
    StringPiece mode_ = "auto";
    StringPiece snapshot_name_ = "";
  };
  SnapshotDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input path, const DataTypeSlice&
                output_types, const gtl::ArraySlice<PartialTensorShape>&
                output_shapes);
  SnapshotDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input path, const DataTypeSlice&
                output_types, const gtl::ArraySlice<PartialTensorShape>&
                output_shapes, const SnapshotDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Compression(StringPiece x) {
    return Attrs().Compression(x);
  }
  static Attrs ReaderPathPrefix(StringPiece x) {
    return Attrs().ReaderPathPrefix(x);
  }
  static Attrs WriterPathPrefix(StringPiece x) {
    return Attrs().WriterPathPrefix(x);
  }
  static Attrs ShardSizeBytes(int64 x) {
    return Attrs().ShardSizeBytes(x);
  }
  static Attrs PendingSnapshotExpirySeconds(int64 x) {
    return Attrs().PendingSnapshotExpirySeconds(x);
  }
  static Attrs NumReaderThreads(int64 x) {
    return Attrs().NumReaderThreads(x);
  }
  static Attrs ReaderBufferSize(int64 x) {
    return Attrs().ReaderBufferSize(x);
  }
  static Attrs NumWriterThreads(int64 x) {
    return Attrs().NumWriterThreads(x);
  }
  static Attrs WriterBufferSize(int64 x) {
    return Attrs().WriterBufferSize(x);
  }
  static Attrs ShuffleOnRead(bool x) {
    return Attrs().ShuffleOnRead(x);
  }
  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }
  static Attrs Mode(StringPiece x) {
    return Attrs().Mode(x);
  }
  static Attrs SnapshotName(StringPiece x) {
    return Attrs().SnapshotName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class SnapshotDatasetReader {
 public:
  /// Optional attribute setters for SnapshotDatasetReader
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Compression(StringPiece x) {
      Attrs ret = *this;
      ret.compression_ = x;
      return ret;
    }

    StringPiece compression_ = "";
  };
  SnapshotDatasetReader(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      shard_dir, ::tensorflow::Input start_index, const
                      DataTypeSlice& output_types, const
                      gtl::ArraySlice<PartialTensorShape>& output_shapes, int64
                      version);
  SnapshotDatasetReader(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      shard_dir, ::tensorflow::Input start_index, const
                      DataTypeSlice& output_types, const
                      gtl::ArraySlice<PartialTensorShape>& output_shapes, int64
                      version, const SnapshotDatasetReader::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Compression(StringPiece x) {
    return Attrs().Compression(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that will write to / read from a snapshot.
///
/// This dataset attempts to determine whether a valid snapshot exists at the
/// `snapshot_path`, and reads from the snapshot in lieu of using `input_dataset`.
/// If not, it will run the preprocessing pipeline as usual, and write out a
/// snapshot of the data processed for future use.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * path: The path we should write snapshots to / read snapshots from.
/// * reader_func: Optional. A function to control how to read data from snapshot shards.
/// * shard_func: Optional. A function to control how to shard data when writing a snapshot.
///
/// Optional attributes (see `Attrs`):
/// * compression: The type of compression to be applied to the saved snapshot files.
///
/// Returns:
/// * `Output`: The handle tensor.
class SnapshotDatasetV2 {
 public:
  /// Optional attribute setters for SnapshotDatasetV2
  struct Attrs {
    /// The type of compression to be applied to the saved snapshot files.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Compression(StringPiece x) {
      Attrs ret = *this;
      ret.compression_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ReaderPrefix(StringPiece x) {
      Attrs ret = *this;
      ret.reader_prefix_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs WriterPrefix(StringPiece x) {
      Attrs ret = *this;
      ret.writer_prefix_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs HashValid(bool x) {
      Attrs ret = *this;
      ret.hash_valid_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Hash(int64 x) {
      Attrs ret = *this;
      ret.hash_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece compression_ = "";
    StringPiece reader_prefix_ = "";
    StringPiece writer_prefix_ = "";
    bool hash_valid_ = false;
    int64 hash_ = 0;
    StringPiece metadata_ = "";
  };
  SnapshotDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input path,
                  ::tensorflow::InputList reader_func_other_args,
                  ::tensorflow::InputList shard_func_other_args, const
                  DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                  NameAttrList& reader_func, const NameAttrList& shard_func);
  SnapshotDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input path,
                  ::tensorflow::InputList reader_func_other_args,
                  ::tensorflow::InputList shard_func_other_args, const
                  DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                  NameAttrList& reader_func, const NameAttrList& shard_func,
                  const SnapshotDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Compression(StringPiece x) {
    return Attrs().Compression(x);
  }
  static Attrs ReaderPrefix(StringPiece x) {
    return Attrs().ReaderPrefix(x);
  }
  static Attrs WriterPrefix(StringPiece x) {
    return Attrs().WriterPrefix(x);
  }
  static Attrs HashValid(bool x) {
    return Attrs().HashValid(x);
  }
  static Attrs Hash(int64 x) {
    return Attrs().Hash(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class SnapshotNestedDatasetReader {
 public:
  SnapshotNestedDatasetReader(const ::tensorflow::Scope& scope,
                            ::tensorflow::InputList inputs, const
                            DataTypeSlice& output_types, const
                            gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that executes a SQL query and emits rows of the result set.
///
/// Args:
/// * scope: A Scope object
/// * driver_name: The database type. Currently, the only supported type is 'sqlite'.
/// * data_source_name: A connection string to connect to the database.
/// * query: A SQL query to execute.
///
/// Returns:
/// * `Output`: The handle tensor.
class SqlDataset {
 public:
  SqlDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input driver_name,
           ::tensorflow::Input data_source_name, ::tensorflow::Input query,
           const DataTypeSlice& output_types, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a statistics manager resource.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class StatsAggregatorHandle {
 public:
  /// Optional attribute setters for StatsAggregatorHandle
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  StatsAggregatorHandle(const ::tensorflow::Scope& scope);
  StatsAggregatorHandle(const ::tensorflow::Scope& scope, const
                      StatsAggregatorHandle::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class StatsAggregatorHandleV2 {
 public:
  /// Optional attribute setters for StatsAggregatorHandleV2
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  StatsAggregatorHandleV2(const ::tensorflow::Scope& scope);
  StatsAggregatorHandleV2(const ::tensorflow::Scope& scope, const
                        StatsAggregatorHandleV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Set a summary_writer_interface to record statistics using given stats_aggregator.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class StatsAggregatorSetSummaryWriter {
 public:
  StatsAggregatorSetSummaryWriter(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input stats_aggregator,
                                ::tensorflow::Input summary);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Produces a summary of any statistics recorded by the given statistics manager.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The summary tensor.
class StatsAggregatorSummary {
 public:
  StatsAggregatorSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       iterator);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Creates a dataset that stops iteration when predicate` is false.
///
/// The `predicate` function must return a scalar boolean and accept the
/// following arguments:
///
/// * One tensor for each component of an element of `input_dataset`.
/// * One tensor for each value in `other_arguments`.
///
/// Args:
/// * scope: A Scope object
/// * other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `predicate`.
/// * predicate: A function returning a scalar boolean.
///
/// Returns:
/// * `Output`: The handle tensor.
class TakeWhileDataset {
 public:
  /// Optional attribute setters for TakeWhileDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  TakeWhileDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::InputList other_arguments, const
                 NameAttrList& predicate, const DataTypeSlice& output_types,
                 const gtl::ArraySlice<PartialTensorShape>& output_shapes);
  TakeWhileDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::InputList other_arguments, const
                 NameAttrList& predicate, const DataTypeSlice& output_types,
                 const gtl::ArraySlice<PartialTensorShape>& output_shapes,
                 const TakeWhileDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that uses a custom thread pool to compute `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * thread_pool: A resource produced by the ThreadPoolHandle op.
///
/// Returns:
/// * `Output`: The handle tensor.
class ThreadPoolDataset {
 public:
  ThreadPoolDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input thread_pool, const
                  DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that uses a custom thread pool to compute `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * num_threads: The number of threads in the thread pool.
/// * display_name: A human-readable name for the threads that may be visible in some
/// visualizations.
/// threadpool.
///
/// Optional attributes (see `Attrs`):
/// * max_intra_op_parallelism: The maximum degree of parallelism to use within operations that execute on this
/// threadpool.
///
/// Returns:
/// * `Output`: A resource that can be consumed by one or more ExperimentalThreadPoolDataset
/// ops.
class ThreadPoolHandle {
 public:
  /// Optional attribute setters for ThreadPoolHandle
  struct Attrs {
    /// The maximum degree of parallelism to use within operations that execute on this
    /// threadpool.
    ///
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs MaxIntraOpParallelism(int64 x) {
      Attrs ret = *this;
      ret.max_intra_op_parallelism_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 max_intra_op_parallelism_ = 1;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  ThreadPoolHandle(const ::tensorflow::Scope& scope, int64 num_threads,
                 StringPiece display_name);
  ThreadPoolHandle(const ::tensorflow::Scope& scope, int64 num_threads,
                 StringPiece display_name, const ThreadPoolHandle::Attrs&
                 attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs MaxIntraOpParallelism(int64 x) {
    return Attrs().MaxIntraOpParallelism(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// A dataset that splits the elements of its input into multiple elements.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class UnbatchDataset {
 public:
  /// Optional attribute setters for UnbatchDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  UnbatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, const DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes);
  UnbatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, const DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes, const
               UnbatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Uncompresses a compressed dataset element.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The components tensor.
class UncompressElement {
 public:
  UncompressElement(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  compressed, const DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  Operation operation;
  ::tensorflow::OutputList components;
};

/// Creates a dataset that contains the unique elements of `input_dataset`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class UniqueDataset {
 public:
  /// Optional attribute setters for UniqueDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  UniqueDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes);
  UniqueDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes, const
              UniqueDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class WeightedFlatMapDataset {
 public:
  /// Optional attribute setters for WeightedFlatMapDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  WeightedFlatMapDataset(const ::tensorflow::Scope& scope,
                       ::tensorflow::InputList input_datasets,
                       ::tensorflow::InputList weights, const DataTypeSlice&
                       output_types, const gtl::ArraySlice<PartialTensorShape>&
                       output_shapes);
  WeightedFlatMapDataset(const ::tensorflow::Scope& scope,
                       ::tensorflow::InputList input_datasets,
                       ::tensorflow::InputList weights, const DataTypeSlice&
                       output_types, const gtl::ArraySlice<PartialTensorShape>&
                       output_shapes, const WeightedFlatMapDataset::Attrs&
                       attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_EXPERIMENTAL_DATASET_OPS_INTERNAL_H_
