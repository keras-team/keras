// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_DATASET_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_DATASET_OPS_INTERNAL_H_

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

/// @defgroup dataset_ops_internal Dataset Ops Internal
/// @{

/// A container for an iterator resource.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` handle: A handle to the iterator that can be passed to a "MakeIterator" or
/// "IteratorGetNext" op. In contrast to Iterator, AnonymousIterator prevents
/// resource sharing by name, and does not keep a reference to the resource
/// container.
/// * `Output` deleter: A variant deleter that should be passed into the op that deletes the iterator.
class AnonymousIteratorV2 {
 public:
  AnonymousIteratorV2(const ::tensorflow::Scope& scope, const DataTypeSlice&
                    output_types, const gtl::ArraySlice<PartialTensorShape>&
                    output_shapes);

  Operation operation;
  ::tensorflow::Output handle;
  ::tensorflow::Output deleter;
};

/// A container for an iterator resource.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: A handle to the iterator that can be passed to a "MakeIterator" or
/// "IteratorGetNext" op. In contrast to Iterator, AnonymousIterator prevents
/// resource sharing by name, and does not keep a reference to the resource
/// container.
class AnonymousIteratorV3 {
 public:
  AnonymousIteratorV3(const ::tensorflow::Scope& scope, const DataTypeSlice&
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
/// * `Output` handle
/// * `Output` deleter
class AnonymousMemoryCache {
 public:
  AnonymousMemoryCache(const ::tensorflow::Scope& scope);

  Operation operation;
  ::tensorflow::Output handle;
  ::tensorflow::Output deleter;
};

/// A container for a multi device iterator resource.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` handle: A handle to a multi device iterator that can be passed to a
/// "MultiDeviceIteratorGetNextFromShard" op. In contrast to MultiDeviceIterator,
/// AnonymousIterator prevents resource sharing by name, and does not keep a
/// reference to the resource container.
/// * `Output` deleter: A variant deleter that should be passed into the op that deletes the iterator.
class AnonymousMultiDeviceIterator {
 public:
  AnonymousMultiDeviceIterator(const ::tensorflow::Scope& scope, const
                             gtl::ArraySlice<::tensorflow::tstring>& devices,
                             const DataTypeSlice& output_types, const
                             gtl::ArraySlice<PartialTensorShape>&
                             output_shapes);

  Operation operation;
  ::tensorflow::Output handle;
  ::tensorflow::Output deleter;
};

/// A container for a multi device iterator resource.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: A handle to a multi device iterator that can be passed to a
/// "MultiDeviceIteratorGetNextFromShard" op. In contrast to MultiDeviceIterator,
/// AnonymousIterator prevents resource sharing by name, and does not keep a
/// reference to the resource container.
class AnonymousMultiDeviceIteratorV3 {
 public:
  AnonymousMultiDeviceIteratorV3(const ::tensorflow::Scope& scope, const
                               gtl::ArraySlice<::tensorflow::tstring>& devices,
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
/// * `Output` handle
/// * `Output` deleter
class AnonymousRandomSeedGenerator {
 public:
  AnonymousRandomSeedGenerator(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input seed, ::tensorflow::Input
                             seed2);

  Operation operation;
  ::tensorflow::Output handle;
  ::tensorflow::Output deleter;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` handle
/// * `Output` deleter
class AnonymousSeedGenerator {
 public:
  AnonymousSeedGenerator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       seed, ::tensorflow::Input seed2, ::tensorflow::Input
                       reshuffle);

  Operation operation;
  ::tensorflow::Output handle;
  ::tensorflow::Output deleter;
};

/// Creates a dataset that batches `batch_size` elements from `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch.
///
/// Returns:
/// * `Output`: The handle tensor.
class BatchDataset {
 public:
  /// Optional attribute setters for BatchDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  BatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, ::tensorflow::Input batch_size, const
             DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes);
  BatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, ::tensorflow::Input batch_size, const
             DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes, const
             BatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that batches `batch_size` elements from `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * batch_size: A scalar representing the number of elements to accumulate in a batch.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
///
/// Returns:
/// * `Output`: The handle tensor.
class BatchDatasetV2 {
 public:
  /// Optional attribute setters for BatchDatasetV2
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ParallelCopy(bool x) {
      Attrs ret = *this;
      ret.parallel_copy_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool parallel_copy_ = false;
    StringPiece metadata_ = "";
  };
  BatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input batch_size,
               ::tensorflow::Input drop_remainder, const DataTypeSlice&
               output_types, const gtl::ArraySlice<PartialTensorShape>&
               output_shapes);
  BatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input batch_size,
               ::tensorflow::Input drop_remainder, const DataTypeSlice&
               output_types, const gtl::ArraySlice<PartialTensorShape>&
               output_shapes, const BatchDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs ParallelCopy(bool x) {
    return Attrs().ParallelCopy(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that caches elements from `input_dataset`.
///
/// A CacheDataset will iterate over the input_dataset, and store tensors. If the
/// cache already exists, the cache will be used. If the cache is inappropriate
/// (e.g. cannot be opened, contains tensors of the wrong shape / size), an error
/// will the returned when used.
///
/// Args:
/// * scope: A Scope object
/// * filename: A path on the filesystem where we should cache the dataset. Note: this
/// will be a directory.
///
/// Returns:
/// * `Output`: The handle tensor.
class CacheDataset {
 public:
  /// Optional attribute setters for CacheDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  CacheDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, ::tensorflow::Input filename, const DataTypeSlice&
             output_types, const gtl::ArraySlice<PartialTensorShape>&
             output_shapes);
  CacheDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, ::tensorflow::Input filename, const DataTypeSlice&
             output_types, const gtl::ArraySlice<PartialTensorShape>&
             output_shapes, const CacheDataset::Attrs& attrs);
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
class CacheDatasetV2 {
 public:
  /// Optional attribute setters for CacheDatasetV2
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  CacheDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input filename, ::tensorflow::Input
               cache, const DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes);
  CacheDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input filename, ::tensorflow::Input
               cache, const DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes, const
               CacheDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that concatenates `input_dataset` with `another_dataset`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ConcatenateDataset {
 public:
  /// Optional attribute setters for ConcatenateDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  ConcatenateDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::Input another_dataset, const
                   DataTypeSlice& output_types, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ConcatenateDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::Input another_dataset, const
                   DataTypeSlice& output_types, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                   ConcatenateDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

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
class DatasetCardinality {
 public:
  /// Optional attribute setters for DatasetCardinality
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CardinalityOptions(StringPiece x) {
      Attrs ret = *this;
      ret.cardinality_options_ = x;
      return ret;
    }

    StringPiece cardinality_options_ = "";
  };
  DatasetCardinality(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset);
  DatasetCardinality(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, const DatasetCardinality::Attrs& attrs);
  operator ::tensorflow::Output() const { return cardinality; }
  operator ::tensorflow::Input() const { return cardinality; }
  ::tensorflow::Node* node() const { return cardinality.node(); }

  static Attrs CardinalityOptions(StringPiece x) {
    return Attrs().CardinalityOptions(x);
  }

  Operation operation;
  ::tensorflow::Output cardinality;
};

/// Returns the fingerprint of `input_dataset`.
///
/// Returns the fingerprint of `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to return fingerprint for.
///
/// Returns:
/// * `Output`: The fingerprint of `input_dataset` in `uint64`
class DatasetFingerprint {
 public:
  DatasetFingerprint(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset);
  operator ::tensorflow::Output() const { return fingerprint; }
  operator ::tensorflow::Input() const { return fingerprint; }
  ::tensorflow::Node* node() const { return fingerprint.node(); }

  Operation operation;
  ::tensorflow::Output fingerprint;
};

/// Returns a serialized GraphDef representing `input_dataset`.
///
/// Returns a graph representation for `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to return the graph representation for.
///
/// Returns:
/// * `Output`: The graph representation of the dataset (as serialized GraphDef).
class DatasetToGraph {
 public:
  /// Optional attribute setters for DatasetToGraph
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs StatefulWhitelist(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.stateful_whitelist_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AllowStateful(bool x) {
      Attrs ret = *this;
      ret.allow_stateful_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs StripDeviceAssignment(bool x) {
      Attrs ret = *this;
      ret.strip_device_assignment_ = x;
      return ret;
    }

    gtl::ArraySlice<::tensorflow::tstring> stateful_whitelist_ = {};
    bool allow_stateful_ = false;
    bool strip_device_assignment_ = false;
  };
  DatasetToGraph(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset);
  DatasetToGraph(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, const DatasetToGraph::Attrs& attrs);
  operator ::tensorflow::Output() const { return graph; }
  operator ::tensorflow::Input() const { return graph; }
  ::tensorflow::Node* node() const { return graph.node(); }

  static Attrs StatefulWhitelist(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().StatefulWhitelist(x);
  }
  static Attrs AllowStateful(bool x) {
    return Attrs().AllowStateful(x);
  }
  static Attrs StripDeviceAssignment(bool x) {
    return Attrs().StripDeviceAssignment(x);
  }

  Operation operation;
  ::tensorflow::Output graph;
};

/// Returns a serialized GraphDef representing `input_dataset`.
///
/// Returns a graph representation for `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to return the graph representation for.
///
/// Returns:
/// * `Output`: The graph representation of the dataset (as serialized GraphDef).
class DatasetToGraphV2 {
 public:
  /// Optional attribute setters for DatasetToGraphV2
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs ExternalStatePolicy(int64 x) {
      Attrs ret = *this;
      ret.external_state_policy_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs StripDeviceAssignment(bool x) {
      Attrs ret = *this;
      ret.strip_device_assignment_ = x;
      return ret;
    }

    int64 external_state_policy_ = 0;
    bool strip_device_assignment_ = false;
  };
  DatasetToGraphV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset);
  DatasetToGraphV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, const DatasetToGraphV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return graph; }
  operator ::tensorflow::Input() const { return graph; }
  ::tensorflow::Node* node() const { return graph.node(); }

  static Attrs ExternalStatePolicy(int64 x) {
    return Attrs().ExternalStatePolicy(x);
  }
  static Attrs StripDeviceAssignment(bool x) {
    return Attrs().StripDeviceAssignment(x);
  }

  Operation operation;
  ::tensorflow::Output graph;
};

/// Outputs the single element from the given dataset.
///
/// Args:
/// * scope: A Scope object
/// * dataset: A handle to a dataset that contains a single element.
///
/// Returns:
/// * `OutputList`: The components of the single element of `input`.
class DatasetToSingleElement {
 public:
  /// Optional attribute setters for DatasetToSingleElement
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  DatasetToSingleElement(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       dataset, const DataTypeSlice& output_types, const
                       gtl::ArraySlice<PartialTensorShape>& output_shapes);
  DatasetToSingleElement(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       dataset, const DataTypeSlice& output_types, const
                       gtl::ArraySlice<PartialTensorShape>& output_shapes,
                       const DatasetToSingleElement::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::OutputList components;
};

/// A container for an iterator resource.
///
/// Args:
/// * scope: A Scope object
/// * handle: A handle to the iterator to delete.
/// * deleter: A variant deleter.
///
/// Returns:
/// * the created `Operation`
class DeleteIterator {
 public:
  DeleteIterator(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
               ::tensorflow::Input deleter);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class DeleteMemoryCache {
 public:
  DeleteMemoryCache(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                  ::tensorflow::Input deleter);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// A container for an iterator resource.
///
/// Args:
/// * scope: A Scope object
/// * multi_device_iterator: A handle to the multi device iterator to delete.
/// * iterators: A list of iterator handles (unused). This is added so that automatic control dependencies get added during function tracing that ensure this op runs after all the dependent iterators are deleted.
/// * deleter: A variant deleter.
///
/// Returns:
/// * the created `Operation`
class DeleteMultiDeviceIterator {
 public:
  DeleteMultiDeviceIterator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          multi_device_iterator, ::tensorflow::InputList
                          iterators, ::tensorflow::Input deleter);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class DeleteRandomSeedGenerator {
 public:
  DeleteRandomSeedGenerator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          handle, ::tensorflow::Input deleter);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class DeleteSeedGenerator {
 public:
  DeleteSeedGenerator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    handle, ::tensorflow::Input deleter);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class DummyMemoryCache {
 public:
  DummyMemoryCache(const ::tensorflow::Scope& scope);
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
class DummySeedGenerator {
 public:
  DummySeedGenerator(const ::tensorflow::Scope& scope);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset containing elements of first component of `input_dataset` having true in the last component.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class FilterByLastComponentDataset {
 public:
  FilterByLastComponentDataset(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input_dataset, const
                             DataTypeSlice& output_types, const
                             gtl::ArraySlice<PartialTensorShape>&
                             output_shapes);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Creates a dataset containing elements of `input_dataset` matching `predicate`.
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
class FilterDataset {
 public:
  /// Optional attribute setters for FilterDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  FilterDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::InputList other_arguments, const
              NameAttrList& predicate, const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes);
  FilterDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::InputList other_arguments, const
              NameAttrList& predicate, const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes, const
              FilterDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset by applying `tf.data.Options` to `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
///
/// Returns:
/// * `Output`: The handle tensor.
class FinalizeDataset {
 public:
  /// Optional attribute setters for FinalizeDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs HasCapturedRef(bool x) {
      Attrs ret = *this;
      ret.has_captured_ref_ = x;
      return ret;
    }

    bool has_captured_ref_ = false;
  };
  FinalizeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, const DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes);
  FinalizeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, const DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                FinalizeDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs HasCapturedRef(bool x) {
    return Attrs().HasCapturedRef(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that emits the records from one or more binary files.
///
/// Args:
/// * scope: A Scope object
/// * filenames: A scalar or a vector containing the name(s) of the file(s) to be
/// read.
/// * header_bytes: A scalar representing the number of bytes to skip at the
/// beginning of a file.
/// * record_bytes: A scalar representing the number of bytes in each record.
/// * footer_bytes: A scalar representing the number of bytes to skip at the end
/// of a file.
/// * buffer_size: A scalar representing the number of bytes to buffer. Must be > 0.
///
/// Returns:
/// * `Output`: The handle tensor.
class FixedLengthRecordDataset {
 public:
  /// Optional attribute setters for FixedLengthRecordDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  FixedLengthRecordDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         filenames, ::tensorflow::Input header_bytes,
                         ::tensorflow::Input record_bytes, ::tensorflow::Input
                         footer_bytes, ::tensorflow::Input buffer_size);
  FixedLengthRecordDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         filenames, ::tensorflow::Input header_bytes,
                         ::tensorflow::Input record_bytes, ::tensorflow::Input
                         footer_bytes, ::tensorflow::Input buffer_size, const
                         FixedLengthRecordDataset::Attrs& attrs);
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
class FixedLengthRecordDatasetV2 {
 public:
  /// Optional attribute setters for FixedLengthRecordDatasetV2
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  FixedLengthRecordDatasetV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input filenames, ::tensorflow::Input
                           header_bytes, ::tensorflow::Input record_bytes,
                           ::tensorflow::Input footer_bytes,
                           ::tensorflow::Input buffer_size, ::tensorflow::Input
                           compression_type);
  FixedLengthRecordDatasetV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input filenames, ::tensorflow::Input
                           header_bytes, ::tensorflow::Input record_bytes,
                           ::tensorflow::Input footer_bytes,
                           ::tensorflow::Input buffer_size, ::tensorflow::Input
                           compression_type, const
                           FixedLengthRecordDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
/// Dataset variant, and FlatMapDataset will flatten successive results
/// into a single Dataset.
///
/// Args:
/// * scope: A Scope object
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Returns:
/// * `Output`: The handle tensor.
class FlatMapDataset {
 public:
  /// Optional attribute setters for FlatMapDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  FlatMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::InputList other_arguments, const
               NameAttrList& f, const DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes);
  FlatMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::InputList other_arguments, const
               NameAttrList& f, const DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes, const
               FlatMapDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that invokes a function to generate elements.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class GeneratorDataset {
 public:
  /// Optional attribute setters for GeneratorDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  GeneratorDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                 init_func_other_args, ::tensorflow::InputList
                 next_func_other_args, ::tensorflow::InputList
                 finalize_func_other_args, const NameAttrList& init_func, const
                 NameAttrList& next_func, const NameAttrList& finalize_func,
                 const DataTypeSlice& output_types, const
                 gtl::ArraySlice<PartialTensorShape>& output_shapes);
  GeneratorDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                 init_func_other_args, ::tensorflow::InputList
                 next_func_other_args, ::tensorflow::InputList
                 finalize_func_other_args, const NameAttrList& init_func, const
                 NameAttrList& next_func, const NameAttrList& finalize_func,
                 const DataTypeSlice& output_types, const
                 gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                 GeneratorDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Returns the `tf.data.Options` attached to `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
///
/// Returns:
/// * `Output`: The serialized_options tensor.
class GetOptions {
 public:
  GetOptions(const ::tensorflow::Scope& scope, ::tensorflow::Input input_dataset);
  operator ::tensorflow::Output() const { return serialized_options; }
  operator ::tensorflow::Input() const { return serialized_options; }
  ::tensorflow::Node* node() const { return serialized_options.node(); }

  Operation operation;
  ::tensorflow::Output serialized_options;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// Unlike MapDataset, the `f` in InterleaveDataset is expected to return
/// a Dataset variant, and InterleaveDataset will flatten successive
/// results into a single Dataset. Unlike FlatMapDataset,
/// InterleaveDataset will interleave sequences of up to `block_length`
/// consecutive elements from `cycle_length` input elements.
///
/// Args:
/// * scope: A Scope object
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Returns:
/// * `Output`: The handle tensor.
class InterleaveDataset {
 public:
  /// Optional attribute setters for InterleaveDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  InterleaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::InputList other_arguments,
                  ::tensorflow::Input cycle_length, ::tensorflow::Input
                  block_length, const NameAttrList& f, const DataTypeSlice&
                  output_types, const gtl::ArraySlice<PartialTensorShape>&
                  output_shapes);
  InterleaveDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::InputList other_arguments,
                  ::tensorflow::Input cycle_length, ::tensorflow::Input
                  block_length, const NameAttrList& f, const DataTypeSlice&
                  output_types, const gtl::ArraySlice<PartialTensorShape>&
                  output_shapes, const InterleaveDataset::Attrs& attrs);
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
/// * `Output`: The resource_handle tensor.
class IteratorFromStringHandleV2 {
 public:
  /// Optional attribute setters for IteratorFromStringHandleV2
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.output_types_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    DataTypeSlice output_types_ = {};
    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
  };
  IteratorFromStringHandleV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input string_handle);
  IteratorFromStringHandleV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input string_handle, const
                           IteratorFromStringHandleV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return resource_handle; }
  operator ::tensorflow::Input() const { return resource_handle; }
  ::tensorflow::Node* node() const { return resource_handle.node(); }

  static Attrs OutputTypes(const DataTypeSlice& x) {
    return Attrs().OutputTypes(x);
  }
  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }

  Operation operation;
  ::tensorflow::Output resource_handle;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class IteratorV2 {
 public:
  IteratorV2(const ::tensorflow::Scope& scope, StringPiece shared_name,
           StringPiece container, const DataTypeSlice& output_types, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

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
class MapDataset {
 public:
  /// Optional attribute setters for MapDataset
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

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool use_inter_op_parallelism_ = true;
    bool preserve_cardinality_ = false;
    bool force_synchronous_ = false;
    StringPiece metadata_ = "";
  };
  MapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input input_dataset,
           ::tensorflow::InputList other_arguments, const NameAttrList& f,
           const DataTypeSlice& output_types, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  MapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input input_dataset,
           ::tensorflow::InputList other_arguments, const NameAttrList& f,
           const DataTypeSlice& output_types, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes, const
           MapDataset::Attrs& attrs);
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
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

///   Maps a function on the list of tensors unpacked from arguments on dimension 0.
///   The function given by `f` is assumed to be stateless, and is executed
///   concurrently on all the slices; up to batch_size (i.e. the size of the 0th
///   dimension of each argument) functions will be scheduled at once.
///
///   The `max_intra_op_parallelism` attr, which defaults to 1, can be used to
///   limit the intra op parallelism. To limit inter-op parallelism, a user can
///   set a private threadpool on the dataset using `tf.data.Options`'s
///   `ThreadingOptions`.
///
///   Note that this op is not exposed to users directly, but is invoked in tf.data
///   rewrites.
///
/// Args:
/// * scope: A Scope object
/// * arguments:     A list of tensors whose types are `Targuments`, corresponding to the inputs
///     the function should be mapped over.
/// * captured_inputs:     A list of tensors whose types are `Tcaptured`, corresponding to the captured
///     inputs of the defun.
/// * output_types: A list of types.
/// * output_shapes: A list of shapes.
///
/// Returns:
/// * `OutputList`:     A list of output tensors whose types are `output_types` and whose dimensions
///     0 are the same as the dimensions 0 of the tensors in `arguments`, and whose
///     remaining dimensions correspond to those in `output_shapes`.
class MapDefun {
 public:
  /// Optional attribute setters for MapDefun
  struct Attrs {
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs MaxIntraOpParallelism(int64 x) {
      Attrs ret = *this;
      ret.max_intra_op_parallelism_ = x;
      return ret;
    }

    int64 max_intra_op_parallelism_ = 1;
  };
  MapDefun(const ::tensorflow::Scope& scope, ::tensorflow::InputList arguments,
         ::tensorflow::InputList captured_inputs, const DataTypeSlice&
         output_types, const gtl::ArraySlice<PartialTensorShape>&
         output_shapes, const NameAttrList& f);
  MapDefun(const ::tensorflow::Scope& scope, ::tensorflow::InputList arguments,
         ::tensorflow::InputList captured_inputs, const DataTypeSlice&
         output_types, const gtl::ArraySlice<PartialTensorShape>&
         output_shapes, const NameAttrList& f, const MapDefun::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs MaxIntraOpParallelism(int64 x) {
    return Attrs().MaxIntraOpParallelism(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// Identity transformation that models performance.
///
/// Identity transformation that models performance.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
///
/// Returns:
/// * `Output`: The handle tensor.
class ModelDataset {
 public:
  /// Optional attribute setters for ModelDataset
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Algorithm(int64 x) {
      Attrs ret = *this;
      ret.algorithm_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs CpuBudget(int64 x) {
      Attrs ret = *this;
      ret.cpu_budget_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs RamBudget(int64 x) {
      Attrs ret = *this;
      ret.ram_budget_ = x;
      return ret;
    }

    int64 algorithm_ = 0;
    int64 cpu_budget_ = 0;
    int64 ram_budget_ = 0;
  };
  ModelDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, const DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ModelDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, const DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes, const
             ModelDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Algorithm(int64 x) {
    return Attrs().Algorithm(x);
  }
  static Attrs CpuBudget(int64 x) {
    return Attrs().CpuBudget(x);
  }
  static Attrs RamBudget(int64 x) {
    return Attrs().RamBudget(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a MultiDeviceIterator resource.
///
/// Args:
/// * scope: A Scope object
/// * devices: A list of devices the iterator works across.
/// * shared_name: If non-empty, this resource will be shared under the given name
/// across multiple sessions.
/// * container: If non-empty, this resource is placed in the given container.
/// Otherwise, a default container is used.
/// * output_types: The type list for the return values.
/// * output_shapes: The list of shapes being produced.
///
/// Returns:
/// * `Output`: Handle to the resource created.
class MultiDeviceIterator {
 public:
  MultiDeviceIterator(const ::tensorflow::Scope& scope, const
                    gtl::ArraySlice<::tensorflow::tstring>& devices,
                    StringPiece shared_name, StringPiece container, const
                    DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Generates a MultiDeviceIterator resource from its provided string handle.
///
/// Args:
/// * scope: A Scope object
/// * string_handle: String representing the resource.
///
/// Optional attributes (see `Attrs`):
/// * output_types: The type list for the return values.
/// * output_shapes: The list of shapes being produced.
///
/// Returns:
/// * `Output`: A MultiDeviceIterator resource.
class MultiDeviceIteratorFromStringHandle {
 public:
  /// Optional attribute setters for MultiDeviceIteratorFromStringHandle
  struct Attrs {
    /// The type list for the return values.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.output_types_ = x;
      return ret;
    }

    /// The list of shapes being produced.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    DataTypeSlice output_types_ = {};
    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
  };
  MultiDeviceIteratorFromStringHandle(const ::tensorflow::Scope& scope,
                                    ::tensorflow::Input string_handle);
  MultiDeviceIteratorFromStringHandle(const ::tensorflow::Scope& scope,
                                    ::tensorflow::Input string_handle, const
                                    MultiDeviceIteratorFromStringHandle::Attrs&
                                    attrs);
  operator ::tensorflow::Output() const { return multi_device_iterator; }
  operator ::tensorflow::Input() const { return multi_device_iterator; }
  ::tensorflow::Node* node() const { return multi_device_iterator.node(); }

  static Attrs OutputTypes(const DataTypeSlice& x) {
    return Attrs().OutputTypes(x);
  }
  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }

  Operation operation;
  ::tensorflow::Output multi_device_iterator;
};

/// Gets next element for the provided shard number.
///
/// Args:
/// * scope: A Scope object
/// * multi_device_iterator: A MultiDeviceIterator resource.
/// * shard_num: Integer representing which shard to fetch data for.
/// * incarnation_id: Which incarnation of the MultiDeviceIterator is running.
/// * output_types: The type list for the return values.
/// * output_shapes: The list of shapes being produced.
///
/// Returns:
/// * `OutputList`: Result of the get_next on the dataset.
class MultiDeviceIteratorGetNextFromShard {
 public:
  MultiDeviceIteratorGetNextFromShard(const ::tensorflow::Scope& scope,
                                    ::tensorflow::Input multi_device_iterator,
                                    ::tensorflow::Input shard_num,
                                    ::tensorflow::Input incarnation_id, const
                                    DataTypeSlice& output_types, const
                                    gtl::ArraySlice<PartialTensorShape>&
                                    output_shapes);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  Operation operation;
  ::tensorflow::OutputList components;
};

/// Initializes the multi device iterator with the given dataset.
///
/// Args:
/// * scope: A Scope object
/// * dataset: Dataset to be iterated upon.
/// * multi_device_iterator: A MultiDeviceIteratorResource.
/// * max_buffer_size: The maximum size of the host side per device buffer to keep.
///
/// Returns:
/// * `Output`: An int64 indicating which incarnation of the MultiDeviceIterator
/// is running.
class MultiDeviceIteratorInit {
 public:
  MultiDeviceIteratorInit(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        dataset, ::tensorflow::Input multi_device_iterator,
                        ::tensorflow::Input max_buffer_size);
  operator ::tensorflow::Output() const { return incarnation_id; }
  operator ::tensorflow::Input() const { return incarnation_id; }
  ::tensorflow::Node* node() const { return incarnation_id.node(); }

  Operation operation;
  ::tensorflow::Output incarnation_id;
};

/// Produces a string handle for the given MultiDeviceIterator.
///
/// Args:
/// * scope: A Scope object
/// * multi_device_iterator: A MultiDeviceIterator resource.
///
/// Returns:
/// * `Output`: A string representing the resource.
class MultiDeviceIteratorToStringHandle {
 public:
  MultiDeviceIteratorToStringHandle(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input multi_device_iterator);
  operator ::tensorflow::Output() const { return string_handle; }
  operator ::tensorflow::Input() const { return string_handle; }
  ::tensorflow::Node* node() const { return string_handle.node(); }

  Operation operation;
  ::tensorflow::Output string_handle;
};

/// Creates a dataset by applying optimizations to `input_dataset`.
///
/// Creates a dataset by applying optimizations to `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * optimizations: A `tf.string` vector `tf.Tensor` identifying optimizations to use.
///
/// Returns:
/// * `Output`: The handle tensor.
class OptimizeDataset {
 public:
  /// Optional attribute setters for OptimizeDataset
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OptimizationConfigs(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.optimization_configs_ = x;
      return ret;
    }

    gtl::ArraySlice<::tensorflow::tstring> optimization_configs_ = {};
  };
  OptimizeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input optimizations, const
                DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes);
  OptimizeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input optimizations, const
                DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                OptimizeDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs OptimizationConfigs(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().OptimizationConfigs(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset by applying related optimizations to `input_dataset`.
///
/// Creates a dataset by applying related optimizations to `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * optimizations_enabled: A `tf.string` vector `tf.Tensor` identifying user enabled optimizations.
/// * optimizations_disabled: A `tf.string` vector `tf.Tensor` identifying user disabled optimizations.
/// * optimizations_default: A `tf.string` vector `tf.Tensor` identifying optimizations by default.
///
/// Returns:
/// * `Output`: The handle tensor.
class OptimizeDatasetV2 {
 public:
  /// Optional attribute setters for OptimizeDatasetV2
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OptimizationConfigs(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.optimization_configs_ = x;
      return ret;
    }

    gtl::ArraySlice<::tensorflow::tstring> optimization_configs_ = {};
  };
  OptimizeDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input optimizations_enabled,
                  ::tensorflow::Input optimizations_disabled,
                  ::tensorflow::Input optimizations_default, const
                  DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes);
  OptimizeDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input optimizations_enabled,
                  ::tensorflow::Input optimizations_disabled,
                  ::tensorflow::Input optimizations_default, const
                  DataTypeSlice& output_types, const
                  gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                  OptimizeDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs OptimizationConfigs(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().OptimizationConfigs(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset by attaching tf.data.Options to `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * serialized_options: A `tf.string` scalar `tf.Tensor` of serialized `tf.data.Options` protocol buffer.
///
/// Returns:
/// * `Output`: The handle tensor.
class OptionsDataset {
 public:
  /// Optional attribute setters for OptionsDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  OptionsDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, StringPiece serialized_options, const
               DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes);
  OptionsDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, StringPiece serialized_options, const
               DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes, const
               OptionsDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that batches and pads `batch_size` elements from the input.
///
/// Args:
/// * scope: A Scope object
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch.
/// * padded_shapes: A list of int64 tensors representing the desired padded shapes
/// of the corresponding output components. These shapes may be partially
/// specified, using `-1` to indicate that a particular dimension should be
/// padded to the maximum size of all batch elements.
/// * padding_values: A list of scalars containing the padding value to use for
/// each of the outputs.
///
/// Returns:
/// * `Output`: The handle tensor.
class PaddedBatchDataset {
 public:
  /// Optional attribute setters for PaddedBatchDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  PaddedBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::Input batch_size,
                   ::tensorflow::InputList padded_shapes,
                   ::tensorflow::InputList padding_values, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes);
  PaddedBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::Input batch_size,
                   ::tensorflow::InputList padded_shapes,
                   ::tensorflow::InputList padding_values, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                   PaddedBatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that batches and pads `batch_size` elements from the input.
///
/// Args:
/// * scope: A Scope object
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch.
/// * padded_shapes: A list of int64 tensors representing the desired padded shapes
/// of the corresponding output components. These shapes may be partially
/// specified, using `-1` to indicate that a particular dimension should be
/// padded to the maximum size of all batch elements.
/// * padding_values: A list of scalars containing the padding value to use for
/// each of the outputs.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
///
/// Returns:
/// * `Output`: The handle tensor.
class PaddedBatchDatasetV2 {
 public:
  /// Optional attribute setters for PaddedBatchDatasetV2
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ParallelCopy(bool x) {
      Attrs ret = *this;
      ret.parallel_copy_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool parallel_copy_ = false;
    StringPiece metadata_ = "";
  };
  PaddedBatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input batch_size,
                     ::tensorflow::InputList padded_shapes,
                     ::tensorflow::InputList padding_values,
                     ::tensorflow::Input drop_remainder, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  PaddedBatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input batch_size,
                     ::tensorflow::InputList padded_shapes,
                     ::tensorflow::InputList padding_values,
                     ::tensorflow::Input drop_remainder, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     PaddedBatchDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs ParallelCopy(bool x) {
    return Attrs().ParallelCopy(x);
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
class ParallelBatchDataset {
 public:
  /// Optional attribute setters for ParallelBatchDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ParallelCopy(bool x) {
      Attrs ret = *this;
      ret.parallel_copy_ = x;
      return ret;
    }

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

    bool parallel_copy_ = false;
    StringPiece deterministic_ = "default";
    StringPiece metadata_ = "";
  };
  ParallelBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input batch_size,
                     ::tensorflow::Input num_parallel_calls,
                     ::tensorflow::Input drop_remainder, const DataTypeSlice&
                     output_types, const gtl::ArraySlice<PartialTensorShape>&
                     output_shapes);
  ParallelBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input batch_size,
                     ::tensorflow::Input num_parallel_calls,
                     ::tensorflow::Input drop_remainder, const DataTypeSlice&
                     output_types, const gtl::ArraySlice<PartialTensorShape>&
                     output_shapes, const ParallelBatchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs ParallelCopy(bool x) {
    return Attrs().ParallelCopy(x);
  }
  static Attrs Deterministic(StringPiece x) {
    return Attrs().Deterministic(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset containing elements of `input_dataset` matching `predicate`.
///
/// The `predicate` function must return a scalar boolean and accept the
/// following arguments:
///
/// * One tensor for each component of an element of `input_dataset`.
/// * One tensor for each value in `other_arguments`.
///
/// Unlike a "FilterDataset", which applies `predicate` sequentially, this dataset
/// invokes up to `num_parallel_calls` copies of `predicate` in parallel.
///
///
/// Args:
/// * scope: A Scope object
/// * other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `predicate`.
/// * num_parallel_calls: The number of concurrent invocations of `predicate` that process
/// elements from `input_dataset` in parallel.
/// * predicate: A function returning a scalar boolean.
///
/// Optional attributes (see `Attrs`):
/// * deterministic: A string indicating the op-level determinism to use. Deterministic controls
/// whether the interleave is allowed to return elements out of order if the next
/// element to be returned isn't available, but a later element is. Options are
/// "true", "false", and "default". "default" indicates that determinism should be
/// decided by the `experimental_deterministic` parameter of `tf.data.Options`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParallelFilterDataset {
 public:
  /// Optional attribute setters for ParallelFilterDataset
  struct Attrs {
    /// A string indicating the op-level determinism to use. Deterministic controls
    /// whether the interleave is allowed to return elements out of order if the next
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

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece deterministic_ = "default";
    StringPiece metadata_ = "";
  };
  ParallelFilterDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_dataset, ::tensorflow::InputList other_arguments,
                      ::tensorflow::Input num_parallel_calls, const
                      NameAttrList& predicate, const DataTypeSlice&
                      output_types, const gtl::ArraySlice<PartialTensorShape>&
                      output_shapes);
  ParallelFilterDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_dataset, ::tensorflow::InputList other_arguments,
                      ::tensorflow::Input num_parallel_calls, const
                      NameAttrList& predicate, const DataTypeSlice&
                      output_types, const gtl::ArraySlice<PartialTensorShape>&
                      output_shapes, const ParallelFilterDataset::Attrs& attrs);
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

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// The resulting dataset is similar to the `InterleaveDataset`, except that the
/// dataset will fetch records from the interleaved datasets in parallel.
///
/// The `tf.data` Python API creates instances of this op from
/// `Dataset.interleave()` when the `num_parallel_calls` parameter of that method
/// is set to any value other than `None`.
///
/// By default, the output of this dataset will be deterministic, which may result
/// in the dataset blocking if the next data item to be returned isn't available.
/// In order to avoid head-of-line blocking, one can set the
/// `experimental_deterministic` parameter of `tf.data.Options` to `False`,
/// which can improve performance at the expense of non-determinism.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: Dataset that produces a stream of arguments for the function `f`.
/// * other_arguments: Additional arguments to pass to `f` beyond those produced by `input_dataset`.
/// Evaluated once when the dataset is instantiated.
/// * cycle_length: Number of datasets (each created by applying `f` to the elements of
/// `input_dataset`) among which the `ParallelInterleaveDatasetV2` will cycle in a
/// round-robin fashion.
/// * block_length: Number of elements at a time to produce from each interleaved invocation of a
/// dataset returned by `f`.
/// * num_parallel_calls: Determines the number of threads that should be used for fetching data from
/// input datasets in parallel. The Python API `tf.data.experimental.AUTOTUNE`
/// constant can be used to indicate that the level of parallelism should be autotuned.
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParallelInterleaveDatasetV2 {
 public:
  /// Optional attribute setters for ParallelInterleaveDatasetV2
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Sloppy(bool x) {
      Attrs ret = *this;
      ret.sloppy_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool sloppy_ = false;
    StringPiece metadata_ = "";
  };
  ParallelInterleaveDatasetV2(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input input_dataset,
                            ::tensorflow::InputList other_arguments,
                            ::tensorflow::Input cycle_length,
                            ::tensorflow::Input block_length,
                            ::tensorflow::Input num_parallel_calls, const
                            NameAttrList& f, const DataTypeSlice& output_types,
                            const gtl::ArraySlice<PartialTensorShape>&
                            output_shapes);
  ParallelInterleaveDatasetV2(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input input_dataset,
                            ::tensorflow::InputList other_arguments,
                            ::tensorflow::Input cycle_length,
                            ::tensorflow::Input block_length,
                            ::tensorflow::Input num_parallel_calls, const
                            NameAttrList& f, const DataTypeSlice& output_types,
                            const gtl::ArraySlice<PartialTensorShape>&
                            output_shapes, const
                            ParallelInterleaveDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Sloppy(bool x) {
    return Attrs().Sloppy(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// The resulting dataset is similar to the `InterleaveDataset`, except that the
/// dataset will fetch records from the interleaved datasets in parallel.
///
/// The `tf.data` Python API creates instances of this op from
/// `Dataset.interleave()` when the `num_parallel_calls` parameter of that method
/// is set to any value other than `None`.
///
/// By default, the output of this dataset will be deterministic, which may result
/// in the dataset blocking if the next data item to be returned isn't available.
/// In order to avoid head-of-line blocking, one can either set the `deterministic`
/// attribute to "false", or leave it as "default" and set the
/// `experimental_deterministic` parameter of `tf.data.Options` to `False`.
/// This can improve performance at the expense of non-determinism.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: Dataset that produces a stream of arguments for the function `f`.
/// * other_arguments: Additional arguments to pass to `f` beyond those produced by `input_dataset`.
/// Evaluated once when the dataset is instantiated.
/// * cycle_length: Number of datasets (each created by applying `f` to the elements of
/// `input_dataset`) among which the `ParallelInterleaveDatasetV2` will cycle in a
/// round-robin fashion.
/// * block_length: Number of elements at a time to produce from each interleaved invocation of a
/// dataset returned by `f`.
/// * num_parallel_calls: Determines the number of threads that should be used for fetching data from
/// input datasets in parallel. The Python API `tf.data.experimental.AUTOTUNE`
/// constant can be used to indicate that the level of parallelism should be autotuned.
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Optional attributes (see `Attrs`):
/// * deterministic: A string indicating the op-level determinism to use. Deterministic controls
/// whether the interleave is allowed to return elements out of order if the next
/// element to be returned isn't available, but a later element is. Options are
/// "true", "false", and "default". "default" indicates that determinism should be
/// decided by the `experimental_deterministic` parameter of `tf.data.Options`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParallelInterleaveDatasetV3 {
 public:
  /// Optional attribute setters for ParallelInterleaveDatasetV3
  struct Attrs {
    /// A string indicating the op-level determinism to use. Deterministic controls
    /// whether the interleave is allowed to return elements out of order if the next
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

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece deterministic_ = "default";
    StringPiece metadata_ = "";
  };
  ParallelInterleaveDatasetV3(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input input_dataset,
                            ::tensorflow::InputList other_arguments,
                            ::tensorflow::Input cycle_length,
                            ::tensorflow::Input block_length,
                            ::tensorflow::Input num_parallel_calls, const
                            NameAttrList& f, const DataTypeSlice& output_types,
                            const gtl::ArraySlice<PartialTensorShape>&
                            output_shapes);
  ParallelInterleaveDatasetV3(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input input_dataset,
                            ::tensorflow::InputList other_arguments,
                            ::tensorflow::Input cycle_length,
                            ::tensorflow::Input block_length,
                            ::tensorflow::Input num_parallel_calls, const
                            NameAttrList& f, const DataTypeSlice& output_types,
                            const gtl::ArraySlice<PartialTensorShape>&
                            output_shapes, const
                            ParallelInterleaveDatasetV3::Attrs& attrs);
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

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// The resulting dataset is similar to the `InterleaveDataset`, except that the
/// dataset will fetch records from the interleaved datasets in parallel.
///
/// The `tf.data` Python API creates instances of this op from
/// `Dataset.interleave()` when the `num_parallel_calls` parameter of that method
/// is set to any value other than `None`.
///
/// By default, the output of this dataset will be deterministic, which may result
/// in the dataset blocking if the next data item to be returned isn't available.
/// In order to avoid head-of-line blocking, one can either set the `deterministic`
/// attribute to "false", or leave it as "default" and set the
/// `experimental_deterministic` parameter of `tf.data.Options` to `False`.
/// This can improve performance at the expense of non-determinism.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: Dataset that produces a stream of arguments for the function `f`.
/// * other_arguments: Additional arguments to pass to `f` beyond those produced by `input_dataset`.
/// Evaluated once when the dataset is instantiated.
/// * cycle_length: Number of datasets (each created by applying `f` to the elements of
/// `input_dataset`) among which the `ParallelInterleaveDatasetV2` will cycle in a
/// round-robin fashion.
/// * block_length: Number of elements at a time to produce from each interleaved invocation of a
/// dataset returned by `f`.
/// * buffer_output_elements: The number of elements each iterator being interleaved should buffer (similar
/// to the `.prefetch()` transformation for each interleaved iterator).
/// * prefetch_input_elements: Determines the number of iterators to prefetch, allowing buffers to warm up and
/// data to be pre-fetched without blocking the main thread.
/// * num_parallel_calls: Determines the number of threads that should be used for fetching data from
/// input datasets in parallel. The Python API `tf.data.experimental.AUTOTUNE`
/// constant can be used to indicate that the level of parallelism should be autotuned.
/// * f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
///
/// Optional attributes (see `Attrs`):
/// * deterministic: A string indicating the op-level determinism to use. Deterministic controls
/// whether the interleave is allowed to return elements out of order if the next
/// element to be returned isn't available, but a later element is. Options are
/// "true", "false", and "default". "default" indicates that determinism should be
/// decided by the `experimental_deterministic` parameter of `tf.data.Options`.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParallelInterleaveDatasetV4 {
 public:
  /// Optional attribute setters for ParallelInterleaveDatasetV4
  struct Attrs {
    /// A string indicating the op-level determinism to use. Deterministic controls
    /// whether the interleave is allowed to return elements out of order if the next
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

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece deterministic_ = "default";
    StringPiece metadata_ = "";
  };
  ParallelInterleaveDatasetV4(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input input_dataset,
                            ::tensorflow::InputList other_arguments,
                            ::tensorflow::Input cycle_length,
                            ::tensorflow::Input block_length,
                            ::tensorflow::Input buffer_output_elements,
                            ::tensorflow::Input prefetch_input_elements,
                            ::tensorflow::Input num_parallel_calls, const
                            NameAttrList& f, const DataTypeSlice& output_types,
                            const gtl::ArraySlice<PartialTensorShape>&
                            output_shapes);
  ParallelInterleaveDatasetV4(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input input_dataset,
                            ::tensorflow::InputList other_arguments,
                            ::tensorflow::Input cycle_length,
                            ::tensorflow::Input block_length,
                            ::tensorflow::Input buffer_output_elements,
                            ::tensorflow::Input prefetch_input_elements,
                            ::tensorflow::Input num_parallel_calls, const
                            NameAttrList& f, const DataTypeSlice& output_types,
                            const gtl::ArraySlice<PartialTensorShape>&
                            output_shapes, const
                            ParallelInterleaveDatasetV4::Attrs& attrs);
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

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
/// to `num_parallel_calls` copies of `f` in parallel.
///
/// Args:
/// * scope: A Scope object
/// * num_parallel_calls: The number of concurrent invocations of `f` that process
/// elements from `input_dataset` in parallel.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParallelMapDataset {
 public:
  /// Optional attribute setters for ParallelMapDataset
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseInterOpParallelism(bool x) {
      Attrs ret = *this;
      ret.use_inter_op_parallelism_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Sloppy(bool x) {
      Attrs ret = *this;
      ret.sloppy_ = x;
      return ret;
    }

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

    bool use_inter_op_parallelism_ = true;
    bool sloppy_ = false;
    bool preserve_cardinality_ = false;
    StringPiece metadata_ = "";
  };
  ParallelMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::InputList other_arguments,
                   ::tensorflow::Input num_parallel_calls, const NameAttrList&
                   f, const DataTypeSlice& output_types, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ParallelMapDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::InputList other_arguments,
                   ::tensorflow::Input num_parallel_calls, const NameAttrList&
                   f, const DataTypeSlice& output_types, const
                   gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                   ParallelMapDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs UseInterOpParallelism(bool x) {
    return Attrs().UseInterOpParallelism(x);
  }
  static Attrs Sloppy(bool x) {
    return Attrs().Sloppy(x);
  }
  static Attrs PreserveCardinality(bool x) {
    return Attrs().PreserveCardinality(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that applies `f` to the outputs of `input_dataset`.
///
/// Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
/// to `num_parallel_calls` copies of `f` in parallel.
///
/// Args:
/// * scope: A Scope object
/// * num_parallel_calls: The number of concurrent invocations of `f` that process
/// elements from `input_dataset` in parallel.
///
/// Returns:
/// * `Output`: The handle tensor.
class ParallelMapDatasetV2 {
 public:
  /// Optional attribute setters for ParallelMapDatasetV2
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseInterOpParallelism(bool x) {
      Attrs ret = *this;
      ret.use_inter_op_parallelism_ = x;
      return ret;
    }

    /// Defaults to "default"
    TF_MUST_USE_RESULT Attrs Deterministic(StringPiece x) {
      Attrs ret = *this;
      ret.deterministic_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs PreserveCardinality(bool x) {
      Attrs ret = *this;
      ret.preserve_cardinality_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseUnboundedThreadpool(bool x) {
      Attrs ret = *this;
      ret.use_unbounded_threadpool_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool use_inter_op_parallelism_ = true;
    StringPiece deterministic_ = "default";
    bool preserve_cardinality_ = false;
    bool use_unbounded_threadpool_ = false;
    StringPiece metadata_ = "";
  };
  ParallelMapDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::InputList other_arguments,
                     ::tensorflow::Input num_parallel_calls, const
                     NameAttrList& f, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ParallelMapDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::InputList other_arguments,
                     ::tensorflow::Input num_parallel_calls, const
                     NameAttrList& f, const DataTypeSlice& output_types, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                     ParallelMapDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs UseInterOpParallelism(bool x) {
    return Attrs().UseInterOpParallelism(x);
  }
  static Attrs Deterministic(StringPiece x) {
    return Attrs().Deterministic(x);
  }
  static Attrs PreserveCardinality(bool x) {
    return Attrs().PreserveCardinality(x);
  }
  static Attrs UseUnboundedThreadpool(bool x) {
    return Attrs().UseUnboundedThreadpool(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that asynchronously prefetches elements from `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * buffer_size: The maximum number of elements to buffer in an iterator over
/// this dataset.
///
/// Returns:
/// * `Output`: The handle tensor.
class PrefetchDataset {
 public:
  /// Optional attribute setters for PrefetchDataset
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs SlackPeriod(int64 x) {
      Attrs ret = *this;
      ret.slack_period_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs LegacyAutotune(bool x) {
      Attrs ret = *this;
      ret.legacy_autotune_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BufferSizeMin(int64 x) {
      Attrs ret = *this;
      ret.buffer_size_min_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    int64 slack_period_ = 0;
    bool legacy_autotune_ = true;
    int64 buffer_size_min_ = 0;
    StringPiece metadata_ = "";
  };
  PrefetchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input buffer_size, const
                DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes);
  PrefetchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input buffer_size, const
                DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                PrefetchDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs SlackPeriod(int64 x) {
    return Attrs().SlackPeriod(x);
  }
  static Attrs LegacyAutotune(bool x) {
    return Attrs().LegacyAutotune(x);
  }
  static Attrs BufferSizeMin(int64 x) {
    return Attrs().BufferSizeMin(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset with a range of values. Corresponds to python's xrange.
///
/// Args:
/// * scope: A Scope object
/// * start: corresponds to start in python's xrange().
/// * stop: corresponds to stop in python's xrange().
/// * step: corresponds to step in python's xrange().
///
/// Returns:
/// * `Output`: The handle tensor.
class RangeDataset {
 public:
  /// Optional attribute setters for RangeDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ReplicateOnSplit(bool x) {
      Attrs ret = *this;
      ret.replicate_on_split_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
    bool replicate_on_split_ = false;
  };
  RangeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input start,
             ::tensorflow::Input stop, ::tensorflow::Input step, const
             DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes);
  RangeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input start,
             ::tensorflow::Input stop, ::tensorflow::Input step, const
             DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes, const
             RangeDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }
  static Attrs ReplicateOnSplit(bool x) {
    return Attrs().ReplicateOnSplit(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Reduces the input dataset to a singleton using a reduce function.
///
/// Args:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * initial_state: A nested structure of tensors, representing the initial state of the
/// transformation.
/// * f: A function that maps `(old_state, input_element)` to `new_state`. It must take
/// two arguments and return a nested structures of tensors. The structure of
/// `new_state` must match the structure of `initial_state`.
///
/// Returns:
/// * `OutputList`: The components tensor.
class ReduceDataset {
 public:
  /// Optional attribute setters for ReduceDataset
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseInterOpParallelism(bool x) {
      Attrs ret = *this;
      ret.use_inter_op_parallelism_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool use_inter_op_parallelism_ = true;
    StringPiece metadata_ = "";
  };
  ReduceDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::InputList initial_state,
              ::tensorflow::InputList other_arguments, const NameAttrList& f,
              const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ReduceDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::InputList initial_state,
              ::tensorflow::InputList other_arguments, const NameAttrList& f,
              const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes, const
              ReduceDataset::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  static Attrs UseInterOpParallelism(bool x) {
    return Attrs().UseInterOpParallelism(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::OutputList components;
};

/// Creates a dataset that emits the outputs of `input_dataset` `count` times.
///
/// Args:
/// * scope: A Scope object
/// * count: A scalar representing the number of times that `input_dataset` should
/// be repeated. A value of `-1` indicates that it should be repeated infinitely.
///
/// Returns:
/// * `Output`: The handle tensor.
class RepeatDataset {
 public:
  /// Optional attribute setters for RepeatDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  RepeatDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::Input count, const DataTypeSlice&
              output_types, const gtl::ArraySlice<PartialTensorShape>&
              output_shapes);
  RepeatDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::Input count, const DataTypeSlice&
              output_types, const gtl::ArraySlice<PartialTensorShape>&
              output_shapes, const RepeatDataset::Attrs& attrs);
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
class RewriteDataset {
 public:
  RewriteDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input rewrite_name, const
               DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a `Dataset` that includes only 1/`num_shards` of this dataset.
///
/// Args:
/// * scope: A Scope object
/// * num_shards: An integer representing the number of shards operating in parallel.
/// * index: An integer representing the current worker index.
///
/// Returns:
/// * `Output`: The handle tensor.
class ShardDataset {
 public:
  /// Optional attribute setters for ShardDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs RequireNonEmpty(bool x) {
      Attrs ret = *this;
      ret.require_non_empty_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    bool require_non_empty_ = false;
    StringPiece metadata_ = "";
  };
  ShardDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, ::tensorflow::Input num_shards, ::tensorflow::Input
             index, const DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ShardDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
             input_dataset, ::tensorflow::Input num_shards, ::tensorflow::Input
             index, const DataTypeSlice& output_types, const
             gtl::ArraySlice<PartialTensorShape>& output_shapes, const
             ShardDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs RequireNonEmpty(bool x) {
    return Attrs().RequireNonEmpty(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that shuffles and repeats elements from `input_dataset`
///
/// pseudorandomly.
///
/// Args:
/// * scope: A Scope object
/// * buffer_size: The number of output elements to buffer in an iterator over
/// this dataset. Compare with the `min_after_dequeue` attr when creating a
/// `RandomShuffleQueue`.
/// * seed: A scalar seed for the random number generator. If either `seed` or
/// `seed2` is set to be non-zero, the random number generator is seeded
/// by the given seed.  Otherwise, a random seed is used.
/// * seed2: A second scalar seed to avoid seed collision.
/// * count: A scalar representing the number of times the underlying dataset
/// should be repeated. The default is `-1`, which results in infinite repetition.
///
/// Returns:
/// * `Output`: The handle tensor.
class ShuffleAndRepeatDataset {
 public:
  /// Optional attribute setters for ShuffleAndRepeatDataset
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
  ShuffleAndRepeatDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input_dataset, ::tensorflow::Input buffer_size,
                        ::tensorflow::Input seed, ::tensorflow::Input seed2,
                        ::tensorflow::Input count, const DataTypeSlice&
                        output_types, const
                        gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ShuffleAndRepeatDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input_dataset, ::tensorflow::Input buffer_size,
                        ::tensorflow::Input seed, ::tensorflow::Input seed2,
                        ::tensorflow::Input count, const DataTypeSlice&
                        output_types, const
                        gtl::ArraySlice<PartialTensorShape>& output_shapes,
                        const ShuffleAndRepeatDataset::Attrs& attrs);
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

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ShuffleAndRepeatDatasetV2 {
 public:
  /// Optional attribute setters for ShuffleAndRepeatDatasetV2
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
  ShuffleAndRepeatDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, ::tensorflow::Input buffer_size,
                          ::tensorflow::Input seed, ::tensorflow::Input seed2,
                          ::tensorflow::Input count, ::tensorflow::Input
                          seed_generator, const DataTypeSlice& output_types,
                          const gtl::ArraySlice<PartialTensorShape>&
                          output_shapes);
  ShuffleAndRepeatDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input_dataset, ::tensorflow::Input buffer_size,
                          ::tensorflow::Input seed, ::tensorflow::Input seed2,
                          ::tensorflow::Input count, ::tensorflow::Input
                          seed_generator, const DataTypeSlice& output_types,
                          const gtl::ArraySlice<PartialTensorShape>&
                          output_shapes, const
                          ShuffleAndRepeatDatasetV2::Attrs& attrs);
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

/// Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.
///
/// Args:
/// * scope: A Scope object
/// * buffer_size: The number of output elements to buffer in an iterator over
/// this dataset. Compare with the `min_after_dequeue` attr when creating a
/// `RandomShuffleQueue`.
/// * seed: A scalar seed for the random number generator. If either `seed` or
/// `seed2` is set to be non-zero, the random number generator is seeded
/// by the given seed.  Otherwise, a random seed is used.
/// * seed2: A second scalar seed to avoid seed collision.
///
/// Optional attributes (see `Attrs`):
/// * reshuffle_each_iteration: If true, each iterator over this dataset will be given
/// a different pseudorandomly generated seed, based on a sequence seeded by the
/// `seed` and `seed2` inputs. If false, each iterator will be given the same
/// seed, and repeated iteration over this dataset will yield the exact same
/// sequence of results.
///
/// Returns:
/// * `Output`: The handle tensor.
class ShuffleDataset {
 public:
  /// Optional attribute setters for ShuffleDataset
  struct Attrs {
    /// If true, each iterator over this dataset will be given
    /// a different pseudorandomly generated seed, based on a sequence seeded by the
    /// `seed` and `seed2` inputs. If false, each iterator will be given the same
    /// seed, and repeated iteration over this dataset will yield the exact same
    /// sequence of results.
    ///
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
  ShuffleDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input buffer_size,
               ::tensorflow::Input seed, ::tensorflow::Input seed2, const
               DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ShuffleDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input buffer_size,
               ::tensorflow::Input seed, ::tensorflow::Input seed2, const
               DataTypeSlice& output_types, const
               gtl::ArraySlice<PartialTensorShape>& output_shapes, const
               ShuffleDataset::Attrs& attrs);
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

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class ShuffleDatasetV2 {
 public:
  /// Optional attribute setters for ShuffleDatasetV2
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  ShuffleDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::Input buffer_size,
                 ::tensorflow::Input seed_generator, const DataTypeSlice&
                 output_types, const gtl::ArraySlice<PartialTensorShape>&
                 output_shapes);
  ShuffleDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::Input buffer_size,
                 ::tensorflow::Input seed_generator, const DataTypeSlice&
                 output_types, const gtl::ArraySlice<PartialTensorShape>&
                 output_shapes, const ShuffleDatasetV2::Attrs& attrs);
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
class ShuffleDatasetV3 {
 public:
  /// Optional attribute setters for ShuffleDatasetV3
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
  ShuffleDatasetV3(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::Input buffer_size,
                 ::tensorflow::Input seed, ::tensorflow::Input seed2,
                 ::tensorflow::Input seed_generator, const DataTypeSlice&
                 output_types, const gtl::ArraySlice<PartialTensorShape>&
                 output_shapes);
  ShuffleDatasetV3(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 input_dataset, ::tensorflow::Input buffer_size,
                 ::tensorflow::Input seed, ::tensorflow::Input seed2,
                 ::tensorflow::Input seed_generator, const DataTypeSlice&
                 output_types, const gtl::ArraySlice<PartialTensorShape>&
                 output_shapes, const ShuffleDatasetV3::Attrs& attrs);
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

/// Creates a dataset that skips `count` elements from the `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * count: A scalar representing the number of elements from the `input_dataset`
/// that should be skipped.  If count is -1, skips everything.
///
/// Returns:
/// * `Output`: The handle tensor.
class SkipDataset {
 public:
  /// Optional attribute setters for SkipDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  SkipDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::Input count, const DataTypeSlice&
            output_types, const gtl::ArraySlice<PartialTensorShape>&
            output_shapes);
  SkipDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::Input count, const DataTypeSlice&
            output_types, const gtl::ArraySlice<PartialTensorShape>&
            output_shapes, const SkipDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that splits a SparseTensor into elements row-wise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class SparseTensorSliceDataset {
 public:
  SparseTensorSliceDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         indices, ::tensorflow::Input values,
                         ::tensorflow::Input dense_shape);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that emits the records from one or more TFRecord files.
///
/// Args:
/// * scope: A Scope object
/// * filenames: A scalar or vector containing the name(s) of the file(s) to be
/// read.
/// * compression_type: A scalar containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
/// * buffer_size: A scalar representing the number of bytes to buffer. A value of
/// 0 means no buffering will be performed.
///
/// Returns:
/// * `Output`: The handle tensor.
class TFRecordDataset {
 public:
  /// Optional attribute setters for TFRecordDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  TFRecordDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                filenames, ::tensorflow::Input compression_type,
                ::tensorflow::Input buffer_size);
  TFRecordDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                filenames, ::tensorflow::Input compression_type,
                ::tensorflow::Input buffer_size, const TFRecordDataset::Attrs&
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

/// Creates a dataset that emits the records from one or more TFRecord files.
///
/// Args:
/// * scope: A Scope object
/// * filenames: A scalar or vector containing the name(s) of the file(s) to be
/// read.
/// * compression_type: A scalar containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
/// * buffer_size: A scalar representing the number of bytes to buffer. A value of
/// 0 means no buffering will be performed.
/// * byte_offsets: A scalar or vector containing the number of bytes for each file
/// that will be skipped prior to reading.
///
/// Returns:
/// * `Output`: The handle tensor.
class TFRecordDatasetV2 {
 public:
  /// Optional attribute setters for TFRecordDatasetV2
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  TFRecordDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  filenames, ::tensorflow::Input compression_type,
                  ::tensorflow::Input buffer_size, ::tensorflow::Input
                  byte_offsets);
  TFRecordDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  filenames, ::tensorflow::Input compression_type,
                  ::tensorflow::Input buffer_size, ::tensorflow::Input
                  byte_offsets, const TFRecordDatasetV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that contains `count` elements from the `input_dataset`.
///
/// Args:
/// * scope: A Scope object
/// * count: A scalar representing the number of elements from the `input_dataset`
/// that should be taken. A value of `-1` indicates that all of `input_dataset`
/// is taken.
///
/// Returns:
/// * `Output`: The handle tensor.
class TakeDataset {
 public:
  /// Optional attribute setters for TakeDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  TakeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::Input count, const DataTypeSlice&
            output_types, const gtl::ArraySlice<PartialTensorShape>&
            output_shapes);
  TakeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset, ::tensorflow::Input count, const DataTypeSlice&
            output_types, const gtl::ArraySlice<PartialTensorShape>&
            output_shapes, const TakeDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that emits `components` as a tuple of tensors once.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class TensorDataset {
 public:
  /// Optional attribute setters for TensorDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  TensorDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
              components, const gtl::ArraySlice<PartialTensorShape>&
              output_shapes);
  TensorDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
              components, const gtl::ArraySlice<PartialTensorShape>&
              output_shapes, const TensorDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that emits each dim-0 slice of `components` once.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class TensorSliceDataset {
 public:
  /// Optional attribute setters for TensorSliceDataset
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs IsFiles(bool x) {
      Attrs ret = *this;
      ret.is_files_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ReplicateOnSplit(bool x) {
      Attrs ret = *this;
      ret.replicate_on_split_ = x;
      return ret;
    }

    bool is_files_ = false;
    StringPiece metadata_ = "";
    bool replicate_on_split_ = false;
  };
  TensorSliceDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                   components, const gtl::ArraySlice<PartialTensorShape>&
                   output_shapes);
  TensorSliceDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                   components, const gtl::ArraySlice<PartialTensorShape>&
                   output_shapes, const TensorSliceDataset::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs IsFiles(bool x) {
    return Attrs().IsFiles(x);
  }
  static Attrs Metadata(StringPiece x) {
    return Attrs().Metadata(x);
  }
  static Attrs ReplicateOnSplit(bool x) {
    return Attrs().ReplicateOnSplit(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Creates a dataset that emits the lines of one or more text files.
///
/// Args:
/// * scope: A Scope object
/// * filenames: A scalar or a vector containing the name(s) of the file(s) to be
/// read.
/// * compression_type: A scalar containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
/// * buffer_size: A scalar containing the number of bytes to buffer.
///
/// Returns:
/// * `Output`: The handle tensor.
class TextLineDataset {
 public:
  /// Optional attribute setters for TextLineDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  TextLineDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                filenames, ::tensorflow::Input compression_type,
                ::tensorflow::Input buffer_size);
  TextLineDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                filenames, ::tensorflow::Input compression_type,
                ::tensorflow::Input buffer_size, const TextLineDataset::Attrs&
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

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output_handle tensor.
class UnwrapDatasetVariant {
 public:
  UnwrapDatasetVariant(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_handle);
  operator ::tensorflow::Output() const { return output_handle; }
  operator ::tensorflow::Input() const { return output_handle; }
  ::tensorflow::Node* node() const { return output_handle.node(); }

  Operation operation;
  ::tensorflow::Output output_handle;
};

///   Combines (nests of) input elements into a dataset of (nests of) windows.
///
///   A "window" is a finite dataset of flat elements of size `size` (or possibly
///   fewer if there are not enough input elements to fill the window and
///   `drop_remainder` evaluates to false).
///
///   The `shift` argument determines the number of input elements by which
///   the window moves on each iteration.  The first element in the `k`th window
///   will be element
///
///   ```
///   1 + (k-1) * shift
///   ```
///
///   of the input dataset. In particular, the first element of the first window
///   will always be the first element of the input dataset.
///
///   If the `stride` parameter is greater than 1, then each window will skip
///   `(stride - 1)` input elements between each element that appears in the
///   window. Output windows will still contain `size` elements regardless of
///   the value of `stride`.
///
///   The `stride` argument determines the stride of the input elements, and the
///   `shift` argument determines the shift of the window.
///
///   For example, letting `{...}` to represent a Dataset:
///
///   - `tf.data.Dataset.range(7).window(2)` produces
///     `{{0, 1}, {2, 3}, {4, 5}, {6}}`
///   - `tf.data.Dataset.range(7).window(3, 2, 1, True)` produces
///     `{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}}`
///   - `tf.data.Dataset.range(7).window(3, 1, 2, True)` produces
///     `{{0, 2, 4}, {1, 3, 5}, {2, 4, 6}}`
///
///   Note that when the `window` transformation is applied to a dataset of
///   nested elements, it produces a dataset of nested windows.
///
///   For example:
///
///   - `tf.data.Dataset.from_tensor_slices((range(4), range(4))).window(2)`
///     produces `{({0, 1}, {0, 1}), ({2, 3}, {2, 3})}`
///   - `tf.data.Dataset.from_tensor_slices({"a": range(4)}).window(2)`
///     produces `{{"a": {0, 1}}, {"a": {2, 3}}}`
///
/// Args:
/// * scope: A Scope object
/// * size: An integer scalar, representing the number of elements
/// of the input dataset to combine into a window. Must be positive.
/// * shift: An integer scalar, representing the number of input elements
/// by which the window moves in each iteration.  Defaults to `size`.
/// Must be positive.
/// * stride: An integer scalar, representing the stride of the input elements
/// in the sliding window. Must be positive. The default value of 1 means
/// "retain every input element".
/// * drop_remainder: A Boolean scalar, representing whether the last window should be
/// dropped if its size is smaller than `window_size`.
///
/// Returns:
/// * `Output`: The handle tensor.
class WindowDataset {
 public:
  /// Optional attribute setters for WindowDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  WindowDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::Input size, ::tensorflow::Input
              shift, ::tensorflow::Input stride, ::tensorflow::Input
              drop_remainder, const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes);
  WindowDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::Input size, ::tensorflow::Input
              shift, ::tensorflow::Input stride, ::tensorflow::Input
              drop_remainder, const DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes, const
              WindowDataset::Attrs& attrs);
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
class WindowOp {
 public:
  WindowOp(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs,
         const DataTypeSlice& output_types, const
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
/// * `Output`: The output_handle tensor.
class WrapDatasetVariant {
 public:
  WrapDatasetVariant(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_handle);
  operator ::tensorflow::Output() const { return output_handle; }
  operator ::tensorflow::Input() const { return output_handle; }
  ::tensorflow::Node* node() const { return output_handle.node(); }

  Operation operation;
  ::tensorflow::Output output_handle;
};

/// Creates a dataset that zips together `input_datasets`.
///
/// The elements of the resulting dataset are created by zipping corresponding
/// elements from each of the input datasets.
///
/// The size of the resulting dataset will match the size of the smallest input
/// dataset, and no error will be raised if input datasets have different sizes.
///
/// Args:
/// * scope: A Scope object
/// * input_datasets: List of `N` variant Tensors representing datasets to be zipped together.
///
/// Returns:
/// * `Output`: The handle tensor.
class ZipDataset {
 public:
  /// Optional attribute setters for ZipDataset
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Metadata(StringPiece x) {
      Attrs ret = *this;
      ret.metadata_ = x;
      return ret;
    }

    StringPiece metadata_ = "";
  };
  ZipDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
           input_datasets, const DataTypeSlice& output_types, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ZipDataset(const ::tensorflow::Scope& scope, ::tensorflow::InputList
           input_datasets, const DataTypeSlice& output_types, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes, const
           ZipDataset::Attrs& attrs);
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

#endif  // TENSORFLOW_CC_OPS_DATASET_OPS_INTERNAL_H_
