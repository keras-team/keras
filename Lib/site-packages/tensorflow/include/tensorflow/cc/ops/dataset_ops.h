// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_DATASET_OPS_H_
#define TENSORFLOW_CC_OPS_DATASET_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup dataset_ops Dataset Ops
/// @{

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
class AnonymousIterator {
 public:
  AnonymousIterator(const ::tensorflow::Scope& scope, const DataTypeSlice&
                  output_types, const gtl::ArraySlice<PartialTensorShape>&
                  output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Converts the given variant tensor to an iterator and stores it in the given resource.
///
/// Args:
/// * scope: A Scope object
/// * resource_handle: A handle to an iterator resource.
/// * serialized: A variant tensor storing the state of the iterator contained in the
/// resource.
///
/// Returns:
/// * the created `Operation`
class DeserializeIterator {
 public:
  DeserializeIterator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    resource_handle, ::tensorflow::Input serialized);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// A container for an iterator resource.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: A handle to the iterator that can be passed to a "MakeIterator"
/// or "IteratorGetNext" op.
class Iterator {
 public:
  Iterator(const ::tensorflow::Scope& scope, StringPiece shared_name, StringPiece
         container, const DataTypeSlice& output_types, const
         gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Converts the given string representing a handle to an iterator to a resource.
///
/// Args:
/// * scope: A Scope object
/// * string_handle: A string representation of the given handle.
///
/// Optional attributes (see `Attrs`):
/// * output_types: If specified, defines the type of each tuple component in an
/// element produced by the resulting iterator.
/// * output_shapes: If specified, defines the shape of each tuple component in an
/// element produced by the resulting iterator.
///
/// Returns:
/// * `Output`: A handle to an iterator resource.
class IteratorFromStringHandle {
 public:
  /// Optional attribute setters for IteratorFromStringHandle
  struct Attrs {
    /// If specified, defines the type of each tuple component in an
    /// element produced by the resulting iterator.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.output_types_ = x;
      return ret;
    }

    /// If specified, defines the shape of each tuple component in an
    /// element produced by the resulting iterator.
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
  IteratorFromStringHandle(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         string_handle);
  IteratorFromStringHandle(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         string_handle, const IteratorFromStringHandle::Attrs&
                         attrs);
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

/// Gets the next output from the given iterator .
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The components tensor.
class IteratorGetNext {
 public:
  IteratorGetNext(const ::tensorflow::Scope& scope, ::tensorflow::Input iterator,
                const DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  Operation operation;
  ::tensorflow::OutputList components;
};

/// Gets the next output from the given iterator as an Optional variant.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The optional tensor.
class IteratorGetNextAsOptional {
 public:
  IteratorGetNextAsOptional(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          iterator, const DataTypeSlice& output_types, const
                          gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return optional; }
  operator ::tensorflow::Input() const { return optional; }
  ::tensorflow::Node* node() const { return optional.node(); }

  Operation operation;
  ::tensorflow::Output optional;
};

/// Gets the next output from the given iterator.
///
/// This operation is a synchronous version IteratorGetNext. It should only be used
/// in situations where the iterator does not block the calling thread, or where
/// the calling thread is not a member of the thread pool used to execute parallel
/// operations (e.g. in eager mode).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The components tensor.
class IteratorGetNextSync {
 public:
  IteratorGetNextSync(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    iterator, const DataTypeSlice& output_types, const
                    gtl::ArraySlice<PartialTensorShape>& output_shapes);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  Operation operation;
  ::tensorflow::OutputList components;
};

/// Converts the given `resource_handle` representing an iterator to a string.
///
/// Args:
/// * scope: A Scope object
/// * resource_handle: A handle to an iterator resource.
///
/// Returns:
/// * `Output`: A string representation of the given handle.
class IteratorToStringHandle {
 public:
  IteratorToStringHandle(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       resource_handle);
  operator ::tensorflow::Output() const { return string_handle; }
  operator ::tensorflow::Input() const { return string_handle; }
  ::tensorflow::Node* node() const { return string_handle.node(); }

  Operation operation;
  ::tensorflow::Output string_handle;
};

/// Makes a new iterator from the given `dataset` and stores it in `iterator`.
///
/// This operation may be executed multiple times. Each execution will reset the
/// iterator in `iterator` to the first element of `dataset`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class MakeIterator {
 public:
  MakeIterator(const ::tensorflow::Scope& scope, ::tensorflow::Input dataset,
             ::tensorflow::Input iterator);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Makes a "one-shot" iterator that can be iterated only once.
///
/// A one-shot iterator bundles the logic for defining the dataset and
/// the state of the iterator in a single op, which allows simple input
/// pipelines to be defined without an additional initialization
/// ("MakeIterator") step.
///
/// One-shot iterators have the following limitations:
///
/// * They do not support parameterization: all logic for creating the underlying
///   dataset must be bundled in the `dataset_factory` function.
/// * They are not resettable. Once a one-shot iterator reaches the end of its
///   underlying dataset, subsequent "IteratorGetNext" operations on that
///   iterator will always produce an `OutOfRange` error.
///
/// For greater flexibility, use "Iterator" and "MakeIterator" to define
/// an iterator using an arbitrary subgraph, which may capture tensors
/// (including fed values) as parameters, and which may be reset multiple
/// times by rerunning "MakeIterator".
///
/// Args:
/// * scope: A Scope object
/// * dataset_factory: A function of type `() -> DT_VARIANT`, where the returned
/// DT_VARIANT is a dataset.
///
/// Returns:
/// * `Output`: A handle to the iterator that can be passed to an "IteratorGetNext"
/// op.
class OneShotIterator {
 public:
  /// Optional attribute setters for OneShotIterator
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
  OneShotIterator(const ::tensorflow::Scope& scope, const NameAttrList&
                dataset_factory, const DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes);
  OneShotIterator(const ::tensorflow::Scope& scope, const NameAttrList&
                dataset_factory, const DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes, const
                OneShotIterator::Attrs& attrs);
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

/// Converts the given `resource_handle` representing an iterator to a variant tensor.
///
/// Args:
/// * scope: A Scope object
/// * resource_handle: A handle to an iterator resource.
///
/// Returns:
/// * `Output`: A variant tensor storing the state of the iterator contained in the
/// resource.
class SerializeIterator {
 public:
  /// Optional attribute setters for SerializeIterator
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs ExternalStatePolicy(int64 x) {
      Attrs ret = *this;
      ret.external_state_policy_ = x;
      return ret;
    }

    int64 external_state_policy_ = 0;
  };
  SerializeIterator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  resource_handle);
  SerializeIterator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  resource_handle, const SerializeIterator::Attrs& attrs);
  operator ::tensorflow::Output() const { return serialized; }
  operator ::tensorflow::Input() const { return serialized; }
  ::tensorflow::Node* node() const { return serialized.node(); }

  static Attrs ExternalStatePolicy(int64 x) {
    return Attrs().ExternalStatePolicy(x);
  }

  Operation operation;
  ::tensorflow::Output serialized;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_DATASET_OPS_H_
