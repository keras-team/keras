// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_

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

/// @defgroup lookup_ops_internal Lookup Ops Internal
/// @{

/// Creates a uninitialized anonymous hash table.
///
/// This op creates a new anonymous hash table (as a resource) everytime
/// it is executed, with the specified dtype of its keys and values,
/// returning the resource handle.  Before using the table you will have
/// to initialize it.  After initialization the table will be
/// immutable. The table is anonymous in the sense that it can only be
/// accessed by the returned resource handle (e.g. it cannot be looked up
/// by a name in a resource manager). The table will be automatically
/// deleted when all resource handles pointing to it are gone.
///
/// Args:
/// * scope: A Scope object
/// * key_dtype: Type of the table keys.
/// * value_dtype: Type of the table values.
///
/// Returns:
/// * `Output`: The resource handle to the newly created hash-table resource.
class AnonymousHashTable {
 public:
  AnonymousHashTable(const ::tensorflow::Scope& scope, DataType key_dtype,
                   DataType value_dtype);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// Creates an empty anonymous mutable hash table that uses tensors as the backing store.
///
/// This op creates a new anonymous mutable hash table (as a resource) everytime
/// it is executed, with the specified dtype of its keys and values,
/// returning the resource handle. Each value must be a scalar.
/// Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
///
/// It uses "open addressing" with quadratic reprobing to resolve
/// collisions.
///
/// The table is anonymous in the sense that it can only be
/// accessed by the returned resource handle (e.g. it cannot be looked up
/// by a name in a resource manager). The table will be automatically
/// deleted when all resource handles pointing to it are gone.
///
/// Args:
/// * scope: A Scope object
/// * empty_key: The key used to represent empty key buckets internally. Must not
/// be used in insert or lookup operations.
/// * value_dtype: Type of the table values.
///
/// Optional attributes (see `Attrs`):
/// * value_shape: The shape of each value.
/// * initial_num_buckets: The initial number of hash table buckets. Must be a power
/// to 2.
/// * max_load_factor: The maximum ratio between number of entries and number of
/// buckets before growing the table. Must be between 0 and 1.
///
/// Returns:
/// * `Output`: The resource handle to the newly created hash-table resource.
class AnonymousMutableDenseHashTable {
 public:
  /// Optional attribute setters for AnonymousMutableDenseHashTable
  struct Attrs {
    /// The shape of each value.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ValueShape(PartialTensorShape x) {
      Attrs ret = *this;
      ret.value_shape_ = x;
      return ret;
    }

    /// The initial number of hash table buckets. Must be a power
    /// to 2.
    ///
    /// Defaults to 131072
    TF_MUST_USE_RESULT Attrs InitialNumBuckets(int64 x) {
      Attrs ret = *this;
      ret.initial_num_buckets_ = x;
      return ret;
    }

    /// The maximum ratio between number of entries and number of
    /// buckets before growing the table. Must be between 0 and 1.
    ///
    /// Defaults to 0.8
    TF_MUST_USE_RESULT Attrs MaxLoadFactor(float x) {
      Attrs ret = *this;
      ret.max_load_factor_ = x;
      return ret;
    }

    PartialTensorShape value_shape_ = {};
    int64 initial_num_buckets_ = 131072;
    float max_load_factor_ = 0.8f;
  };
  AnonymousMutableDenseHashTable(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input empty_key,
                               ::tensorflow::Input deleted_key, DataType
                               value_dtype);
  AnonymousMutableDenseHashTable(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input empty_key,
                               ::tensorflow::Input deleted_key, DataType
                               value_dtype, const
                               AnonymousMutableDenseHashTable::Attrs& attrs);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  static Attrs ValueShape(PartialTensorShape x) {
    return Attrs().ValueShape(x);
  }
  static Attrs InitialNumBuckets(int64 x) {
    return Attrs().InitialNumBuckets(x);
  }
  static Attrs MaxLoadFactor(float x) {
    return Attrs().MaxLoadFactor(x);
  }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// Creates an empty anonymous mutable hash table.
///
/// This op creates a new anonymous mutable hash table (as a resource) everytime
/// it is executed, with the specified dtype of its keys and values,
/// returning the resource handle. Each value must be a scalar.
/// Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// The table is anonymous in the sense that it can only be
/// accessed by the returned resource handle (e.g. it cannot be looked up
/// by a name in a resource manager). The table will be automatically
/// deleted when all resource handles pointing to it are gone.
///
/// Args:
/// * scope: A Scope object
/// * key_dtype: Type of the table keys.
/// * value_dtype: Type of the table values.
///
/// Returns:
/// * `Output`: The resource handle to the newly created hash-table resource.
class AnonymousMutableHashTable {
 public:
  AnonymousMutableHashTable(const ::tensorflow::Scope& scope, DataType key_dtype,
                          DataType value_dtype);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// Creates an empty anonymous mutable hash table of vector values.
///
/// This op creates a new anonymous mutable hash table (as a resource) everytime
/// it is executed, with the specified dtype of its keys and values,
/// returning the resource handle. Each value must be a vector.
/// Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// The table is anonymous in the sense that it can only be
/// accessed by the returned resource handle (e.g. it cannot be looked up
/// by a name in a resource manager). The table will be automatically
/// deleted when all resource handles pointing to it are gone.
///
/// Args:
/// * scope: A Scope object
/// * key_dtype: Type of the table keys.
/// * value_dtype: Type of the table values.
///
/// Returns:
/// * `Output`: The resource handle to the newly created hash-table resource.
class AnonymousMutableHashTableOfTensors {
 public:
  /// Optional attribute setters for AnonymousMutableHashTableOfTensors
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ValueShape(PartialTensorShape x) {
      Attrs ret = *this;
      ret.value_shape_ = x;
      return ret;
    }

    PartialTensorShape value_shape_ = {};
  };
  AnonymousMutableHashTableOfTensors(const ::tensorflow::Scope& scope, DataType
                                   key_dtype, DataType value_dtype);
  AnonymousMutableHashTableOfTensors(const ::tensorflow::Scope& scope, DataType
                                   key_dtype, DataType value_dtype, const
                                   AnonymousMutableHashTableOfTensors::Attrs&
                                   attrs);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  static Attrs ValueShape(PartialTensorShape x) {
    return Attrs().ValueShape(x);
  }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// Removes keys and its associated values from a table.
///
/// The tensor `keys` must of the same type as the keys of the table. Keys not
/// already in the table are silently ignored.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
/// * keys: Any shape.  Keys of the elements to remove.
///
/// Returns:
/// * the created `Operation`
class LookupTableRemove {
 public:
  LookupTableRemove(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  table_handle, ::tensorflow::Input keys);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LOOKUP_OPS_INTERNAL_H_
