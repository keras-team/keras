// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LOOKUP_OPS_H_
#define TENSORFLOW_CC_OPS_LOOKUP_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup lookup_ops Lookup Ops
/// @{

/// Creates a non-initialized hash table.
///
/// This op creates a hash table, specifying the type of its keys and values.
/// Before using the table you will have to initialize it.  After initialization the
/// table will be immutable.
///
/// Args:
/// * scope: A Scope object
/// * key_dtype: Type of the table keys.
/// * value_dtype: Type of the table values.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// * use_node_name_sharing: If true and shared_name is empty, the table is shared
/// using the node name.
///
/// Returns:
/// * `Output`: Handle to a table.
class HashTable {
 public:
  /// Optional attribute setters for HashTable
  struct Attrs {
    /// If non-empty, this table is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this table is shared under the given name across
    /// multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// If true and shared_name is empty, the table is shared
    /// using the node name.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNodeNameSharing(bool x) {
      Attrs ret = *this;
      ret.use_node_name_sharing_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    bool use_node_name_sharing_ = false;
  };
  HashTable(const ::tensorflow::Scope& scope, DataType key_dtype, DataType
          value_dtype);
  HashTable(const ::tensorflow::Scope& scope, DataType key_dtype, DataType
          value_dtype, const HashTable::Attrs& attrs);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs UseNodeNameSharing(bool x) {
    return Attrs().UseNodeNameSharing(x);
  }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// Initializes a table from a text file.
///
/// It inserts one key-value pair into the table for each line of the file.
/// The key and value is extracted from the whole line content, elements from the
/// split line based on `delimiter` or the line number (starting from zero).
/// Where to extract the key and value from a line is specified by `key_index` and
/// `value_index`.
///
/// - A value of -1 means use the line number(starting from zero), expects `int64`.
/// - A value of -2 means use the whole line content, expects `string`.
/// - A value >= 0 means use the index (starting at zero) of the split line based
///   on `delimiter`.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to a table which will be initialized.
/// * filename: Filename of a vocabulary text file.
/// * key_index: Column index in a line to get the table `key` values from.
/// * value_index: Column index that represents information of a line to get the table
/// `value` values from.
///
/// Optional attributes (see `Attrs`):
/// * vocab_size: Number of elements of the file, use -1 if unknown.
/// * delimiter: Delimiter to separate fields in a line.
///
/// Returns:
/// * the created `Operation`
class InitializeTableFromTextFile {
 public:
  /// Optional attribute setters for InitializeTableFromTextFile
  struct Attrs {
    /// Number of elements of the file, use -1 if unknown.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs VocabSize(int64 x) {
      Attrs ret = *this;
      ret.vocab_size_ = x;
      return ret;
    }

    /// Delimiter to separate fields in a line.
    ///
    /// Defaults to "\t"
    TF_MUST_USE_RESULT Attrs Delimiter(StringPiece x) {
      Attrs ret = *this;
      ret.delimiter_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Offset(int64 x) {
      Attrs ret = *this;
      ret.offset_ = x;
      return ret;
    }

    int64 vocab_size_ = -1;
    StringPiece delimiter_ = "\t";
    int64 offset_ = 0;
  };
  InitializeTableFromTextFile(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input table_handle,
                            ::tensorflow::Input filename, int64 key_index,
                            int64 value_index);
  InitializeTableFromTextFile(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input table_handle,
                            ::tensorflow::Input filename, int64 key_index,
                            int64 value_index, const
                            InitializeTableFromTextFile::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs VocabSize(int64 x) {
    return Attrs().VocabSize(x);
  }
  static Attrs Delimiter(StringPiece x) {
    return Attrs().Delimiter(x);
  }
  static Attrs Offset(int64 x) {
    return Attrs().Offset(x);
  }

  Operation operation;
};

/// Table initializer that takes two tensors for keys and values respectively.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to a table which will be initialized.
/// * keys: Keys of type Tkey.
/// * values: Values of type Tval.
///
/// Returns:
/// * the created `Operation`
class InitializeTable {
 public:
  InitializeTable(const ::tensorflow::Scope& scope, ::tensorflow::Input
                table_handle, ::tensorflow::Input keys, ::tensorflow::Input
                values);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Outputs all keys and values in the table.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
///
/// Returns:
/// * `Output` keys: Vector of all keys present in the table.
/// * `Output` values: Tensor of all values in the table. Indexed in parallel with `keys`.
class LookupTableExport {
 public:
  LookupTableExport(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  table_handle, DataType Tkeys, DataType Tvalues);

  Operation operation;
  ::tensorflow::Output keys;
  ::tensorflow::Output values;
};

/// Looks up keys in a table, outputs the corresponding values.
///
/// The tensor `keys` must of the same type as the keys of the table.
/// The output `values` is of the type of the table values.
///
/// The scalar `default_value` is the value output for keys not present in the
/// table. It must also be of the same type as the table values.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
/// * keys: Any shape.  Keys to look up.
///
/// Returns:
/// * `Output`: Same shape as `keys`.  Values found in the table, or `default_values`
/// for missing keys.
class LookupTableFind {
 public:
  LookupTableFind(const ::tensorflow::Scope& scope, ::tensorflow::Input
                table_handle, ::tensorflow::Input keys, ::tensorflow::Input
                default_value);
  operator ::tensorflow::Output() const { return values; }
  operator ::tensorflow::Input() const { return values; }
  ::tensorflow::Node* node() const { return values.node(); }

  Operation operation;
  ::tensorflow::Output values;
};

/// Replaces the contents of the table with the specified keys and values.
///
/// The tensor `keys` must be of the same type as the keys of the table.
/// The tensor `values` must be of the type of the table values.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
/// * keys: Any shape.  Keys to look up.
/// * values: Values to associate with keys.
///
/// Returns:
/// * the created `Operation`
class LookupTableImport {
 public:
  LookupTableImport(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  table_handle, ::tensorflow::Input keys, ::tensorflow::Input
                  values);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Updates the table to associates keys with values.
///
/// The tensor `keys` must be of the same type as the keys of the table.
/// The tensor `values` must be of the type of the table values.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
/// * keys: Any shape.  Keys to look up.
/// * values: Values to associate with keys.
///
/// Returns:
/// * the created `Operation`
class LookupTableInsert {
 public:
  LookupTableInsert(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  table_handle, ::tensorflow::Input keys, ::tensorflow::Input
                  values);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Computes the number of elements in the given table.
///
/// Args:
/// * scope: A Scope object
/// * table_handle: Handle to the table.
///
/// Returns:
/// * `Output`: Scalar that contains number of elements in the table.
class LookupTableSize {
 public:
  LookupTableSize(const ::tensorflow::Scope& scope, ::tensorflow::Input
                table_handle);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  Operation operation;
  ::tensorflow::Output size;
};

/// Creates an empty hash table that uses tensors as the backing store.
///
/// It uses "open addressing" with quadratic reprobing to resolve
/// collisions.
///
/// This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a scalar. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
///
/// Args:
/// * scope: A Scope object
/// * empty_key: The key used to represent empty key buckets internally. Must not
/// be used in insert or lookup operations.
/// * value_dtype: Type of the table values.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// * value_shape: The shape of each value.
/// * initial_num_buckets: The initial number of hash table buckets. Must be a power
/// to 2.
/// * max_load_factor: The maximum ratio between number of entries and number of
/// buckets before growing the table. Must be between 0 and 1.
///
/// Returns:
/// * `Output`: Handle to a table.
class MutableDenseHashTable {
 public:
  /// Optional attribute setters for MutableDenseHashTable
  struct Attrs {
    /// If non-empty, this table is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this table is shared under the given name across
    /// multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNodeNameSharing(bool x) {
      Attrs ret = *this;
      ret.use_node_name_sharing_ = x;
      return ret;
    }

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

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    bool use_node_name_sharing_ = false;
    PartialTensorShape value_shape_ = {};
    int64 initial_num_buckets_ = 131072;
    float max_load_factor_ = 0.8f;
  };
  MutableDenseHashTable(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      empty_key, ::tensorflow::Input deleted_key, DataType
                      value_dtype);
  MutableDenseHashTable(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      empty_key, ::tensorflow::Input deleted_key, DataType
                      value_dtype, const MutableDenseHashTable::Attrs& attrs);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs UseNodeNameSharing(bool x) {
    return Attrs().UseNodeNameSharing(x);
  }
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

/// Creates an empty hash table.
///
/// This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a vector. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
///
/// Args:
/// * scope: A Scope object
/// * key_dtype: Type of the table keys.
/// * value_dtype: Type of the table values.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this table is shared under the given name across
/// multiple sessions.
///
/// Returns:
/// * `Output`: Handle to a table.
class MutableHashTableOfTensors {
 public:
  /// Optional attribute setters for MutableHashTableOfTensors
  struct Attrs {
    /// If non-empty, this table is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this table is shared under the given name across
    /// multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNodeNameSharing(bool x) {
      Attrs ret = *this;
      ret.use_node_name_sharing_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ValueShape(PartialTensorShape x) {
      Attrs ret = *this;
      ret.value_shape_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    bool use_node_name_sharing_ = false;
    PartialTensorShape value_shape_ = {};
  };
  MutableHashTableOfTensors(const ::tensorflow::Scope& scope, DataType key_dtype,
                          DataType value_dtype);
  MutableHashTableOfTensors(const ::tensorflow::Scope& scope, DataType key_dtype,
                          DataType value_dtype, const
                          MutableHashTableOfTensors::Attrs& attrs);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs UseNodeNameSharing(bool x) {
    return Attrs().UseNodeNameSharing(x);
  }
  static Attrs ValueShape(PartialTensorShape x) {
    return Attrs().ValueShape(x);
  }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// Creates an empty hash table.
///
/// This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a scalar. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
///
/// Args:
/// * scope: A Scope object
/// * key_dtype: Type of the table keys.
/// * value_dtype: Type of the table values.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// * use_node_name_sharing: If true and shared_name is empty, the table is shared
/// using the node name.
///
/// Returns:
/// * `Output`: Handle to a table.
class MutableHashTable {
 public:
  /// Optional attribute setters for MutableHashTable
  struct Attrs {
    /// If non-empty, this table is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this table is shared under the given name across
    /// multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// If true and shared_name is empty, the table is shared
    /// using the node name.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNodeNameSharing(bool x) {
      Attrs ret = *this;
      ret.use_node_name_sharing_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    bool use_node_name_sharing_ = false;
  };
  MutableHashTable(const ::tensorflow::Scope& scope, DataType key_dtype, DataType
                 value_dtype);
  MutableHashTable(const ::tensorflow::Scope& scope, DataType key_dtype, DataType
                 value_dtype, const MutableHashTable::Attrs& attrs);
  operator ::tensorflow::Output() const { return table_handle; }
  operator ::tensorflow::Input() const { return table_handle; }
  ::tensorflow::Node* node() const { return table_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs UseNodeNameSharing(bool x) {
    return Attrs().UseNodeNameSharing(x);
  }

  Operation operation;
  ::tensorflow::Output table_handle;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LOOKUP_OPS_H_
