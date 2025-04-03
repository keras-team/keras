// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_MAP_OPS_H_
#define TENSORFLOW_CC_OPS_MAP_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup map_ops Map Ops
/// @{

/// Creates and returns an empty tensor map.
///
/// handle: an empty tensor map
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class EmptyTensorMap {
 public:
  EmptyTensorMap(const ::tensorflow::Scope& scope);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Returns a tensor map with item from given key erased.
///
/// input_handle: the original map
/// output_handle: the map with value from given key removed
/// key: the key of the value to be erased
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output_handle tensor.
class TensorMapErase {
 public:
  TensorMapErase(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_handle, ::tensorflow::Input key, DataType value_dtype);
  operator ::tensorflow::Output() const { return output_handle; }
  operator ::tensorflow::Input() const { return output_handle; }
  ::tensorflow::Node* node() const { return output_handle.node(); }

  Operation operation;
  ::tensorflow::Output output_handle;
};

/// Returns whether the given key exists in the map.
///
/// input_handle: the input map
/// key: the key to check
/// has_key: whether the key is already in the map or not
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The has_key tensor.
class TensorMapHasKey {
 public:
  TensorMapHasKey(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_handle, ::tensorflow::Input key);
  operator ::tensorflow::Output() const { return has_key; }
  operator ::tensorflow::Input() const { return has_key; }
  ::tensorflow::Node* node() const { return has_key.node(); }

  Operation operation;
  ::tensorflow::Output has_key;
};

/// Returns a map that is the 'input_handle' with the given key-value pair inserted.
///
/// input_handle: the original map
/// output_handle: the map with key and value inserted
/// key: the key to be inserted
/// value: the value to be inserted
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output_handle tensor.
class TensorMapInsert {
 public:
  TensorMapInsert(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_handle, ::tensorflow::Input key, ::tensorflow::Input
                value);
  operator ::tensorflow::Output() const { return output_handle; }
  operator ::tensorflow::Input() const { return output_handle; }
  ::tensorflow::Node* node() const { return output_handle.node(); }

  Operation operation;
  ::tensorflow::Output output_handle;
};

/// Returns the value from a given key in a tensor map.
///
/// input_handle: the input map
/// key: the key to be looked up
/// value: the value found from the given key
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The value tensor.
class TensorMapLookup {
 public:
  TensorMapLookup(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_handle, ::tensorflow::Input key, DataType value_dtype);
  operator ::tensorflow::Output() const { return value; }
  operator ::tensorflow::Input() const { return value; }
  ::tensorflow::Node* node() const { return value.node(); }

  Operation operation;
  ::tensorflow::Output value;
};

/// Returns the number of tensors in the input tensor map.
///
/// input_handle: the input map
/// size: the number of tensors in the map
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The size tensor.
class TensorMapSize {
 public:
  TensorMapSize(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_handle);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  Operation operation;
  ::tensorflow::Output size;
};

/// Returns a Tensor stack of all keys in a tensor map.
///
/// input_handle: the input map
/// keys: the returned Tensor of all keys in the map
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The keys tensor.
class TensorMapStackKeys {
 public:
  TensorMapStackKeys(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_handle, DataType key_dtype);
  operator ::tensorflow::Output() const { return keys; }
  operator ::tensorflow::Input() const { return keys; }
  ::tensorflow::Node* node() const { return keys.node(); }

  Operation operation;
  ::tensorflow::Output keys;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_MAP_OPS_H_
