// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_RESOURCE_VARIABLE_OPS_H_
#define TENSORFLOW_CC_OPS_RESOURCE_VARIABLE_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup resource_variable_ops Resource Variable Ops
/// @{

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class AssignAddVariableOp {
 public:
  AssignAddVariableOp(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    resource, ::tensorflow::Input value);
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
class AssignSubVariableOp {
 public:
  AssignSubVariableOp(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    resource, ::tensorflow::Input value);
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
class AssignVariableOp {
 public:
  /// Optional attribute setters for AssignVariableOp
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ValidateShape(bool x) {
      Attrs ret = *this;
      ret.validate_shape_ = x;
      return ret;
    }

    bool validate_shape_ = false;
  };
  AssignVariableOp(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 resource, ::tensorflow::Input value);
  AssignVariableOp(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 resource, ::tensorflow::Input value, const
                 AssignVariableOp::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs ValidateShape(bool x) {
    return Attrs().ValidateShape(x);
  }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class ConsumeMutexLock {
 public:
  ConsumeMutexLock(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 mutex_lock);
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
class DestroyResourceOp {
 public:
  /// Optional attribute setters for DestroyResourceOp
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs IgnoreLookupError(bool x) {
      Attrs ret = *this;
      ret.ignore_lookup_error_ = x;
      return ret;
    }

    bool ignore_lookup_error_ = true;
  };
  DestroyResourceOp(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  resource);
  DestroyResourceOp(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  resource, const DestroyResourceOp::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs IgnoreLookupError(bool x) {
    return Attrs().IgnoreLookupError(x);
  }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class DisableCopyOnRead {
 public:
  DisableCopyOnRead(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  resource);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The mutex_lock tensor.
class MutexLock {
 public:
  MutexLock(const ::tensorflow::Scope& scope, ::tensorflow::Input mutex);
  operator ::tensorflow::Output() const { return mutex_lock; }
  operator ::tensorflow::Input() const { return mutex_lock; }
  ::tensorflow::Node* node() const { return mutex_lock.node(); }

  Operation operation;
  ::tensorflow::Output mutex_lock;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The resource tensor.
class MutexV2 {
 public:
  /// Optional attribute setters for MutexV2
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
  MutexV2(const ::tensorflow::Scope& scope);
  MutexV2(const ::tensorflow::Scope& scope, const MutexV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return resource; }
  operator ::tensorflow::Input() const { return resource; }
  ::tensorflow::Node* node() const { return resource.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output resource;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The value tensor.
class ReadVariableOp {
 public:
  ReadVariableOp(const ::tensorflow::Scope& scope, ::tensorflow::Input resource,
               DataType dtype);
  operator ::tensorflow::Output() const { return value; }
  operator ::tensorflow::Input() const { return value; }
  ::tensorflow::Node* node() const { return value.node(); }

  Operation operation;
  ::tensorflow::Output value;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class ResourceGather {
 public:
  /// Optional attribute setters for ResourceGather
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BatchDims(int64 x) {
      Attrs ret = *this;
      ret.batch_dims_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ValidateIndices(bool x) {
      Attrs ret = *this;
      ret.validate_indices_ = x;
      return ret;
    }

    int64 batch_dims_ = 0;
    bool validate_indices_ = true;
  };
  ResourceGather(const ::tensorflow::Scope& scope, ::tensorflow::Input resource,
               ::tensorflow::Input indices, DataType dtype);
  ResourceGather(const ::tensorflow::Scope& scope, ::tensorflow::Input resource,
               ::tensorflow::Input indices, DataType dtype, const
               ResourceGather::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BatchDims(int64 x) {
    return Attrs().BatchDims(x);
  }
  static Attrs ValidateIndices(bool x) {
    return Attrs().ValidateIndices(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class ResourceGatherNd {
 public:
  ResourceGatherNd(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 resource, ::tensorflow::Input indices, DataType dtype);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class ResourceScatterAdd {
 public:
  ResourceScatterAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   resource, ::tensorflow::Input indices, ::tensorflow::Input
                   updates);
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
class ResourceScatterDiv {
 public:
  ResourceScatterDiv(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   resource, ::tensorflow::Input indices, ::tensorflow::Input
                   updates);
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
class ResourceScatterMax {
 public:
  ResourceScatterMax(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   resource, ::tensorflow::Input indices, ::tensorflow::Input
                   updates);
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
class ResourceScatterMin {
 public:
  ResourceScatterMin(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   resource, ::tensorflow::Input indices, ::tensorflow::Input
                   updates);
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
class ResourceScatterMul {
 public:
  ResourceScatterMul(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   resource, ::tensorflow::Input indices, ::tensorflow::Input
                   updates);
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
class ResourceScatterSub {
 public:
  ResourceScatterSub(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   resource, ::tensorflow::Input indices, ::tensorflow::Input
                   updates);
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
class ResourceScatterUpdate {
 public:
  ResourceScatterUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      resource, ::tensorflow::Input indices,
                      ::tensorflow::Input updates);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The resource tensor.
class VarHandleOp {
 public:
  /// Optional attribute setters for VarHandleOp
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

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs DebugName(StringPiece x) {
      Attrs ret = *this;
      ret.debug_name_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs AllowedDevices(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.allowed_devices_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    StringPiece debug_name_ = "";
    gtl::ArraySlice<::tensorflow::tstring> allowed_devices_ = {};
  };
  VarHandleOp(const ::tensorflow::Scope& scope, DataType dtype,
            PartialTensorShape shape);
  VarHandleOp(const ::tensorflow::Scope& scope, DataType dtype,
            PartialTensorShape shape, const VarHandleOp::Attrs& attrs);
  operator ::tensorflow::Output() const { return resource; }
  operator ::tensorflow::Input() const { return resource; }
  ::tensorflow::Node* node() const { return resource.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs DebugName(StringPiece x) {
    return Attrs().DebugName(x);
  }
  static Attrs AllowedDevices(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().AllowedDevices(x);
  }

  Operation operation;
  ::tensorflow::Output resource;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The is_initialized tensor.
class VarIsInitializedOp {
 public:
  VarIsInitializedOp(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   resource);
  operator ::tensorflow::Output() const { return is_initialized; }
  operator ::tensorflow::Input() const { return is_initialized; }
  ::tensorflow::Node* node() const { return is_initialized.node(); }

  Operation operation;
  ::tensorflow::Output is_initialized;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class VariableShape {
 public:
  /// Optional attribute setters for VariableShape
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_INT32;
  };
  VariableShape(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  VariableShape(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              const VariableShape::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The values tensor.
class _ReadVariablesOp {
 public:
  _ReadVariablesOp(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                 resources, const DataTypeSlice& dtypes);
  ::tensorflow::Output operator[](size_t index) const { return values[index]; }


  Operation operation;
  ::tensorflow::OutputList values;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The resources tensor.
class _VarHandlesOp {
 public:
  _VarHandlesOp(const ::tensorflow::Scope& scope, const
              gtl::ArraySlice<::tensorflow::tstring>& containers, const
              gtl::ArraySlice<::tensorflow::tstring>& shared_names, int64 N,
              const DataTypeSlice& dtypes, const
              gtl::ArraySlice<PartialTensorShape>& shapes);
  ::tensorflow::Output operator[](size_t index) const { return resources[index]; }


  Operation operation;
  ::tensorflow::OutputList resources;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_RESOURCE_VARIABLE_OPS_H_
