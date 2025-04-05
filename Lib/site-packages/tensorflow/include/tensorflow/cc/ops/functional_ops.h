// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_FUNCTIONAL_OPS_H_
#define TENSORFLOW_CC_OPS_FUNCTIONAL_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup functional_ops Functional Ops
/// @{

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class Case {
 public:
  /// Optional attribute setters for Case
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
  };
  Case(const ::tensorflow::Scope& scope, ::tensorflow::Input branch_index,
     ::tensorflow::InputList input, const DataTypeSlice& Tout, const
     gtl::ArraySlice<NameAttrList>& branches);
  Case(const ::tensorflow::Scope& scope, ::tensorflow::Input branch_index,
     ::tensorflow::InputList input, const DataTypeSlice& Tout, const
     gtl::ArraySlice<NameAttrList>& branches, const Case::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The index tensor.
class DeviceIndex {
 public:
  DeviceIndex(const ::tensorflow::Scope& scope, const
            gtl::ArraySlice<::tensorflow::tstring>& device_names);
  operator ::tensorflow::Output() const { return index; }
  operator ::tensorflow::Input() const { return index; }
  ::tensorflow::Node* node() const { return index.node(); }

  Operation operation;
  ::tensorflow::Output index;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class FakeParam {
 public:
  FakeParam(const ::tensorflow::Scope& scope, DataType dtype, PartialTensorShape
          shape);
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
/// * `OutputList`: The output tensor.
class For {
 public:
  For(const ::tensorflow::Scope& scope, ::tensorflow::Input start,
    ::tensorflow::Input limit, ::tensorflow::Input delta,
    ::tensorflow::InputList input, const NameAttrList& body);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class If {
 public:
  /// Optional attribute setters for If
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
  };
  If(const ::tensorflow::Scope& scope, ::tensorflow::Input cond,
   ::tensorflow::InputList input, const DataTypeSlice& Tout, const
   NameAttrList& then_branch, const NameAttrList& else_branch);
  If(const ::tensorflow::Scope& scope, ::tensorflow::Input cond,
   ::tensorflow::InputList input, const DataTypeSlice& Tout, const
   NameAttrList& then_branch, const NameAttrList& else_branch, const If::Attrs&
   attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class PartitionedCall {
 public:
  /// Optional attribute setters for PartitionedCall
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Config(StringPiece x) {
      Attrs ret = *this;
      ret.config_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ConfigProto(StringPiece x) {
      Attrs ret = *this;
      ret.config_proto_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ExecutorType(StringPiece x) {
      Attrs ret = *this;
      ret.executor_type_ = x;
      return ret;
    }

    StringPiece config_ = "";
    StringPiece config_proto_ = "";
    StringPiece executor_type_ = "";
  };
  PartitionedCall(const ::tensorflow::Scope& scope, ::tensorflow::InputList args,
                const DataTypeSlice& Tout, const NameAttrList& f);
  PartitionedCall(const ::tensorflow::Scope& scope, ::tensorflow::InputList args,
                const DataTypeSlice& Tout, const NameAttrList& f, const
                PartitionedCall::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs Config(StringPiece x) {
    return Attrs().Config(x);
  }
  static Attrs ConfigProto(StringPiece x) {
    return Attrs().ConfigProto(x);
  }
  static Attrs ExecutorType(StringPiece x) {
    return Attrs().ExecutorType(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class RemoteCall {
 public:
  RemoteCall(const ::tensorflow::Scope& scope, ::tensorflow::Input target,
           ::tensorflow::InputList args, const DataTypeSlice& Tout, const
           NameAttrList& f);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class StatefulPartitionedCall {
 public:
  /// Optional attribute setters for StatefulPartitionedCall
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Config(StringPiece x) {
      Attrs ret = *this;
      ret.config_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ConfigProto(StringPiece x) {
      Attrs ret = *this;
      ret.config_proto_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ExecutorType(StringPiece x) {
      Attrs ret = *this;
      ret.executor_type_ = x;
      return ret;
    }

    StringPiece config_ = "";
    StringPiece config_proto_ = "";
    StringPiece executor_type_ = "";
  };
  StatefulPartitionedCall(const ::tensorflow::Scope& scope,
                        ::tensorflow::InputList args, const DataTypeSlice&
                        Tout, const NameAttrList& f);
  StatefulPartitionedCall(const ::tensorflow::Scope& scope,
                        ::tensorflow::InputList args, const DataTypeSlice&
                        Tout, const NameAttrList& f, const
                        StatefulPartitionedCall::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs Config(StringPiece x) {
    return Attrs().Config(x);
  }
  static Attrs ConfigProto(StringPiece x) {
    return Attrs().ConfigProto(x);
  }
  static Attrs ExecutorType(StringPiece x) {
    return Attrs().ExecutorType(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class StatelessCase {
 public:
  /// Optional attribute setters for StatelessCase
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
  };
  StatelessCase(const ::tensorflow::Scope& scope, ::tensorflow::Input
              branch_index, ::tensorflow::InputList input, const DataTypeSlice&
              Tout, const gtl::ArraySlice<NameAttrList>& branches);
  StatelessCase(const ::tensorflow::Scope& scope, ::tensorflow::Input
              branch_index, ::tensorflow::InputList input, const DataTypeSlice&
              Tout, const gtl::ArraySlice<NameAttrList>& branches, const
              StatelessCase::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class StatelessIf {
 public:
  /// Optional attribute setters for StatelessIf
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
  };
  StatelessIf(const ::tensorflow::Scope& scope, ::tensorflow::Input cond,
            ::tensorflow::InputList input, const DataTypeSlice& Tout, const
            NameAttrList& then_branch, const NameAttrList& else_branch);
  StatelessIf(const ::tensorflow::Scope& scope, ::tensorflow::Input cond,
            ::tensorflow::InputList input, const DataTypeSlice& Tout, const
            NameAttrList& then_branch, const NameAttrList& else_branch, const
            StatelessIf::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class StatelessWhile {
 public:
  /// Optional attribute setters for StatelessWhile
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    /// Defaults to 10
    TF_MUST_USE_RESULT Attrs ParallelIterations(int64 x) {
      Attrs ret = *this;
      ret.parallel_iterations_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
    int64 parallel_iterations_ = 10;
  };
  StatelessWhile(const ::tensorflow::Scope& scope, ::tensorflow::InputList input,
               const NameAttrList& cond, const NameAttrList& body);
  StatelessWhile(const ::tensorflow::Scope& scope, ::tensorflow::InputList input,
               const NameAttrList& cond, const NameAttrList& body, const
               StatelessWhile::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }
  static Attrs ParallelIterations(int64 x) {
    return Attrs().ParallelIterations(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class SymbolicGradient {
 public:
  SymbolicGradient(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                 input, const DataTypeSlice& Tout, const NameAttrList& f);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class ToBool {
 public:
  ToBool(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
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
/// * `OutputList`: The output tensor.
class While {
 public:
  /// Optional attribute setters for While
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    /// Defaults to 10
    TF_MUST_USE_RESULT Attrs ParallelIterations(int64 x) {
      Attrs ret = *this;
      ret.parallel_iterations_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
    int64 parallel_iterations_ = 10;
  };
  While(const ::tensorflow::Scope& scope, ::tensorflow::InputList input, const
      NameAttrList& cond, const NameAttrList& body);
  While(const ::tensorflow::Scope& scope, ::tensorflow::InputList input, const
      NameAttrList& cond, const NameAttrList& body, const While::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }
  static Attrs ParallelIterations(int64 x) {
    return Attrs().ParallelIterations(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// output = cond ? then_branch(input) : else_branch(input)
///
/// Args:
/// * scope: A Scope object
/// * cond: A Tensor. If the tensor is a scalar of non-boolean type, the
/// scalar is converted to a boolean according to the
/// following rule: if the scalar is a numerical value, non-zero means
/// True and zero means False; if the scalar is a string, non-empty
/// means True and empty means False. If the tensor is not a scalar,
/// being empty means False and being non-empty means True.
/// * input: A list of input tensors.
/// * then_branch: A function that takes 'inputs' and returns a list of
/// tensors, whose types are the same as what else_branch returns.
/// * else_branch: A function that takes 'inputs' and returns a list of
/// tensors.  whose types are the same as what then_branch returns.
///
/// Returns:
/// * `OutputList`: The output tensor.
class _If {
 public:
  _If(const ::tensorflow::Scope& scope, ::tensorflow::Input cond,
    ::tensorflow::InputList input, const DataTypeSlice& Tout, const
    NameAttrList& then_branch, const NameAttrList& else_branch);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// output = input; While (Cond(output)) { output = Body(output) }
///
/// Args:
/// * scope: A Scope object
/// * input: A list of input tensors whose types are T.
/// * cond: A function takes 'input' and returns a tensor.  If the tensor is
/// a scalar of non-boolean, the scalar is converted to a boolean
/// according to the following rule: if the scalar is a numerical
/// value, non-zero means True and zero means False; if the scalar is
/// a string, non-empty means True and empty means False. If the
/// tensor is not a scalar, non-emptiness means True and False
/// otherwise.
/// * body: A function that takes a list of tensors and returns another
/// list of tensors. Both lists have the same types as specified
/// by T.
///
/// Returns:
/// * `OutputList`: A list of output tensors whose types are T.
class _While {
 public:
  _While(const ::tensorflow::Scope& scope, ::tensorflow::InputList input, const
       NameAttrList& cond, const NameAttrList& body);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_FUNCTIONAL_OPS_H_
