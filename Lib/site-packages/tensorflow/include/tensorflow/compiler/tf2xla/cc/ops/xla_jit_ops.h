// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_COMPILER_TF_XLA_CC_OPS_XLA_JIT_OPS_H_
#define TENSORFLOW_COMPILER_TF_XLA_CC_OPS_XLA_JIT_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup xla_jit_ops Xla Jit Ops
/// @{

/// Operator that connects the output of an XLA computation to other consumer graph nodes.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The outputs tensor.
class XlaClusterOutput {
 public:
  XlaClusterOutput(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return outputs; }
  operator ::tensorflow::Input() const { return outputs; }
  ::tensorflow::Node* node() const { return outputs.node(); }

  Operation operation;
  ::tensorflow::Output outputs;
};

/// XLA Launch Op. For use by the XLA JIT only.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The results tensor.
class XlaLaunch {
 public:
  XlaLaunch(const ::tensorflow::Scope& scope, ::tensorflow::InputList constants,
          ::tensorflow::InputList args, ::tensorflow::InputList resources,
          const DataTypeSlice& Tresults, const NameAttrList& function);
  ::tensorflow::Output operator[](size_t index) const { return results[index]; }


  Operation operation;
  ::tensorflow::OutputList results;
};

/// XLA Launch Op. For use by the XLA JIT only.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The results tensor.
class XlaLaunchV2 {
 public:
  XlaLaunchV2(const ::tensorflow::Scope& scope, ::tensorflow::InputList args,
            const DataTypeSlice& Tresults, const gtl::ArraySlice<int>&
            constants, const gtl::ArraySlice<int>& resources, const
            NameAttrList& function);
  ::tensorflow::Output operator[](size_t index) const { return results[index]; }


  Operation operation;
  ::tensorflow::OutputList results;
};

/// XLA Compile Op. For use by the XLA JIT only.
///
/// Compiles a TensorFlow function into an XLA LocalExecutable and returns a key
/// that _XlaRun can use to look up the LocalExecutable and execute it.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` key: A key that can be used to look up the local executable compiled by the
/// node and associated metadata.
/// * `Output` compilation_successful: If the `must_compile` attr is false the _XlaCompile op
/// can decide not to compile the clusters based on some profitability
/// heuristics.  In that case `compilation_successful` is false if _XlaCompile
/// chose not to compile the cluster.  If the `must_compile` attr is true then
/// _XlaCompile always attempts to compile the cluster and
/// `compilation_successful` is always true.
class _XlaCompile {
 public:
  _XlaCompile(const ::tensorflow::Scope& scope, ::tensorflow::InputList
            constants, ::tensorflow::InputList args, ::tensorflow::InputList
            resources, bool must_compile, const NameAttrList& function);

  Operation operation;
  ::tensorflow::Output key;
  ::tensorflow::Output compilation_successful;
};

/// XLA Merge Op. For use by the XLA JIT only.
///
/// Merges the outputs from the PartitionedCall node and the _XlaRun node.
/// Unlike the TensorFlow Merge op, which requires inputs of some types to be
/// placed on the host, the _XlaMerge op can merge inputs of all types when
/// placed on the device. This prevents the need for copy operations, in
/// particular when an XLA cluster has int32 outputs. The _XlaMerge up does not
/// have a value_index output that identifies the chosen input.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class _XlaMerge {
 public:
  _XlaMerge(const ::tensorflow::Scope& scope, ::tensorflow::Input
          partitioned_call, ::tensorflow::Input xla_run);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// XLA Run Op. For use by the XLA JIT only.
///
/// Executes a TensorFlow function previously compiled into a LocalExecutable by an
/// _XlaCompile op.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The results tensor.
class _XlaRun {
 public:
  _XlaRun(const ::tensorflow::Scope& scope, ::tensorflow::InputList args,
        ::tensorflow::Input key, const DataTypeSlice& Tresults);
  ::tensorflow::Output operator[](size_t index) const { return results[index]; }


  Operation operation;
  ::tensorflow::OutputList results;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF_XLA_CC_OPS_XLA_JIT_OPS_H_
