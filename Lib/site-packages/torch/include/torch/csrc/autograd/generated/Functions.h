#pragma once

// @generated from ..\tools\autograd\templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include <torch/csrc/Export.h>

#include <c10/core/SymIntArrayRef.h>

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using std::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [&saved_for](const SavedVariable& x) {
    // TODO(crcrpar): Use `std::move(saved_for)` to avoid incrementing refcount, which would need refactoring.
    return static_cast<Tensor>(x.unpack(saved_for));
  });
}

inline c10::List<std::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr) {
  torch::List<std::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    auto var = v.unpack(saved_for);
    result.push_back(var.defined() ? std::optional<Tensor>(var) : ::std::nullopt);
  }
  return result;
}

using torch::autograd::TypeAndSize;

#ifdef _WIN32
struct AbsBackward0 : public TraceableFunction {
  TORCH_API AbsBackward0() = default;
#else
struct TORCH_API AbsBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AbsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AcosBackward0 : public TraceableFunction {
  TORCH_API AcosBackward0() = default;
#else
struct TORCH_API AcosBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcosBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AddBackward0 : public TraceableFunction {
  TORCH_API AddBackward0() = default;
#else
struct TORCH_API AddBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct AddBackward1 : public TraceableFunction {
  TORCH_API AddBackward1() = default;
#else
struct TORCH_API AddBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct AddbmmBackward0 : public TraceableFunction {
  TORCH_API AddbmmBackward0() = default;
#else
struct TORCH_API AddbmmBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddbmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch1_.reset_data();
    batch2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable batch1_;
  c10::SymInt batch1_sym_argsize_0;
  c10::SymInt batch1_sym_argsize_1;
  SavedVariable batch2_;
  c10::SymInt batch2_sym_argsize_2;
  at::Scalar beta;

};
#ifdef _WIN32
struct AddcdivBackward0 : public TraceableFunction {
  TORCH_API AddcdivBackward0() = default;
#else
struct TORCH_API AddcdivBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcdivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor1_.reset_data();
    tensor2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
  SavedVariable tensor1_;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::ScalarType tensor2_scalar_type;
  at::Scalar value;

};
#ifdef _WIN32
struct AddcmulBackward0 : public TraceableFunction {
  TORCH_API AddcmulBackward0() = default;
#else
struct TORCH_API AddcmulBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcmulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor1_.reset_data();
    tensor2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
  SavedVariable tensor1_;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::ScalarType tensor2_scalar_type;
  at::Scalar value;

};
#ifdef _WIN32
struct AddmmBackward0 : public TraceableFunction {
  TORCH_API AddmmBackward0() = default;
#else
struct TORCH_API AddmmBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat1_.reset_data();
    mat2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat1_;
  at::Layout mat1_layout;
  std::vector<c10::SymInt> mat1_sym_sizes;
  std::vector<c10::SymInt> mat1_sym_strides;
  SavedVariable mat2_;
  at::Layout mat2_layout;
  std::vector<c10::SymInt> mat2_sym_sizes;
  std::vector<c10::SymInt> mat2_sym_strides;

};
#ifdef _WIN32
struct SparseAddmmBackward0 : public TraceableFunction {
  TORCH_API SparseAddmmBackward0() = default;
#else
struct TORCH_API SparseAddmmBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseAddmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat1_.reset_data();
    mat2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat1_;
  SavedVariable mat2_;
  at::Layout mat2_layout;
  std::vector<c10::SymInt> mat2_sym_sizes;
  std::vector<c10::SymInt> mat2_sym_strides;

};
#ifdef _WIN32
struct AddmvBackward0 : public TraceableFunction {
  TORCH_API AddmvBackward0() = default;
#else
struct TORCH_API AddmvBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmvBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat_.reset_data();
    vec_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat_;
  SavedVariable vec_;

};
#ifdef _WIN32
struct AddrBackward0 : public TraceableFunction {
  TORCH_API AddrBackward0() = default;
#else
struct TORCH_API AddrBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec1_.reset_data();
    vec2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable vec1_;
  SavedVariable vec2_;

};
#ifdef _WIN32
struct AffineGridGeneratorBackward0 : public TraceableFunction {
  TORCH_API AffineGridGeneratorBackward0() = default;
#else
struct TORCH_API AffineGridGeneratorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AffineGridGeneratorBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> size;

};
#ifdef _WIN32
struct AliasBackward0 : public Node {
  TORCH_API AliasBackward0() = default;
#else
struct TORCH_API AliasBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AliasBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct AngleBackward0 : public TraceableFunction {
  TORCH_API AngleBackward0() = default;
#else
struct TORCH_API AngleBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AngleBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AcoshBackward0 : public TraceableFunction {
  TORCH_API AcoshBackward0() = default;
#else
struct TORCH_API AcoshBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcoshBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AcoshBackward1 : public TraceableFunction {
  TORCH_API AcoshBackward1() = default;
#else
struct TORCH_API AcoshBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcoshBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct AsinhBackward0 : public TraceableFunction {
  TORCH_API AsinhBackward0() = default;
#else
struct TORCH_API AsinhBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AsinhBackward1 : public TraceableFunction {
  TORCH_API AsinhBackward1() = default;
#else
struct TORCH_API AsinhBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinhBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct AtanhBackward0 : public TraceableFunction {
  TORCH_API AtanhBackward0() = default;
#else
struct TORCH_API AtanhBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AtanhBackward1 : public TraceableFunction {
  TORCH_API AtanhBackward1() = default;
#else
struct TORCH_API AtanhBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanhBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct AsStridedBackward0 : public Node {
  TORCH_API AsStridedBackward0() = default;
#else
struct TORCH_API AsStridedBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  ::std::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;

};
#ifdef _WIN32
struct AsStridedBackward1 : public TraceableFunction {
  TORCH_API AsStridedBackward1() = default;
#else
struct TORCH_API AsStridedBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  ::std::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;

};
#ifdef _WIN32
struct AsinBackward0 : public TraceableFunction {
  TORCH_API AsinBackward0() = default;
#else
struct TORCH_API AsinBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AtanBackward0 : public TraceableFunction {
  TORCH_API AtanBackward0() = default;
#else
struct TORCH_API AtanBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct Atan2Backward0 : public TraceableFunction {
  TORCH_API Atan2Backward0() = default;
#else
struct TORCH_API Atan2Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Atan2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct BaddbmmBackward0 : public TraceableFunction {
  TORCH_API BaddbmmBackward0() = default;
#else
struct TORCH_API BaddbmmBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BaddbmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch1_.reset_data();
    batch2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable batch1_;
  SavedVariable batch2_;
  at::Scalar beta;

};
#ifdef _WIN32
struct BernoulliBackward0 : public TraceableFunction {
  TORCH_API BernoulliBackward0() = default;
#else
struct TORCH_API BernoulliBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct BernoulliBackward1 : public TraceableFunction {
  TORCH_API BernoulliBackward1() = default;
#else
struct TORCH_API BernoulliBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize p_info;

};
#ifdef _WIN32
struct BernoulliBackward2 : public TraceableFunction {
  TORCH_API BernoulliBackward2() = default;
#else
struct TORCH_API BernoulliBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward2"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct BmmBackward0 : public TraceableFunction {
  TORCH_API BmmBackward0() = default;
#else
struct TORCH_API BmmBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat2_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mat2_;
  SavedVariable self_;

};
#ifdef _WIN32
struct MatmulBackward0 : public TraceableFunction {
  TORCH_API MatmulBackward0() = default;
#else
struct TORCH_API MatmulBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MatmulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct CatBackward0 : public TraceableFunction {
  TORCH_API CatBackward0() = default;
#else
struct TORCH_API CatBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CatBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::vector<at::ScalarType> tensors_args_scalartypes;
  ::std::vector<::std::vector<c10::SymInt>> tensors_args_sizes_symint;
  size_t tensors_size_;
};
#ifdef _WIN32
struct CauchyBackward0 : public TraceableFunction {
  TORCH_API CauchyBackward0() = default;
#else
struct TORCH_API CauchyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CauchyBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct CeilBackward0 : public TraceableFunction {
  TORCH_API CeilBackward0() = default;
#else
struct TORCH_API CeilBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeilBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct CholeskyBackward0 : public TraceableFunction {
  TORCH_API CholeskyBackward0() = default;
#else
struct TORCH_API CholeskyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool upper;
  SavedVariable result_;

};
#ifdef _WIN32
struct ChunkBackward0 : public TraceableFunction {
  TORCH_API ChunkBackward0() = default;
#else
struct TORCH_API ChunkBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ChunkBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ChunkBackwardAutogradNestedTensor0 : public TraceableFunction {
  TORCH_API ChunkBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API ChunkBackwardAutogradNestedTensor0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ChunkBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t chunks = 0;
  int64_t dim = 0;
  SavedVariable self_;

};
#ifdef _WIN32
struct LinalgCholeskyExBackward0 : public TraceableFunction {
  TORCH_API LinalgCholeskyExBackward0() = default;
#else
struct TORCH_API LinalgCholeskyExBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgCholeskyExBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    L_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool upper;
  SavedVariable L_;

};
#ifdef _WIN32
struct CholeskySolveBackward0 : public TraceableFunction {
  TORCH_API CholeskySolveBackward0() = default;
#else
struct TORCH_API CholeskySolveBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskySolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input2_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable input2_;
  SavedVariable self_;
  bool upper;
  SavedVariable result_;

};
#ifdef _WIN32
struct CholeskyInverseBackward0 : public TraceableFunction {
  TORCH_API CholeskyInverseBackward0() = default;
#else
struct TORCH_API CholeskyInverseBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyInverseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  bool upper;
  SavedVariable result_;

};
#ifdef _WIN32
struct ClampBackward0 : public TraceableFunction {
  TORCH_API ClampBackward0() = default;
#else
struct TORCH_API ClampBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    max_.reset_data();
    min_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable max_;
  SavedVariable min_;
  SavedVariable self_;

};
#ifdef _WIN32
struct ClampBackward1 : public TraceableFunction {
  TORCH_API ClampBackward1() = default;
#else
struct TORCH_API ClampBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<at::Scalar> max;
  ::std::optional<at::Scalar> min;
  SavedVariable self_;

};
#ifdef _WIN32
struct ClampMinBackward0 : public TraceableFunction {
  TORCH_API ClampMinBackward0() = default;
#else
struct TORCH_API ClampMinBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar min;
  SavedVariable self_;

};
#ifdef _WIN32
struct ClampMinBackward1 : public TraceableFunction {
  TORCH_API ClampMinBackward1() = default;
#else
struct TORCH_API ClampMinBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMinBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    min_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable min_;
  SavedVariable self_;

};
#ifdef _WIN32
struct ClampMaxBackward0 : public TraceableFunction {
  TORCH_API ClampMaxBackward0() = default;
#else
struct TORCH_API ClampMaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar max;
  SavedVariable self_;

};
#ifdef _WIN32
struct ClampMaxBackward1 : public TraceableFunction {
  TORCH_API ClampMaxBackward1() = default;
#else
struct TORCH_API ClampMaxBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMaxBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    max_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable max_;
  SavedVariable self_;

};
#ifdef _WIN32
struct CloneBackward0 : public TraceableFunction {
  TORCH_API CloneBackward0() = default;
#else
struct TORCH_API CloneBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CloneBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct LazyCloneBackward0 : public TraceableFunction {
  TORCH_API LazyCloneBackward0() = default;
#else
struct TORCH_API LazyCloneBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LazyCloneBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ToCopyBackward0 : public TraceableFunction {
  TORCH_API ToCopyBackward0() = default;
#else
struct TORCH_API ToCopyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToCopyBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::TensorOptions self_options;

};
#ifdef _WIN32
struct CoalesceBackward0 : public TraceableFunction {
  TORCH_API CoalesceBackward0() = default;
#else
struct TORCH_API CoalesceBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoalesceBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ComplexBackward0 : public TraceableFunction {
  TORCH_API ComplexBackward0() = default;
#else
struct TORCH_API ComplexBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ComplexBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    imag_.reset_data();
    real_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable imag_;
  SavedVariable real_;

};
#ifdef _WIN32
struct PolarBackward0 : public TraceableFunction {
  TORCH_API PolarBackward0() = default;
#else
struct TORCH_API PolarBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolarBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct ConjBackward0 : public Node {
  TORCH_API ConjBackward0() = default;
#else
struct TORCH_API ConjBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NegViewBackward0 : public Node {
  TORCH_API NegViewBackward0() = default;
#else
struct TORCH_API NegViewBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegViewBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ConjPhysicalBackward0 : public TraceableFunction {
  TORCH_API ConjPhysicalBackward0() = default;
#else
struct TORCH_API ConjPhysicalBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjPhysicalBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ConjPhysicalBackward1 : public TraceableFunction {
  TORCH_API ConjPhysicalBackward1() = default;
#else
struct TORCH_API ConjPhysicalBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjPhysicalBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct CopysignBackward0 : public TraceableFunction {
  TORCH_API CopysignBackward0() = default;
#else
struct TORCH_API CopysignBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CopysignBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct CopysignBackward1 : public TraceableFunction {
  TORCH_API CopysignBackward1() = default;
#else
struct TORCH_API CopysignBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CopysignBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct CosBackward0 : public TraceableFunction {
  TORCH_API CosBackward0() = default;
#else
struct TORCH_API CosBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CosBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct CoshBackward0 : public TraceableFunction {
  TORCH_API CoshBackward0() = default;
#else
struct TORCH_API CoshBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoshBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct LinalgCrossBackward0 : public TraceableFunction {
  TORCH_API LinalgCrossBackward0() = default;
#else
struct TORCH_API LinalgCrossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgCrossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct LogcumsumexpBackward0 : public TraceableFunction {
  TORCH_API LogcumsumexpBackward0() = default;
#else
struct TORCH_API LogcumsumexpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogcumsumexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct CumprodBackward0 : public TraceableFunction {
  TORCH_API CumprodBackward0() = default;
#else
struct TORCH_API CumprodBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumprodBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::ScalarType self_scalar_type;
  SavedVariable result_;

};
#ifdef _WIN32
struct CumsumBackward0 : public TraceableFunction {
  TORCH_API CumsumBackward0() = default;
#else
struct TORCH_API CumsumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumsumBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct CummaxBackward0 : public TraceableFunction {
  TORCH_API CummaxBackward0() = default;
#else
struct TORCH_API CummaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CummaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable indices_;

};
#ifdef _WIN32
struct CumminBackward0 : public TraceableFunction {
  TORCH_API CumminBackward0() = default;
#else
struct TORCH_API CumminBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumminBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable indices_;

};
#ifdef _WIN32
struct ConvTbcBackward0 : public TraceableFunction {
  TORCH_API ConvTbcBackward0() = default;
#else
struct TORCH_API ConvTbcBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvTbcBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    bias_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable bias_;
  int64_t pad = 0;
  SavedVariable self_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct CtcLossBackward0 : public TraceableFunction {
  TORCH_API CtcLossBackward0() = default;
#else
struct TORCH_API CtcLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CtcLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    log_probs_.reset_data();
    targets_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t blank = 0;
  std::vector<int64_t> input_lengths;
  SavedVariable log_probs_;
  std::vector<int64_t> target_lengths;
  SavedVariable targets_;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct CtcLossBackward1 : public TraceableFunction {
  TORCH_API CtcLossBackward1() = default;
#else
struct TORCH_API CtcLossBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CtcLossBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_lengths_.reset_data();
    log_probs_.reset_data();
    target_lengths_.reset_data();
    targets_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t blank = 0;
  SavedVariable input_lengths_;
  SavedVariable log_probs_;
  SavedVariable target_lengths_;
  SavedVariable targets_;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct Deg2RadBackward0 : public TraceableFunction {
  TORCH_API Deg2RadBackward0() = default;
#else
struct TORCH_API Deg2RadBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Deg2RadBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct LinalgDetBackward0 : public TraceableFunction {
  TORCH_API LinalgDetBackward0() = default;
#else
struct TORCH_API LinalgDetBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgDetBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    LU_.reset_data();
    pivots_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable A_;
  SavedVariable LU_;
  SavedVariable pivots_;
  SavedVariable result_;

};
#ifdef _WIN32
struct LinalgSlogdetBackward0 : public TraceableFunction {
  TORCH_API LinalgSlogdetBackward0() = default;
#else
struct TORCH_API LinalgSlogdetBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSlogdetBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    LU_.reset_data();
    pivots_.reset_data();
    sign_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable A_;
  SavedVariable LU_;
  SavedVariable pivots_;
  SavedVariable sign_;

};
#ifdef _WIN32
struct BlockDiagBackward0 : public TraceableFunction {
  TORCH_API BlockDiagBackward0() = default;
#else
struct TORCH_API BlockDiagBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BlockDiagBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::vector<at::ScalarType> tensors_args_scalartypes;
  ::std::vector<::std::vector<int64_t>> tensors_args_sizes;
  size_t tensors_size_;
};
#ifdef _WIN32
struct DiagEmbedBackward0 : public TraceableFunction {
  TORCH_API DiagEmbedBackward0() = default;
#else
struct TORCH_API DiagEmbedBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagEmbedBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;

};
#ifdef _WIN32
struct DiagonalBackward0 : public Node {
  TORCH_API DiagonalBackward0() = default;
#else
struct TORCH_API DiagonalBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct DiagonalBackwardBackward0 : public TraceableFunction {
  TORCH_API DiagonalBackwardBackward0() = default;
#else
struct TORCH_API DiagonalBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;

};
#ifdef _WIN32
struct DistBackward0 : public TraceableFunction {
  TORCH_API DistBackward0() = default;
#else
struct TORCH_API DistBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar p;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct DivBackward0 : public TraceableFunction {
  TORCH_API DivBackward0() = default;
#else
struct TORCH_API DivBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct DivBackward1 : public TraceableFunction {
  TORCH_API DivBackward1() = default;
#else
struct TORCH_API DivBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar other;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct DivBackward2 : public TraceableFunction {
  TORCH_API DivBackward2() = default;
#else
struct TORCH_API DivBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  std::optional<std::string> rounding_mode;
  SavedVariable self_;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct DivBackward3 : public TraceableFunction {
  TORCH_API DivBackward3() = default;
#else
struct TORCH_API DivBackward3 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward3"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar other;
  std::optional<std::string> rounding_mode;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct DotBackward0 : public TraceableFunction {
  TORCH_API DotBackward0() = default;
#else
struct TORCH_API DotBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DotBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    tensor_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable tensor_;

};
#ifdef _WIN32
struct VdotBackward0 : public TraceableFunction {
  TORCH_API VdotBackward0() = default;
#else
struct TORCH_API VdotBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VdotBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct FusedDropoutBackward0 : public TraceableFunction {
  TORCH_API FusedDropoutBackward0() = default;
#else
struct TORCH_API FusedDropoutBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FusedDropoutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double p;
  SavedVariable result1_;

};
#ifdef _WIN32
struct NativeDropoutBackward0 : public TraceableFunction {
  TORCH_API NativeDropoutBackward0() = default;
#else
struct TORCH_API NativeDropoutBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeDropoutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double p;
  ::std::optional<bool> train;
  SavedVariable result1_;

};
#ifdef _WIN32
struct NativeDropoutBackwardBackward0 : public TraceableFunction {
  TORCH_API NativeDropoutBackwardBackward0() = default;
#else
struct TORCH_API NativeDropoutBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeDropoutBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable mask_;
  double scale;

};
#ifdef _WIN32
struct EqBackward0 : public TraceableFunction {
  TORCH_API EqBackward0() = default;
#else
struct TORCH_API EqBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct EqBackward1 : public TraceableFunction {
  TORCH_API EqBackward1() = default;
#else
struct TORCH_API EqBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct ErfBackward0 : public TraceableFunction {
  TORCH_API ErfBackward0() = default;
#else
struct TORCH_API ErfBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ErfcBackward0 : public TraceableFunction {
  TORCH_API ErfcBackward0() = default;
#else
struct TORCH_API ErfcBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfcBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SpecialErfcxBackward0 : public TraceableFunction {
  TORCH_API SpecialErfcxBackward0() = default;
#else
struct TORCH_API SpecialErfcxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialErfcxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct ErfinvBackward0 : public TraceableFunction {
  TORCH_API ErfinvBackward0() = default;
#else
struct TORCH_API ErfinvBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfinvBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ExpBackward0 : public TraceableFunction {
  TORCH_API ExpBackward0() = default;
#else
struct TORCH_API ExpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct Exp2Backward0 : public TraceableFunction {
  TORCH_API Exp2Backward0() = default;
#else
struct TORCH_API Exp2Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Exp2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct Expm1Backward0 : public TraceableFunction {
  TORCH_API Expm1Backward0() = default;
#else
struct TORCH_API Expm1Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Expm1Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct ExpandBackward0 : public Node {
  TORCH_API ExpandBackward0() = default;
#else
struct TORCH_API ExpandBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpandBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct ExponentialBackward0 : public TraceableFunction {
  TORCH_API ExponentialBackward0() = default;
#else
struct TORCH_API ExponentialBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExponentialBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FakeQuantizePerTensorAffineCachemaskBackward0 : public TraceableFunction {
  TORCH_API FakeQuantizePerTensorAffineCachemaskBackward0() = default;
#else
struct TORCH_API FakeQuantizePerTensorAffineCachemaskBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerTensorAffineCachemaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;

};
#ifdef _WIN32
struct FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0 : public TraceableFunction {
  TORCH_API FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0() = default;
#else
struct TORCH_API FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;

};
#ifdef _WIN32
struct FakeQuantizeLearnablePerTensorAffineBackward0 : public TraceableFunction {
  TORCH_API FakeQuantizeLearnablePerTensorAffineBackward0() = default;
#else
struct TORCH_API FakeQuantizeLearnablePerTensorAffineBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizeLearnablePerTensorAffineBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scale_.reset_data();
    self_.reset_data();
    zero_point_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double grad_factor;
  int64_t quant_max = 0;
  int64_t quant_min = 0;
  SavedVariable scale_;
  SavedVariable self_;
  SavedVariable zero_point_;

};
#ifdef _WIN32
struct FakeQuantizePerChannelAffineCachemaskBackward0 : public TraceableFunction {
  TORCH_API FakeQuantizePerChannelAffineCachemaskBackward0() = default;
#else
struct TORCH_API FakeQuantizePerChannelAffineCachemaskBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerChannelAffineCachemaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;

};
#ifdef _WIN32
struct FakeQuantizeLearnablePerChannelAffineBackward0 : public TraceableFunction {
  TORCH_API FakeQuantizeLearnablePerChannelAffineBackward0() = default;
#else
struct TORCH_API FakeQuantizeLearnablePerChannelAffineBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizeLearnablePerChannelAffineBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scale_.reset_data();
    self_.reset_data();
    zero_point_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t axis = 0;
  double grad_factor;
  int64_t quant_max = 0;
  int64_t quant_min = 0;
  SavedVariable scale_;
  SavedVariable self_;
  SavedVariable zero_point_;

};
#ifdef _WIN32
struct FusedMovingAvgObsFqHelperBackward0 : public TraceableFunction {
  TORCH_API FusedMovingAvgObsFqHelperBackward0() = default;
#else
struct TORCH_API FusedMovingAvgObsFqHelperBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FusedMovingAvgObsFqHelperBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;

};
#ifdef _WIN32
struct FillBackward0 : public TraceableFunction {
  TORCH_API FillBackward0() = default;
#else
struct TORCH_API FillBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FillBackward1 : public TraceableFunction {
  TORCH_API FillBackward1() = default;
#else
struct TORCH_API FillBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FillBackward2 : public TraceableFunction {
  TORCH_API FillBackward2() = default;
#else
struct TORCH_API FillBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward2"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FillBackward3 : public TraceableFunction {
  TORCH_API FillBackward3() = default;
#else
struct TORCH_API FillBackward3 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward3"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FloorBackward0 : public TraceableFunction {
  TORCH_API FloorBackward0() = default;
#else
struct TORCH_API FloorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FloorBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FmodBackward0 : public TraceableFunction {
  TORCH_API FmodBackward0() = default;
#else
struct TORCH_API FmodBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FmodBackward1 : public TraceableFunction {
  TORCH_API FmodBackward1() = default;
#else
struct TORCH_API FmodBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct FracBackward0 : public TraceableFunction {
  TORCH_API FracBackward0() = default;
#else
struct TORCH_API FracBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FracBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FrexpBackward0 : public TraceableFunction {
  TORCH_API FrexpBackward0() = default;
#else
struct TORCH_API FrexpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FrexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable exponent_;

};
#ifdef _WIN32
struct GatherBackward0 : public TraceableFunction {
  TORCH_API GatherBackward0() = default;
#else
struct TORCH_API GatherBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GatherBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
  SavedVariable self_;
  bool sparse_grad;

};
#ifdef _WIN32
struct GeBackward0 : public TraceableFunction {
  TORCH_API GeBackward0() = default;
#else
struct TORCH_API GeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct GeBackward1 : public TraceableFunction {
  TORCH_API GeBackward1() = default;
#else
struct TORCH_API GeBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct GeometricBackward0 : public TraceableFunction {
  TORCH_API GeometricBackward0() = default;
#else
struct TORCH_API GeometricBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeometricBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct GeqrfBackward0 : public TraceableFunction {
  TORCH_API GeqrfBackward0() = default;
#else
struct TORCH_API GeqrfBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeqrfBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct GridSampler2DBackward0 : public TraceableFunction {
  TORCH_API GridSampler2DBackward0() = default;
#else
struct TORCH_API GridSampler2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    input_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  SavedVariable grid_;
  SavedVariable input_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;

};
#ifdef _WIN32
struct GridSampler3DBackward0 : public TraceableFunction {
  TORCH_API GridSampler3DBackward0() = default;
#else
struct TORCH_API GridSampler3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    input_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  SavedVariable grid_;
  SavedVariable input_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;

};
#ifdef _WIN32
struct GridSampler2DCpuFallbackBackward0 : public TraceableFunction {
  TORCH_API GridSampler2DCpuFallbackBackward0() = default;
#else
struct TORCH_API GridSampler2DCpuFallbackBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler2DCpuFallbackBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    input_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  SavedVariable grid_;
  SavedVariable input_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;

};
#ifdef _WIN32
struct GtBackward0 : public TraceableFunction {
  TORCH_API GtBackward0() = default;
#else
struct TORCH_API GtBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct GtBackward1 : public TraceableFunction {
  TORCH_API GtBackward1() = default;
#else
struct TORCH_API GtBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct HardsigmoidBackward0 : public TraceableFunction {
  TORCH_API HardsigmoidBackward0() = default;
#else
struct TORCH_API HardsigmoidBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardsigmoidBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct HardswishBackward0 : public TraceableFunction {
  TORCH_API HardswishBackward0() = default;
#else
struct TORCH_API HardswishBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardswishBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct HardswishBackwardBackward0 : public TraceableFunction {
  TORCH_API HardswishBackwardBackward0() = default;
#else
struct TORCH_API HardswishBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardswishBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable self_;
  at::TensorOptions self_options;

};
#ifdef _WIN32
struct HypotBackward0 : public TraceableFunction {
  TORCH_API HypotBackward0() = default;
#else
struct TORCH_API HypotBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HypotBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct I0Backward0 : public TraceableFunction {
  TORCH_API I0Backward0() = default;
#else
struct TORCH_API I0Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "I0Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SpecialI0EBackward0 : public TraceableFunction {
  TORCH_API SpecialI0EBackward0() = default;
#else
struct TORCH_API SpecialI0EBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialI0EBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct SpecialI1Backward0 : public TraceableFunction {
  TORCH_API SpecialI1Backward0() = default;
#else
struct TORCH_API SpecialI1Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialI1Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct SpecialI1EBackward0 : public TraceableFunction {
  TORCH_API SpecialI1EBackward0() = default;
#else
struct TORCH_API SpecialI1EBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialI1EBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct IgammaBackward0 : public TraceableFunction {
  TORCH_API IgammaBackward0() = default;
#else
struct TORCH_API IgammaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IgammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct IgammacBackward0 : public TraceableFunction {
  TORCH_API IgammacBackward0() = default;
#else
struct TORCH_API IgammacBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IgammacBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct IndexBackward0 : public TraceableFunction {
  TORCH_API IndexBackward0() = default;
#else
struct TORCH_API IndexBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UnsafeIndexBackward0 : public TraceableFunction {
  TORCH_API UnsafeIndexBackward0() = default;
#else
struct TORCH_API UnsafeIndexBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeIndexBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UnsafeMaskedIndexBackward0 : public TraceableFunction {
  TORCH_API UnsafeMaskedIndexBackward0() = default;
#else
struct TORCH_API UnsafeMaskedIndexBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeMaskedIndexBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  SavedVariable mask_;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UnsafeMaskedIndexPutAccumulateBackward0 : public TraceableFunction {
  TORCH_API UnsafeMaskedIndexPutAccumulateBackward0() = default;
#else
struct TORCH_API UnsafeMaskedIndexPutAccumulateBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeMaskedIndexPutAccumulateBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  SavedVariable mask_;

};
#ifdef _WIN32
struct IndexAddBackward0 : public TraceableFunction {
  TORCH_API IndexAddBackward0() = default;
#else
struct TORCH_API IndexAddBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexAddBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  int64_t dim = 0;
  SavedVariable index_;
  SavedVariable source_;
  int64_t source_dim = 0;

};
#ifdef _WIN32
struct IndexReduceBackward0 : public TraceableFunction {
  TORCH_API IndexReduceBackward0() = default;
#else
struct TORCH_API IndexReduceBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexReduceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
    source_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool include_self;
  SavedVariable index_;
  std::string reduce;
  SavedVariable self_;
  SavedVariable source_;
  SavedVariable result_;

};
#ifdef _WIN32
struct IndexCopyBackward0 : public TraceableFunction {
  TORCH_API IndexCopyBackward0() = default;
#else
struct TORCH_API IndexCopyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexCopyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
  SavedVariable source_;
  int64_t source_dim = 0;

};
#ifdef _WIN32
struct IndexFillBackward0 : public TraceableFunction {
  TORCH_API IndexFillBackward0() = default;
#else
struct TORCH_API IndexFillBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;

};
#ifdef _WIN32
struct IndexFillBackward1 : public TraceableFunction {
  TORCH_API IndexFillBackward1() = default;
#else
struct TORCH_API IndexFillBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;

};
#ifdef _WIN32
struct IndexPutBackward0 : public TraceableFunction {
  TORCH_API IndexPutBackward0() = default;
#else
struct TORCH_API IndexPutBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool accumulate;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;

};
#ifdef _WIN32
struct UnsafeIndexPutBackward0 : public TraceableFunction {
  TORCH_API UnsafeIndexPutBackward0() = default;
#else
struct TORCH_API UnsafeIndexPutBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeIndexPutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool accumulate;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;

};
#ifdef _WIN32
struct IndexPutImplBackward0 : public TraceableFunction {
  TORCH_API IndexPutImplBackward0() = default;
#else
struct TORCH_API IndexPutImplBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutImplBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool accumulate;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;

};
#ifdef _WIN32
struct IndexSelectBackward0 : public TraceableFunction {
  TORCH_API IndexSelectBackward0() = default;
#else
struct TORCH_API IndexSelectBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexSelectBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct LinalgInvExBackward0 : public TraceableFunction {
  TORCH_API LinalgInvExBackward0() = default;
#else
struct TORCH_API LinalgInvExBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgInvExBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    inverse_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable inverse_;

};
#ifdef _WIN32
struct LinalgPinvBackward0 : public TraceableFunction {
  TORCH_API LinalgPinvBackward0() = default;
#else
struct TORCH_API LinalgPinvBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgPinvBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct KthvalueBackward0 : public TraceableFunction {
  TORCH_API KthvalueBackward0() = default;
#else
struct TORCH_API KthvalueBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KthvalueBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct LeBackward0 : public TraceableFunction {
  TORCH_API LeBackward0() = default;
#else
struct TORCH_API LeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct LeBackward1 : public TraceableFunction {
  TORCH_API LeBackward1() = default;
#else
struct TORCH_API LeBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct LerpBackward0 : public TraceableFunction {
  TORCH_API LerpBackward0() = default;
#else
struct TORCH_API LerpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar weight;

};
#ifdef _WIN32
struct LerpBackward1 : public TraceableFunction {
  TORCH_API LerpBackward1() = default;
#else
struct TORCH_API LerpBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    end_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable end_;
  SavedVariable self_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct LgammaBackward0 : public TraceableFunction {
  TORCH_API LgammaBackward0() = default;
#else
struct TORCH_API LgammaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LgammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct DigammaBackward0 : public TraceableFunction {
  TORCH_API DigammaBackward0() = default;
#else
struct TORCH_API DigammaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DigammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct PolygammaBackward0 : public TraceableFunction {
  TORCH_API PolygammaBackward0() = default;
#else
struct TORCH_API PolygammaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolygammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t n = 0;
  SavedVariable self_;

};
#ifdef _WIN32
struct PolygammaBackward1 : public TraceableFunction {
  TORCH_API PolygammaBackward1() = default;
#else
struct TORCH_API PolygammaBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolygammaBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t n = 0;
  SavedVariable self_;

};
#ifdef _WIN32
struct LogBackward0 : public TraceableFunction {
  TORCH_API LogBackward0() = default;
#else
struct TORCH_API LogBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct Log10Backward0 : public TraceableFunction {
  TORCH_API Log10Backward0() = default;
#else
struct TORCH_API Log10Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log10Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct Log1PBackward0 : public TraceableFunction {
  TORCH_API Log1PBackward0() = default;
#else
struct TORCH_API Log1PBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log1PBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct Log2Backward0 : public TraceableFunction {
  TORCH_API Log2Backward0() = default;
#else
struct TORCH_API Log2Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct LogaddexpBackward0 : public TraceableFunction {
  TORCH_API LogaddexpBackward0() = default;
#else
struct TORCH_API LogaddexpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogaddexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct Logaddexp2Backward0 : public TraceableFunction {
  TORCH_API Logaddexp2Backward0() = default;
#else
struct TORCH_API Logaddexp2Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Logaddexp2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct XlogyBackward0 : public TraceableFunction {
  TORCH_API XlogyBackward0() = default;
#else
struct TORCH_API XlogyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct XlogyBackward1 : public TraceableFunction {
  TORCH_API XlogyBackward1() = default;
#else
struct TORCH_API XlogyBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar self;

};
#ifdef _WIN32
struct XlogyBackward2 : public TraceableFunction {
  TORCH_API XlogyBackward2() = default;
#else
struct TORCH_API XlogyBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar other;
  SavedVariable self_;

};
#ifdef _WIN32
struct SpecialXlog1PyBackward0 : public TraceableFunction {
  TORCH_API SpecialXlog1PyBackward0() = default;
#else
struct TORCH_API SpecialXlog1PyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct SpecialXlog1PyBackward1 : public TraceableFunction {
  TORCH_API SpecialXlog1PyBackward1() = default;
#else
struct TORCH_API SpecialXlog1PyBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar self;

};
#ifdef _WIN32
struct SpecialXlog1PyBackward2 : public TraceableFunction {
  TORCH_API SpecialXlog1PyBackward2() = default;
#else
struct TORCH_API SpecialXlog1PyBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar other;
  SavedVariable self_;

};
#ifdef _WIN32
struct SpecialZetaBackward0 : public TraceableFunction {
  TORCH_API SpecialZetaBackward0() = default;
#else
struct TORCH_API SpecialZetaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialZetaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct SpecialZetaBackward1 : public TraceableFunction {
  TORCH_API SpecialZetaBackward1() = default;
#else
struct TORCH_API SpecialZetaBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialZetaBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar self;

};
#ifdef _WIN32
struct SpecialZetaBackward2 : public TraceableFunction {
  TORCH_API SpecialZetaBackward2() = default;
#else
struct TORCH_API SpecialZetaBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialZetaBackward2"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct LogNormalBackward0 : public TraceableFunction {
  TORCH_API LogNormalBackward0() = default;
#else
struct TORCH_API LogNormalBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogNormalBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct LogsumexpBackward0 : public TraceableFunction {
  TORCH_API LogsumexpBackward0() = default;
#else
struct TORCH_API LogsumexpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogsumexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct LinalgLstsqBackward0 : public TraceableFunction {
  TORCH_API LinalgLstsqBackward0() = default;
#else
struct TORCH_API LinalgLstsqBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgLstsqBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    b_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable b_;
  SavedVariable self_;

};
#ifdef _WIN32
struct LtBackward0 : public TraceableFunction {
  TORCH_API LtBackward0() = default;
#else
struct TORCH_API LtBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct LtBackward1 : public TraceableFunction {
  TORCH_API LtBackward1() = default;
#else
struct TORCH_API LtBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct LinalgLuFactorExBackward0 : public TraceableFunction {
  TORCH_API LinalgLuFactorExBackward0() = default;
#else
struct TORCH_API LinalgLuFactorExBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgLuFactorExBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    LU_.reset_data();
    pivots_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool pivot;
  SavedVariable LU_;
  SavedVariable pivots_;

};
#ifdef _WIN32
struct LinalgLuFactorBackward0 : public TraceableFunction {
  TORCH_API LinalgLuFactorBackward0() = default;
#else
struct TORCH_API LinalgLuFactorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgLuFactorBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    LU_.reset_data();
    pivots_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool pivot;
  SavedVariable LU_;
  SavedVariable pivots_;

};
#ifdef _WIN32
struct LinalgLuBackward0 : public TraceableFunction {
  TORCH_API LinalgLuBackward0() = default;
#else
struct TORCH_API LinalgLuBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgLuBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    L_.reset_data();
    P_.reset_data();
    U_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool pivot;
  SavedVariable L_;
  SavedVariable P_;
  SavedVariable U_;

};
#ifdef _WIN32
struct LinalgLuSolveBackward0 : public TraceableFunction {
  TORCH_API LinalgLuSolveBackward0() = default;
#else
struct TORCH_API LinalgLuSolveBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgLuSolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    LU_.reset_data();
    pivots_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable LU_;
  bool adjoint;
  bool left;
  SavedVariable pivots_;
  SavedVariable result_;

};
#ifdef _WIN32
struct LuUnpackBackward0 : public TraceableFunction {
  TORCH_API LuUnpackBackward0() = default;
#else
struct TORCH_API LuUnpackBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuUnpackBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt LU_data_sym_argsize_minus_1;
  c10::SymInt LU_data_sym_argsize_minus_2;

};
#ifdef _WIN32
struct MaskedFillBackward0 : public TraceableFunction {
  TORCH_API MaskedFillBackward0() = default;
#else
struct TORCH_API MaskedFillBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;

};
#ifdef _WIN32
struct MaskedFillBackward1 : public TraceableFunction {
  TORCH_API MaskedFillBackward1() = default;
#else
struct TORCH_API MaskedFillBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;

};
#ifdef _WIN32
struct MaskedScatterBackward0 : public TraceableFunction {
  TORCH_API MaskedScatterBackward0() = default;
#else
struct TORCH_API MaskedScatterBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedScatterBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;
  std::vector<c10::SymInt> source_sym_sizes;

};
#ifdef _WIN32
struct MaskedScatterBackwardBackward0 : public TraceableFunction {
  TORCH_API MaskedScatterBackwardBackward0() = default;
#else
struct TORCH_API MaskedScatterBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedScatterBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize grad_output_info;
  SavedVariable mask_;

};
#ifdef _WIN32
struct MaskedSelectBackward0 : public TraceableFunction {
  TORCH_API MaskedSelectBackward0() = default;
#else
struct TORCH_API MaskedSelectBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedSelectBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;
  SavedVariable self_;

};
#ifdef _WIN32
struct LinalgMatrixExpBackward0 : public TraceableFunction {
  TORCH_API LinalgMatrixExpBackward0() = default;
#else
struct TORCH_API LinalgMatrixExpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgMatrixExpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct MaxBackward0 : public TraceableFunction {
  TORCH_API MaxBackward0() = default;
#else
struct TORCH_API MaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct MaxBackward1 : public TraceableFunction {
  TORCH_API MaxBackward1() = default;
#else
struct TORCH_API MaxBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct MaximumBackward0 : public TraceableFunction {
  TORCH_API MaximumBackward0() = default;
#else
struct TORCH_API MaximumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaximumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct FmaxBackward0 : public TraceableFunction {
  TORCH_API FmaxBackward0() = default;
#else
struct TORCH_API FmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct MeanBackward0 : public TraceableFunction {
  TORCH_API MeanBackward0() = default;
#else
struct TORCH_API MeanBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt self_sym_numel;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct MeanBackwardAutogradNestedTensor0 : public TraceableFunction {
  TORCH_API MeanBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API MeanBackwardAutogradNestedTensor0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  c10::SymInt self_sym_numel;

};
#ifdef _WIN32
struct MeanBackward1 : public TraceableFunction {
  TORCH_API MeanBackward1() = default;
#else
struct TORCH_API MeanBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  c10::SymInt self_sym_numel;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct MedianBackward0 : public TraceableFunction {
  TORCH_API MedianBackward0() = default;
#else
struct TORCH_API MedianBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct NanmedianBackward0 : public TraceableFunction {
  TORCH_API NanmedianBackward0() = default;
#else
struct TORCH_API NanmedianBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanmedianBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct MedianBackward1 : public TraceableFunction {
  TORCH_API MedianBackward1() = default;
#else
struct TORCH_API MedianBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct NanmedianBackward1 : public TraceableFunction {
  TORCH_API NanmedianBackward1() = default;
#else
struct TORCH_API NanmedianBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanmedianBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct MinBackward0 : public TraceableFunction {
  TORCH_API MinBackward0() = default;
#else
struct TORCH_API MinBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct MinBackward1 : public TraceableFunction {
  TORCH_API MinBackward1() = default;
#else
struct TORCH_API MinBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct MinimumBackward0 : public TraceableFunction {
  TORCH_API MinimumBackward0() = default;
#else
struct TORCH_API MinimumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinimumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct FminBackward0 : public TraceableFunction {
  TORCH_API FminBackward0() = default;
#else
struct TORCH_API FminBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FminBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct AmaxBackward0 : public TraceableFunction {
  TORCH_API AmaxBackward0() = default;
#else
struct TORCH_API AmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct AminBackward0 : public TraceableFunction {
  TORCH_API AminBackward0() = default;
#else
struct TORCH_API AminBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AminBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct MmBackward0 : public TraceableFunction {
  TORCH_API MmBackward0() = default;
#else
struct TORCH_API MmBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat2_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mat2_;
  at::Layout mat2_layout;
  std::vector<c10::SymInt> mat2_sym_sizes;
  std::vector<c10::SymInt> mat2_sym_strides;
  SavedVariable self_;
  at::Layout self_layout;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> self_sym_strides;

};
#ifdef _WIN32
struct ModeBackward0 : public TraceableFunction {
  TORCH_API ModeBackward0() = default;
#else
struct TORCH_API ModeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ModeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct MulBackward0 : public TraceableFunction {
  TORCH_API MulBackward0() = default;
#else
struct TORCH_API MulBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::ScalarType other_scalar_type;
  SavedVariable self_;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct MulBackward1 : public TraceableFunction {
  TORCH_API MulBackward1() = default;
#else
struct TORCH_API MulBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar other;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct MvBackward0 : public TraceableFunction {
  TORCH_API MvBackward0() = default;
#else
struct TORCH_API MvBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    vec_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable vec_;

};
#ifdef _WIN32
struct MvlgammaBackward0 : public TraceableFunction {
  TORCH_API MvlgammaBackward0() = default;
#else
struct TORCH_API MvlgammaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvlgammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t p = 0;
  SavedVariable self_;

};
#ifdef _WIN32
struct NanToNumBackward0 : public TraceableFunction {
  TORCH_API NanToNumBackward0() = default;
#else
struct TORCH_API NanToNumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanToNumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct NativeBatchNormBackward0 : public TraceableFunction {
  TORCH_API NativeBatchNormBackward0() = default;
#else
struct TORCH_API NativeBatchNormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct NativeBatchNormLegitBackward0 : public TraceableFunction {
  TORCH_API NativeBatchNormLegitBackward0() = default;
#else
struct TORCH_API NativeBatchNormLegitBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormLegitBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct NativeBatchNormLegitNoTrainingBackward0 : public TraceableFunction {
  TORCH_API NativeBatchNormLegitNoTrainingBackward0() = default;
#else
struct TORCH_API NativeBatchNormLegitNoTrainingBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormLegitNoTrainingBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct NativeBatchNormLegitBackward1 : public TraceableFunction {
  TORCH_API NativeBatchNormLegitBackward1() = default;
#else
struct TORCH_API NativeBatchNormLegitBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormLegitBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct NativeBatchNormBackwardBackward0 : public TraceableFunction {
  TORCH_API NativeBatchNormBackwardBackward0() = default;
#else
struct TORCH_API NativeBatchNormBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_invstd_.reset_data();
    save_mean_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_invstd_;
  SavedVariable save_mean_;
  bool train;
  SavedVariable weight_;

};
#ifdef _WIN32
struct NativeLayerNormBackward0 : public TraceableFunction {
  TORCH_API NativeLayerNormBackward0() = default;
#else
struct TORCH_API NativeLayerNormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeLayerNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    bias_.reset_data();
    input_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable bias_;
  SavedVariable input_;
  std::vector<c10::SymInt> normalized_shape;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct NativeLayerNormBackwardBackward0 : public TraceableFunction {
  TORCH_API NativeLayerNormBackwardBackward0() = default;
#else
struct TORCH_API NativeLayerNormBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeLayerNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    input_.reset_data();
    mean_.reset_data();
    rstd_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable mean_;
  std::vector<c10::SymInt> normalized_shape;
  SavedVariable rstd_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct NativeGroupNormBackward0 : public TraceableFunction {
  TORCH_API NativeGroupNormBackward0() = default;
#else
struct TORCH_API NativeGroupNormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeGroupNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt C;
  c10::SymInt HxW;
  c10::SymInt N;
  double eps;
  int64_t group = 0;
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct NeBackward0 : public TraceableFunction {
  TORCH_API NeBackward0() = default;
#else
struct TORCH_API NeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct NeBackward1 : public TraceableFunction {
  TORCH_API NeBackward1() = default;
#else
struct TORCH_API NeBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct NegBackward0 : public TraceableFunction {
  TORCH_API NegBackward0() = default;
#else
struct TORCH_API NegBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct BatchNormWithUpdateBackward0 : public TraceableFunction {
  TORCH_API BatchNormWithUpdateBackward0() = default;
#else
struct TORCH_API BatchNormWithUpdateBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BatchNormWithUpdateBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
#ifdef _WIN32
struct BatchNormNoUpdateBackward0 : public TraceableFunction {
  TORCH_API BatchNormNoUpdateBackward0() = default;
#else
struct TORCH_API BatchNormNoUpdateBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BatchNormNoUpdateBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
#ifdef _WIN32
struct BatchNormBackwardBackward0 : public TraceableFunction {
  TORCH_API BatchNormBackwardBackward0() = default;
#else
struct TORCH_API BatchNormBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BatchNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    input_.reset_data();
    reserve_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double eps;
  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable reserve_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  bool update;
  SavedVariable weight_;

};
#ifdef _WIN32
struct NextafterBackward0 : public TraceableFunction {
  TORCH_API NextafterBackward0() = default;
#else
struct TORCH_API NextafterBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NextafterBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NormBackward0 : public TraceableFunction {
  TORCH_API NormBackward0() = default;
#else
struct TORCH_API NormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar p;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct NormBackward1 : public TraceableFunction {
  TORCH_API NormBackward1() = default;
#else
struct TORCH_API NormBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  ::std::optional<at::Scalar> p;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct NormBackward2 : public TraceableFunction {
  TORCH_API NormBackward2() = default;
#else
struct TORCH_API NormBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<at::Scalar> p;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct NormBackward3 : public TraceableFunction {
  TORCH_API NormBackward3() = default;
#else
struct TORCH_API NormBackward3 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward3"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  ::std::optional<at::Scalar> p;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct LinalgVectorNormBackward0 : public TraceableFunction {
  TORCH_API LinalgVectorNormBackward0() = default;
#else
struct TORCH_API LinalgVectorNormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgVectorNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  at::Scalar ord;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct PdistBackward0 : public TraceableFunction {
  TORCH_API PdistBackward0() = default;
#else
struct TORCH_API PdistBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double p;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct PdistBackwardBackward0 : public TraceableFunction {
  TORCH_API PdistBackwardBackward0() = default;
#else
struct TORCH_API PdistBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct EuclideanDistBackward0 : public TraceableFunction {
  TORCH_API EuclideanDistBackward0() = default;
#else
struct TORCH_API EuclideanDistBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EuclideanDistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x2_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable x1_;
  SavedVariable x2_;
  SavedVariable result_;

};
#ifdef _WIN32
struct CdistBackward0 : public TraceableFunction {
  TORCH_API CdistBackward0() = default;
#else
struct TORCH_API CdistBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x2_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double p;
  SavedVariable x1_;
  SavedVariable x2_;
  SavedVariable result_;

};
#ifdef _WIN32
struct CdistBackwardBackward0 : public TraceableFunction {
  TORCH_API CdistBackwardBackward0() = default;
#else
struct TORCH_API CdistBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NormalBackward0 : public TraceableFunction {
  TORCH_API NormalBackward0() = default;
#else
struct TORCH_API NormalBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NormalBackward1 : public TraceableFunction {
  TORCH_API NormalBackward1() = default;
#else
struct TORCH_API NormalBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> mean_sym_sizes;

};
#ifdef _WIN32
struct NormalBackward2 : public TraceableFunction {
  TORCH_API NormalBackward2() = default;
#else
struct TORCH_API NormalBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward2"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> std_sym_sizes;

};
#ifdef _WIN32
struct NormalBackward3 : public TraceableFunction {
  TORCH_API NormalBackward3() = default;
#else
struct TORCH_API NormalBackward3 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward3"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> mean_sym_sizes;
  std::vector<c10::SymInt> std_sym_sizes;

};
#ifdef _WIN32
struct LinalgHouseholderProductBackward0 : public TraceableFunction {
  TORCH_API LinalgHouseholderProductBackward0() = default;
#else
struct TORCH_API LinalgHouseholderProductBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgHouseholderProductBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    tau_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable input_;
  SavedVariable tau_;
  SavedVariable result_;

};
#ifdef _WIN32
struct OrmqrBackward0 : public TraceableFunction {
  TORCH_API OrmqrBackward0() = default;
#else
struct TORCH_API OrmqrBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "OrmqrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input2_.reset_data();
    input3_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable input2_;
  SavedVariable input3_;
  bool left;
  SavedVariable self_;
  bool transpose;
  SavedVariable result_;

};
#ifdef _WIN32
struct PermuteBackward0 : public Node {
  TORCH_API PermuteBackward0() = default;
#else
struct TORCH_API PermuteBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PermuteBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;

};
#ifdef _WIN32
struct PoissonBackward0 : public TraceableFunction {
  TORCH_API PoissonBackward0() = default;
#else
struct TORCH_API PoissonBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PoissonBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct PowBackward0 : public TraceableFunction {
  TORCH_API PowBackward0() = default;
#else
struct TORCH_API PowBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar exponent;
  SavedVariable self_;

};
#ifdef _WIN32
struct PowBackward1 : public TraceableFunction {
  TORCH_API PowBackward1() = default;
#else
struct TORCH_API PowBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable exponent_;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct PowBackward2 : public TraceableFunction {
  TORCH_API PowBackward2() = default;
#else
struct TORCH_API PowBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable exponent_;
  at::Scalar self;
  SavedVariable result_;

};
#ifdef _WIN32
struct ProdBackward0 : public TraceableFunction {
  TORCH_API ProdBackward0() = default;
#else
struct TORCH_API ProdBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct ProdBackward1 : public TraceableFunction {
  TORCH_API ProdBackward1() = default;
#else
struct TORCH_API ProdBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct PutBackward0 : public TraceableFunction {
  TORCH_API PutBackward0() = default;
#else
struct TORCH_API PutBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool accumulate;
  SavedVariable index_;
  SavedVariable source_;
  torch::autograd::generated::TypeAndSize source_info;

};
#ifdef _WIN32
struct LinalgQrBackward0 : public TraceableFunction {
  TORCH_API LinalgQrBackward0() = default;
#else
struct TORCH_API LinalgQrBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgQrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    Q_.reset_data();
    R_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::string mode;
  SavedVariable Q_;
  SavedVariable R_;

};
#ifdef _WIN32
struct Rad2DegBackward0 : public TraceableFunction {
  TORCH_API Rad2DegBackward0() = default;
#else
struct TORCH_API Rad2DegBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rad2DegBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct RandomBackward0 : public TraceableFunction {
  TORCH_API RandomBackward0() = default;
#else
struct TORCH_API RandomBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct RandomBackward1 : public TraceableFunction {
  TORCH_API RandomBackward1() = default;
#else
struct TORCH_API RandomBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct RandomBackward2 : public TraceableFunction {
  TORCH_API RandomBackward2() = default;
#else
struct TORCH_API RandomBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward2"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ReciprocalBackward0 : public TraceableFunction {
  TORCH_API ReciprocalBackward0() = default;
#else
struct TORCH_API ReciprocalBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReciprocalBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct RemainderBackward0 : public TraceableFunction {
  TORCH_API RemainderBackward0() = default;
#else
struct TORCH_API RemainderBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct RemainderBackward1 : public TraceableFunction {
  TORCH_API RemainderBackward1() = default;
#else
struct TORCH_API RemainderBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct RenormBackward0 : public TraceableFunction {
  TORCH_API RenormBackward0() = default;
#else
struct TORCH_API RenormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RenormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::Scalar maxnorm;
  at::Scalar p;
  SavedVariable self_;

};
#ifdef _WIN32
struct RepeatBackward0 : public TraceableFunction {
  TORCH_API RepeatBackward0() = default;
#else
struct TORCH_API RepeatBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RepeatBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> repeats;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SpecialEntrBackward0 : public TraceableFunction {
  TORCH_API SpecialEntrBackward0() = default;
#else
struct TORCH_API SpecialEntrBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialEntrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SpecialNdtriBackward0 : public TraceableFunction {
  TORCH_API SpecialNdtriBackward0() = default;
#else
struct TORCH_API SpecialNdtriBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialNdtriBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct SpecialLogNdtrBackward0 : public TraceableFunction {
  TORCH_API SpecialLogNdtrBackward0() = default;
#else
struct TORCH_API SpecialLogNdtrBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialLogNdtrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct ReshapeAliasBackward0 : public Node {
  TORCH_API ReshapeAliasBackward0() = default;
#else
struct TORCH_API ReshapeAliasBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReshapeAliasBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct RoundBackward0 : public TraceableFunction {
  TORCH_API RoundBackward0() = default;
#else
struct TORCH_API RoundBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RoundBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct RoundBackward1 : public TraceableFunction {
  TORCH_API RoundBackward1() = default;
#else
struct TORCH_API RoundBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RoundBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct RsqrtBackward0 : public TraceableFunction {
  TORCH_API RsqrtBackward0() = default;
#else
struct TORCH_API RsqrtBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsqrtBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct ScatterBackward0 : public TraceableFunction {
  TORCH_API ScatterBackward0() = default;
#else
struct TORCH_API ScatterBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;

};
#ifdef _WIN32
struct ScatterBackward1 : public TraceableFunction {
  TORCH_API ScatterBackward1() = default;
#else
struct TORCH_API ScatterBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;

};
#ifdef _WIN32
struct ScatterAddBackward0 : public TraceableFunction {
  TORCH_API ScatterAddBackward0() = default;
#else
struct TORCH_API ScatterAddBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterAddBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;

};
#ifdef _WIN32
struct SelectBackward0 : public Node {
  TORCH_API SelectBackward0() = default;
#else
struct TORCH_API SelectBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SelectBackwardAutogradNestedTensor0 : public Node {
  TORCH_API SelectBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API SelectBackwardAutogradNestedTensor0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  SavedVariable self_;

};
#ifdef _WIN32
struct SelectBackwardBackward0 : public TraceableFunction {
  TORCH_API SelectBackwardBackward0() = default;
#else
struct TORCH_API SelectBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;

};
#ifdef _WIN32
struct SigmoidBackward0 : public TraceableFunction {
  TORCH_API SigmoidBackward0() = default;
#else
struct TORCH_API SigmoidBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct LogitBackward0 : public TraceableFunction {
  TORCH_API LogitBackward0() = default;
#else
struct TORCH_API LogitBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogitBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<double> eps;
  SavedVariable self_;

};
#ifdef _WIN32
struct SignBackward0 : public TraceableFunction {
  TORCH_API SignBackward0() = default;
#else
struct TORCH_API SignBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SignBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct SgnBackward0 : public TraceableFunction {
  TORCH_API SgnBackward0() = default;
#else
struct TORCH_API SgnBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SgnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct SinBackward0 : public TraceableFunction {
  TORCH_API SinBackward0() = default;
#else
struct TORCH_API SinBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SincBackward0 : public TraceableFunction {
  TORCH_API SincBackward0() = default;
#else
struct TORCH_API SincBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SincBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SinhBackward0 : public TraceableFunction {
  TORCH_API SinhBackward0() = default;
#else
struct TORCH_API SinhBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SliceBackward0 : public Node {
  TORCH_API SliceBackward0() = default;
#else
struct TORCH_API SliceBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::optional<c10::SymInt> end;
  std::vector<c10::SymInt> self_sym_sizes;
  ::std::optional<c10::SymInt> start;
  c10::SymInt step;

};
#ifdef _WIN32
struct SliceBackwardBackward0 : public TraceableFunction {
  TORCH_API SliceBackwardBackward0() = default;
#else
struct TORCH_API SliceBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt end;
  c10::SymInt start;
  c10::SymInt step;

};
#ifdef _WIN32
struct SliceInverseBackward0 : public Node {
  TORCH_API SliceInverseBackward0() = default;
#else
struct TORCH_API SliceInverseBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceInverseBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::optional<c10::SymInt> end;
  torch::autograd::generated::TypeAndSize self_info;
  ::std::optional<c10::SymInt> start;
  c10::SymInt step;

};
#ifdef _WIN32
struct SliceScatterBackward0 : public TraceableFunction {
  TORCH_API SliceScatterBackward0() = default;
#else
struct TORCH_API SliceScatterBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceScatterBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::optional<c10::SymInt> end;
  torch::autograd::generated::TypeAndSize src_info;
  ::std::optional<c10::SymInt> start;
  c10::SymInt step;

};
#ifdef _WIN32
struct SelectScatterBackward0 : public TraceableFunction {
  TORCH_API SelectScatterBackward0() = default;
#else
struct TORCH_API SelectScatterBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectScatterBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  torch::autograd::generated::TypeAndSize src_info;

};
#ifdef _WIN32
struct DiagonalScatterBackward0 : public TraceableFunction {
  TORCH_API DiagonalScatterBackward0() = default;
#else
struct TORCH_API DiagonalScatterBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalScatterBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
  torch::autograd::generated::TypeAndSize src_info;

};
#ifdef _WIN32
struct AsStridedScatterBackward0 : public TraceableFunction {
  TORCH_API AsStridedScatterBackward0() = default;
#else
struct TORCH_API AsStridedScatterBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedScatterBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  at::TensorGeometry src_geometry;
  ::std::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;

};
#ifdef _WIN32
struct LinalgSolveExBackward0 : public TraceableFunction {
  TORCH_API LinalgSolveExBackward0() = default;
#else
struct TORCH_API LinalgSolveExBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSolveExBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    LU_.reset_data();
    pivots_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable A_;
  bool left;
  SavedVariable LU_;
  SavedVariable pivots_;
  SavedVariable result_;

};
#ifdef _WIN32
struct SortBackward0 : public TraceableFunction {
  TORCH_API SortBackward0() = default;
#else
struct TORCH_API SortBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct SortBackward1 : public TraceableFunction {
  TORCH_API SortBackward1() = default;
#else
struct TORCH_API SortBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct SplitBackward0 : public Node {
  TORCH_API SplitBackward0() = default;
#else
struct TORCH_API SplitBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::SymInt split_size;

};
#ifdef _WIN32
struct UnsafeSplitBackward0 : public TraceableFunction {
  TORCH_API UnsafeSplitBackward0() = default;
#else
struct TORCH_API UnsafeSplitBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeSplitBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::SymInt split_size;

};
#ifdef _WIN32
struct SplitWithSizesBackward0 : public Node {
  TORCH_API SplitWithSizesBackward0() = default;
#else
struct TORCH_API SplitWithSizesBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitWithSizesBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> split_sizes;

};
#ifdef _WIN32
struct SplitWithSizesBackwardAutogradNestedTensor0 : public Node {
  TORCH_API SplitWithSizesBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API SplitWithSizesBackwardAutogradNestedTensor0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitWithSizesBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> split_sizes;

};
#ifdef _WIN32
struct UnsafeSplitWithSizesBackward0 : public TraceableFunction {
  TORCH_API UnsafeSplitWithSizesBackward0() = default;
#else
struct TORCH_API UnsafeSplitWithSizesBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeSplitWithSizesBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> split_sizes;

};
#ifdef _WIN32
struct SqrtBackward0 : public TraceableFunction {
  TORCH_API SqrtBackward0() = default;
#else
struct TORCH_API SqrtBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqrtBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct SqueezeBackward0 : public Node {
  TORCH_API SqueezeBackward0() = default;
#else
struct TORCH_API SqueezeBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackward1 : public Node {
  TORCH_API SqueezeBackward1() = default;
#else
struct TORCH_API SqueezeBackward1 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackwardAutogradNestedTensor0 : public Node {
  TORCH_API SqueezeBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API SqueezeBackwardAutogradNestedTensor0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackwardAutogradNestedTensor0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;

};
#ifdef _WIN32
struct SqueezeBackward2 : public Node {
  TORCH_API SqueezeBackward2() = default;
#else
struct TORCH_API SqueezeBackward2 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward2"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackwardAutogradNestedTensor1 : public Node {
  TORCH_API SqueezeBackwardAutogradNestedTensor1() = default;
#else
struct TORCH_API SqueezeBackwardAutogradNestedTensor1 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackwardAutogradNestedTensor1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t self_dim = 0;

};
#ifdef _WIN32
struct SqueezeBackward3 : public TraceableFunction {
  TORCH_API SqueezeBackward3() = default;
#else
struct TORCH_API SqueezeBackward3 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward3"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackward4 : public TraceableFunction {
  TORCH_API SqueezeBackward4() = default;
#else
struct TORCH_API SqueezeBackward4 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward4"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackward5 : public TraceableFunction {
  TORCH_API SqueezeBackward5() = default;
#else
struct TORCH_API SqueezeBackward5 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward5"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct StdBackward0 : public TraceableFunction {
  TORCH_API StdBackward0() = default;
#else
struct TORCH_API StdBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct StdMeanBackward0 : public TraceableFunction {
  TORCH_API StdMeanBackward0() = default;
#else
struct TORCH_API StdMeanBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdMeanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result0_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result0_;

};
#ifdef _WIN32
struct SubBackward0 : public TraceableFunction {
  TORCH_API SubBackward0() = default;
#else
struct TORCH_API SubBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct SubBackward1 : public TraceableFunction {
  TORCH_API SubBackward1() = default;
#else
struct TORCH_API SubBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct RsubBackward0 : public TraceableFunction {
  TORCH_API RsubBackward0() = default;
#else
struct TORCH_API RsubBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct RsubBackward1 : public TraceableFunction {
  TORCH_API RsubBackward1() = default;
#else
struct TORCH_API RsubBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct SumBackward0 : public TraceableFunction {
  TORCH_API SumBackward0() = default;
#else
struct TORCH_API SumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SumBackwardAutogradNestedTensor0 : public TraceableFunction {
  TORCH_API SumBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API SumBackwardAutogradNestedTensor0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SumBackward1 : public TraceableFunction {
  TORCH_API SumBackward1() = default;
#else
struct TORCH_API SumBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SumBackwardAutogradNestedTensor1 : public TraceableFunction {
  TORCH_API SumBackwardAutogradNestedTensor1() = default;
#else
struct TORCH_API SumBackwardAutogradNestedTensor1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackwardAutogradNestedTensor1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;

};
#ifdef _WIN32
struct NansumBackward0 : public TraceableFunction {
  TORCH_API NansumBackward0() = default;
#else
struct TORCH_API NansumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NansumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  at::ScalarType self_scalar_type;

};
#ifdef _WIN32
struct LinalgSvdBackward0 : public TraceableFunction {
  TORCH_API LinalgSvdBackward0() = default;
#else
struct TORCH_API LinalgSvdBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSvdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    S_.reset_data();
    U_.reset_data();
    Vh_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool full_matrices;
  SavedVariable S_;
  c10::SymInt S_sym_argsize_minus_1;
  SavedVariable U_;
  SavedVariable Vh_;

};
#ifdef _WIN32
struct LinalgEighBackward0 : public TraceableFunction {
  TORCH_API LinalgEighBackward0() = default;
#else
struct TORCH_API LinalgEighBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgEighBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    eigenvalues_.reset_data();
    eigenvectors_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;

};
#ifdef _WIN32
struct LinalgEigBackward0 : public TraceableFunction {
  TORCH_API LinalgEigBackward0() = default;
#else
struct TORCH_API LinalgEigBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgEigBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    eigenvalues_.reset_data();
    eigenvectors_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;

};
#ifdef _WIN32
struct TBackward0 : public Node {
  TORCH_API TBackward0() = default;
#else
struct TORCH_API TBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct TBackward1 : public TraceableFunction {
  TORCH_API TBackward1() = default;
#else
struct TORCH_API TBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct FlipBackward0 : public TraceableFunction {
  TORCH_API FlipBackward0() = default;
#else
struct TORCH_API FlipBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FlipBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;

};
#ifdef _WIN32
struct RollBackward0 : public TraceableFunction {
  TORCH_API RollBackward0() = default;
#else
struct TORCH_API RollBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RollBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;
  std::vector<c10::SymInt> shifts;

};
#ifdef _WIN32
struct Rot90Backward0 : public TraceableFunction {
  TORCH_API Rot90Backward0() = default;
#else
struct TORCH_API Rot90Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rot90Backward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;
  int64_t k = 0;

};
#ifdef _WIN32
struct TakeBackward0 : public TraceableFunction {
  TORCH_API TakeBackward0() = default;
#else
struct TORCH_API TakeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TakeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable index_;
  SavedVariable self_;

};
#ifdef _WIN32
struct TanBackward0 : public TraceableFunction {
  TORCH_API TanBackward0() = default;
#else
struct TORCH_API TanBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct TanhBackward0 : public TraceableFunction {
  TORCH_API TanhBackward0() = default;
#else
struct TORCH_API TanhBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct TopkBackward0 : public TraceableFunction {
  TORCH_API TopkBackward0() = default;
#else
struct TORCH_API TopkBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TopkBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;

};
#ifdef _WIN32
struct TraceBackward0 : public TraceableFunction {
  TORCH_API TraceBackward0() = default;
#else
struct TORCH_API TraceBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TraceBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct TransposeBackward0 : public Node {
  TORCH_API TransposeBackward0() = default;
#else
struct TORCH_API TransposeBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
#ifdef _WIN32
struct TransposeBackward1 : public TraceableFunction {
  TORCH_API TransposeBackward1() = default;
#else
struct TORCH_API TransposeBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
#ifdef _WIN32
struct TriangularSolveBackward0 : public TraceableFunction {
  TORCH_API TriangularSolveBackward0() = default;
#else
struct TORCH_API TriangularSolveBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriangularSolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    self_.reset_data();
    solution_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable A_;
  SavedVariable self_;
  bool transpose;
  bool unitriangular;
  bool upper;
  SavedVariable solution_;

};
#ifdef _WIN32
struct LinalgSolveTriangularBackward0 : public TraceableFunction {
  TORCH_API LinalgSolveTriangularBackward0() = default;
#else
struct TORCH_API LinalgSolveTriangularBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSolveTriangularBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool left;
  SavedVariable self_;
  bool unitriangular;
  bool upper;
  SavedVariable result_;

};
#ifdef _WIN32
struct TrilBackward0 : public TraceableFunction {
  TORCH_API TrilBackward0() = default;
#else
struct TORCH_API TrilBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t diagonal = 0;

};
#ifdef _WIN32
struct TriuBackward0 : public TraceableFunction {
  TORCH_API TriuBackward0() = default;
#else
struct TORCH_API TriuBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriuBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t diagonal = 0;

};
#ifdef _WIN32
struct TruncBackward0 : public TraceableFunction {
  TORCH_API TruncBackward0() = default;
#else
struct TORCH_API TruncBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TruncBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ToDenseBackward0 : public TraceableFunction {
  TORCH_API ToDenseBackward0() = default;
#else
struct TORCH_API ToDenseBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToDenseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<bool> masked_grad;
  SavedVariable self_;

};
#ifdef _WIN32
struct ToSparseBackward0 : public TraceableFunction {
  TORCH_API ToSparseBackward0() = default;
#else
struct TORCH_API ToSparseBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;

};
#ifdef _WIN32
struct ToSparseBackward1 : public TraceableFunction {
  TORCH_API ToSparseBackward1() = default;
#else
struct TORCH_API ToSparseBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;

};
#ifdef _WIN32
struct ToSparseCsrBackward0 : public TraceableFunction {
  TORCH_API ToSparseCsrBackward0() = default;
#else
struct TORCH_API ToSparseCsrBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseCsrBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;

};
#ifdef _WIN32
struct ToSparseCscBackward0 : public TraceableFunction {
  TORCH_API ToSparseCscBackward0() = default;
#else
struct TORCH_API ToSparseCscBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseCscBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;

};
#ifdef _WIN32
struct ToSparseBsrBackward0 : public TraceableFunction {
  TORCH_API ToSparseBsrBackward0() = default;
#else
struct TORCH_API ToSparseBsrBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBsrBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;

};
#ifdef _WIN32
struct ToSparseBscBackward0 : public TraceableFunction {
  TORCH_API ToSparseBscBackward0() = default;
#else
struct TORCH_API ToSparseBscBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBscBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;

};
#ifdef _WIN32
struct ToMkldnnBackward0 : public TraceableFunction {
  TORCH_API ToMkldnnBackward0() = default;
#else
struct TORCH_API ToMkldnnBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToMkldnnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct UnfoldBackward0 : public Node {
  TORCH_API UnfoldBackward0() = default;
#else
struct TORCH_API UnfoldBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dimension = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  int64_t size = 0;
  int64_t step = 0;

};
#ifdef _WIN32
struct UnfoldBackwardBackward0 : public TraceableFunction {
  TORCH_API UnfoldBackwardBackward0() = default;
#else
struct TORCH_API UnfoldBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  int64_t size = 0;
  int64_t step = 0;

};
#ifdef _WIN32
struct UniformBackward0 : public TraceableFunction {
  TORCH_API UniformBackward0() = default;
#else
struct TORCH_API UniformBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniformBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct UniqueBackward0 : public TraceableFunction {
  TORCH_API UniqueBackward0() = default;
#else
struct TORCH_API UniqueBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct UniqueDimBackward0 : public TraceableFunction {
  TORCH_API UniqueDimBackward0() = default;
#else
struct TORCH_API UniqueDimBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueDimBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct UniqueConsecutiveBackward0 : public TraceableFunction {
  TORCH_API UniqueConsecutiveBackward0() = default;
#else
struct TORCH_API UniqueConsecutiveBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueConsecutiveBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct UniqueDimConsecutiveBackward0 : public TraceableFunction {
  TORCH_API UniqueDimConsecutiveBackward0() = default;
#else
struct TORCH_API UniqueDimConsecutiveBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueDimConsecutiveBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct Unique2Backward0 : public TraceableFunction {
  TORCH_API Unique2Backward0() = default;
#else
struct TORCH_API Unique2Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Unique2Backward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct UnsafeViewBackward0 : public TraceableFunction {
  TORCH_API UnsafeViewBackward0() = default;
#else
struct TORCH_API UnsafeViewBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeViewBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct LiftBackward0 : public TraceableFunction {
  TORCH_API LiftBackward0() = default;
#else
struct TORCH_API LiftBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LiftBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct LiftFreshBackward0 : public TraceableFunction {
  TORCH_API LiftFreshBackward0() = default;
#else
struct TORCH_API LiftFreshBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LiftFreshBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct UnsqueezeBackward0 : public Node {
  TORCH_API UnsqueezeBackward0() = default;
#else
struct TORCH_API UnsqueezeBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;

};
#ifdef _WIN32
struct UnsqueezeBackward1 : public TraceableFunction {
  TORCH_API UnsqueezeBackward1() = default;
#else
struct TORCH_API UnsqueezeBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;

};
#ifdef _WIN32
struct VarBackward0 : public TraceableFunction {
  TORCH_API VarBackward0() = default;
#else
struct TORCH_API VarBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;

};
#ifdef _WIN32
struct VarMeanBackward0 : public TraceableFunction {
  TORCH_API VarMeanBackward0() = default;
#else
struct TORCH_API VarMeanBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarMeanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;

};
#ifdef _WIN32
struct ViewBackward0 : public Node {
  TORCH_API ViewBackward0() = default;
#else
struct TORCH_API ViewBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct ViewBackwardAutogradNestedTensor0 : public Node {
  TORCH_API ViewBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API ViewBackwardAutogradNestedTensor0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ViewAsRealBackward0 : public Node {
  TORCH_API ViewAsRealBackward0() = default;
#else
struct TORCH_API ViewAsRealBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsRealBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ViewAsComplexBackward0 : public Node {
  TORCH_API ViewAsComplexBackward0() = default;
#else
struct TORCH_API ViewAsComplexBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsComplexBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct WhereBackward0 : public TraceableFunction {
  TORCH_API WhereBackward0() = default;
#else
struct TORCH_API WhereBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "WhereBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    condition_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable condition_;

};
#ifdef _WIN32
struct WeightNormInterfaceBackward0 : public TraceableFunction {
  TORCH_API WeightNormInterfaceBackward0() = default;
#else
struct TORCH_API WeightNormInterfaceBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "WeightNormInterfaceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    g_.reset_data();
    v_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable g_;
  SavedVariable v_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct ZeroBackward0 : public TraceableFunction {
  TORCH_API ZeroBackward0() = default;
#else
struct TORCH_API ZeroBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ZeroBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct SparseMaskBackward0 : public TraceableFunction {
  TORCH_API SparseMaskBackward0() = default;
#else
struct TORCH_API SparseMaskBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseMaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable mask_;
  at::Layout self_layout;

};
#ifdef _WIN32
struct SparseCooTensorWithDimsAndTensorsBackward0 : public TraceableFunction {
  TORCH_API SparseCooTensorWithDimsAndTensorsBackward0() = default;
#else
struct TORCH_API SparseCooTensorWithDimsAndTensorsBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseCooTensorWithDimsAndTensorsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct SparseCompressedTensorBackward0 : public TraceableFunction {
  TORCH_API SparseCompressedTensorBackward0() = default;
#else
struct TORCH_API SparseCompressedTensorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseCompressedTensorBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    values_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable values_;
  SavedVariable result_;

};
#ifdef _WIN32
struct SparseSumBackward0 : public TraceableFunction {
  TORCH_API SparseSumBackward0() = default;
#else
struct TORCH_API SparseSumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  SavedVariable self_;

};
#ifdef _WIN32
struct StandardGammaBackward0 : public TraceableFunction {
  TORCH_API StandardGammaBackward0() = default;
#else
struct TORCH_API StandardGammaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct StandardGammaGradBackward0 : public TraceableFunction {
  TORCH_API StandardGammaGradBackward0() = default;
#else
struct TORCH_API StandardGammaGradBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaGradBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ValuesBackward0 : public Node {
  TORCH_API ValuesBackward0() = default;
#else
struct TORCH_API ValuesBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ValuesBackwardAutogradNestedTensor0 : public Node {
  TORCH_API ValuesBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API ValuesBackwardAutogradNestedTensor0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct TrilinearBackward0 : public TraceableFunction {
  TORCH_API TrilinearBackward0() = default;
#else
struct TORCH_API TrilinearBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilinearBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    i1_.reset_data();
    i2_.reset_data();
    i3_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> expand1;
  std::vector<int64_t> expand2;
  std::vector<int64_t> expand3;
  SavedVariable i1_;
  SavedVariable i2_;
  SavedVariable i3_;
  std::vector<int64_t> sumdim;

};
#ifdef _WIN32
struct ConstantPadNdBackward0 : public TraceableFunction {
  TORCH_API ConstantPadNdBackward0() = default;
#else
struct TORCH_API ConstantPadNdBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConstantPadNdBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> pad;

};
#ifdef _WIN32
struct BinaryCrossEntropyBackward0 : public TraceableFunction {
  TORCH_API BinaryCrossEntropyBackward0() = default;
#else
struct TORCH_API BinaryCrossEntropyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct BinaryCrossEntropyBackwardBackward0 : public TraceableFunction {
  TORCH_API BinaryCrossEntropyBackwardBackward0() = default;
#else
struct TORCH_API BinaryCrossEntropyBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct BinaryCrossEntropyWithLogitsBackward0 : public TraceableFunction {
  TORCH_API BinaryCrossEntropyWithLogitsBackward0() = default;
#else
struct TORCH_API BinaryCrossEntropyWithLogitsBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyWithLogitsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    pos_weight_.reset_data();
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable pos_weight_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct EmbeddingBackward0 : public TraceableFunction {
  TORCH_API EmbeddingBackward0() = default;
#else
struct TORCH_API EmbeddingBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  c10::SymInt padding_idx;
  bool scale_grad_by_freq;
  bool sparse;
  c10::SymInt weight_sym_argsize_0;

};
#ifdef _WIN32
struct EmbeddingDenseBackwardBackward0 : public TraceableFunction {
  TORCH_API EmbeddingDenseBackwardBackward0() = default;
#else
struct TORCH_API EmbeddingDenseBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingDenseBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  c10::SymInt padding_idx;

};
#ifdef _WIN32
struct EmbeddingBagBackward0 : public TraceableFunction {
  TORCH_API EmbeddingBagBackward0() = default;
#else
struct TORCH_API EmbeddingBagBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBagBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    offsets_.reset_data();
    per_sample_weights_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  int64_t mode = 0;
  SavedVariable offsets_;
  int64_t padding_idx = 0;
  SavedVariable per_sample_weights_;
  bool scale_grad_by_freq;
  bool sparse;
  SavedVariable weight_;
  c10::SymInt weight_sym_argsize_0;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
#ifdef _WIN32
struct EmbeddingBagBackwardBackward0 : public TraceableFunction {
  TORCH_API EmbeddingBagBackwardBackward0() = default;
#else
struct TORCH_API EmbeddingBagBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBagBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct EmbeddingBagDenseBackwardBackward0 : public TraceableFunction {
  TORCH_API EmbeddingBagDenseBackwardBackward0() = default;
#else
struct TORCH_API EmbeddingBagDenseBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBagDenseBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct EmbeddingRenormBackward0 : public TraceableFunction {
  TORCH_API EmbeddingRenormBackward0() = default;
#else
struct TORCH_API EmbeddingRenormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingRenormBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct MseLossBackward0 : public TraceableFunction {
  TORCH_API MseLossBackward0() = default;
#else
struct TORCH_API MseLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct MultiMarginLossBackward0 : public TraceableFunction {
  TORCH_API MultiMarginLossBackward0() = default;
#else
struct TORCH_API MultiMarginLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultiMarginLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar margin;
  at::Scalar p;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MultilabelMarginLossBackward0 : public TraceableFunction {
  TORCH_API MultilabelMarginLossBackward0() = default;
#else
struct TORCH_API MultilabelMarginLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultilabelMarginLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    is_target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable is_target_;

};
#ifdef _WIN32
struct NllLossBackward0 : public TraceableFunction {
  TORCH_API NllLossBackward0() = default;
#else
struct TORCH_API NllLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable total_weight_;

};
#ifdef _WIN32
struct NllLoss2DBackward0 : public TraceableFunction {
  TORCH_API NllLoss2DBackward0() = default;
#else
struct TORCH_API NllLoss2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable total_weight_;

};
#ifdef _WIN32
struct SmoothL1LossBackward0 : public TraceableFunction {
  TORCH_API SmoothL1LossBackward0() = default;
#else
struct TORCH_API SmoothL1LossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double beta;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct HuberLossBackward0 : public TraceableFunction {
  TORCH_API HuberLossBackward0() = default;
#else
struct TORCH_API HuberLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HuberLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double delta;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct SoftMarginLossBackward0 : public TraceableFunction {
  TORCH_API SoftMarginLossBackward0() = default;
#else
struct TORCH_API SoftMarginLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct ReluBackward0 : public TraceableFunction {
  TORCH_API ReluBackward0() = default;
#else
struct TORCH_API ReluBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable result_;

};
#ifdef _WIN32
struct SiluBackward0 : public TraceableFunction {
  TORCH_API SiluBackward0() = default;
#else
struct TORCH_API SiluBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SiluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct MishBackward0 : public TraceableFunction {
  TORCH_API MishBackward0() = default;
#else
struct TORCH_API MishBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MishBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct EluBackward0 : public TraceableFunction {
  TORCH_API EluBackward0() = default;
#else
struct TORCH_API EluBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar input_scale;
  at::Scalar scale;
  SavedVariable self_;

};
#ifdef _WIN32
struct EluBackward1 : public TraceableFunction {
  TORCH_API EluBackward1() = default;
#else
struct TORCH_API EluBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar input_scale;
  at::Scalar scale;
  SavedVariable result_;

};
#ifdef _WIN32
struct CeluBackward0 : public TraceableFunction {
  TORCH_API CeluBackward0() = default;
#else
struct TORCH_API CeluBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable self_;

};
#ifdef _WIN32
struct CeluBackward1 : public TraceableFunction {
  TORCH_API CeluBackward1() = default;
#else
struct TORCH_API CeluBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable result_;

};
#ifdef _WIN32
struct GeluBackward0 : public TraceableFunction {
  TORCH_API GeluBackward0() = default;
#else
struct TORCH_API GeluBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::string approximate;
  SavedVariable self_;

};
#ifdef _WIN32
struct GeluBackwardBackward0 : public TraceableFunction {
  TORCH_API GeluBackwardBackward0() = default;
#else
struct TORCH_API GeluBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::string approximate;
  SavedVariable grad_output_;
  SavedVariable self_;

};
#ifdef _WIN32
struct GluBackward0 : public TraceableFunction {
  TORCH_API GluBackward0() = default;
#else
struct TORCH_API GluBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;

};
#ifdef _WIN32
struct HardshrinkBackward0 : public TraceableFunction {
  TORCH_API HardshrinkBackward0() = default;
#else
struct TORCH_API HardshrinkBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;

};
#ifdef _WIN32
struct HardshrinkBackwardBackward0 : public TraceableFunction {
  TORCH_API HardshrinkBackwardBackward0() = default;
#else
struct TORCH_API HardshrinkBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;

};
#ifdef _WIN32
struct HardtanhBackward0 : public TraceableFunction {
  TORCH_API HardtanhBackward0() = default;
#else
struct TORCH_API HardtanhBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar max_val;
  at::Scalar min_val;
  SavedVariable self_;

};
#ifdef _WIN32
struct LeakyReluBackward0 : public TraceableFunction {
  TORCH_API LeakyReluBackward0() = default;
#else
struct TORCH_API LeakyReluBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar negative_slope;
  SavedVariable self_;

};
#ifdef _WIN32
struct LeakyReluBackward1 : public TraceableFunction {
  TORCH_API LeakyReluBackward1() = default;
#else
struct TORCH_API LeakyReluBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar negative_slope;
  SavedVariable result_;

};
#ifdef _WIN32
struct LogSigmoidBackward0 : public TraceableFunction {
  TORCH_API LogSigmoidBackward0() = default;
#else
struct TORCH_API LogSigmoidBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    buffer_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable buffer_;

};
#ifdef _WIN32
struct LogSoftmaxBackward0 : public TraceableFunction {
  TORCH_API LogSoftmaxBackward0() = default;
#else
struct TORCH_API LogSoftmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::ScalarType self_scalar_type;
  SavedVariable result_;

};
#ifdef _WIN32
struct SparseLogSoftmaxBackward0 : public TraceableFunction {
  TORCH_API SparseLogSoftmaxBackward0() = default;
#else
struct TORCH_API SparseLogSoftmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseLogSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct MaskedSoftmaxBackward0 : public TraceableFunction {
  TORCH_API MaskedSoftmaxBackward0() = default;
#else
struct TORCH_API MaskedSoftmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ::std::optional<int64_t> dim;
  SavedVariable mask_;
  SavedVariable result_;

};
#ifdef _WIN32
struct PreluKernelBackward0 : public TraceableFunction {
  TORCH_API PreluKernelBackward0() = default;
#else
struct TORCH_API PreluKernelBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluKernelBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct PreluKernelBackwardBackward0 : public TraceableFunction {
  TORCH_API PreluKernelBackwardBackward0() = default;
#else
struct TORCH_API PreluKernelBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluKernelBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  at::TensorOptions grad_output_options;
  SavedVariable self_;
  torch::autograd::generated::TypeAndSize self_info;
  at::TensorOptions self_options;
  SavedVariable weight_;
  at::TensorOptions weight_options;

};
#ifdef _WIN32
struct RreluWithNoiseBackward0 : public TraceableFunction {
  TORCH_API RreluWithNoiseBackward0() = default;
#else
struct TORCH_API RreluWithNoiseBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lower;
  SavedVariable noise_;
  SavedVariable self_;
  bool training;
  at::Scalar upper;

};
#ifdef _WIN32
struct RreluWithNoiseBackward1 : public TraceableFunction {
  TORCH_API RreluWithNoiseBackward1() = default;
#else
struct TORCH_API RreluWithNoiseBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lower;
  SavedVariable noise_;
  bool training;
  at::Scalar upper;
  SavedVariable result_;

};
#ifdef _WIN32
struct RreluWithNoiseFunctionalBackward0 : public TraceableFunction {
  TORCH_API RreluWithNoiseFunctionalBackward0() = default;
#else
struct TORCH_API RreluWithNoiseFunctionalBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseFunctionalBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lower;
  SavedVariable noise_;
  SavedVariable self_;
  bool training;
  at::Scalar upper;

};
#ifdef _WIN32
struct SoftmaxBackward0 : public TraceableFunction {
  TORCH_API SoftmaxBackward0() = default;
#else
struct TORCH_API SoftmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::ScalarType self_scalar_type;
  SavedVariable result_;

};
#ifdef _WIN32
struct SparseSoftmaxBackward0 : public TraceableFunction {
  TORCH_API SparseSoftmaxBackward0() = default;
#else
struct TORCH_API SparseSoftmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable result_;

};
#ifdef _WIN32
struct SparseSparseMatmulBackward0 : public TraceableFunction {
  TORCH_API SparseSparseMatmulBackward0() = default;
#else
struct TORCH_API SparseSparseMatmulBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSparseMatmulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;

};
#ifdef _WIN32
struct SoftplusBackward0 : public TraceableFunction {
  TORCH_API SoftplusBackward0() = default;
#else
struct TORCH_API SoftplusBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar beta;
  SavedVariable self_;
  at::Scalar threshold;

};
#ifdef _WIN32
struct SoftshrinkBackward0 : public TraceableFunction {
  TORCH_API SoftshrinkBackward0() = default;
#else
struct TORCH_API SoftshrinkBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;

};
#ifdef _WIN32
struct ThresholdBackward0 : public TraceableFunction {
  TORCH_API ThresholdBackward0() = default;
#else
struct TORCH_API ThresholdBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  at::Scalar threshold;

};
#ifdef _WIN32
struct ThresholdBackward1 : public TraceableFunction {
  TORCH_API ThresholdBackward1() = default;
#else
struct TORCH_API ThresholdBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  at::Scalar threshold;

};
#ifdef _WIN32
struct ReflectionPad1DBackward0 : public TraceableFunction {
  TORCH_API ReflectionPad1DBackward0() = default;
#else
struct TORCH_API ReflectionPad1DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;

};
#ifdef _WIN32
struct ReflectionPad2DBackward0 : public TraceableFunction {
  TORCH_API ReflectionPad2DBackward0() = default;
#else
struct TORCH_API ReflectionPad2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;

};
#ifdef _WIN32
struct ReflectionPad3DBackward0 : public TraceableFunction {
  TORCH_API ReflectionPad3DBackward0() = default;
#else
struct TORCH_API ReflectionPad3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;

};
#ifdef _WIN32
struct ReplicationPad1DBackward0 : public TraceableFunction {
  TORCH_API ReplicationPad1DBackward0() = default;
#else
struct TORCH_API ReplicationPad1DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;

};
#ifdef _WIN32
struct ReplicationPad2DBackward0 : public TraceableFunction {
  TORCH_API ReplicationPad2DBackward0() = default;
#else
struct TORCH_API ReplicationPad2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;

};
#ifdef _WIN32
struct ReplicationPad3DBackward0 : public TraceableFunction {
  TORCH_API ReplicationPad3DBackward0() = default;
#else
struct TORCH_API ReplicationPad3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;

};
#ifdef _WIN32
struct UpsampleLinear1DBackward0 : public TraceableFunction {
  TORCH_API UpsampleLinear1DBackward0() = default;
#else
struct TORCH_API UpsampleLinear1DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleBilinear2DBackward0 : public TraceableFunction {
  TORCH_API UpsampleBilinear2DBackward0() = default;
#else
struct TORCH_API UpsampleBilinear2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleBilinear2DAaBackward0 : public TraceableFunction {
  TORCH_API UpsampleBilinear2DAaBackward0() = default;
#else
struct TORCH_API UpsampleBilinear2DAaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DAaBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleBicubic2DBackward0 : public TraceableFunction {
  TORCH_API UpsampleBicubic2DBackward0() = default;
#else
struct TORCH_API UpsampleBicubic2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleBicubic2DAaBackward0 : public TraceableFunction {
  TORCH_API UpsampleBicubic2DAaBackward0() = default;
#else
struct TORCH_API UpsampleBicubic2DAaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DAaBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleTrilinear3DBackward0 : public TraceableFunction {
  TORCH_API UpsampleTrilinear3DBackward0() = default;
#else
struct TORCH_API UpsampleTrilinear3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_d;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleNearest1DBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearest1DBackward0() = default;
#else
struct TORCH_API UpsampleNearest1DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleNearestExact1DBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearestExact1DBackward0() = default;
#else
struct TORCH_API UpsampleNearestExact1DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearestExact1DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleNearest2DBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearest2DBackward0() = default;
#else
struct TORCH_API UpsampleNearest2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleNearestExact2DBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearestExact2DBackward0() = default;
#else
struct TORCH_API UpsampleNearestExact2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearestExact2DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleNearest3DBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearest3DBackward0() = default;
#else
struct TORCH_API UpsampleNearest3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_d;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct UpsampleNearestExact3DBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearestExact3DBackward0() = default;
#else
struct TORCH_API UpsampleNearestExact3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearestExact3DBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_d;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct PixelShuffleBackward0 : public TraceableFunction {
  TORCH_API PixelShuffleBackward0() = default;
#else
struct TORCH_API PixelShuffleBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PixelShuffleBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t upscale_factor = 0;

};
#ifdef _WIN32
struct PixelUnshuffleBackward0 : public TraceableFunction {
  TORCH_API PixelUnshuffleBackward0() = default;
#else
struct TORCH_API PixelUnshuffleBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PixelUnshuffleBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t downscale_factor = 0;

};
#ifdef _WIN32
struct ChannelShuffleBackward0 : public TraceableFunction {
  TORCH_API ChannelShuffleBackward0() = default;
#else
struct TORCH_API ChannelShuffleBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ChannelShuffleBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt groups;

};
#ifdef _WIN32
struct AdaptiveAvgPool2DBackward0 : public TraceableFunction {
  TORCH_API AdaptiveAvgPool2DBackward0() = default;
#else
struct TORCH_API AdaptiveAvgPool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AdaptiveAvgPool3DBackward0 : public TraceableFunction {
  TORCH_API AdaptiveAvgPool3DBackward0() = default;
#else
struct TORCH_API AdaptiveAvgPool3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct AdaptiveMaxPool2DBackward0 : public TraceableFunction {
  TORCH_API AdaptiveMaxPool2DBackward0() = default;
#else
struct TORCH_API AdaptiveMaxPool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct AdaptiveMaxPool3DBackward0 : public TraceableFunction {
  TORCH_API AdaptiveMaxPool3DBackward0() = default;
#else
struct TORCH_API AdaptiveMaxPool3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct AvgPool2DBackward0 : public TraceableFunction {
  TORCH_API AvgPool2DBackward0() = default;
#else
struct TORCH_API AvgPool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  ::std::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;

};
#ifdef _WIN32
struct AvgPool3DBackward0 : public TraceableFunction {
  TORCH_API AvgPool3DBackward0() = default;
#else
struct TORCH_API AvgPool3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  ::std::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;

};
#ifdef _WIN32
struct FractionalMaxPool2DBackward0 : public TraceableFunction {
  TORCH_API FractionalMaxPool2DBackward0() = default;
#else
struct TORCH_API FractionalMaxPool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable self_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct FractionalMaxPool3DBackward0 : public TraceableFunction {
  TORCH_API FractionalMaxPool3DBackward0() = default;
#else
struct TORCH_API FractionalMaxPool3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable self_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct LinearBackward0 : public TraceableFunction {
  TORCH_API LinearBackward0() = default;
#else
struct TORCH_API LinearBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinearBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable input_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct LinearBackwardBackward0 : public TraceableFunction {
  TORCH_API LinearBackwardBackward0() = default;
#else
struct TORCH_API LinearBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinearBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MaxPool2DBackward0 : public TraceableFunction {
  TORCH_API MaxPool2DBackward0() = default;
#else
struct TORCH_API MaxPool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;

};
#ifdef _WIN32
struct MpsConvolutionBackward0 : public TraceableFunction {
  TORCH_API MpsConvolutionBackward0() = default;
#else
struct TORCH_API MpsConvolutionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MpsConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MpsConvolutionBackwardBackward0 : public TraceableFunction {
  TORCH_API MpsConvolutionBackwardBackward0() = default;
#else
struct TORCH_API MpsConvolutionBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MpsConvolutionBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  SavedVariable grad_output_;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MaxPool2DWithIndicesBackward0 : public TraceableFunction {
  TORCH_API MaxPool2DWithIndicesBackward0() = default;
#else
struct TORCH_API MaxPool2DWithIndicesBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result1_;

};
#ifdef _WIN32
struct MaxPool3DWithIndicesBackward0 : public TraceableFunction {
  TORCH_API MaxPool3DWithIndicesBackward0() = default;
#else
struct TORCH_API MaxPool3DWithIndicesBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result1_;

};
#ifdef _WIN32
struct MaxUnpool2DBackward0 : public TraceableFunction {
  TORCH_API MaxUnpool2DBackward0() = default;
#else
struct TORCH_API MaxUnpool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;

};
#ifdef _WIN32
struct MaxUnpool3DBackward0 : public TraceableFunction {
  TORCH_API MaxUnpool3DBackward0() = default;
#else
struct TORCH_API MaxUnpool3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;

};
#ifdef _WIN32
struct ConvolutionBackward0 : public TraceableFunction {
  TORCH_API ConvolutionBackward0() = default;
#else
struct TORCH_API ConvolutionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;

};
#ifdef _WIN32
struct ConvolutionBackward1 : public TraceableFunction {
  TORCH_API ConvolutionBackward1() = default;
#else
struct TORCH_API ConvolutionBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;

};
#ifdef _WIN32
struct ConvolutionBackwardBackward0 : public TraceableFunction {
  TORCH_API ConvolutionBackwardBackward0() = default;
#else
struct TORCH_API ConvolutionBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  SavedVariable grad_output_;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;

};
#ifdef _WIN32
struct ConvolutionOverrideableBackward0 : public TraceableFunction {
  TORCH_API ConvolutionOverrideableBackward0() = default;
#else
struct TORCH_API ConvolutionOverrideableBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionOverrideableBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;

};
#ifdef _WIN32
struct ConvolutionBackwardOverrideableBackward0 : public TraceableFunction {
  TORCH_API ConvolutionBackwardOverrideableBackward0() = default;
#else
struct TORCH_API ConvolutionBackwardOverrideableBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionBackwardOverrideableBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  SavedVariable grad_output_;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;

};
#ifdef _WIN32
struct SlowConvTranspose2DBackward0 : public TraceableFunction {
  TORCH_API SlowConvTranspose2DBackward0() = default;
#else
struct TORCH_API SlowConvTranspose2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct SlowConvTranspose3DBackward0 : public TraceableFunction {
  TORCH_API SlowConvTranspose3DBackward0() = default;
#else
struct TORCH_API SlowConvTranspose3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct SlowConv2DBackward0 : public TraceableFunction {
  TORCH_API SlowConv2DBackward0() = default;
#else
struct TORCH_API SlowConv2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConv2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> kernel_size;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct SlowConv2DBackwardBackward0 : public TraceableFunction {
  TORCH_API SlowConv2DBackwardBackward0() = default;
#else
struct TORCH_API SlowConv2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConv2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct ConvDepthwise2DBackward0 : public TraceableFunction {
  TORCH_API ConvDepthwise2DBackward0() = default;
#else
struct TORCH_API ConvDepthwise2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct ConvDepthwise3DBackward0 : public TraceableFunction {
  TORCH_API ConvDepthwise3DBackward0() = default;
#else
struct TORCH_API ConvDepthwise3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct SlowConv3DBackward0 : public TraceableFunction {
  TORCH_API SlowConv3DBackward0() = default;
#else
struct TORCH_API SlowConv3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConv3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct SlowConvDilated2DBackward0 : public TraceableFunction {
  TORCH_API SlowConvDilated2DBackward0() = default;
#else
struct TORCH_API SlowConvDilated2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct SlowConvDilated3DBackward0 : public TraceableFunction {
  TORCH_API SlowConvDilated3DBackward0() = default;
#else
struct TORCH_API SlowConvDilated3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct Col2ImBackward0 : public TraceableFunction {
  TORCH_API Col2ImBackward0() = default;
#else
struct TORCH_API Col2ImBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Col2ImBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
#ifdef _WIN32
struct Im2ColBackward0 : public TraceableFunction {
  TORCH_API Im2ColBackward0() = default;
#else
struct TORCH_API Im2ColBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Im2ColBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  c10::SymInt self_sym_argsize_minus_1;
  c10::SymInt self_sym_argsize_minus_2;
  std::vector<int64_t> stride;

};
#ifdef _WIN32
struct AdaptiveAvgPool2DBackwardBackward0 : public TraceableFunction {
  TORCH_API AdaptiveAvgPool2DBackwardBackward0() = default;
#else
struct TORCH_API AdaptiveAvgPool2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt grad_output_sym_argsize_minus_1;
  c10::SymInt grad_output_sym_argsize_minus_2;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct AdaptiveAvgPool3DBackwardBackward0 : public TraceableFunction {
  TORCH_API AdaptiveAvgPool3DBackwardBackward0() = default;
#else
struct TORCH_API AdaptiveAvgPool3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt grad_output_sym_argsize_minus_1;
  c10::SymInt grad_output_sym_argsize_minus_2;
  c10::SymInt grad_output_sym_argsize_minus_3;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct AdaptiveMaxPool2DBackwardBackward0 : public TraceableFunction {
  TORCH_API AdaptiveMaxPool2DBackwardBackward0() = default;
#else
struct TORCH_API AdaptiveMaxPool2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct AdaptiveMaxPool3DBackwardBackward0 : public TraceableFunction {
  TORCH_API AdaptiveMaxPool3DBackwardBackward0() = default;
#else
struct TORCH_API AdaptiveMaxPool3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct AvgPool2DBackwardBackward0 : public TraceableFunction {
  TORCH_API AvgPool2DBackwardBackward0() = default;
#else
struct TORCH_API AvgPool2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  ::std::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;
  std::vector<int64_t> stride;

};
#ifdef _WIN32
struct AvgPool3DBackwardBackward0 : public TraceableFunction {
  TORCH_API AvgPool3DBackwardBackward0() = default;
#else
struct TORCH_API AvgPool3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  ::std::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;
  std::vector<int64_t> stride;

};
#ifdef _WIN32
struct EluBackwardBackward0 : public TraceableFunction {
  TORCH_API EluBackwardBackward0() = default;
#else
struct TORCH_API EluBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_or_result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable grad_output_;
  at::Scalar input_scale;
  bool is_result;
  at::Scalar scale;
  SavedVariable self_or_result_;

};
#ifdef _WIN32
struct FractionalMaxPool2DBackwardBackward0 : public TraceableFunction {
  TORCH_API FractionalMaxPool2DBackwardBackward0() = default;
#else
struct TORCH_API FractionalMaxPool2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct FractionalMaxPool3DBackwardBackward0 : public TraceableFunction {
  TORCH_API FractionalMaxPool3DBackwardBackward0() = default;
#else
struct TORCH_API FractionalMaxPool3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct GluBackwardBackward0 : public TraceableFunction {
  TORCH_API GluBackwardBackward0() = default;
#else
struct TORCH_API GluBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable grad_output_;
  SavedVariable self_;

};
#ifdef _WIN32
struct HardtanhBackwardBackward0 : public TraceableFunction {
  TORCH_API HardtanhBackwardBackward0() = default;
#else
struct TORCH_API HardtanhBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar max_val;
  at::Scalar min_val;
  SavedVariable self_;

};
#ifdef _WIN32
struct LogSigmoidBackwardBackward0 : public TraceableFunction {
  TORCH_API LogSigmoidBackwardBackward0() = default;
#else
struct TORCH_API LogSigmoidBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.reset_data();
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable buffer_;
  SavedVariable grad_output_;
  SavedVariable self_;

};
#ifdef _WIN32
struct LogSoftmaxBackwardDataBackward0 : public TraceableFunction {
  TORCH_API LogSoftmaxBackwardDataBackward0() = default;
#else
struct TORCH_API LogSoftmaxBackwardDataBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackwardDataBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable grad_output_;
  SavedVariable output_;

};
#ifdef _WIN32
struct LeakyReluBackwardBackward0 : public TraceableFunction {
  TORCH_API LeakyReluBackwardBackward0() = default;
#else
struct TORCH_API LeakyReluBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar negative_slope;
  SavedVariable self_;

};
#ifdef _WIN32
struct MaxPool2DBackwardBackward0 : public TraceableFunction {
  TORCH_API MaxPool2DBackwardBackward0() = default;
#else
struct TORCH_API MaxPool2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct MaxPool2DWithIndicesBackwardBackward0 : public TraceableFunction {
  TORCH_API MaxPool2DWithIndicesBackwardBackward0() = default;
#else
struct TORCH_API MaxPool2DWithIndicesBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct MaxPool3DWithIndicesBackwardBackward0 : public TraceableFunction {
  TORCH_API MaxPool3DWithIndicesBackwardBackward0() = default;
#else
struct TORCH_API MaxPool3DWithIndicesBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct MseLossBackwardBackward0 : public TraceableFunction {
  TORCH_API MseLossBackwardBackward0() = default;
#else
struct TORCH_API MseLossBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct NllLossBackwardBackward0 : public TraceableFunction {
  TORCH_API NllLossBackwardBackward0() = default;
#else
struct TORCH_API NllLossBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable target_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct NllLoss2DBackwardBackward0 : public TraceableFunction {
  TORCH_API NllLoss2DBackwardBackward0() = default;
#else
struct TORCH_API NllLoss2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable target_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct RreluWithNoiseBackwardBackward0 : public TraceableFunction {
  TORCH_API RreluWithNoiseBackwardBackward0() = default;
#else
struct TORCH_API RreluWithNoiseBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lower;
  SavedVariable noise_;
  SavedVariable self_;
  bool training;
  at::Scalar upper;

};
#ifdef _WIN32
struct ReflectionPad1DBackwardBackward0 : public TraceableFunction {
  TORCH_API ReflectionPad1DBackwardBackward0() = default;
#else
struct TORCH_API ReflectionPad1DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct ReflectionPad2DBackwardBackward0 : public TraceableFunction {
  TORCH_API ReflectionPad2DBackwardBackward0() = default;
#else
struct TORCH_API ReflectionPad2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct ReflectionPad3DBackwardBackward0 : public TraceableFunction {
  TORCH_API ReflectionPad3DBackwardBackward0() = default;
#else
struct TORCH_API ReflectionPad3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad3DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct ReplicationPad1DBackwardBackward0 : public TraceableFunction {
  TORCH_API ReplicationPad1DBackwardBackward0() = default;
#else
struct TORCH_API ReplicationPad1DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct ReplicationPad2DBackwardBackward0 : public TraceableFunction {
  TORCH_API ReplicationPad2DBackwardBackward0() = default;
#else
struct TORCH_API ReplicationPad2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct ReplicationPad3DBackwardBackward0 : public TraceableFunction {
  TORCH_API ReplicationPad3DBackwardBackward0() = default;
#else
struct TORCH_API ReplicationPad3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
#ifdef _WIN32
struct SparseSampledAddmmBackward0 : public TraceableFunction {
  TORCH_API SparseSampledAddmmBackward0() = default;
#else
struct TORCH_API SparseSampledAddmmBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSampledAddmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat1_.reset_data();
    mat2_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat1_;
  SavedVariable mat2_;
  SavedVariable self_;

};
#ifdef _WIN32
struct SparseMmReduceImplBackward0 : public TraceableFunction {
  TORCH_API SparseMmReduceImplBackward0() = default;
#else
struct TORCH_API SparseMmReduceImplBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseMmReduceImplBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  std::string reduce;
  SavedVariable self_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct SmoothL1LossBackwardBackward0 : public TraceableFunction {
  TORCH_API SmoothL1LossBackwardBackward0() = default;
#else
struct TORCH_API SmoothL1LossBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double beta;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct HuberLossBackwardBackward0 : public TraceableFunction {
  TORCH_API HuberLossBackwardBackward0() = default;
#else
struct TORCH_API HuberLossBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HuberLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double delta;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct SoftplusBackwardBackward0 : public TraceableFunction {
  TORCH_API SoftplusBackwardBackward0() = default;
#else
struct TORCH_API SoftplusBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar beta;
  SavedVariable grad_output_;
  SavedVariable self_;
  at::Scalar threshold;

};
#ifdef _WIN32
struct SoftmaxBackwardDataBackward0 : public TraceableFunction {
  TORCH_API SoftmaxBackwardDataBackward0() = default;
#else
struct TORCH_API SoftmaxBackwardDataBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackwardDataBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable grad_output_;
  at::ScalarType input_dtype;
  SavedVariable output_;

};
#ifdef _WIN32
struct SoftMarginLossBackwardBackward0 : public TraceableFunction {
  TORCH_API SoftMarginLossBackwardBackward0() = default;
#else
struct TORCH_API SoftMarginLossBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;

};
#ifdef _WIN32
struct SoftshrinkBackwardBackward0 : public TraceableFunction {
  TORCH_API SoftshrinkBackwardBackward0() = default;
#else
struct TORCH_API SoftshrinkBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;

};
#ifdef _WIN32
struct ThresholdBackwardBackward0 : public TraceableFunction {
  TORCH_API ThresholdBackwardBackward0() = default;
#else
struct TORCH_API ThresholdBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  at::Scalar threshold;

};
#ifdef _WIN32
struct UpsampleLinear1DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleLinear1DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleLinear1DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales;

};
#ifdef _WIN32
struct UpsampleBilinear2DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleBilinear2DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleBilinear2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleBilinear2DAaBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleBilinear2DAaBackwardBackward0() = default;
#else
struct TORCH_API UpsampleBilinear2DAaBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DAaBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleBicubic2DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleBicubic2DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleBicubic2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleBicubic2DAaBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleBicubic2DAaBackwardBackward0() = default;
#else
struct TORCH_API UpsampleBicubic2DAaBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DAaBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleTrilinear3DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleTrilinear3DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleTrilinear3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_d;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleNearest1DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearest1DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleNearest1DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales;

};
#ifdef _WIN32
struct UpsampleNearestExact1DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearestExact1DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleNearestExact1DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearestExact1DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales;

};
#ifdef _WIN32
struct UpsampleNearest2DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearest2DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleNearest2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleNearestExact2DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearestExact2DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleNearestExact2DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearestExact2DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleNearest3DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearest3DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleNearest3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_d;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct UpsampleNearestExact3DBackwardBackward0 : public TraceableFunction {
  TORCH_API UpsampleNearestExact3DBackwardBackward0() = default;
#else
struct TORCH_API UpsampleNearestExact3DBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearestExact3DBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  ::std::optional<double> scales_d;
  ::std::optional<double> scales_h;
  ::std::optional<double> scales_w;

};
#ifdef _WIN32
struct SigmoidBackwardBackward0 : public TraceableFunction {
  TORCH_API SigmoidBackwardBackward0() = default;
#else
struct TORCH_API SigmoidBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable output_;

};
#ifdef _WIN32
struct TanhBackwardBackward0 : public TraceableFunction {
  TORCH_API TanhBackwardBackward0() = default;
#else
struct TORCH_API TanhBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable output_;

};
#ifdef _WIN32
struct CudnnCtcLossBackward0 : public TraceableFunction {
  TORCH_API CudnnCtcLossBackward0() = default;
#else
struct TORCH_API CudnnCtcLossBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnCtcLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct CudnnCtcLossBackward1 : public TraceableFunction {
  TORCH_API CudnnCtcLossBackward1() = default;
#else
struct TORCH_API CudnnCtcLossBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnCtcLossBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct CudnnConvolutionTransposeBackward0 : public TraceableFunction {
  TORCH_API CudnnConvolutionTransposeBackward0() = default;
#else
struct TORCH_API CudnnConvolutionTransposeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionTransposeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MpsConvolutionTransposeBackward0 : public TraceableFunction {
  TORCH_API MpsConvolutionTransposeBackward0() = default;
#else
struct TORCH_API MpsConvolutionTransposeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MpsConvolutionTransposeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct CudnnConvolutionBackward0 : public TraceableFunction {
  TORCH_API CudnnConvolutionBackward0() = default;
#else
struct TORCH_API CudnnConvolutionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct CudnnGridSamplerBackward0 : public TraceableFunction {
  TORCH_API CudnnGridSamplerBackward0() = default;
#else
struct TORCH_API CudnnGridSamplerBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnGridSamplerBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable grid_;
  SavedVariable self_;

};
#ifdef _WIN32
struct CudnnAffineGridGeneratorBackward0 : public TraceableFunction {
  TORCH_API CudnnAffineGridGeneratorBackward0() = default;
#else
struct TORCH_API CudnnAffineGridGeneratorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnAffineGridGeneratorBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t C = 0;
  int64_t H = 0;
  int64_t N = 0;
  int64_t W = 0;

};
#ifdef _WIN32
struct CudnnBatchNormBackward0 : public TraceableFunction {
  TORCH_API CudnnBatchNormBackward0() = default;
#else
struct TORCH_API CudnnBatchNormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
#ifdef _WIN32
struct CudnnBatchNormBackwardBackward0 : public TraceableFunction {
  TORCH_API CudnnBatchNormBackwardBackward0() = default;
#else
struct TORCH_API CudnnBatchNormBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    reserveSpace_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable grad_output_;
  SavedVariable input_;
  SavedVariable reserveSpace_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct NnpackSpatialConvolutionBackward0 : public TraceableFunction {
  TORCH_API NnpackSpatialConvolutionBackward0() = default;
#else
struct TORCH_API NnpackSpatialConvolutionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NnpackSpatialConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  SavedVariable input_;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct LstmMpsBackward0 : public TraceableFunction {
  TORCH_API LstmMpsBackward0() = default;
#else
struct TORCH_API LstmMpsBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LstmMpsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    hx_.clear();
    hx_released_ = true;
    input_.reset_data();
    params_.clear();
    params_released_ = true;
    result3_.reset_data();
    result4_.reset_data();
    result5_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool batch_first;
  bool bidirectional;
  double dropout;
  bool has_biases;
  std::vector<SavedVariable> hx_;
  bool hx_released_ = false;
  SavedVariable input_;
  int64_t num_layers = 0;
  std::vector<SavedVariable> params_;
  bool params_released_ = false;
  bool train;
  SavedVariable result3_;
  SavedVariable result4_;
  SavedVariable result5_;
  size_t hx_size_;
  size_t params_size_;
};
#ifdef _WIN32
struct CudnnRnnBackward0 : public TraceableFunction {
  TORCH_API CudnnRnnBackward0() = default;
#else
struct TORCH_API CudnnRnnBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx_.reset_data();
    dropout_state_.reset_data();
    hx_.reset_data();
    input_.reset_data();
    weight_.clear();
    weight_released_ = true;
    result0_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<c10::SymInt> batch_sizes;
  bool bidirectional;
  SavedVariable cx_;
  double dropout;
  SavedVariable dropout_state_;
  c10::SymInt hidden_size;
  SavedVariable hx_;
  SavedVariable input_;
  int64_t mode = 0;
  int64_t num_layers = 0;
  c10::SymInt proj_size;
  bool train;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
#ifdef _WIN32
struct CudnnRnnBackwardBackward0 : public TraceableFunction {
  TORCH_API CudnnRnnBackwardBackward0() = default;
#else
struct TORCH_API CudnnRnnBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackwardBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t weight_size_;
};
#ifdef _WIN32
struct MiopenConvolutionTransposeBackward0 : public TraceableFunction {
  TORCH_API MiopenConvolutionTransposeBackward0() = default;
#else
struct TORCH_API MiopenConvolutionTransposeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionTransposeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MiopenConvolutionBackward0 : public TraceableFunction {
  TORCH_API MiopenConvolutionBackward0() = default;
#else
struct TORCH_API MiopenConvolutionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MiopenDepthwiseConvolutionBackward0 : public TraceableFunction {
  TORCH_API MiopenDepthwiseConvolutionBackward0() = default;
#else
struct TORCH_API MiopenDepthwiseConvolutionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenDepthwiseConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MiopenBatchNormBackward0 : public TraceableFunction {
  TORCH_API MiopenBatchNormBackward0() = default;
#else
struct TORCH_API MiopenBatchNormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct MiopenBatchNormBackwardBackward0 : public TraceableFunction {
  TORCH_API MiopenBatchNormBackwardBackward0() = default;
#else
struct TORCH_API MiopenBatchNormBackwardBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable grad_output_;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MiopenRnnBackward0 : public TraceableFunction {
  TORCH_API MiopenRnnBackward0() = default;
#else
struct TORCH_API MiopenRnnBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenRnnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx_.reset_data();
    dropout_state_.reset_data();
    hx_.reset_data();
    input_.reset_data();
    weight_.clear();
    weight_released_ = true;
    result0_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<int64_t> batch_sizes;
  bool bidirectional;
  SavedVariable cx_;
  double dropout;
  SavedVariable dropout_state_;
  int64_t hidden_size = 0;
  SavedVariable hx_;
  SavedVariable input_;
  int64_t mode = 0;
  int64_t num_layers = 0;
  bool train;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
#ifdef _WIN32
struct MkldnnRnnLayerBackward0 : public TraceableFunction {
  TORCH_API MkldnnRnnLayerBackward0() = default;
#else
struct TORCH_API MkldnnRnnLayerBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnRnnLayerBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx__.reset_data();
    hx__.reset_data();
    input_.reset_data();
    weight0_.reset_data();
    weight1_.reset_data();
    weight2_.reset_data();
    weight3_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<int64_t> batch_sizes;
  bool bidirectional;
  SavedVariable cx__;
  bool has_biases;
  int64_t hidden_size = 0;
  SavedVariable hx__;
  SavedVariable input_;
  int64_t mode = 0;
  int64_t num_layers = 0;
  bool reverse;
  bool train;
  SavedVariable weight0_;
  SavedVariable weight1_;
  SavedVariable weight2_;
  SavedVariable weight3_;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
#ifdef _WIN32
struct MkldnnConvolutionBackward0 : public TraceableFunction {
  TORCH_API MkldnnConvolutionBackward0() = default;
#else
struct TORCH_API MkldnnConvolutionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MkldnnLinearBackward0 : public TraceableFunction {
  TORCH_API MkldnnLinearBackward0() = default;
#else
struct TORCH_API MkldnnLinearBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnLinearBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable weight_;

};
#ifdef _WIN32
struct MkldnnMaxPool2DBackward0 : public TraceableFunction {
  TORCH_API MkldnnMaxPool2DBackward0() = default;
#else
struct TORCH_API MkldnnMaxPool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnMaxPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result_;

};
#ifdef _WIN32
struct MkldnnMaxPool3DBackward0 : public TraceableFunction {
  TORCH_API MkldnnMaxPool3DBackward0() = default;
#else
struct TORCH_API MkldnnMaxPool3DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnMaxPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result_;

};
#ifdef _WIN32
struct MkldnnAdaptiveAvgPool2DBackward0 : public TraceableFunction {
  TORCH_API MkldnnAdaptiveAvgPool2DBackward0() = default;
#else
struct TORCH_API MkldnnAdaptiveAvgPool2DBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnAdaptiveAvgPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct MkldnnReshapeBackward0 : public TraceableFunction {
  TORCH_API MkldnnReshapeBackward0() = default;
#else
struct TORCH_API MkldnnReshapeBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnReshapeBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct NestedTensorFromTensorListBackward0 : public TraceableFunction {
  TORCH_API NestedTensorFromTensorListBackward0() = default;
#else
struct TORCH_API NestedTensorFromTensorListBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedTensorFromTensorListBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    list_.clear();
    list_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> list_;
  bool list_released_ = false;
  size_t list_size_;
};
#ifdef _WIN32
struct NestedTensorFromMaskBackward0 : public TraceableFunction {
  TORCH_API NestedTensorFromMaskBackward0() = default;
#else
struct TORCH_API NestedTensorFromMaskBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedTensorFromMaskBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> t_sym_sizes;

};
#ifdef _WIN32
struct NestedFromPaddedBackward0 : public TraceableFunction {
  TORCH_API NestedFromPaddedBackward0() = default;
#else
struct TORCH_API NestedFromPaddedBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedFromPaddedBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    padded_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool fuse_transform_0213;
  SavedVariable padded_;

};
#ifdef _WIN32
struct ToPaddedTensorBackward0 : public TraceableFunction {
  TORCH_API ToPaddedTensorBackward0() = default;
#else
struct TORCH_API ToPaddedTensorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToPaddedTensorBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;
  at::Layout self_layout;

};
#ifdef _WIN32
struct NestedFromPaddedTensorBackward0 : public TraceableFunction {
  TORCH_API NestedFromPaddedTensorBackward0() = default;
#else
struct TORCH_API NestedFromPaddedTensorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedFromPaddedTensorBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padded_sym_sizes;

};
#ifdef _WIN32
struct NestedViewFromBufferBackward0 : public Node {
  TORCH_API NestedViewFromBufferBackward0() = default;
#else
struct TORCH_API NestedViewFromBufferBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedViewFromBufferBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NestedViewFromJaggedBackward0 : public Node {
  TORCH_API NestedViewFromJaggedBackward0() = default;
#else
struct TORCH_API NestedViewFromJaggedBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedViewFromJaggedBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NestedGetValuesBackward0 : public Node {
  TORCH_API NestedGetValuesBackward0() = default;
#else
struct TORCH_API NestedGetValuesBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedGetValuesBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct SafeSoftmaxBackward0 : public TraceableFunction {
  TORCH_API SafeSoftmaxBackward0() = default;
#else
struct TORCH_API SafeSoftmaxBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SafeSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::ScalarType self_scalar_type;
  SavedVariable result_;

};
#ifdef _WIN32
struct ScaledDotProductEfficientAttentionBackward0 : public TraceableFunction {
  TORCH_API ScaledDotProductEfficientAttentionBackward0() = default;
#else
struct TORCH_API ScaledDotProductEfficientAttentionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScaledDotProductEfficientAttentionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    attn_bias_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    log_sumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable attn_bias_;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  SavedVariable query_;
  ::std::optional<double> scale;
  SavedVariable value_;
  SavedVariable log_sumexp_;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;

};
#ifdef _WIN32
struct ScaledDotProductFlashAttentionBackward0 : public TraceableFunction {
  TORCH_API ScaledDotProductFlashAttentionBackward0() = default;
#else
struct TORCH_API ScaledDotProductFlashAttentionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScaledDotProductFlashAttentionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    cum_seq_k_.reset_data();
    cum_seq_q_.reset_data();
    logsumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  SavedVariable query_;
  ::std::optional<double> scale;
  SavedVariable value_;
  SavedVariable cum_seq_k_;
  SavedVariable cum_seq_q_;
  SavedVariable logsumexp_;
  c10::SymInt max_k;
  c10::SymInt max_q;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;

};
#ifdef _WIN32
struct ScaledDotProductFlashAttentionForCpuBackward0 : public TraceableFunction {
  TORCH_API ScaledDotProductFlashAttentionForCpuBackward0() = default;
#else
struct TORCH_API ScaledDotProductFlashAttentionForCpuBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScaledDotProductFlashAttentionForCpuBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    attn_mask_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    logsumexp_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable attn_mask_;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  SavedVariable query_;
  ::std::optional<double> scale;
  SavedVariable value_;
  SavedVariable logsumexp_;
  SavedVariable output_;

};
#ifdef _WIN32
struct FlashAttentionBackward0 : public TraceableFunction {
  TORCH_API FlashAttentionBackward0() = default;
#else
struct TORCH_API FlashAttentionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FlashAttentionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cum_seq_k_.reset_data();
    cum_seq_q_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
    softmax_logsumexp_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable cum_seq_k_;
  SavedVariable cum_seq_q_;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  c10::SymInt max_k;
  c10::SymInt max_q;
  SavedVariable query_;
  ::std::optional<double> scale;
  SavedVariable value_;
  ::std::optional<c10::SymInt> window_size_left;
  ::std::optional<c10::SymInt> window_size_right;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;
  SavedVariable softmax_logsumexp_;

};
#ifdef _WIN32
struct EfficientAttentionBackward0 : public TraceableFunction {
  TORCH_API EfficientAttentionBackward0() = default;
#else
struct TORCH_API EfficientAttentionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EfficientAttentionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    bias_.reset_data();
    cu_seqlens_k_.reset_data();
    cu_seqlens_q_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    logsumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable bias_;
  SavedVariable cu_seqlens_k_;
  SavedVariable cu_seqlens_q_;
  int64_t custom_mask_type = 0;
  double dropout_p;
  SavedVariable key_;
  SavedVariable query_;
  ::std::optional<double> scale;
  SavedVariable value_;
  SavedVariable logsumexp_;
  c10::SymInt max_seqlen_batch_k;
  c10::SymInt max_seqlen_batch_q;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;

};
#ifdef _WIN32
struct ScaledDotProductCudnnAttentionBackward0 : public TraceableFunction {
  TORCH_API ScaledDotProductCudnnAttentionBackward0() = default;
#else
struct TORCH_API ScaledDotProductCudnnAttentionBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScaledDotProductCudnnAttentionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    attn_bias_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    cum_seq_k_.reset_data();
    cum_seq_q_.reset_data();
    logsumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable attn_bias_;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  SavedVariable query_;
  ::std::optional<double> scale;
  SavedVariable value_;
  SavedVariable cum_seq_k_;
  SavedVariable cum_seq_q_;
  SavedVariable logsumexp_;
  c10::SymInt max_k;
  c10::SymInt max_q;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;

};
#ifdef _WIN32
struct ScaledDotProductFusedAttentionOverrideableBackward0 : public TraceableFunction {
  TORCH_API ScaledDotProductFusedAttentionOverrideableBackward0() = default;
#else
struct TORCH_API ScaledDotProductFusedAttentionOverrideableBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScaledDotProductFusedAttentionOverrideableBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    attn_bias_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    cum_seq_k_.reset_data();
    cum_seq_q_.reset_data();
    logsumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable attn_bias_;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  SavedVariable query_;
  ::std::optional<double> scale;
  SavedVariable value_;
  SavedVariable cum_seq_k_;
  SavedVariable cum_seq_q_;
  SavedVariable logsumexp_;
  c10::SymInt max_k;
  c10::SymInt max_q;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;

};
#ifdef _WIN32
struct FftR2CBackward0 : public TraceableFunction {
  TORCH_API FftR2CBackward0() = default;
#else
struct TORCH_API FftR2CBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftR2CBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t normalization = 0;
  bool onesided;
  SavedVariable self_;

};
#ifdef _WIN32
struct FftC2RBackward0 : public TraceableFunction {
  TORCH_API FftC2RBackward0() = default;
#else
struct TORCH_API FftC2RBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftC2RBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t normalization = 0;

};
#ifdef _WIN32
struct FftC2CBackward0 : public TraceableFunction {
  TORCH_API FftC2CBackward0() = default;
#else
struct TORCH_API FftC2CBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftC2CBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dim;
  bool forward;
  int64_t normalization = 0;

};
#ifdef _WIN32
struct UnbindBackward0 : public Node {
  TORCH_API UnbindBackward0() = default;
#else
struct TORCH_API UnbindBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;

};
#ifdef _WIN32
struct UnbindBackwardAutogradNestedTensor0 : public Node {
  TORCH_API UnbindBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API UnbindBackwardAutogradNestedTensor0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackwardAutogradNestedTensor0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::Layout self_layout;
  at::TensorOptions self_options;

};
#ifdef _WIN32
struct StackBackward0 : public TraceableFunction {
  TORCH_API StackBackward0() = default;
#else
struct TORCH_API StackBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StackBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::vector<at::ScalarType> tensors_args_scalartypes;
  size_t tensors_size_;
};
#ifdef _WIN32
struct ThnnFusedLstmCellBackward0 : public TraceableFunction {
  TORCH_API ThnnFusedLstmCellBackward0() = default;
#else
struct TORCH_API ThnnFusedLstmCellBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedLstmCellBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx_.reset_data();
    hidden_bias_.reset_data();
    hidden_gates_.reset_data();
    input_bias_.reset_data();
    input_gates_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable cx_;
  SavedVariable hidden_bias_;
  SavedVariable hidden_gates_;
  SavedVariable input_bias_;
  SavedVariable input_gates_;
  SavedVariable result1_;
  SavedVariable result2_;

};
#ifdef _WIN32
struct ThnnFusedGruCellBackward0 : public TraceableFunction {
  TORCH_API ThnnFusedGruCellBackward0() = default;
#else
struct TORCH_API ThnnFusedGruCellBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedGruCellBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    hidden_bias_.reset_data();
    hidden_gates_.reset_data();
    hx_.reset_data();
    input_bias_.reset_data();
    input_gates_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable hidden_bias_;
  SavedVariable hidden_gates_;
  SavedVariable hx_;
  SavedVariable input_bias_;
  SavedVariable input_gates_;
  SavedVariable result1_;

};
#ifdef _WIN32
struct PackPaddedSequenceBackward0 : public TraceableFunction {
  TORCH_API PackPaddedSequenceBackward0() = default;
#else
struct TORCH_API PackPaddedSequenceBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PackPaddedSequenceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<c10::SymInt> input_sym_sizes;
  SavedVariable result1_;

};
#ifdef _WIN32
struct SegmentReduceBackward0 : public TraceableFunction {
  TORCH_API SegmentReduceBackward0() = default;
#else
struct TORCH_API SegmentReduceBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SegmentReduceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.reset_data();
    lengths_.reset_data();
    offsets_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t axis = 0;
  SavedVariable data_;
  ::std::optional<at::Scalar> initial;
  SavedVariable lengths_;
  SavedVariable offsets_;
  std::string reduce;
  SavedVariable result_;

};
#ifdef _WIN32
struct PinMemoryBackward0 : public TraceableFunction {
  TORCH_API PinMemoryBackward0() = default;
#else
struct TORCH_API PinMemoryBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PinMemoryBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct TestWarnInAutogradBackward0 : public TraceableFunction {
  TORCH_API TestWarnInAutogradBackward0() = default;
#else
struct TORCH_API TestWarnInAutogradBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestWarnInAutogradBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct TestAutogradMultipleDispatchBackward0 : public TraceableFunction {
  TORCH_API TestAutogradMultipleDispatchBackward0() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct TestAutogradMultipleDispatchBackwardAutogradNestedTensor0 : public TraceableFunction {
  TORCH_API TestAutogradMultipleDispatchBackwardAutogradNestedTensor0() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchBackwardAutogradNestedTensor0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchBackwardAutogradNestedTensor0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct TestAutogradMultipleDispatchBackwardAutogradCUDA0 : public TraceableFunction {
  TORCH_API TestAutogradMultipleDispatchBackwardAutogradCUDA0() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchBackwardAutogradCUDA0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchBackwardAutogradCUDA0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct TestAutogradMultipleDispatchBackwardAutogradNestedTensor1 : public TraceableFunction {
  TORCH_API TestAutogradMultipleDispatchBackwardAutogradNestedTensor1() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchBackwardAutogradNestedTensor1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchBackwardAutogradNestedTensor1"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct TestAutogradMultipleDispatchViewBackward0 : public Node {
  TORCH_API TestAutogradMultipleDispatchViewBackward0() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchViewBackward0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchViewBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct TestAutogradMultipleDispatchViewBackwardAutogradCUDA0 : public Node {
  TORCH_API TestAutogradMultipleDispatchViewBackwardAutogradCUDA0() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchViewBackwardAutogradCUDA0 : public Node {
#endif
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchViewBackwardAutogradCUDA0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ScatterReduceBackward0 : public TraceableFunction {
  TORCH_API ScatterReduceBackward0() = default;
#else
struct TORCH_API ScatterReduceBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterReduceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
    src_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool include_self;
  SavedVariable index_;
  std::string reduce;
  SavedVariable self_;
  SavedVariable src_;
  SavedVariable result_;

};
#ifdef _WIN32
struct ReshapeCopyBackward0 : public TraceableFunction {
  TORCH_API ReshapeCopyBackward0() = default;
#else
struct TORCH_API ReshapeCopyBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReshapeCopyBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct ForeachDivBackward0 : public TraceableFunction {
  TORCH_API ForeachDivBackward0() = default;
#else
struct TORCH_API ForeachDivBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachDivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachPowBackward0 : public TraceableFunction {
  TORCH_API ForeachPowBackward0() = default;
#else
struct TORCH_API ForeachPowBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachPowBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.clear();
    exponent_released_ = true;
    self_.clear();
    self_released_ = true;
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> exponent_;
  bool exponent_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
  size_t exponent_size_;
};
#ifdef _WIN32
struct ForeachPowBackward1 : public TraceableFunction {
  TORCH_API ForeachPowBackward1() = default;
#else
struct TORCH_API ForeachPowBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachPowBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> exponent;
  bool exponent_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachPowBackward2 : public TraceableFunction {
  TORCH_API ForeachPowBackward2() = default;
#else
struct TORCH_API ForeachPowBackward2 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachPowBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.clear();
    exponent_released_ = true;
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> exponent_;
  bool exponent_released_ = false;
  at::Scalar self;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t exponent_size_;
};
#ifdef _WIN32
struct ForeachMinimumBackward0 : public TraceableFunction {
  TORCH_API ForeachMinimumBackward0() = default;
#else
struct TORCH_API ForeachMinimumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMinimumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachMinimumBackward1 : public TraceableFunction {
  TORCH_API ForeachMinimumBackward1() = default;
#else
struct TORCH_API ForeachMinimumBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMinimumBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachMaximumBackward0 : public TraceableFunction {
  TORCH_API ForeachMaximumBackward0() = default;
#else
struct TORCH_API ForeachMaximumBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMaximumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachMaximumBackward1 : public TraceableFunction {
  TORCH_API ForeachMaximumBackward1() = default;
#else
struct TORCH_API ForeachMaximumBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMaximumBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachNormBackward0 : public TraceableFunction {
  TORCH_API ForeachNormBackward0() = default;
#else
struct TORCH_API ForeachNormBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar ord;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct AliasBackward0_copy : public TraceableFunction {
  TORCH_API AliasBackward0_copy() = default;
#else
struct TORCH_API AliasBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AliasBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct AsStridedBackward0_copy : public TraceableFunction {
  TORCH_API AsStridedBackward0_copy() = default;
#else
struct TORCH_API AsStridedBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  ::std::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;

};
#ifdef _WIN32
struct ConjBackward0_copy : public TraceableFunction {
  TORCH_API ConjBackward0_copy() = default;
#else
struct TORCH_API ConjBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NegViewBackward0_copy : public TraceableFunction {
  TORCH_API NegViewBackward0_copy() = default;
#else
struct TORCH_API NegViewBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegViewBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct DiagonalBackward0_copy : public TraceableFunction {
  TORCH_API DiagonalBackward0_copy() = default;
#else
struct TORCH_API DiagonalBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct ExpandBackward0_copy : public TraceableFunction {
  TORCH_API ExpandBackward0_copy() = default;
#else
struct TORCH_API ExpandBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpandBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct PermuteBackward0_copy : public TraceableFunction {
  TORCH_API PermuteBackward0_copy() = default;
#else
struct TORCH_API PermuteBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PermuteBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;

};
#ifdef _WIN32
struct ReshapeAliasBackward0_copy : public TraceableFunction {
  TORCH_API ReshapeAliasBackward0_copy() = default;
#else
struct TORCH_API ReshapeAliasBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReshapeAliasBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SelectBackward0_copy : public TraceableFunction {
  TORCH_API SelectBackward0_copy() = default;
#else
struct TORCH_API SelectBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SelectBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  TORCH_API SelectBackwardAutogradNestedTensor0_copy() = default;
#else
struct TORCH_API SelectBackwardAutogradNestedTensor0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackwardAutogradNestedTensor0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  SavedVariable self_;

};
#ifdef _WIN32
struct SliceBackward0_copy : public TraceableFunction {
  TORCH_API SliceBackward0_copy() = default;
#else
struct TORCH_API SliceBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::optional<c10::SymInt> end;
  std::vector<c10::SymInt> self_sym_sizes;
  ::std::optional<c10::SymInt> start;
  c10::SymInt step;

};
#ifdef _WIN32
struct SplitBackward0_copy : public TraceableFunction {
  TORCH_API SplitBackward0_copy() = default;
#else
struct TORCH_API SplitBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::SymInt split_size;

};
#ifdef _WIN32
struct SplitWithSizesBackward0_copy : public TraceableFunction {
  TORCH_API SplitWithSizesBackward0_copy() = default;
#else
struct TORCH_API SplitWithSizesBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitWithSizesBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> split_sizes;

};
#ifdef _WIN32
struct SplitWithSizesBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  TORCH_API SplitWithSizesBackwardAutogradNestedTensor0_copy() = default;
#else
struct TORCH_API SplitWithSizesBackwardAutogradNestedTensor0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitWithSizesBackwardAutogradNestedTensor0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> split_sizes;

};
#ifdef _WIN32
struct SqueezeBackward0_copy : public TraceableFunction {
  TORCH_API SqueezeBackward0_copy() = default;
#else
struct TORCH_API SqueezeBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackward1_copy : public TraceableFunction {
  TORCH_API SqueezeBackward1_copy() = default;
#else
struct TORCH_API SqueezeBackward1_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward1_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  TORCH_API SqueezeBackwardAutogradNestedTensor0_copy() = default;
#else
struct TORCH_API SqueezeBackwardAutogradNestedTensor0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackwardAutogradNestedTensor0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;

};
#ifdef _WIN32
struct SqueezeBackward2_copy : public TraceableFunction {
  TORCH_API SqueezeBackward2_copy() = default;
#else
struct TORCH_API SqueezeBackward2_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward2_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct SqueezeBackwardAutogradNestedTensor1_copy : public TraceableFunction {
  TORCH_API SqueezeBackwardAutogradNestedTensor1_copy() = default;
#else
struct TORCH_API SqueezeBackwardAutogradNestedTensor1_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackwardAutogradNestedTensor1_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t self_dim = 0;

};
#ifdef _WIN32
struct TBackward0_copy : public TraceableFunction {
  TORCH_API TBackward0_copy() = default;
#else
struct TORCH_API TBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct TransposeBackward0_copy : public TraceableFunction {
  TORCH_API TransposeBackward0_copy() = default;
#else
struct TORCH_API TransposeBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
#ifdef _WIN32
struct UnfoldBackward0_copy : public TraceableFunction {
  TORCH_API UnfoldBackward0_copy() = default;
#else
struct TORCH_API UnfoldBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dimension = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  int64_t size = 0;
  int64_t step = 0;

};
#ifdef _WIN32
struct LiftFreshBackward0_copy : public TraceableFunction {
  TORCH_API LiftFreshBackward0_copy() = default;
#else
struct TORCH_API LiftFreshBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LiftFreshBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct UnsqueezeBackward0_copy : public TraceableFunction {
  TORCH_API UnsqueezeBackward0_copy() = default;
#else
struct TORCH_API UnsqueezeBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;

};
#ifdef _WIN32
struct ViewBackward0_copy : public TraceableFunction {
  TORCH_API ViewBackward0_copy() = default;
#else
struct TORCH_API ViewBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;

};
#ifdef _WIN32
struct ViewBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  TORCH_API ViewBackwardAutogradNestedTensor0_copy() = default;
#else
struct TORCH_API ViewBackwardAutogradNestedTensor0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackwardAutogradNestedTensor0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ViewAsRealBackward0_copy : public TraceableFunction {
  TORCH_API ViewAsRealBackward0_copy() = default;
#else
struct TORCH_API ViewAsRealBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsRealBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ViewAsComplexBackward0_copy : public TraceableFunction {
  TORCH_API ViewAsComplexBackward0_copy() = default;
#else
struct TORCH_API ViewAsComplexBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsComplexBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct ValuesBackward0_copy : public TraceableFunction {
  TORCH_API ValuesBackward0_copy() = default;
#else
struct TORCH_API ValuesBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackward0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ValuesBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  TORCH_API ValuesBackwardAutogradNestedTensor0_copy() = default;
#else
struct TORCH_API ValuesBackwardAutogradNestedTensor0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackwardAutogradNestedTensor0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct NestedViewFromBufferBackward0_copy : public TraceableFunction {
  TORCH_API NestedViewFromBufferBackward0_copy() = default;
#else
struct TORCH_API NestedViewFromBufferBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedViewFromBufferBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NestedViewFromJaggedBackward0_copy : public TraceableFunction {
  TORCH_API NestedViewFromJaggedBackward0_copy() = default;
#else
struct TORCH_API NestedViewFromJaggedBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedViewFromJaggedBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;


};
#ifdef _WIN32
struct NestedGetValuesBackward0_copy : public TraceableFunction {
  TORCH_API NestedGetValuesBackward0_copy() = default;
#else
struct TORCH_API NestedGetValuesBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NestedGetValuesBackward0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct UnbindBackward0_copy : public TraceableFunction {
  TORCH_API UnbindBackward0_copy() = default;
#else
struct TORCH_API UnbindBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackward0_copy"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;

};
#ifdef _WIN32
struct UnbindBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  TORCH_API UnbindBackwardAutogradNestedTensor0_copy() = default;
#else
struct TORCH_API UnbindBackwardAutogradNestedTensor0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackwardAutogradNestedTensor0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::Layout self_layout;
  at::TensorOptions self_options;

};
#ifdef _WIN32
struct TestAutogradMultipleDispatchViewBackward0_copy : public TraceableFunction {
  TORCH_API TestAutogradMultipleDispatchViewBackward0_copy() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchViewBackward0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchViewBackward0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy : public TraceableFunction {
  TORCH_API TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy() = default;
#else
struct TORCH_API TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable self_;

};
#ifdef _WIN32
struct ForeachAbsBackward0 : public TraceableFunction {
  TORCH_API ForeachAbsBackward0() = default;
#else
struct TORCH_API ForeachAbsBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAbsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachAcosBackward0 : public TraceableFunction {
  TORCH_API ForeachAcosBackward0() = default;
#else
struct TORCH_API ForeachAcosBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAcosBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachAddBackward1Scalar : public TraceableFunction {
  TORCH_API ForeachAddBackward1Scalar() = default;
#else
struct TORCH_API ForeachAddBackward1Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddBackward1Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachAddBackward0List : public TraceableFunction {
  TORCH_API ForeachAddBackward0List() = default;
#else
struct TORCH_API ForeachAddBackward0List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddBackward0List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachAddBackward1ScalarList : public TraceableFunction {
  TORCH_API ForeachAddBackward1ScalarList() = default;
#else
struct TORCH_API ForeachAddBackward1ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddBackward1ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachAddBackward0Tensor : public TraceableFunction {
  TORCH_API ForeachAddBackward0Tensor() = default;
#else
struct TORCH_API ForeachAddBackward0Tensor : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddBackward0Tensor"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable other_;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachAddcdivBackward0Scalar : public TraceableFunction {
  TORCH_API ForeachAddcdivBackward0Scalar() = default;
#else
struct TORCH_API ForeachAddcdivBackward0Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddcdivBackward0Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  at::Scalar value;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
#ifdef _WIN32
struct ForeachAddcdivBackward0ScalarList : public TraceableFunction {
  TORCH_API ForeachAddcdivBackward0ScalarList() = default;
#else
struct TORCH_API ForeachAddcdivBackward0ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddcdivBackward0ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
#ifdef _WIN32
struct ForeachAddcmulBackward0Scalar : public TraceableFunction {
  TORCH_API ForeachAddcmulBackward0Scalar() = default;
#else
struct TORCH_API ForeachAddcmulBackward0Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddcmulBackward0Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  at::Scalar value;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
#ifdef _WIN32
struct ForeachAddcmulBackward0ScalarList : public TraceableFunction {
  TORCH_API ForeachAddcmulBackward0ScalarList() = default;
#else
struct TORCH_API ForeachAddcmulBackward0ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAddcmulBackward0ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
#ifdef _WIN32
struct ForeachAsinBackward0 : public TraceableFunction {
  TORCH_API ForeachAsinBackward0() = default;
#else
struct TORCH_API ForeachAsinBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAsinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachAtanBackward0 : public TraceableFunction {
  TORCH_API ForeachAtanBackward0() = default;
#else
struct TORCH_API ForeachAtanBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachAtanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachCeilBackward0 : public TraceableFunction {
  TORCH_API ForeachCeilBackward0() = default;
#else
struct TORCH_API ForeachCeilBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachCeilBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t self_size_;
};
#ifdef _WIN32
struct ForeachClampMaxBackward0Scalar : public TraceableFunction {
  TORCH_API ForeachClampMaxBackward0Scalar() = default;
#else
struct TORCH_API ForeachClampMaxBackward0Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachClampMaxBackward0Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachClampMaxBackward1List : public TraceableFunction {
  TORCH_API ForeachClampMaxBackward1List() = default;
#else
struct TORCH_API ForeachClampMaxBackward1List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachClampMaxBackward1List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachClampMaxBackward0ScalarList : public TraceableFunction {
  TORCH_API ForeachClampMaxBackward0ScalarList() = default;
#else
struct TORCH_API ForeachClampMaxBackward0ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachClampMaxBackward0ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachClampMinBackward0Scalar : public TraceableFunction {
  TORCH_API ForeachClampMinBackward0Scalar() = default;
#else
struct TORCH_API ForeachClampMinBackward0Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachClampMinBackward0Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachClampMinBackward1List : public TraceableFunction {
  TORCH_API ForeachClampMinBackward1List() = default;
#else
struct TORCH_API ForeachClampMinBackward1List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachClampMinBackward1List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachClampMinBackward0ScalarList : public TraceableFunction {
  TORCH_API ForeachClampMinBackward0ScalarList() = default;
#else
struct TORCH_API ForeachClampMinBackward0ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachClampMinBackward0ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachCosBackward0 : public TraceableFunction {
  TORCH_API ForeachCosBackward0() = default;
#else
struct TORCH_API ForeachCosBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachCosBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachCoshBackward0 : public TraceableFunction {
  TORCH_API ForeachCoshBackward0() = default;
#else
struct TORCH_API ForeachCoshBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachCoshBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachDivBackward1Scalar : public TraceableFunction {
  TORCH_API ForeachDivBackward1Scalar() = default;
#else
struct TORCH_API ForeachDivBackward1Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachDivBackward1Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachDivBackward1ScalarList : public TraceableFunction {
  TORCH_API ForeachDivBackward1ScalarList() = default;
#else
struct TORCH_API ForeachDivBackward1ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachDivBackward1ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachDivBackward0Tensor : public TraceableFunction {
  TORCH_API ForeachDivBackward0Tensor() = default;
#else
struct TORCH_API ForeachDivBackward0Tensor : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachDivBackward0Tensor"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachErfBackward0 : public TraceableFunction {
  TORCH_API ForeachErfBackward0() = default;
#else
struct TORCH_API ForeachErfBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachErfBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachErfcBackward0 : public TraceableFunction {
  TORCH_API ForeachErfcBackward0() = default;
#else
struct TORCH_API ForeachErfcBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachErfcBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachExpBackward0 : public TraceableFunction {
  TORCH_API ForeachExpBackward0() = default;
#else
struct TORCH_API ForeachExpBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachExpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachExpm1Backward0 : public TraceableFunction {
  TORCH_API ForeachExpm1Backward0() = default;
#else
struct TORCH_API ForeachExpm1Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachExpm1Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachFloorBackward0 : public TraceableFunction {
  TORCH_API ForeachFloorBackward0() = default;
#else
struct TORCH_API ForeachFloorBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachFloorBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t self_size_;
};
#ifdef _WIN32
struct ForeachFracBackward0 : public TraceableFunction {
  TORCH_API ForeachFracBackward0() = default;
#else
struct TORCH_API ForeachFracBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachFracBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t self_size_;
};
#ifdef _WIN32
struct ForeachLerpBackward1List : public TraceableFunction {
  TORCH_API ForeachLerpBackward1List() = default;
#else
struct TORCH_API ForeachLerpBackward1List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLerpBackward1List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    tensors1_.clear();
    tensors1_released_ = true;
    weights_.clear();
    weights_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensors1_;
  bool tensors1_released_ = false;
  std::vector<SavedVariable> weights_;
  bool weights_released_ = false;
  size_t self_size_;
  size_t tensors1_size_;
  size_t weights_size_;
};
#ifdef _WIN32
struct ForeachLerpBackward0Scalar : public TraceableFunction {
  TORCH_API ForeachLerpBackward0Scalar() = default;
#else
struct TORCH_API ForeachLerpBackward0Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLerpBackward0Scalar"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar weight;
  size_t self_size_;
  size_t tensors1_size_;
};
#ifdef _WIN32
struct ForeachLerpBackward0ScalarList : public TraceableFunction {
  TORCH_API ForeachLerpBackward0ScalarList() = default;
#else
struct TORCH_API ForeachLerpBackward0ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLerpBackward0ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    weight.clear();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> weight;
  bool weight_released_ = false;
  size_t self_size_;
  size_t tensors1_size_;
};
#ifdef _WIN32
struct ForeachLgammaBackward0 : public TraceableFunction {
  TORCH_API ForeachLgammaBackward0() = default;
#else
struct TORCH_API ForeachLgammaBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLgammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachLogBackward0 : public TraceableFunction {
  TORCH_API ForeachLogBackward0() = default;
#else
struct TORCH_API ForeachLogBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLogBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachLog10Backward0 : public TraceableFunction {
  TORCH_API ForeachLog10Backward0() = default;
#else
struct TORCH_API ForeachLog10Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLog10Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachLog1PBackward0 : public TraceableFunction {
  TORCH_API ForeachLog1PBackward0() = default;
#else
struct TORCH_API ForeachLog1PBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLog1PBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachLog2Backward0 : public TraceableFunction {
  TORCH_API ForeachLog2Backward0() = default;
#else
struct TORCH_API ForeachLog2Backward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachLog2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachMaxBackward1 : public TraceableFunction {
  TORCH_API ForeachMaxBackward1() = default;
#else
struct TORCH_API ForeachMaxBackward1 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMaxBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachMaximumBackward0List : public TraceableFunction {
  TORCH_API ForeachMaximumBackward0List() = default;
#else
struct TORCH_API ForeachMaximumBackward0List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMaximumBackward0List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachMinimumBackward0List : public TraceableFunction {
  TORCH_API ForeachMinimumBackward0List() = default;
#else
struct TORCH_API ForeachMinimumBackward0List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMinimumBackward0List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachMulBackward1Scalar : public TraceableFunction {
  TORCH_API ForeachMulBackward1Scalar() = default;
#else
struct TORCH_API ForeachMulBackward1Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMulBackward1Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachMulBackward0List : public TraceableFunction {
  TORCH_API ForeachMulBackward0List() = default;
#else
struct TORCH_API ForeachMulBackward0List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMulBackward0List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachMulBackward1ScalarList : public TraceableFunction {
  TORCH_API ForeachMulBackward1ScalarList() = default;
#else
struct TORCH_API ForeachMulBackward1ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMulBackward1ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachMulBackward0Tensor : public TraceableFunction {
  TORCH_API ForeachMulBackward0Tensor() = default;
#else
struct TORCH_API ForeachMulBackward0Tensor : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachMulBackward0Tensor"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  SavedVariable other_;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachNegBackward0 : public TraceableFunction {
  TORCH_API ForeachNegBackward0() = default;
#else
struct TORCH_API ForeachNegBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachNegBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t self_size_;
};
#ifdef _WIN32
struct ForeachPowBackward0Scalar : public TraceableFunction {
  TORCH_API ForeachPowBackward0Scalar() = default;
#else
struct TORCH_API ForeachPowBackward0Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachPowBackward0Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar exponent;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachReciprocalBackward0 : public TraceableFunction {
  TORCH_API ForeachReciprocalBackward0() = default;
#else
struct TORCH_API ForeachReciprocalBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachReciprocalBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachRoundBackward0 : public TraceableFunction {
  TORCH_API ForeachRoundBackward0() = default;
#else
struct TORCH_API ForeachRoundBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachRoundBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t self_size_;
};
#ifdef _WIN32
struct ForeachRsqrtBackward0 : public TraceableFunction {
  TORCH_API ForeachRsqrtBackward0() = default;
#else
struct TORCH_API ForeachRsqrtBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachRsqrtBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachSigmoidBackward0 : public TraceableFunction {
  TORCH_API ForeachSigmoidBackward0() = default;
#else
struct TORCH_API ForeachSigmoidBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSigmoidBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachSignBackward0 : public TraceableFunction {
  TORCH_API ForeachSignBackward0() = default;
#else
struct TORCH_API ForeachSignBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSignBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t self_size_;
};
#ifdef _WIN32
struct ForeachSinBackward0 : public TraceableFunction {
  TORCH_API ForeachSinBackward0() = default;
#else
struct TORCH_API ForeachSinBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachSinhBackward0 : public TraceableFunction {
  TORCH_API ForeachSinhBackward0() = default;
#else
struct TORCH_API ForeachSinhBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSinhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachSqrtBackward0 : public TraceableFunction {
  TORCH_API ForeachSqrtBackward0() = default;
#else
struct TORCH_API ForeachSqrtBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSqrtBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachSubBackward1Scalar : public TraceableFunction {
  TORCH_API ForeachSubBackward1Scalar() = default;
#else
struct TORCH_API ForeachSubBackward1Scalar : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSubBackward1Scalar"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachSubBackward0List : public TraceableFunction {
  TORCH_API ForeachSubBackward0List() = default;
#else
struct TORCH_API ForeachSubBackward0List : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSubBackward0List"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  at::Scalar alpha;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
#ifdef _WIN32
struct ForeachSubBackward1ScalarList : public TraceableFunction {
  TORCH_API ForeachSubBackward1ScalarList() = default;
#else
struct TORCH_API ForeachSubBackward1ScalarList : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachSubBackward1ScalarList"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachTanBackward0 : public TraceableFunction {
  TORCH_API ForeachTanBackward0() = default;
#else
struct TORCH_API ForeachTanBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachTanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachTanhBackward0 : public TraceableFunction {
  TORCH_API ForeachTanhBackward0() = default;
#else
struct TORCH_API ForeachTanhBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachTanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
#ifdef _WIN32
struct ForeachTruncBackward0 : public TraceableFunction {
  TORCH_API ForeachTruncBackward0() = default;
#else
struct TORCH_API ForeachTruncBackward0 : public TraceableFunction {
#endif
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ForeachTruncBackward0"; }
  void release_variables() override {


  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;

  size_t self_size_;
};

}}} // namespace torch::autograd::generated
