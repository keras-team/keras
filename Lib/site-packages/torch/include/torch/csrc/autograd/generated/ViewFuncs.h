#pragma once

// @generated from ..\tools\autograd\templates/ViewFuncs.h

#include <torch/library.h>
#include <torch/csrc/autograd/variable.h>
#include <c10/core/SymIntArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/_conj_ops.h>
#include <ATen/ops/_indices_ops.h>
#include <ATen/ops/_neg_view_ops.h>
#include <ATen/ops/_nested_get_values_ops.h>
#include <ATen/ops/_nested_view_from_buffer_ops.h>
#include <ATen/ops/_nested_view_from_jagged_ops.h>
#include <ATen/ops/_reshape_alias_ops.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_view_ops.h>
#include <ATen/ops/_values_ops.h>
#include <ATen/ops/alias_ops.h>
#include <ATen/ops/as_strided_ops.h>
#include <ATen/ops/ccol_indices_ops.h>
#include <ATen/ops/chunk_ops.h>
#include <ATen/ops/col_indices_ops.h>
#include <ATen/ops/crow_indices_ops.h>
#include <ATen/ops/diagonal_ops.h>
#include <ATen/ops/expand_ops.h>
#include <ATen/ops/indices_ops.h>
#include <ATen/ops/narrow_ops.h>
#include <ATen/ops/permute_ops.h>
#include <ATen/ops/row_indices_ops.h>
#include <ATen/ops/select_ops.h>
#include <ATen/ops/slice_ops.h>
#include <ATen/ops/slice_inverse_ops.h>
#include <ATen/ops/split_ops.h>
#include <ATen/ops/split_with_sizes_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/t_ops.h>
#include <ATen/ops/transpose_ops.h>
#include <ATen/ops/unbind_ops.h>
#include <ATen/ops/unfold_ops.h>
#include <ATen/ops/unsqueeze_ops.h>
#include <ATen/ops/values_ops.h>
#include <ATen/ops/view_ops.h>
#include <ATen/ops/view_ops.h>
#include <ATen/ops/view_as_complex_ops.h>
#include <ATen/ops/view_as_real_ops.h>
#endif

namespace torch::autograd::generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::ScalarType;
using std::optional;
using c10::fmap;

#define _CONJ_VIEW_FUNC_AVAILABLE
struct _ConjViewFunc : public torch::autograd::ViewFunc {
  _ConjViewFunc() 
  {}
  virtual ~_ConjViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define _INDICES_VIEW_FUNC_AVAILABLE
struct _IndicesViewFunc : public torch::autograd::ViewFunc {
  _IndicesViewFunc() 
  {}
  virtual ~_IndicesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define _NEG_VIEW_VIEW_FUNC_AVAILABLE
struct _NegViewViewFunc : public torch::autograd::ViewFunc {
  _NegViewViewFunc() 
  {}
  virtual ~_NegViewViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define _NESTED_GET_VALUES_VIEW_FUNC_AVAILABLE
struct _NestedGetValuesViewFunc : public torch::autograd::ViewFunc {
  _NestedGetValuesViewFunc() 
  {}
  virtual ~_NestedGetValuesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define _NESTED_VIEW_FROM_BUFFER_VIEW_FUNC_AVAILABLE
struct _NestedViewFromBufferViewFunc : public torch::autograd::ViewFunc {
  _NestedViewFromBufferViewFunc(const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets) : nested_size(nested_size), nested_strides(nested_strides), offsets(offsets)
  {}
  virtual ~_NestedViewFromBufferViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  at::Tensor nested_size;
  at::Tensor nested_strides;
  at::Tensor offsets;
};

#define _NESTED_VIEW_FROM_JAGGED_VIEW_FUNC_AVAILABLE
struct _NestedViewFromJaggedViewFunc : public torch::autograd::ViewFunc {
  _NestedViewFromJaggedViewFunc(const at::Tensor & offsets, const at::Tensor & dummy, const ::std::optional<at::Tensor> & lengths, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen) : offsets(offsets), dummy(dummy), lengths(lengths), ragged_idx(ragged_idx), min_seqlen(min_seqlen), max_seqlen(max_seqlen)
  {}
  virtual ~_NestedViewFromJaggedViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  at::Tensor offsets;
  at::Tensor dummy;
  ::std::optional<at::Tensor> lengths;
  int64_t ragged_idx;
  ::std::optional<at::Tensor> min_seqlen;
  ::std::optional<at::Tensor> max_seqlen;
};

#define _RESHAPE_ALIAS_VIEW_FUNC_AVAILABLE
struct _ReshapeAliasViewFunc : public torch::autograd::ViewFunc {
  _ReshapeAliasViewFunc(c10::SymIntArrayRef size, c10::SymIntArrayRef stride) : size(size.vec()), stride(stride.vec())
  {}
  virtual ~_ReshapeAliasViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ::std::vector<c10::SymInt> size;
  ::std::vector<c10::SymInt> stride;
};

#define _TEST_AUTOGRAD_MULTIPLE_DISPATCH_VIEW_VIEW_FUNC_AVAILABLE
struct _TestAutogradMultipleDispatchViewViewFunc : public torch::autograd::ViewFunc {
  _TestAutogradMultipleDispatchViewViewFunc() 
  {}
  virtual ~_TestAutogradMultipleDispatchViewViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define _VALUES_VIEW_FUNC_AVAILABLE
struct _ValuesViewFunc : public torch::autograd::ViewFunc {
  _ValuesViewFunc() 
  {}
  virtual ~_ValuesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define ALIAS_VIEW_FUNC_AVAILABLE
struct AliasViewFunc : public torch::autograd::ViewFunc {
  AliasViewFunc() 
  {}
  virtual ~AliasViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define AS_STRIDED_VIEW_FUNC_AVAILABLE
struct AsStridedViewFunc : public torch::autograd::ViewFunc {
  AsStridedViewFunc(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) : size(size.vec()), stride(stride.vec()), storage_offset(storage_offset)
  {}
  virtual ~AsStridedViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ::std::vector<c10::SymInt> size;
  ::std::vector<c10::SymInt> stride;
  ::std::optional<c10::SymInt> storage_offset;
};

#define CCOL_INDICES_VIEW_FUNC_AVAILABLE
struct CcolIndicesViewFunc : public torch::autograd::ViewFunc {
  CcolIndicesViewFunc() 
  {}
  virtual ~CcolIndicesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define CHUNK_VIEW_FUNC_AVAILABLE
struct ChunkViewFunc : public torch::autograd::ViewFunc {
  ChunkViewFunc(int64_t chunks, int64_t dim, int64_t view_idx) : chunks(chunks), dim(dim), view_idx(view_idx)
  {}
  virtual ~ChunkViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t chunks;
  int64_t dim;
  int64_t view_idx;
};

#define COL_INDICES_VIEW_FUNC_AVAILABLE
struct ColIndicesViewFunc : public torch::autograd::ViewFunc {
  ColIndicesViewFunc() 
  {}
  virtual ~ColIndicesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define CROW_INDICES_VIEW_FUNC_AVAILABLE
struct CrowIndicesViewFunc : public torch::autograd::ViewFunc {
  CrowIndicesViewFunc() 
  {}
  virtual ~CrowIndicesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define DIAGONAL_VIEW_FUNC_AVAILABLE
struct DiagonalViewFunc : public torch::autograd::ViewFunc {
  DiagonalViewFunc(int64_t offset, int64_t dim1, int64_t dim2) : offset(offset), dim1(dim1), dim2(dim2)
  {}
  virtual ~DiagonalViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t offset;
  int64_t dim1;
  int64_t dim2;
};

#define EXPAND_VIEW_FUNC_AVAILABLE
struct ExpandViewFunc : public torch::autograd::ViewFunc {
  ExpandViewFunc(c10::SymIntArrayRef size, bool implicit) : size(size.vec()), implicit(implicit)
  {}
  virtual ~ExpandViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ::std::vector<c10::SymInt> size;
  bool implicit;
};

#define INDICES_VIEW_FUNC_AVAILABLE
struct IndicesViewFunc : public torch::autograd::ViewFunc {
  IndicesViewFunc() 
  {}
  virtual ~IndicesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define NARROW_VIEW_FUNC_AVAILABLE
struct NarrowViewFunc : public torch::autograd::ViewFunc {
  NarrowViewFunc(int64_t dim, c10::SymInt start, c10::SymInt length) : dim(dim), start(start), length(length)
  {}
  virtual ~NarrowViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dim;
  c10::SymInt start;
  c10::SymInt length;
};

#define PERMUTE_VIEW_FUNC_AVAILABLE
struct PermuteViewFunc : public torch::autograd::ViewFunc {
  PermuteViewFunc(at::IntArrayRef dims) : dims(dims.vec())
  {}
  virtual ~PermuteViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ::std::vector<int64_t> dims;
};

#define ROW_INDICES_VIEW_FUNC_AVAILABLE
struct RowIndicesViewFunc : public torch::autograd::ViewFunc {
  RowIndicesViewFunc() 
  {}
  virtual ~RowIndicesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define SELECT_INT_VIEW_FUNC_AVAILABLE
struct SelectIntViewFunc : public torch::autograd::ViewFunc {
  SelectIntViewFunc(int64_t dim, c10::SymInt index) : dim(dim), index(index)
  {}
  virtual ~SelectIntViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dim;
  c10::SymInt index;
};

#define SLICE_TENSOR_VIEW_FUNC_AVAILABLE
struct SliceTensorViewFunc : public torch::autograd::ViewFunc {
  SliceTensorViewFunc(int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) : dim(dim), start(start), end(end), step(step)
  {}
  virtual ~SliceTensorViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dim;
  ::std::optional<c10::SymInt> start;
  ::std::optional<c10::SymInt> end;
  c10::SymInt step;
};

#define SLICE_INVERSE_VIEW_FUNC_AVAILABLE
struct SliceInverseViewFunc : public torch::autograd::ViewFunc {
  SliceInverseViewFunc(const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) : src(src), dim(dim), start(start), end(end), step(step)
  {}
  virtual ~SliceInverseViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  at::Tensor src;
  int64_t dim;
  ::std::optional<c10::SymInt> start;
  ::std::optional<c10::SymInt> end;
  c10::SymInt step;
};

#define SPLIT_TENSOR_VIEW_FUNC_AVAILABLE
struct SplitTensorViewFunc : public torch::autograd::ViewFunc {
  SplitTensorViewFunc(c10::SymInt split_size, int64_t dim, int64_t view_idx) : split_size(split_size), dim(dim), view_idx(view_idx)
  {}
  virtual ~SplitTensorViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  c10::SymInt split_size;
  int64_t dim;
  int64_t view_idx;
};

#define SPLIT_WITH_SIZES_VIEW_FUNC_AVAILABLE
struct SplitWithSizesViewFunc : public torch::autograd::ViewFunc {
  SplitWithSizesViewFunc(c10::SymIntArrayRef split_sizes, int64_t dim, int64_t view_idx) : split_sizes(split_sizes.vec()), dim(dim), view_idx(view_idx)
  {}
  virtual ~SplitWithSizesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ::std::vector<c10::SymInt> split_sizes;
  int64_t dim;
  int64_t view_idx;
};

#define SQUEEZE_VIEW_FUNC_AVAILABLE
struct SqueezeViewFunc : public torch::autograd::ViewFunc {
  SqueezeViewFunc() 
  {}
  virtual ~SqueezeViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define SQUEEZE_DIM_VIEW_FUNC_AVAILABLE
struct SqueezeDimViewFunc : public torch::autograd::ViewFunc {
  SqueezeDimViewFunc(int64_t dim) : dim(dim)
  {}
  virtual ~SqueezeDimViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dim;
};

#define SQUEEZE_DIMS_VIEW_FUNC_AVAILABLE
struct SqueezeDimsViewFunc : public torch::autograd::ViewFunc {
  SqueezeDimsViewFunc(at::IntArrayRef dim) : dim(dim.vec())
  {}
  virtual ~SqueezeDimsViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ::std::vector<int64_t> dim;
};

#define T_VIEW_FUNC_AVAILABLE
struct TViewFunc : public torch::autograd::ViewFunc {
  TViewFunc() 
  {}
  virtual ~TViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define TRANSPOSE_INT_VIEW_FUNC_AVAILABLE
struct TransposeIntViewFunc : public torch::autograd::ViewFunc {
  TransposeIntViewFunc(int64_t dim0, int64_t dim1) : dim0(dim0), dim1(dim1)
  {}
  virtual ~TransposeIntViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dim0;
  int64_t dim1;
};

#define UNBIND_INT_VIEW_FUNC_AVAILABLE
struct UnbindIntViewFunc : public torch::autograd::ViewFunc {
  UnbindIntViewFunc(int64_t dim, int64_t view_idx) : dim(dim), view_idx(view_idx)
  {}
  virtual ~UnbindIntViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dim;
  int64_t view_idx;
};

#define UNFOLD_VIEW_FUNC_AVAILABLE
struct UnfoldViewFunc : public torch::autograd::ViewFunc {
  UnfoldViewFunc(int64_t dimension, int64_t size, int64_t step) : dimension(dimension), size(size), step(step)
  {}
  virtual ~UnfoldViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dimension;
  int64_t size;
  int64_t step;
};

#define UNSQUEEZE_VIEW_FUNC_AVAILABLE
struct UnsqueezeViewFunc : public torch::autograd::ViewFunc {
  UnsqueezeViewFunc(int64_t dim) : dim(dim)
  {}
  virtual ~UnsqueezeViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  int64_t dim;
};

#define VALUES_VIEW_FUNC_AVAILABLE
struct ValuesViewFunc : public torch::autograd::ViewFunc {
  ValuesViewFunc() 
  {}
  virtual ~ValuesViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define VIEW_VIEW_FUNC_AVAILABLE
struct ViewViewFunc : public torch::autograd::ViewFunc {
  ViewViewFunc(c10::SymIntArrayRef size) : size(size.vec())
  {}
  virtual ~ViewViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ::std::vector<c10::SymInt> size;
};

#define VIEW_DTYPE_VIEW_FUNC_AVAILABLE
struct ViewDtypeViewFunc : public torch::autograd::ViewFunc {
  ViewDtypeViewFunc(at::ScalarType dtype) : dtype(dtype)
  {}
  virtual ~ViewDtypeViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  at::ScalarType dtype;
};

#define VIEW_AS_COMPLEX_VIEW_FUNC_AVAILABLE
struct ViewAsComplexViewFunc : public torch::autograd::ViewFunc {
  ViewAsComplexViewFunc() 
  {}
  virtual ~ViewAsComplexViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

#define VIEW_AS_REAL_VIEW_FUNC_AVAILABLE
struct ViewAsRealViewFunc : public torch::autograd::ViewFunc {
  ViewAsRealViewFunc() 
  {}
  virtual ~ViewAsRealViewFunc() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:

};

} // namespace torch::autograd::generated
