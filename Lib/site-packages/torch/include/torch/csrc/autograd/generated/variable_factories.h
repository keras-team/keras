#pragma once

// @generated from ..\tools\autograd\templates/variable_factories.h

#include <ATen/core/Tensor.h>
#include <ATen/TracerMode.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/MemoryFormat.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <torch/csrc/autograd/variable.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/from_blob.h>
#include <ATen/ops/_make_dep_token.h>
#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/bartlett_window.h>
#include <ATen/ops/bartlett_window.h>
#include <ATen/ops/blackman_window.h>
#include <ATen/ops/blackman_window.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_permuted.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/from_file.h>
#include <ATen/ops/hann_window.h>
#include <ATen/ops/hann_window.h>
#include <ATen/ops/hamming_window.h>
#include <ATen/ops/hamming_window.h>
#include <ATen/ops/hamming_window.h>
#include <ATen/ops/hamming_window.h>
#include <ATen/ops/kaiser_window.h>
#include <ATen/ops/kaiser_window.h>
#include <ATen/ops/kaiser_window.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/rand_like.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randint_like.h>
#include <ATen/ops/randint_like.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randn_like.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/range.h>
#include <ATen/ops/range.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/_sparse_compressed_tensor_with_dims.h>
#include <ATen/ops/sparse_compressed_tensor.h>
#include <ATen/ops/sparse_csr_tensor.h>
#include <ATen/ops/sparse_csc_tensor.h>
#include <ATen/ops/sparse_bsr_tensor.h>
#include <ATen/ops/sparse_bsc_tensor.h>
#include <ATen/ops/sparse_compressed_tensor.h>
#include <ATen/ops/sparse_csr_tensor.h>
#include <ATen/ops/sparse_csc_tensor.h>
#include <ATen/ops/sparse_bsr_tensor.h>
#include <ATen/ops/sparse_bsc_tensor.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/tril_indices.h>
#include <ATen/ops/triu_indices.h>
#include <ATen/ops/normal.h>
#include <ATen/ops/fft_fftfreq.h>
#include <ATen/ops/fft_rfftfreq.h>
#endif

#include <functional>
#include <initializer_list>
#include <utility>

namespace torch {

/// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
/// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
/// support it in the future by iterating over all sub-lists to find
/// the largest data type that can represent all of the elements, or by using
/// variadic templates.
///
/// NOTE: C++ `torch::tensor` with a floating-point type or an `at::ArrayRef` / `std::vector` /
/// (nested) braced-init-list of floating-point types always produces a tensor of dtype
/// `torch::get_default_dtype()`, matching Python `torch.tensor` behavior.
///
/// NOTE: C++ `torch::tensor` with an integer type or an `at::ArrayRef` / `std::vector` /
/// (nested) braced-init-list of integer types always produces a tensor of dtype `at::kLong`
/// (aka. int64_t), matching Python `torch.tensor` behavior.
///
/// NOTE: The following dtypes are not supported by `torch::tensor` currently:
/// - `unsigned int`
/// - `unsigned long int`
/// - `unsigned long long int`
/// - `long long int`
inline at::Tensor tensor(detail::TensorDataContainer tensor_data_container, const at::TensorOptions& options = {}) {
  return autograd::make_variable(
    // note: we remove the requires_grad setting from the TensorOptions because
    // it is ignored anyways (and we actually have an assertion that it isn't set
    // which would fail otherwise). We handle requires_grad explicitly here
    // instead of passing it through to the kernel.
    tensor_data_container.convert_to_tensor(options.requires_grad(::std::nullopt)),
    options.requires_grad());
}

/// A generic deleter function.
using Deleter = std::function<void(void*)>;
using at::MemoryFormat;

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `deleter` function (a
/// `std::function<void(void*)>`) will be called on the `data` when the Tensor
/// data would normally be deallocated. The `TensorOptions` specify additional
/// configuration options for the returned tensor, such as what type to
/// interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, strides, deleter, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `TensorOptions`
/// specify additional configuration options for the returned tensor, such as
/// what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, strides, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The `deleter`
/// (a `std::function<void(void*)>`) function will be called on the `data` when
/// the Tensor data would normally be deallocated. The `TensorOptions` specify
/// additional configuration options for the returned tensor, such as what type
/// to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, deleter, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The
/// `TensorOptions` specify additional configuration options for the returned
/// tensor, such as what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

inline at::Tensor _make_dep_token(at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_make_dep_token(at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_cudnn_init_dropout_state(dropout, train, dropout_seed, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor arange(const at::Scalar & end, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::arange(end, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor arange(const at::Scalar & start, const at::Scalar & end, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::arange(start, end, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor arange(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::arange(start, end, step, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor bartlett_window(int64_t window_length, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::bartlett_window(window_length, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor bartlett_window(int64_t window_length, bool periodic, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::bartlett_window(window_length, periodic, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor blackman_window(int64_t window_length, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::blackman_window(window_length, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor blackman_window(int64_t window_length, bool periodic, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::blackman_window(window_length, periodic, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty(size, names, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty(at::IntArrayRef size, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty(size, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty_symint(c10::SymIntArrayRef size, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty_symint(size, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty_permuted(at::IntArrayRef size, at::IntArrayRef physical_layout, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty_permuted(size, physical_layout, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty_permuted_symint(c10::SymIntArrayRef size, at::IntArrayRef physical_layout, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty_permuted_symint(size, physical_layout, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _empty_affine_quantized(at::IntArrayRef size, at::TensorOptions options = {}, double scale = 1, int64_t zero_point = 0, ::std::optional<at::MemoryFormat> memory_format = c10::MemoryFormat::Contiguous) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_empty_affine_quantized(size, at::TensorOptions(options).requires_grad(::std::nullopt), scale, zero_point, memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _empty_affine_quantized_symint(c10::SymIntArrayRef size, at::TensorOptions options = {}, double scale = 1, int64_t zero_point = 0, ::std::optional<at::MemoryFormat> memory_format = c10::MemoryFormat::Contiguous) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_empty_affine_quantized_symint(size, at::TensorOptions(options).requires_grad(::std::nullopt), scale, zero_point, memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = c10::MemoryFormat::Contiguous) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_empty_per_channel_affine_quantized(size, scales, zero_points, axis, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _empty_per_channel_affine_quantized_symint(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = c10::MemoryFormat::Contiguous) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_empty_per_channel_affine_quantized_symint(size, scales, zero_points, axis, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty_quantized(at::IntArrayRef size, const at::Tensor & qtensor, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty_quantized(size, qtensor, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty_like(const at::Tensor & self, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty_like(self, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty_strided(size, stride, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor empty_strided_symint(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::empty_strided_symint(size, stride, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor eye(int64_t n, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::eye(n, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor eye_symint(c10::SymInt n, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::eye_symint(n, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor eye(int64_t n, int64_t m, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::eye(n, m, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor eye_symint(c10::SymInt n, c10::SymInt m, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::eye_symint(n, m, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::full(size, fill_value, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::full(size, fill_value, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor full_symint(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::full_symint(size, fill_value, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor full_like(const at::Tensor & self, const at::Scalar & fill_value, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::full_like(self, fill_value, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor from_file(c10::string_view filename, ::std::optional<bool> shared = ::std::nullopt, ::std::optional<int64_t> size = 0, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::from_file(filename, shared, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor hann_window(int64_t window_length, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::hann_window(window_length, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor hann_window(int64_t window_length, bool periodic, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::hann_window(window_length, periodic, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor hamming_window(int64_t window_length, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::hamming_window(window_length, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor hamming_window(int64_t window_length, bool periodic, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::hamming_window(window_length, periodic, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor hamming_window(int64_t window_length, bool periodic, double alpha, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::hamming_window(window_length, periodic, alpha, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor hamming_window(int64_t window_length, bool periodic, double alpha, double beta, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::hamming_window(window_length, periodic, alpha, beta, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor kaiser_window(int64_t window_length, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::kaiser_window(window_length, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor kaiser_window(int64_t window_length, bool periodic, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::kaiser_window(window_length, periodic, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor kaiser_window(int64_t window_length, bool periodic, double beta, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::kaiser_window(window_length, periodic, beta, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor linspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::linspace(start, end, steps, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor linspace(const at::Tensor & start, const at::Tensor & end, int64_t steps, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::linspace(start, end, steps, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor linspace(const at::Tensor & start, const at::Scalar & end, int64_t steps, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::linspace(start, end, steps, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor linspace(const at::Scalar & start, const at::Tensor & end, int64_t steps, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::linspace(start, end, steps, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor logspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, double base = 10.0, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::logspace(start, end, steps, base, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor logspace(const at::Tensor & start, const at::Tensor & end, int64_t steps, double base = 10.0, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::logspace(start, end, steps, base, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor logspace(const at::Tensor & start, const at::Scalar & end, int64_t steps, double base = 10.0, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::logspace(start, end, steps, base, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor logspace(const at::Scalar & start, const at::Tensor & end, int64_t steps, double base = 10.0, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::logspace(start, end, steps, base, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor ones(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::ones(size, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor ones(at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::ones(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor ones_symint(c10::SymIntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::ones_symint(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor ones_like(const at::Tensor & self, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::ones_like(self, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor scalar_tensor(const at::Scalar & s, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::scalar_tensor(s, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand(size, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand_symint(size, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand(size, generator, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand_symint(size, generator, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand(at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand_symint(c10::SymIntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand_symint(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand(size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand_symint(size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor rand_like(const at::Tensor & self, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::rand_like(self, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint(int64_t high, at::IntArrayRef size, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint(high, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_symint(c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_symint(high, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint(int64_t high, at::IntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint(high, size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_symint(c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_symint(high, size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint(low, high, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_symint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_symint(low, high, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint(low, high, size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_symint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_symint(low, high, size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t high, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_like(self, high, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_like_symint(const at::Tensor & self, c10::SymInt high, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_like_symint(self, high, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_like(self, low, high, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randint_like_symint(const at::Tensor & self, c10::SymInt low, c10::SymInt high, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randint_like_symint(self, low, high, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn(at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn_symint(c10::SymIntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn_symint(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn(at::IntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn(size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn_symint(size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn(size, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn_symint(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn_symint(size, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn(size, generator, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn_symint(size, generator, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randn_like(const at::Tensor & self, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randn_like(self, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randperm(int64_t n, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randperm(n, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randperm_symint(c10::SymInt n, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randperm_symint(n, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randperm(int64_t n, ::std::optional<at::Generator> generator, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randperm(n, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor randperm_symint(c10::SymInt n, ::std::optional<at::Generator> generator, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::randperm_symint(n, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor range(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step = 1, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::range(start, end, step, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor range(const at::Scalar & start, const at::Scalar & end, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::range(start, end, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor zeros(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::zeros(size, names, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _efficientzerotensor(at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_efficientzerotensor(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _efficientzerotensor_symint(c10::SymIntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_efficientzerotensor_symint(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::zeros(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor zeros_symint(c10::SymIntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::zeros_symint(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor zeros_like(const at::Tensor & self, at::TensorOptions options = {}, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::zeros_like(self, at::TensorOptions(options).requires_grad(::std::nullopt), memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_compressed_tensor_with_dims(int64_t nnz, int64_t dense_dim, at::IntArrayRef size, at::IntArrayRef blocksize, at::ScalarType index_dtype, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_compressed_tensor_with_dims(nnz, dense_dim, size, blocksize, index_dtype, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_compressed_tensor(const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_compressed_tensor(compressed_indices, plain_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_compressed_tensor_symint(const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, c10::SymIntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_compressed_tensor_symint(compressed_indices, plain_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_csr_tensor(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_csr_tensor(crow_indices, col_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_csc_tensor(const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_csc_tensor(ccol_indices, row_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_bsr_tensor(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_bsr_tensor(crow_indices, col_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_bsc_tensor(const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_bsc_tensor(ccol_indices, row_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_compressed_tensor(const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_compressed_tensor(compressed_indices, plain_indices, values, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_csr_tensor(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_csr_tensor(crow_indices, col_indices, values, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_csc_tensor(const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_csc_tensor(ccol_indices, row_indices, values, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_bsr_tensor(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_bsr_tensor(crow_indices, col_indices, values, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_bsc_tensor(const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_bsc_tensor(ccol_indices, row_indices, values, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_compressed_tensor_unsafe(const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_compressed_tensor_unsafe(compressed_indices, plain_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_compressed_tensor_unsafe_symint(const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, c10::SymIntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_compressed_tensor_unsafe_symint(compressed_indices, plain_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_csr_tensor_unsafe(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_csr_tensor_unsafe(crow_indices, col_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_csc_tensor_unsafe(const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_csc_tensor_unsafe(ccol_indices, row_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_bsr_tensor_unsafe(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_bsr_tensor_unsafe(crow_indices, col_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_bsc_tensor_unsafe(const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_bsc_tensor_unsafe(ccol_indices, row_indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_coo_tensor(at::IntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_coo_tensor(size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, at::TensorOptions options = {}, ::std::optional<bool> is_coalesced = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_coo_tensor(indices, values, at::TensorOptions(options).requires_grad(::std::nullopt), is_coalesced), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor sparse_coo_tensor(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options = {}, ::std::optional<bool> is_coalesced = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::sparse_coo_tensor(indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt), is_coalesced), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options = {}, ::std::optional<bool> is_coalesced = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_coo_tensor_unsafe(indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt), is_coalesced), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_coo_tensor_unsafe_symint(const at::Tensor & indices, const at::Tensor & values, c10::SymIntArrayRef size, at::TensorOptions options = {}, ::std::optional<bool> is_coalesced = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_coo_tensor_unsafe_symint(indices, values, size, at::TensorOptions(options).requires_grad(::std::nullopt), is_coalesced), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, at::TensorOptions options) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, at::TensorOptions options, ::std::optional<bool> is_coalesced = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, at::TensorOptions(options).requires_grad(::std::nullopt), is_coalesced), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _sparse_coo_tensor_with_dims_and_tensors_symint(int64_t sparse_dim, int64_t dense_dim, c10::SymIntArrayRef size, const at::Tensor & indices, const at::Tensor & values, at::TensorOptions options, ::std::optional<bool> is_coalesced = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_sparse_coo_tensor_with_dims_and_tensors_symint(sparse_dim, dense_dim, size, indices, values, at::TensorOptions(options).requires_grad(::std::nullopt), is_coalesced), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor _to_copy(const at::Tensor & self, at::TensorOptions options = {}, bool non_blocking = false, ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::_to_copy(self, at::TensorOptions(options).requires_grad(::std::nullopt), non_blocking, memory_format), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor tril_indices(int64_t row, int64_t col, int64_t offset = 0, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::tril_indices(row, col, offset, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor triu_indices(int64_t row, int64_t col, int64_t offset = 0, at::TensorOptions options = at::kLong) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::triu_indices(row, col, offset, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor normal(double mean, double std, at::IntArrayRef size, ::std::optional<at::Generator> generator = ::std::nullopt, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::normal(mean, std, size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor normal_symint(double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator = ::std::nullopt, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::normal_symint(mean, std, size, generator, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor fft_fftfreq(int64_t n, double d = 1.0, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::fft_fftfreq(n, d, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}
inline at::Tensor fft_rfftfreq(int64_t n, double d = 1.0, at::TensorOptions options = {}) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::fft_rfftfreq(n, d, at::TensorOptions(options).requires_grad(::std::nullopt)), /*requires_grad=*/options.requires_grad());
}

} // namespace torch
