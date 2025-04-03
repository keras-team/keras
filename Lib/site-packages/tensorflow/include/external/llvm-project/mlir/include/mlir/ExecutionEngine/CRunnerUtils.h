//===- CRunnerUtils.h - Utils for debugging MLIR execution ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to manipulate structured MLIR
// types at runtime. Entities in this file must be compliant with C++11 and be
// retargetable, including on targets without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_CRUNNERUTILS_H
#define MLIR_EXECUTIONENGINE_CRUNNERUTILS_H

#ifdef _WIN32
#ifndef MLIR_CRUNNERUTILS_EXPORT
#ifdef mlir_c_runner_utils_EXPORTS
// We are building this library
#define MLIR_CRUNNERUTILS_EXPORT __declspec(dllexport)
#define MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
#else
// We are using this library
#define MLIR_CRUNNERUTILS_EXPORT __declspec(dllimport)
#endif // mlir_c_runner_utils_EXPORTS
#endif // MLIR_CRUNNERUTILS_EXPORT
#else  // _WIN32
// Non-windows: use visibility attributes.
#define MLIR_CRUNNERUTILS_EXPORT __attribute__((visibility("default")))
#define MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
#endif // _WIN32

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

//===----------------------------------------------------------------------===//
// Codegen-compatible structures for Vector type.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {

constexpr bool isPowerOf2(int n) { return (!(n & (n - 1))); }

constexpr unsigned nextPowerOf2(int n) {
  return (n <= 1) ? 1 : (isPowerOf2(n) ? n : (2 * nextPowerOf2((n + 1) / 2)));
}

template <typename T, int Dim, bool IsPowerOf2>
struct Vector1D;

template <typename T, int Dim>
struct Vector1D<T, Dim, /*IsPowerOf2=*/true> {
  Vector1D() {
    static_assert(detail::nextPowerOf2(sizeof(T[Dim])) == sizeof(T[Dim]),
                  "size error");
  }
  inline T &operator[](unsigned i) { return vector[i]; }
  inline const T &operator[](unsigned i) const { return vector[i]; }

private:
  T vector[Dim];
};

// 1-D vector, padded to the next power of 2 allocation.
// Specialization occurs to avoid zero size arrays (which fail in -Werror).
template <typename T, int Dim>
struct Vector1D<T, Dim, /*IsPowerOf2=*/false> {
  Vector1D() {
    static_assert(nextPowerOf2(sizeof(T[Dim])) > sizeof(T[Dim]), "size error");
    static_assert(nextPowerOf2(sizeof(T[Dim])) < 2 * sizeof(T[Dim]),
                  "size error");
  }
  inline T &operator[](unsigned i) { return vector[i]; }
  inline const T &operator[](unsigned i) const { return vector[i]; }

private:
  T vector[Dim];
  char padding[nextPowerOf2(sizeof(T[Dim])) - sizeof(T[Dim])];
};
} // namespace detail
} // namespace mlir

// N-D vectors recurse down to 1-D.
template <typename T, int Dim, int... Dims>
struct Vector {
  inline Vector<T, Dims...> &operator[](unsigned i) { return vector[i]; }
  inline const Vector<T, Dims...> &operator[](unsigned i) const {
    return vector[i];
  }

private:
  Vector<T, Dims...> vector[Dim];
};

// 1-D vectors in LLVM are automatically padded to the next power of 2.
// We insert explicit padding in to account for this.
template <typename T, int Dim>
struct Vector<T, Dim>
    : public mlir::detail::Vector1D<T, Dim,
                                    mlir::detail::isPowerOf2(sizeof(T[Dim]))> {
};

template <int D1, typename T>
using Vector1D = Vector<T, D1>;
template <int D1, int D2, typename T>
using Vector2D = Vector<T, D1, D2>;
template <int D1, int D2, int D3, typename T>
using Vector3D = Vector<T, D1, D2, D3>;
template <int D1, int D2, int D3, int D4, typename T>
using Vector4D = Vector<T, D1, D2, D3, D4>;

template <int N>
void dropFront(int64_t arr[N], int64_t *res) {
  for (unsigned i = 1; i < N; ++i)
    *(res + i - 1) = arr[i];
}

//===----------------------------------------------------------------------===//
// Codegen-compatible structures for StridedMemRef type.
//===----------------------------------------------------------------------===//
template <typename T, int Rank>
class StridedMemrefIterator;

/// StridedMemRef descriptor type with static rank.
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];

  template <typename Range,
            typename sfinae = decltype(std::declval<Range>().begin())>
  T &operator[](Range &&indices) {
    assert(indices.size() == N &&
           "indices should match rank in memref subscript");
    int64_t curOffset = offset;
    for (int dim = N - 1; dim >= 0; --dim) {
      int64_t currentIndex = *(indices.begin() + dim);
      assert(currentIndex < sizes[dim] && "Index overflow");
      curOffset += currentIndex * strides[dim];
    }
    return data[curOffset];
  }

  StridedMemrefIterator<T, N> begin() { return {*this, offset}; }
  StridedMemrefIterator<T, N> end() { return {*this, -1}; }

  // This operator[] is extremely slow and only for sugaring purposes.
  StridedMemRefType<T, N - 1> operator[](int64_t idx) {
    StridedMemRefType<T, N - 1> res;
    res.basePtr = basePtr;
    res.data = data;
    res.offset = offset + idx * strides[0];
    dropFront<N>(sizes, res.sizes);
    dropFront<N>(strides, res.strides);
    return res;
  }
};

/// StridedMemRef descriptor type specialized for rank 1.
template <typename T>
struct StridedMemRefType<T, 1> {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];

  template <typename Range,
            typename sfinae = decltype(std::declval<Range>().begin())>
  T &operator[](Range indices) {
    assert(indices.size() == 1 &&
           "indices should match rank in memref subscript");
    return (*this)[*indices.begin()];
  }

  StridedMemrefIterator<T, 1> begin() { return {*this, offset}; }
  StridedMemrefIterator<T, 1> end() { return {*this, -1}; }

  T &operator[](int64_t idx) { return *(data + offset + idx * strides[0]); }
};

/// StridedMemRef descriptor type specialized for rank 0.
template <typename T>
struct StridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  int64_t offset;

  template <typename Range,
            typename sfinae = decltype(std::declval<Range>().begin())>
  T &operator[](Range indices) {
    assert((indices.size() == 0) &&
           "Expect empty indices for 0-rank memref subscript");
    return data[offset];
  }

  StridedMemrefIterator<T, 0> begin() { return {*this, offset}; }
  StridedMemrefIterator<T, 0> end() { return {*this, offset + 1}; }
};

/// Iterate over all elements in a strided memref.
template <typename T, int Rank>
class StridedMemrefIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;

  StridedMemrefIterator(StridedMemRefType<T, Rank> &descriptor,
                        int64_t offset = 0)
      : offset(offset), descriptor(&descriptor) {}
  StridedMemrefIterator<T, Rank> &operator++() {
    int dim = Rank - 1;
    while (dim >= 0 && indices[dim] == (descriptor->sizes[dim] - 1)) {
      offset -= indices[dim] * descriptor->strides[dim];
      indices[dim] = 0;
      --dim;
    }
    if (dim < 0) {
      offset = -1;
      return *this;
    }
    ++indices[dim];
    offset += descriptor->strides[dim];
    return *this;
  }

  reference operator*() { return descriptor->data[offset]; }
  pointer operator->() { return &descriptor->data[offset]; }

  const std::array<int64_t, Rank> &getIndices() { return indices; }

  bool operator==(const StridedMemrefIterator &other) const {
    return other.offset == offset && other.descriptor == descriptor;
  }

  bool operator!=(const StridedMemrefIterator &other) const {
    return !(*this == other);
  }

private:
  /// Offset in the buffer. This can be derived from the indices and the
  /// descriptor.
  int64_t offset = 0;

  /// Array of indices in the multi-dimensional memref.
  std::array<int64_t, Rank> indices = {};

  /// Descriptor for the strided memref.
  StridedMemRefType<T, Rank> *descriptor;
};

/// Iterate over all elements in a 0-ranked strided memref.
template <typename T>
class StridedMemrefIterator<T, 0> {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;

  StridedMemrefIterator(StridedMemRefType<T, 0> &descriptor, int64_t offset = 0)
      : elt(descriptor.data + offset) {}

  StridedMemrefIterator<T, 0> &operator++() {
    ++elt;
    return *this;
  }

  reference operator*() { return *elt; }
  pointer operator->() { return elt; }

  // There are no indices for a 0-ranked memref, but this API is provided for
  // consistency with the general case.
  const std::array<int64_t, 0> &getIndices() {
    // Since this is a 0-array of indices we can keep a single global const
    // copy.
    static const std::array<int64_t, 0> indices = {};
    return indices;
  }

  bool operator==(const StridedMemrefIterator &other) const {
    return other.elt == elt;
  }

  bool operator!=(const StridedMemrefIterator &other) const {
    return !(*this == other);
  }

private:
  /// Pointer to the single element in the zero-ranked memref.
  T *elt;
};

//===----------------------------------------------------------------------===//
// Codegen-compatible structure for UnrankedMemRef type.
//===----------------------------------------------------------------------===//
// Unranked MemRef
template <typename T>
struct UnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};

//===----------------------------------------------------------------------===//
// DynamicMemRefType type.
//===----------------------------------------------------------------------===//
template <typename T>
class DynamicMemRefIterator;

// A reference to one of the StridedMemRef types.
template <typename T>
class DynamicMemRefType {
public:
  int64_t rank;
  T *basePtr;
  T *data;
  int64_t offset;
  const int64_t *sizes;
  const int64_t *strides;

  explicit DynamicMemRefType(const StridedMemRefType<T, 0> &memRef)
      : rank(0), basePtr(memRef.basePtr), data(memRef.data),
        offset(memRef.offset), sizes(nullptr), strides(nullptr) {}
  template <int N>
  explicit DynamicMemRefType(const StridedMemRefType<T, N> &memRef)
      : rank(N), basePtr(memRef.basePtr), data(memRef.data),
        offset(memRef.offset), sizes(memRef.sizes), strides(memRef.strides) {}
  explicit DynamicMemRefType(const ::UnrankedMemRefType<T> &memRef)
      : rank(memRef.rank) {
    auto *desc = static_cast<StridedMemRefType<T, 1> *>(memRef.descriptor);
    basePtr = desc->basePtr;
    data = desc->data;
    offset = desc->offset;
    sizes = rank == 0 ? nullptr : desc->sizes;
    strides = sizes + rank;
  }

  template <typename Range,
            typename sfinae = decltype(std::declval<Range>().begin())>
  T &operator[](Range &&indices) {
    assert(indices.size() == rank &&
           "indices should match rank in memref subscript");
    if (rank == 0)
      return data[offset];

    int64_t curOffset = offset;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t currentIndex = *(indices.begin() + dim);
      assert(currentIndex < sizes[dim] && "Index overflow");
      curOffset += currentIndex * strides[dim];
    }
    return data[curOffset];
  }

  DynamicMemRefIterator<T> begin() { return {*this, offset}; }
  DynamicMemRefIterator<T> end() { return {*this, -1}; }

  // This operator[] is extremely slow and only for sugaring purposes.
  DynamicMemRefType<T> operator[](int64_t idx) {
    assert(rank > 0 && "can't make a subscript of a zero ranked array");

    DynamicMemRefType<T> res(*this);
    --res.rank;
    res.offset += idx * res.strides[0];
    ++res.sizes;
    ++res.strides;
    return res;
  }

  // This operator* can be used in conjunction with the previous operator[] in
  // order to access the underlying value in case of zero-ranked memref.
  T &operator*() {
    assert(rank == 0 && "not a zero-ranked memRef");
    return data[offset];
  }
};

/// Iterate over all elements in a dynamic memref.
template <typename T>
class DynamicMemRefIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;

  DynamicMemRefIterator(DynamicMemRefType<T> &descriptor, int64_t offset = 0)
      : offset(offset), descriptor(&descriptor) {
    indices.resize(descriptor.rank, 0);
  }

  DynamicMemRefIterator<T> &operator++() {
    if (descriptor->rank == 0) {
      offset = -1;
      return *this;
    }

    int dim = descriptor->rank - 1;

    while (dim >= 0 && indices[dim] == (descriptor->sizes[dim] - 1)) {
      offset -= indices[dim] * descriptor->strides[dim];
      indices[dim] = 0;
      --dim;
    }

    if (dim < 0) {
      offset = -1;
      return *this;
    }

    ++indices[dim];
    offset += descriptor->strides[dim];
    return *this;
  }

  reference operator*() { return descriptor->data[offset]; }
  pointer operator->() { return &descriptor->data[offset]; }

  const std::vector<int64_t> &getIndices() { return indices; }

  bool operator==(const DynamicMemRefIterator &other) const {
    return other.offset == offset && other.descriptor == descriptor;
  }

  bool operator!=(const DynamicMemRefIterator &other) const {
    return !(*this == other);
  }

private:
  /// Offset in the buffer. This can be derived from the indices and the
  /// descriptor.
  int64_t offset = 0;

  /// Array of indices in the multi-dimensional memref.
  std::vector<int64_t> indices = {};

  /// Descriptor for the dynamic memref.
  DynamicMemRefType<T> *descriptor;
};

//===----------------------------------------------------------------------===//
// Small runtime support library for memref.copy lowering during codegen.
//===----------------------------------------------------------------------===//
extern "C" MLIR_CRUNNERUTILS_EXPORT void
memrefCopy(int64_t elemSize, ::UnrankedMemRefType<char> *src,
           ::UnrankedMemRefType<char> *dst);

//===----------------------------------------------------------------------===//
// Small runtime support library for vector.print lowering during codegen.
//===----------------------------------------------------------------------===//
extern "C" MLIR_CRUNNERUTILS_EXPORT void printI64(int64_t i);
extern "C" MLIR_CRUNNERUTILS_EXPORT void printU64(uint64_t u);
extern "C" MLIR_CRUNNERUTILS_EXPORT void printF32(float f);
extern "C" MLIR_CRUNNERUTILS_EXPORT void printF64(double d);
extern "C" MLIR_CRUNNERUTILS_EXPORT void printString(char const *s);
extern "C" MLIR_CRUNNERUTILS_EXPORT void printOpen();
extern "C" MLIR_CRUNNERUTILS_EXPORT void printClose();
extern "C" MLIR_CRUNNERUTILS_EXPORT void printComma();
extern "C" MLIR_CRUNNERUTILS_EXPORT void printNewline();

//===----------------------------------------------------------------------===//
// Small runtime support library for timing execution and printing GFLOPS
//===----------------------------------------------------------------------===//
extern "C" MLIR_CRUNNERUTILS_EXPORT void printFlops(double flops);
extern "C" MLIR_CRUNNERUTILS_EXPORT double rtclock();

//===----------------------------------------------------------------------===//
// Runtime support library for random number generation.
//===----------------------------------------------------------------------===//
// Uses a seed to initialize a random generator and returns the generator.
extern "C" MLIR_CRUNNERUTILS_EXPORT void *rtsrand(uint64_t s);
// Uses a random number generator g and returns a random number
// in the range of [0, m).
extern "C" MLIR_CRUNNERUTILS_EXPORT uint64_t rtrand(void *g, uint64_t m);
// Deletes the random number generator.
extern "C" MLIR_CRUNNERUTILS_EXPORT void rtdrand(void *g);
// Uses a random number generator g and std::shuffle to modify mref
// in place. Memref mref will be a permutation of all numbers
// in the range of [0, size of mref).
extern "C" MLIR_CRUNNERUTILS_EXPORT void
_mlir_ciface_shuffle(StridedMemRefType<uint64_t, 1> *mref, void *g);

//===----------------------------------------------------------------------===//
// Runtime support library to allow the use of std::sort in MLIR program.
//===----------------------------------------------------------------------===//
extern "C" MLIR_CRUNNERUTILS_EXPORT void
_mlir_ciface_stdSortI64(uint64_t n, StridedMemRefType<int64_t, 1> *vref);
extern "C" MLIR_CRUNNERUTILS_EXPORT void
_mlir_ciface_stdSortF64(uint64_t n, StridedMemRefType<double, 1> *vref);
extern "C" MLIR_CRUNNERUTILS_EXPORT void
_mlir_ciface_stdSortF32(uint64_t n, StridedMemRefType<float, 1> *vref);
#endif // MLIR_EXECUTIONENGINE_CRUNNERUTILS_H
