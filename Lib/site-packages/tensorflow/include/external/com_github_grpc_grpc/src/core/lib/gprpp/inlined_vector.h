/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_CORE_LIB_GPRPP_INLINED_VECTOR_H
#define GRPC_CORE_LIB_GPRPP_INLINED_VECTOR_H

#include <grpc/support/port_platform.h>

#include <cassert>
#include <cstring>

#include "src/core/lib/gprpp/memory.h"

#if GRPC_USE_ABSL
#include "absl/container/inlined_vector.h"
#endif

namespace grpc_core {

#if GRPC_USE_ABSL

template <typename T, size_t N, typename A = std::allocator<T>>
using InlinedVector = absl::InlinedVector<T, N, A>;

#else

// NOTE: We eventually want to use absl::InlinedVector here.  However,
// there are currently build problems that prevent us from using absl.
// In the interim, we define a custom implementation as a place-holder,
// with the intent to eventually replace this with the absl
// implementation.
//
// This place-holder implementation does not implement the full set of
// functionality from the absl version; it has just the methods that we
// currently happen to need in gRPC.  If additional functionality is
// needed before this gets replaced with the absl version, it can be
// added, with the following proviso:
//
// ANY METHOD ADDED HERE MUST COMPLY WITH THE INTERFACE IN THE absl
// IMPLEMENTATION!
//
// TODO(nnoble, roth): Replace this with absl::InlinedVector once we
// integrate absl into the gRPC build system in a usable way.
template <typename T, size_t N>
class InlinedVector {
 public:
  InlinedVector() { init_data(); }
  ~InlinedVector() { destroy_elements(); }

  // copy constructor
  InlinedVector(const InlinedVector& v) {
    init_data();
    copy_from(v);
  }

  InlinedVector& operator=(const InlinedVector& v) {
    if (this != &v) {
      clear();
      copy_from(v);
    }
    return *this;
  }

  // move constructor
  InlinedVector(InlinedVector&& v) {
    init_data();
    move_from(v);
  }

  InlinedVector& operator=(InlinedVector&& v) {
    if (this != &v) {
      clear();
      move_from(v);
    }
    return *this;
  }

  T* data() {
    return dynamic_ != nullptr ? dynamic_ : reinterpret_cast<T*>(inline_);
  }

  const T* data() const {
    return dynamic_ != nullptr ? dynamic_ : reinterpret_cast<const T*>(inline_);
  }

  T& operator[](size_t offset) {
    assert(offset < size_);
    return data()[offset];
  }

  const T& operator[](size_t offset) const {
    assert(offset < size_);
    return data()[offset];
  }

  bool operator==(const InlinedVector& other) const {
    if (size_ != other.size_) return false;
    for (size_t i = 0; i < size_; ++i) {
      // Note that this uses == instead of != so that the data class doesn't
      // have to implement !=.
      if (!(data()[i] == other.data()[i])) return false;
    }
    return true;
  }

  bool operator!=(const InlinedVector& other) const {
    return !(*this == other);
  }

  void reserve(size_t capacity) {
    if (capacity > capacity_) {
      T* new_dynamic =
          std::alignment_of<T>::value == 0
              ? static_cast<T*>(gpr_malloc(sizeof(T) * capacity))
              : static_cast<T*>(gpr_malloc_aligned(
                    sizeof(T) * capacity, std::alignment_of<T>::value));
      move_elements(data(), new_dynamic, size_);
      free_dynamic();
      dynamic_ = new_dynamic;
      capacity_ = capacity;
    }
  }

  void resize(size_t new_size) {
    while (new_size > size_) emplace_back();
    while (new_size < size_) pop_back();
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    if (size_ == capacity_) {
      reserve(capacity_ * 2);
    }
    new (&(data()[size_])) T(std::forward<Args>(args)...);
    ++size_;
  }

  void push_back(const T& value) { emplace_back(value); }

  void push_back(T&& value) { emplace_back(std::move(value)); }

  void pop_back() {
    assert(!empty());
    size_t s = size();
    T& value = data()[s - 1];
    value.~T();
    size_--;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  size_t capacity() const { return capacity_; }

  void clear() {
    destroy_elements();
    init_data();
  }

 private:
  void copy_from(const InlinedVector& v) {
    // if v is allocated, make sure we have enough capacity.
    if (v.dynamic_ != nullptr) {
      reserve(v.capacity_);
    }
    // copy over elements
    for (size_t i = 0; i < v.size_; ++i) {
      new (&(data()[i])) T(v[i]);
    }
    // copy over metadata
    size_ = v.size_;
    capacity_ = v.capacity_;
  }

  void move_from(InlinedVector& v) {
    // if v is allocated, then we steal its dynamic array; otherwise, we
    // move the elements individually.
    if (v.dynamic_ != nullptr) {
      dynamic_ = v.dynamic_;
    } else {
      move_elements(v.data(), data(), v.size_);
    }
    // copy over metadata
    size_ = v.size_;
    capacity_ = v.capacity_;
    // null out the original
    v.init_data();
  }

  static void move_elements(T* src, T* dst, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
      new (&dst[i]) T(std::move(src[i]));
      src[i].~T();
    }
  }

  void init_data() {
    dynamic_ = nullptr;
    size_ = 0;
    capacity_ = N;
  }

  void destroy_elements() {
    for (size_t i = 0; i < size_; ++i) {
      T& value = data()[i];
      value.~T();
    }
    free_dynamic();
  }

  void free_dynamic() {
    if (dynamic_ != nullptr) {
      if (std::alignment_of<T>::value == 0) {
        gpr_free(dynamic_);
      } else {
        gpr_free_aligned(dynamic_);
      }
    }
  }

  typename std::aligned_storage<sizeof(T)>::type inline_[N];
  T* dynamic_;
  size_t size_;
  size_t capacity_;
};

#endif

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_GPRPP_INLINED_VECTOR_H */
