/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file introduces utilities for working with pointer.

#ifndef TFRT_SUPPORT_POINTER_UTIL_H_
#define TFRT_SUPPORT_POINTER_UTIL_H_

#include <memory>

#include "llvm/ADT/PointerIntPair.h"

namespace tfrt {

template <typename T>
class OwnedOrUnownedPtr {
 public:
  OwnedOrUnownedPtr() : ptr_(nullptr, empty_deleter) {}
  explicit OwnedOrUnownedPtr(T* object) : ptr_(object, empty_deleter) {}
  explicit OwnedOrUnownedPtr(std::unique_ptr<T> object)
      : ptr_(object.release(), default_deleter) {}

  // Support implicit conversion from OwnedOrUnownedPtr<Derived> to
  // OwnedOrUnownedPtr<Base>.
  template <typename DerivedT,
            std::enable_if_t<std::is_base_of<T, DerivedT>::value, int> = 0>
  OwnedOrUnownedPtr(OwnedOrUnownedPtr<DerivedT>&& u)
      : ptr_(std::move(u.ptr_)) {}

  ~OwnedOrUnownedPtr() {}

  OwnedOrUnownedPtr& operator=(OwnedOrUnownedPtr&& other) {
    reset(other.ptr_.release(), /*owned=*/true);
    return *this;
  }

  OwnedOrUnownedPtr& operator=(std::unique_ptr<T>&& other) {
    reset(other.release(), /*owned=*/true);
    return *this;
  }

  void reset(T* object, const bool owned = true) {
    if (owned) {
      ptr_ = std::unique_ptr<T, void (*)(T*)>(object, default_deleter);
    } else {
      ptr_ = std::unique_ptr<T, void (*)(T*)>(object, empty_deleter);
    }
  }

  T* get() const { return ptr_.get(); }

  T* operator->() const { return get(); }

  T& operator*() const { return *get(); }

  T* release() { return ptr_.release(); }

 private:
  static void empty_deleter(T* t) {}
  static void default_deleter(T* t) { std::default_delete<T>()(t); }
  std::unique_ptr<T, void (*)(T*)> ptr_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_POINTER_UTIL_H_
