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

// This file defines a mapping from type to numeric id.

#ifndef TFRT_SUPPORT_TYPE_ID_H_
#define TFRT_SUPPORT_TYPE_ID_H_

#include <atomic>

namespace tfrt {

// Forward declare
template <typename IdSet>
class DenseTypeId;

namespace internal {
template <typename IdSet, typename T>
class DenseTypeIdResolver {
  friend DenseTypeId<IdSet>;
  static size_t get();
};
}  // namespace internal

// Use this as DenseTypeId<some_type_specific_to_your_use>, that way you are
// guaranteed to get contiguous IDs starting at 0 unique to your particular
// use-case, as would be appropriate to use for indexes into a vector.
// 'some_type_specific_to_your_use' could (e.g.) be the class that contains
// that particular vector.
template <typename IdSet>
class DenseTypeId {
 public:
  template <typename T>
  static size_t get() {
    return internal::DenseTypeIdResolver<IdSet, T>::get();
  }

 private:
  // Partial template specialization can't be declared as a friend, so we
  // declare all `DenseTypeIdResolver` as a friend.
  template <typename OtherIdSet, typename T>
  friend class internal::DenseTypeIdResolver;

  static size_t next_id() {
    return next_id_.fetch_add(1, std::memory_order_relaxed);
  }

  static std::atomic<size_t> next_id_;
};

template <typename IdSet>
std::atomic<size_t> DenseTypeId<IdSet>::next_id_;

namespace internal {
template <typename IdSet, typename T>
size_t DenseTypeIdResolver<IdSet, T>::get() {
  static const size_t id = DenseTypeId<IdSet>::next_id();
  return id;
}
}  // namespace internal
}  // namespace tfrt

// Declare/define an explicit specialization for DenseTypeId.
//
// This forces the compiler to assign a dense type id for the given type and
// avoids checking the static initilization guard if the type id defined as a
// static variable (default implementation of the DenseTypeIdResolver).
//
//  Example:
//
//  // Foo.h
//  struct FooIdSet {};
//
//  TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(FooIdSet, int32_t);
//
//  // Foo.cpp
//  TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(FooIdSet, int32_t);
//
#define TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(ID_SET, T) \
  namespace tfrt {                                     \
  namespace internal {                                 \
  template <>                                          \
  class DenseTypeIdResolver<ID_SET, T> {               \
   public:                                             \
    static size_t get() { return id; }                 \
                                                       \
   private:                                            \
    static size_t id;                                  \
  };                                                   \
  } /* namespace internal */                           \
  } /* namespace tfrt */

#define TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(ID_SET, T)                         \
  namespace tfrt {                                                            \
  namespace internal {                                                        \
  size_t DenseTypeIdResolver<ID_SET, T>::id = DenseTypeId<ID_SET>::next_id(); \
  } /* namespace internal */                                                  \
  } /* namespace tfrt */

#endif  // TFRT_SUPPORT_TYPE_ID_H_
