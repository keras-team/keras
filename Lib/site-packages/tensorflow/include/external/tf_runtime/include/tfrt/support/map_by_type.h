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

// This file defines a container type that can store arbitrary data type keyed
// by the data type.

#ifndef TFRT_SUPPORT_MAP_BY_TYPE_H_
#define TFRT_SUPPORT_MAP_BY_TYPE_H_

#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm_derived/Support/unique_any.h"
#include "tfrt/support/type_id.h"

namespace tfrt {

/**
 * MapByType<IdSet> stores arbitrary data type keyed by the data type.
 *
 * Example usage:
 * struct MyMapTagType {};
 *
 * MapByType<MyMapTagType> map;
 *
 * map.insert(2);
 * assert(map.contains<int>());
 * assert(map.get<int>(), 2);
 *
 * When a data type is inserted more than once, the previous value is replaced.
 *
 * Example:
 * map.insert(2);
 * map.insert(3);
 * assert(map.get<int>(), 3);
 */
template <typename IdSet>
class MapByType {
  struct StorageBase {
    virtual ~StorageBase() = default;
  };

  template <typename ConcreteT>
  struct Storage : StorageBase {
    template <typename... Args>
    explicit Storage(Args&&... args) : value(std::forward<Args>(args)...) {}

    ConcreteT value;
  };

 public:
  template <typename T, typename... Args, typename VT = std::decay_t<T>>
  VT& emplace(Args&&... args) {
    auto id = getTypeId<VT>();
    if (id >= data_.size()) data_.resize(id + 1);

    data_[id] = std::make_unique<Storage<VT>>(std::forward<Args>(args)...);

    return cast<VT>(data_[id].get());
  }

  template <typename T>
  std::decay_t<T>& insert(T&& t) {
    return emplace<T>(std::forward<T>(t));
  }

  template <typename... Ts>
  void insert_all(Ts&&... values) {
    static constexpr size_t n = sizeof...(Ts);
    if (n == 0) return;

    // Resize the `data_` to prepare the storage for inserted values.
    std::array<size_t, n> ids = {getTypeId<Ts>()...};
    data_.resize(1 + *std::max_element(ids.begin(), ids.end()));

    // Insert all values into the map.
    // TODO(ezhulenev): C++17: (insert<Ts>(std::forward<Ts>(values)), ...);
    std::tuple<std::decay_t<Ts>&...> refs = {
        insert<Ts>(std::forward<Ts>(values))...};
    (void)refs;
  }

  template <typename T>
  T& get() {
    return const_cast<T&>(static_cast<const MapByType*>(this)->get<T>());
  }

  template <typename T>
  const T& get() const {
    using VT = std::decay_t<T>;
    auto id = getTypeId<VT>();
    assert(id < data_.size());
    return cast<VT>(data_[id].get());
  }

  template <typename T>
  T* getIfExists() {
    return const_cast<T*>(
        static_cast<const MapByType*>(this)->getIfExists<T>());
  }

  template <typename T>
  const T* getIfExists() const {
    using VT = std::decay_t<T>;

    auto id = getTypeId<VT>();
    if (id >= data_.size()) return nullptr;

    auto& value = data_[id];
    if (value) return &cast<VT>(value.get());

    return nullptr;
  }

  template <typename T>
  bool contains() const {
    using VT = std::decay_t<T>;
    auto id = getTypeId<VT>();
    if (id >= data_.size()) return false;
    return data_[id] != nullptr;
  }

 private:
  template <typename T>
  static size_t getTypeId() {
    return DenseTypeId<IdSet>::template get<std::decay_t<T>>();
  }

  template <typename T>
  static T& cast(StorageBase* base) {
    return static_cast<Storage<T>*>(base)->value;
  }

  llvm::SmallVector<std::unique_ptr<StorageBase>> data_;
};

// Optimized MapByType container for storing pointers to data.
//
// In contrast to the MapByType the `const T` and `T` are different keys,
// because the data is not owned by this container, and we can't just cast const
// pointer to the non-const pointer.
template <typename IdSet>
class PtrMapByType {
 public:
  template <typename T>
  T* insert(T* value) {
    size_t id = getTypeId<T>();
    if (id >= data_.size()) data_.resize(id + 1);
    data_[id] = const_cast<std::decay_t<T>*>(value);
    return value;
  }

  template <typename... Ts>
  void insert_all(Ts*... values) {
    static constexpr size_t n = sizeof...(Ts);
    if (n == 0) return;

    // Resize the `data_` to prepare the storage for inserted values.
    std::array<size_t, n> ids = {getTypeId<Ts>()...};
    data_.resize(1 + *std::max_element(ids.begin(), ids.end()), nullptr);

    // Insert all values into the map.
    // TODO(ezhulenev): C++17: (insert<Ts>(std::forward<Ts>(values)), ...);
    std::tuple<Ts*...> ptrs = {insert<Ts>(values)...};
    (void)ptrs;
  }

  template <typename T>
  T* get() const {
    size_t id = getTypeId<T>();
    assert(id < data_.size());
    return reinterpret_cast<T*>(data_[id]);
  }

  template <typename T>
  T* getIfExists() const {
    size_t id = getTypeId<T>();
    return LLVM_LIKELY(id < data_.size()) ? reinterpret_cast<T*>(data_[id])
                                          : nullptr;
  }

  template <typename T>
  bool contains() const {
    size_t id = getTypeId<T>();
    return id < data_.size() && data_[id] != nullptr;
  }

 private:
  template <typename T>
  static size_t getTypeId() {
    return DenseTypeId<IdSet>::template get<T>();
  }

  llvm::SmallVector<void*> data_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_MAP_BY_TYPE_H_
