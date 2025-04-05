//===- UniqueAny.h - Generic type erased holder of any type -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file provides UniqueAny, a non-template class adapted from llvm::Any.
//  The idea is to provide a type-safe replacement for C's void*. In contrast to
//  llvm::Any, tfrt::UniqueAny can hold a non-copy-constructible type such as
//  std::unique_ptr, and as a result, tfrt::UniqueAny itself is no longer
//  copyable. As an improvement to llvm::Any, tfrt::UniqueAny also supports
//  in-place construction and make_unique_any, a la std::any and absl::any.
//  However, tfrt::UniqueAny does not yet support emplace.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LLVM_DERIVED_UNIQUE_ANY_H_
#define TFRT_LLVM_DERIVED_UNIQUE_ANY_H_

#include <cassert>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "llvm_derived/Support/in_place.h"

namespace tfrt {

class UniqueAny;

//===----------------------------------------------------------------------===//
// any_isa and any_cast.
//===----------------------------------------------------------------------===//

// Returns whether UniqueAny contains an instance of the specified class.
template <typename T>
bool any_isa(const UniqueAny& Value);

// Statically cast to a given type. T has to be a reference type.
//   e.g., int& i = any_cast<int&>(any_val);
template <class T>
T any_cast(const UniqueAny& Value);

// Overload of any_cast to statically cast non-const UniqueAny type to the given
// type. T has to be a reference type. This function will assert fail or crash
// (optimized build) if the stored value type does not match the cast.
template <class T>
T any_cast(UniqueAny& Value);

// Overload of any_cast to statically cast rvalue UniqueAny type to the given
// type. This function will assert fail or crash (optimized build) if the stored
// value type does not match the cast.
template <class T>
T any_cast(UniqueAny&& Value);

// Overload of any_cast to statically cast a const pointer UniqueAny type to the
// given type or nullptr if the stored value type does not match the cast.
template <class T>
const T* any_cast(const UniqueAny* Value);

// Overload of any_cast to statically cast a pointer UniqueAny type to the
// given type or nullptr if the stored value type does not match the cast.
template <class T>
T* any_cast(UniqueAny* Value);

//===----------------------------------------------------------------------===//
// UniqueAny and make_unique_any.
//===----------------------------------------------------------------------===//

// Construct a tfrt::UniqueAny of type T with the given arguments.
template <typename T, typename... Args>
UniqueAny make_unique_any(Args&&... args);

// Overload of tfrt::make_unique_any() for constructing a tfrt::UniqueAny type
// from an initializer list.
template <typename T, typename U, typename... Args>
UniqueAny make_unique_any(std::initializer_list<U> il, Args&&... args);

class UniqueAny {
  template <typename T>
  struct IsInPlaceType;

  template <typename T>
  struct TypeId {
    static const char Id;
  };

  struct StorageBase {
    virtual ~StorageBase() = default;
    virtual const void* id() const = 0;
  };

  template <typename T>
  struct StorageImpl : public StorageBase {
    explicit StorageImpl(const T& Value) : Value(Value) {}

    template <typename... Args>
    explicit StorageImpl(in_place_t /*tag*/, Args&&... args)
        : Value(std::forward<Args>(args)...) {}

    const void* id() const override { return &TypeId<T>::Id; }

    T Value;

   private:
    StorageImpl& operator=(const StorageImpl& Other) = delete;
    StorageImpl(const StorageImpl& Other) = delete;
  };

 public:
  UniqueAny() = default;

  // When T is UniqueAny we need to explicitly disable the forwarding
  // constructor.
  template <
      typename T, typename VT = std::decay_t<T>,
      std::enable_if_t<(!std::is_same<VT, UniqueAny>() && !IsInPlaceType<VT>()),
                       int> = 0>
  UniqueAny(T&& Value) {
    Storage =
        std::make_unique<StorageImpl<VT>>(in_place, std::forward<T>(Value));
  }

  template <typename T, typename... Args, typename VT = std::decay_t<T>>
  explicit UniqueAny(in_place_type_t<T> /*tag*/, Args&&... args) {
    Storage = std::make_unique<StorageImpl<VT>>(in_place,
                                                std::forward<Args>(args)...);
  }

  template <
      typename T, typename U, typename... Args, typename VT = std::decay_t<T>,
      std::enable_if_t<
          std::is_constructible<VT, std::initializer_list<U>&, Args...>::value,
          int> = 0>
  explicit UniqueAny(in_place_type_t<T> /*tag*/, std::initializer_list<U> ilist,
                     Args&&... args) {
    Storage = std::make_unique<StorageImpl<VT>>(in_place, ilist,
                                                std::forward<Args>(args)...);
  }

  UniqueAny(UniqueAny&& Other) : Storage(std::move(Other.Storage)) {}

  UniqueAny& swap(UniqueAny& Other) {
    std::swap(Storage, Other.Storage);
    return *this;
  }

  UniqueAny& operator=(UniqueAny Other) {
    Storage = std::move(Other.Storage);
    return *this;
  }

  bool hasValue() const { return !!Storage; }

  void reset() { Storage.reset(); }

 private:
  template <class T>
  friend T any_cast(const UniqueAny& Value);
  template <class T>
  friend T any_cast(UniqueAny& Value);
  template <class T>
  friend T any_cast(UniqueAny&& Value);
  template <class T>
  friend const T* any_cast(const UniqueAny* Value);
  template <class T>
  friend T* any_cast(UniqueAny* Value);
  template <typename T>
  friend bool any_isa(const UniqueAny& Value);

  std::unique_ptr<StorageBase> Storage;
};

//===----------------------------------------------------------------------===//
// Implementation details.
//===----------------------------------------------------------------------===//

template <typename T>
struct UniqueAny::IsInPlaceType : std::false_type {};

template <typename T>
struct UniqueAny::IsInPlaceType<in_place_type_t<T>> : std::true_type {};

template <typename T, typename... Args>
UniqueAny make_unique_any(Args&&... args) {
  return UniqueAny(in_place_type_t<T>(), std::forward<Args>(args)...);
}

template <typename T, typename U, typename... Args>
UniqueAny make_unique_any(std::initializer_list<U> il, Args&&... args) {
  return UniqueAny(in_place_type_t<T>(), il, std::forward<Args>(args)...);
}

template <typename T>
const char UniqueAny::TypeId<T>::Id = 0;

template <typename T>
bool any_isa(const UniqueAny& Value) {
  if (!Value.Storage) return false;
  return Value.Storage->id() ==
         &UniqueAny::TypeId<std::remove_cv_t<std::remove_reference_t<T>>>::Id;
}

template <class T>
T any_cast(const UniqueAny& Value) {
  return static_cast<T>(
      *any_cast<std::remove_cv_t<std::remove_reference_t<T>>>(&Value));
}

template <class T>
T any_cast(UniqueAny& Value) {
  return static_cast<T>(
      *any_cast<std::remove_cv_t<std::remove_reference_t<T>>>(&Value));
}

template <class T>
T any_cast(UniqueAny&& Value) {
  return static_cast<T>(std::move(
      *any_cast<std::remove_cv_t<std::remove_reference_t<T>>>(&Value)));
}

template <class T>
const T* any_cast(const UniqueAny* Value) {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  assert(Value && any_isa<T>(*Value) && "Bad any cast!");
  if (!Value || !any_isa<U>(*Value)) return nullptr;
  return &static_cast<UniqueAny::StorageImpl<U>&>(*Value->Storage).Value;
}

template <class T>
T* any_cast(UniqueAny* Value) {
  using U = std::decay_t<T>;
  assert(Value && any_isa<U>(*Value) && "Bad any cast!");
  if (!Value || !any_isa<U>(*Value)) return nullptr;
  return &static_cast<UniqueAny::StorageImpl<U>&>(*Value->Storage).Value;
}

}  // namespace tfrt

#endif  // TFRT_LLVM_DERIVED_UNIQUE_ANY_H_
