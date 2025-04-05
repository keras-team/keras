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

// Generic synchronous value type
//
// This file declares Value, a generic type-erased value type designed for use
// for synchronous kernels and TFRT interpreter.

#ifndef TFRT_HOST_CONTEXT_VALUE_H_
#define TFRT_HOST_CONTEXT_VALUE_H_

#include <cassert>

#include "tfrt/support/type_traits.h"

namespace tfrt {

namespace internal {
class TypeTraits;
template <class T>
class InPlaceTypeTraits;
template <class T>
class OutOfPlaceTypeTraits;
class PointerPayloadTypeTraits;
}  // namespace internal

// Value is a type-erased data type for representing synchronous values and is
// used for defining synchronous kernels and in TFRT interpreter. This is the
// counterpart of AsyncValue for defining asynchronous kernels.
//
// The Value class has an inplace storage that avoids heap allocations for small
// object types. Currently, we set the inplace storage size to 56 bytes to
// avoid heap allocations for storing DenseHostTensors which takes 48 bytes
// while keeping the Value object to be the same size of a cache line (64
// bytes).
//
// The Value class can store any type regardless of their move/copy-ability. The
// memory layout for different payload types is as follows:
// * For moveable types with size <= 56 bytes, the payload is stored in place
//   with no heap allocation.
// * For moveable types with size > 56 bytes, the payload is stored out of place
//   with a heap allocation.
// * For non-moveable types regardless of size, the payload is stored out of
//   place with a heap allocation.
//
// The Value class also allows storing a non-owning non-const pointer to the
// payload. This allows the client to use Value without copying/moving the
// payload into the Value object. Note that Value does not allow storing const
// pointer (trying to do so will cause a compiler error), as this is
// incompatible with Value::get() which return a non-const ref to the payload.
//
// Unlike std::any, Value supports getting a reference to the base class of the
// stored type. For example, the following code works:
//
// struct BaseClass {};
// struct DerivedClass : BaseClass {};
//
// Value v{DerivedClass()};
// auto& base = v.get<BaseClass>();
//
// The restriction to this capability is that casting from a derived class to a
// base class should not need pointer adjustment. More specifically, all of the
// following cases are illegal:
//
// * Get a non-polymorphic base class when the payload is a polymorphic derived
//   class.
// * Get non-first base class when the payload inherits from multiple bases.
// * Get a virtual base class of the payload class.
//
// TODO: Add type assertion to prevent getting a base class reference for the
// stored type in the presense of multiple inheritance or virtual inheritance.
// This will make the debug type checking more robust.
//
// The Value class is thread-compatible.
//
// TODO: We need to add the conversion between AsyncValue and Value to allow for
// the interoperation of TFRT interpreter and executor.
class Value {
 public:
  struct PointerPayload {};

  // Value is default constructible. The payload is unset in the default
  // constructed Value.
  Value() = default;

  // Value is not copyable or copy-assignable.
  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;

  // Value is movable and move-assignable.
  Value(Value&&);
  Value& operator=(Value&&);

  // Construct Value and store `t` as the payload.
  template <typename T>
  explicit Value(T&& t);

  // Construct Value that stores a pointer to the payload. With Value(ptr,
  // PointerPayload{}), Value::get() returns a ref to the pointee object. This
  // is unlike Value(ptr) where Value::get() returns a ref to the pointer.
  template <typename T>
  explicit Value(T* t, PointerPayload);

  ~Value();

  // get() function returns the payload of the Value object in the requested
  // type.
  //
  // Dynamic type checking is performed in the debug mode.
  template <typename T>
  T& get();

  template <typename T>
  const T& get() const;

  // emplace() constructs the payload object of type T in place with the given
  // args.
  template <typename T, typename... Args>
  void emplace(Args&&... args);

  // set() stores the argument `t` as the payload of Value.
  template <typename T>
  void set(T&& t);

  template <typename T>
  void set(T* t, PointerPayload);

  // Reset the Value object to empty.
  void reset();

  // Check if Value contains a payload.
  bool HasValue() const { return traits_; }

  // Check if Value contains object of type T.
  template <typename T>
  bool IsType() const;

  // MaybeTypeCompatible returns true if the type value stored in this Value
  // instance can be safely cast to `T`.  MaybeTypeCompatible may return true
  // even if the value cannot be safely cast to `T`. However, if it returns
  // false then the value definitely cannot be safely cast to `T`. This means it
  // is useful mainly as a debugging aid for use in assert() etc.

  template <typename T,
            typename std::enable_if<MaybeBase<T>::value>::type* = nullptr>
  bool MaybeTypeCompatible() const;

  template <typename T,
            typename std::enable_if<!MaybeBase<T>::value>::type* = nullptr>
  bool MaybeTypeCompatible() const;

  // Check if object of type T is stored in place.
  template <typename T>
  static constexpr bool IsInPlace() {
    return sizeof(T) <= sizeof(InPlaceStorageT) &&
           alignof(T) <= kInPlaceAlignment && std::is_move_constructible<T>();
  }

 private:
  // DenseHostTensor is 48 bytes. We want to avoid heap allocation for
  // DenseHostTensor and keep the size of Value objects to be the size of a
  // cache line size (64 bytes).
  static constexpr int kInPlaceSize = 48;
  static constexpr int kInPlaceAlignment = 8;

  template <class T>
  friend class internal::InPlaceTypeTraits;

  template <class T>
  friend class internal::OutOfPlaceTypeTraits;

  friend class internal::PointerPayloadTypeTraits;

  template <typename T, typename... Args>
  void fill(Args&&... args);

  // In place storage for the payload
  using InPlaceStorageT =
      std::aligned_storage_t<kInPlaceSize, kInPlaceAlignment>;

  void* value_;  // Always point to the payload.
  const internal::TypeTraits* traits_ = nullptr;
  InPlaceStorageT storage_;
};

// We only optimize the code for 64-bit architectures for now.
static_assert(sizeof(Value) == 64 || sizeof(void*) != 8,
              "Value is not one cache line size");

// -----------------------------------------------------------
// Implementation details.

namespace internal {

template <class T>
struct InPlaceTypeTraits {
  // Clear the payload in `v`. `v` should be non-empty.
  static void Clear(Value* v) {
    assert(v->HasValue());

    T& t = v->get<T>();
    t.~T();
    v->traits_ = nullptr;
  }

  // Move construct `from` to `to`. `to` should be an empty Value and `from`
  // should be a non-empty Value.
  static void MoveConstruct(Value* to, Value* from) {
    assert(!to->HasValue() && from->HasValue());

    T& t = from->get<T>();
    new (&to->storage_) T(std::move(t));
    to->value_ = &to->storage_;
    to->traits_ = from->traits_;

    t.~T();
    from->traits_ = nullptr;
  }
};

template <class T>
struct OutOfPlaceTypeTraits {
  // Clear the payload in `v`. `v` should be non-empty.
  static void Clear(Value* v) {
    assert(v->HasValue());

    T& t = v->get<T>();
    delete &t;
    v->traits_ = nullptr;
  }

  // Move construct `from` to `to`. `to` should be an empty Value and `from`
  // should be a non-empty Value.
  static void MoveConstruct(Value* to, Value* from) {
    assert(!to->HasValue() && from->HasValue());

    T& t = from->get<T>();
    to->value_ = &t;
    to->traits_ = from->traits_;
    from->traits_ = nullptr;
  }
};

struct PointerPayloadTypeTraits {
  // Clear the payload in `v`. `v` should be non-empty.
  static void Clear(Value* v) {
    assert(v->HasValue());
    v->traits_ = nullptr;
  }

  // Move construct `from` to `to`. `to` should be an empty Value and `from`
  // should be a non-empty Value.
  static void MoveConstruct(Value* to, Value* from) {
    assert(!to->HasValue() && from->HasValue());

    to->value_ = from->value_;
    to->traits_ = from->traits_;
    from->traits_ = nullptr;
  }
};

struct TypeTraits {
  using ClearFn = void (*)(Value*);
  using MoveConstructFn = void (*)(Value*, Value*);

  template <typename T>
  TypeTraits(TypeTag<T>) {
    using TypeTraitFns =
        std::conditional_t<Value::IsInPlace<T>(), InPlaceTypeTraits<T>,
                           OutOfPlaceTypeTraits<T>>;
    clear = &TypeTraitFns::Clear;
    move_construct = &TypeTraitFns::MoveConstruct;
    is_polymorphic = std::is_polymorphic<T>::value;
    is_pointer_payload = false;
  }

  template <typename T>
  TypeTraits(TypeTag<T>, Value::PointerPayload) {
    using TypeTraitFns = PointerPayloadTypeTraits;
    clear = &TypeTraitFns::Clear;
    move_construct = &TypeTraitFns::MoveConstruct;
    is_polymorphic = std::is_polymorphic<T>::value;
    is_pointer_payload = true;
  }

  ClearFn clear;
  MoveConstructFn move_construct;
  bool is_polymorphic;
  bool is_pointer_payload;
};

template <typename T>
TypeTraits* GetTypeTraits() {
  static TypeTraits* traits = new TypeTraits(TypeTag<T>());
  return traits;
}

template <typename T>
TypeTraits* GetTypeTraits(Value::PointerPayload) {
  static TypeTraits* traits =
      new TypeTraits(TypeTag<T>(), Value::PointerPayload{});
  return traits;
}

}  // namespace internal

template <typename T>
Value::Value(T&& t) {
  fill<T>(std::forward<T>(t));
}

template <typename T>
Value::Value(T* t, PointerPayload)
    : value_{t}, traits_{internal::GetTypeTraits<T>(PointerPayload{})} {}

inline Value::Value(Value&& v) {
  if (v.HasValue()) v.traits_->move_construct(this, &v);
}

inline Value& Value::operator=(Value&& v) {
  reset();
  if (v.HasValue()) v.traits_->move_construct(this, &v);
  return *this;
}

inline Value::~Value() { reset(); }

template <typename T>
T& Value::get() {
  return const_cast<T&>(static_cast<const Value*>(this)->get<T>());
}

template <typename T>
const T& Value::get() const {
  assert(MaybeTypeCompatible<T>());

  return *static_cast<const T*>(value_);
}

// emplace() constructs the payload object of type T in place with the given
// args.
template <typename T, typename... Args>
void Value::emplace(Args&&... args) {
  reset();
  fill<T>(std::forward<Args>(args)...);
}

template <typename T>
void Value::set(T* t, PointerPayload) {
  reset();
  value_ = t;
  traits_ = internal::GetTypeTraits<T>(PointerPayload{});
}

// set() stores the argument `t` as the payload of Value.
template <typename T>
void Value::set(T&& t) {
  emplace<T>(std::forward<T>(t));
}

template <typename T, typename... Args>
void Value::fill(Args&&... args) {
  traits_ = internal::GetTypeTraits<T>();

  if (IsInPlace<T>()) {
    static_assert(alignof(T) <= kInPlaceAlignment,
                  "Alignment requirement too big for Value");
    new (&storage_) T(std::forward<Args>(args)...);
    value_ = &storage_;
  } else {
    value_ = new T(std::forward<Args>(args)...);
  }
}

// Reset the Value object to empty.
inline void Value::reset() {
  if (!traits_) return;
  traits_->clear(this);
}

template <typename T>
bool Value::IsType() const {
  if (traits_->is_pointer_payload)
    return internal::GetTypeTraits<T>(PointerPayload{}) == traits_;
  else
    return internal::GetTypeTraits<T>() == traits_;
}

template <typename T, typename std::enable_if<MaybeBase<T>::value>::type*>
bool Value::MaybeTypeCompatible() const {
  // We can't do a IsType<T>() in this case because `T` might be an base class.
  // So we conservatively just check the polymorphic-ness are consistent.
  return std::is_polymorphic<T>::value == traits_->is_polymorphic;
}

template <typename T, typename std::enable_if<!MaybeBase<T>::value>::type*>
bool Value::MaybeTypeCompatible() const {
  return IsType<T>() &&
         std::is_polymorphic<T>::value == traits_->is_polymorphic;
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_VALUE_H_
