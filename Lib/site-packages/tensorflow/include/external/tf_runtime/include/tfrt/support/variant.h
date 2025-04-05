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

// This file implements the variant data structure similar to std::variant in
// C++17.
#ifndef TFRT_SUPPORT_VARIANT_H_
#define TFRT_SUPPORT_VARIANT_H_

#include <tuple>
#include <type_traits>

#include "tfrt/support/type_traits.h"

namespace tfrt {

// A Variant similar to std::variant in C++17.
//
// Example usage:
//
// Variant<int, float, double> v;
//
// v = 1;
// assert(v.get<int>() == 1);
// assert(v.is<int>());
// assert(v.get_if<float>() == nullptr);
//
// // Print the variant.
// visit([](auto& t) { std::cout << t; }, v);
//
// v.emplace<float>(3);
//
template <typename... Ts>
class Variant {
  // Convenient constant to check if a type is a variant.
  template <typename T>
  static constexpr bool IsVariant =
      std::is_same<std::decay_t<T>, Variant>::value;

  using Types = std::tuple<Ts...>;

  // Convenient constant to check if T is part of the Variant.
  template <typename T>
  static constexpr bool HasType = TupleHasType<T, Types>::value;

 public:
  using IndexT = int8_t;
  template <int N>
  using TypeOf = typename std::tuple_element<N, Types>::type;
  static constexpr size_t kNTypes = sizeof...(Ts);

  // Default constructor sets the Variant to the default constructed fisrt type.
  Variant() {
    using Type0 = TypeOf<0>;
    index_ = 0;
    new (&storage_) Type0();
  }

  // Support implicit conversion from T to Variant.
  template <typename T, std::enable_if_t<
                            !IsVariant<T> && HasType<std::decay_t<T>>, int> = 0>
  Variant(T&& t) {
    fillValue(std::forward<T>(t));
  }

  // TODO(b/187739825): tfrt::Variant should have trivial constructors when Ts
  // are trivial types.
  Variant(const Variant& v) {
    visit([this](auto&& t) { this->fillValue(t); }, v);
  }

  Variant(Variant&& v) {
    visit([this](auto&& t) { this->fillValue(std::move(t)); }, v);
  }

  ~Variant() { destroy(); }

  Variant& operator=(Variant&& v) {
    visit([this](auto&& t) { *this = std::move(t); }, v);
    return *this;
  }

  Variant& operator=(const Variant& v) {
    visit([this](auto&& t) { *this = t; }, v);
    return *this;
  }

  template <typename T, std::enable_if_t<!IsVariant<T>, int> = 0>
  Variant& operator=(T&& t) {
    destroy();
    fillValue(std::forward<T>(t));

    return *this;
  }

  template <typename T, typename... Args>
  T& emplace(Args&&... args) {
    AssertHasType<T>();

    destroy();
    index_ = IndexOf<T>;
    auto* t = new (&storage_) T(std::forward<Args>(args)...);
    return *t;
  }

  template <typename T>
  bool is() const {
    AssertHasType<T>();
    return IndexOf<T> == index_;
  }

  template <typename T>
  const T& get() const {
    AssertHasType<T>();
    return *reinterpret_cast<const T*>(&storage_);
  }

  template <typename T>
  T& get() {
    AssertHasType<T>();
    return *reinterpret_cast<T*>(&storage_);
  }

  template <typename T>
  const T* get_if() const {
    if (is<T>()) return &get<T>();
    return nullptr;
  }

  template <typename T>
  T* get_if() {
    if (is<T>()) return &get<T>();
    return nullptr;
  }

 private:
  template <typename T>
  static constexpr size_t IndexOf = TupleIndexOf<T, Types>::value;

  static constexpr size_t kStorageSize = std::max({sizeof(Ts)...});
  static constexpr size_t kAlignment = std::max({alignof(Ts)...});

  template <typename T>
  static constexpr void AssertHasType() {
    static_assert(HasType<T>, "Invalid Type used for Variant");
  }

  void destroy() {
    visit(
        [](auto&& t) {
          using T = std::decay_t<decltype(t)>;
          t.~T();
        },
        *this);
  }

  template <typename T>
  void fillValue(T&& t) {
    using Type = std::decay_t<T>;
    AssertHasType<Type>();

    index_ = IndexOf<Type>;
    new (&storage_) Type(std::forward<T>(t));
  }

  using StorageT = std::aligned_storage_t<kStorageSize, kAlignment>;

  StorageT storage_;
  IndexT index_ = -1;
};

struct Monostate {};

namespace internal {

// The return type when applying visitor of type `F` to a variant of type
// `Variant`. `F()` should return the same type for all alternative types in the
// variant.  We use the return value of `F()(FirstAlternativeType)` as the
// return type.
template <typename F, typename Variant>
using VariantVisitorResultT = decltype(std::declval<std::decay_t<F>>()(
    std::declval<typename std::decay_t<Variant>::template TypeOf<0>&>()));

template <typename F, typename Variant>
auto visitHelper(F&& f, Variant&& v,
                 std::integral_constant<int, std::decay_t<Variant>::kNTypes>)
    -> VariantVisitorResultT<F, Variant> {
  assert(false && "Unexpected index_ in Variant");
}

template <typename F, typename Variant, int N,
          std::enable_if_t<N<std::decay_t<Variant>::kNTypes, int> = 0> auto
              visitHelper(F&& f, Variant&& v, std::integral_constant<int, N>)
                  ->VariantVisitorResultT<F, Variant> {
  using VariantT = std::decay_t<Variant>;
  using T = typename VariantT::template TypeOf<N>;
  if (auto* t = v.template get_if<T>()) {
    return f(*t);
  } else {
    return visitHelper(std::forward<F>(f), std::forward<Variant>(v),
                       std::integral_constant<int, N + 1>());
  }
}

}  // namespace internal

template <typename F, typename Variant>
auto visit(F&& f, Variant&& v) -> internal::VariantVisitorResultT<F, Variant> {
  return internal::visitHelper(std::forward<F>(f), std::forward<Variant>(v),
                               std::integral_constant<int, 0>());
}

}  // namespace tfrt

#endif  // TFRT_SUPPORT_VARIANT_H_
