// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TUPLE_GPU
#define EIGEN_TUPLE_GPU

#include <type_traits>
#include <utility>

// This is a replacement of std::tuple that can be used in device code.

namespace Eigen {
namespace internal {
namespace tuple_impl {

// Internal tuple implementation.
template <size_t N, typename... Types>
class TupleImpl;

// Generic recursive tuple.
template <size_t N, typename T1, typename... Ts>
class TupleImpl<N, T1, Ts...> {
 public:
  // Tuple may contain Eigen types.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor, enable if all types are default-constructible.
  template <typename U1 = T1,
            typename EnableIf = std::enable_if_t<std::is_default_constructible<U1>::value &&
                                                 reduce_all<std::is_default_constructible<Ts>::value...>::value>>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC TupleImpl() : head_{}, tail_{} {}

  // Element constructor.
  template <typename U1, typename... Us,
            // Only enable if...
            typename EnableIf = std::enable_if_t<
                // the number of input arguments match, and ...
                sizeof...(Us) == sizeof...(Ts) && (
                                                      // this does not look like a copy/move constructor.
                                                      N > 1 || std::is_convertible<U1, T1>::value)>>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC TupleImpl(U1&& arg1, Us&&... args)
      : head_(std::forward<U1>(arg1)), tail_(std::forward<Us>(args)...) {}

  // The first stored value.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T1& head() { return head_; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE const T1& head() const { return head_; }

  // The tail values.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE TupleImpl<N - 1, Ts...>& tail() { return tail_; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE const TupleImpl<N - 1, Ts...>& tail() const { return tail_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(TupleImpl& other) {
    using numext::swap;
    swap(head_, other.head_);
    swap(tail_, other.tail_);
  }

  template <typename... UTypes>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TupleImpl& operator=(const TupleImpl<N, UTypes...>& other) {
    head_ = other.head_;
    tail_ = other.tail_;
    return *this;
  }

  template <typename... UTypes>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TupleImpl& operator=(TupleImpl<N, UTypes...>&& other) {
    head_ = std::move(other.head_);
    tail_ = std::move(other.tail_);
    return *this;
  }

 private:
  // Allow related tuples to reference head_/tail_.
  template <size_t M, typename... UTypes>
  friend class TupleImpl;

  T1 head_;
  TupleImpl<N - 1, Ts...> tail_;
};

// Empty tuple specialization.
template <>
class TupleImpl<size_t(0)> {};

template <typename TupleType>
struct is_tuple : std::false_type {};

template <typename... Types>
struct is_tuple<TupleImpl<sizeof...(Types), Types...>> : std::true_type {};

// Gets an element from a tuple.
template <size_t Idx, typename T1, typename... Ts>
struct tuple_get_impl {
  using TupleType = TupleImpl<sizeof...(Ts) + 1, T1, Ts...>;
  using ReturnType = typename tuple_get_impl<Idx - 1, Ts...>::ReturnType;

  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ReturnType& run(TupleType& tuple) {
    return tuple_get_impl<Idx - 1, Ts...>::run(tuple.tail());
  }

  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE const ReturnType& run(const TupleType& tuple) {
    return tuple_get_impl<Idx - 1, Ts...>::run(tuple.tail());
  }
};

// Base case, getting the head element.
template <typename T1, typename... Ts>
struct tuple_get_impl<0, T1, Ts...> {
  using TupleType = TupleImpl<sizeof...(Ts) + 1, T1, Ts...>;
  using ReturnType = T1;

  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T1& run(TupleType& tuple) { return tuple.head(); }

  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE const T1& run(const TupleType& tuple) {
    return tuple.head();
  }
};

// Concatenates N Tuples.
template <size_t NTuples, typename... Tuples>
struct tuple_cat_impl;

template <size_t NTuples, size_t N1, typename... Args1, size_t N2, typename... Args2, typename... Tuples>
struct tuple_cat_impl<NTuples, TupleImpl<N1, Args1...>, TupleImpl<N2, Args2...>, Tuples...> {
  using TupleType1 = TupleImpl<N1, Args1...>;
  using TupleType2 = TupleImpl<N2, Args2...>;
  using MergedTupleType = TupleImpl<N1 + N2, Args1..., Args2...>;

  using ReturnType = typename tuple_cat_impl<NTuples - 1, MergedTupleType, Tuples...>::ReturnType;

  // Uses the index sequences to extract and merge elements from tuple1 and tuple2,
  // then recursively calls again.
  template <typename Tuple1, size_t... I1s, typename Tuple2, size_t... I2s, typename... MoreTuples>
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ReturnType run(Tuple1&& tuple1,
                                                                              std::index_sequence<I1s...>,
                                                                              Tuple2&& tuple2,
                                                                              std::index_sequence<I2s...>,
                                                                              MoreTuples&&... tuples) {
    return tuple_cat_impl<NTuples - 1, MergedTupleType, Tuples...>::run(
        MergedTupleType(tuple_get_impl<I1s, Args1...>::run(std::forward<Tuple1>(tuple1))...,
                        tuple_get_impl<I2s, Args2...>::run(std::forward<Tuple2>(tuple2))...),
        std::forward<MoreTuples>(tuples)...);
  }

  // Concatenates the first two tuples.
  template <typename Tuple1, typename Tuple2, typename... MoreTuples>
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ReturnType run(Tuple1&& tuple1, Tuple2&& tuple2,
                                                                              MoreTuples&&... tuples) {
    return run(std::forward<Tuple1>(tuple1), std::make_index_sequence<N1>{}, std::forward<Tuple2>(tuple2),
               std::make_index_sequence<N2>{}, std::forward<MoreTuples>(tuples)...);
  }
};

// Base case with a single tuple.
template <size_t N, typename... Args>
struct tuple_cat_impl<1, TupleImpl<N, Args...>> {
  using ReturnType = TupleImpl<N, Args...>;

  template <typename Tuple1>
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ReturnType run(Tuple1&& tuple1) {
    return tuple1;
  }
};

// Special case of no tuples.
template <>
struct tuple_cat_impl<0> {
  using ReturnType = TupleImpl<0>;
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ReturnType run() { return ReturnType{}; }
};

// For use in make_tuple, unwraps a reference_wrapper.
template <typename T>
struct unwrap_reference_wrapper {
  using type = T;
};

template <typename T>
struct unwrap_reference_wrapper<std::reference_wrapper<T>> {
  using type = T&;
};

// For use in make_tuple, decays a type and unwraps a reference_wrapper.
template <typename T>
struct unwrap_decay {
  using type = typename unwrap_reference_wrapper<typename std::decay<T>::type>::type;
};

/**
 * Utility for determining a tuple's size.
 */
template <typename Tuple>
struct tuple_size;

template <typename... Types>
struct tuple_size<TupleImpl<sizeof...(Types), Types...>> : std::integral_constant<size_t, sizeof...(Types)> {};

/**
 * Gets an element of a tuple.
 * \tparam Idx index of the element.
 * \tparam Types ... tuple element types.
 * \param tuple the tuple.
 * \return a reference to the desired element.
 */
template <size_t Idx, typename... Types>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const typename tuple_get_impl<Idx, Types...>::ReturnType& get(
    const TupleImpl<sizeof...(Types), Types...>& tuple) {
  return tuple_get_impl<Idx, Types...>::run(tuple);
}

template <size_t Idx, typename... Types>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename tuple_get_impl<Idx, Types...>::ReturnType& get(
    TupleImpl<sizeof...(Types), Types...>& tuple) {
  return tuple_get_impl<Idx, Types...>::run(tuple);
}

/**
 * Concatenate multiple tuples.
 * \param tuples ... list of tuples.
 * \return concatenated tuple.
 */
template <typename... Tuples, typename EnableIf = std::enable_if_t<
                                  internal::reduce_all<is_tuple<typename std::decay<Tuples>::type>::value...>::value>>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    typename tuple_cat_impl<sizeof...(Tuples), typename std::decay<Tuples>::type...>::ReturnType
    tuple_cat(Tuples&&... tuples) {
  return tuple_cat_impl<sizeof...(Tuples), typename std::decay<Tuples>::type...>::run(std::forward<Tuples>(tuples)...);
}

/**
 * Tie arguments together into a tuple.
 */
template <typename... Args, typename ReturnType = TupleImpl<sizeof...(Args), Args&...>>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ReturnType tie(Args&... args) EIGEN_NOEXCEPT {
  return ReturnType{args...};
}

/**
 * Create a tuple of l-values with the supplied arguments.
 */
template <typename... Args, typename ReturnType = TupleImpl<sizeof...(Args), typename unwrap_decay<Args>::type...>>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ReturnType make_tuple(Args&&... args) {
  return ReturnType{std::forward<Args>(args)...};
}

/**
 * Forward a set of arguments as a tuple.
 */
template <typename... Args>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TupleImpl<sizeof...(Args), Args...> forward_as_tuple(
    Args&&... args) {
  return TupleImpl<sizeof...(Args), Args...>(std::forward<Args>(args)...);
}

/**
 * Alternative to std::tuple that can be used on device.
 */
template <typename... Types>
using tuple = TupleImpl<sizeof...(Types), Types...>;

}  // namespace tuple_impl
}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_TUPLE_GPU
