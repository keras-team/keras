// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SYMBOLIC_INDEX_H
#define EIGEN_SYMBOLIC_INDEX_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

/** \namespace Eigen::symbolic
 * \ingroup Core_Module
 *
 * This namespace defines a set of classes and functions to build and evaluate symbolic expressions of scalar type
 * Index. Here is a simple example:
 *
 * \code
 * // First step, defines symbols:
 * struct x_tag {};  static const symbolic::SymbolExpr<x_tag> x;
 * struct y_tag {};  static const symbolic::SymbolExpr<y_tag> y;
 * struct z_tag {};  static const symbolic::SymbolExpr<z_tag> z;
 *
 * // Defines an expression:
 * auto expr = (x+3)/y+z;
 *
 * // And evaluate it: (c++14)
 * std::cout << expr.eval(x=6,y=3,z=-13) << "\n";
 *
 * \endcode
 *
 * It is currently only used internally to define and manipulate the
 * Eigen::placeholders::last and Eigen::placeholders::lastp1 symbols in
 * Eigen::seq and Eigen::seqN.
 *
 */
namespace symbolic {

template <typename Tag>
class Symbol;
template <typename Tag, typename Type>
class SymbolValue;
template <typename Arg0>
class NegateExpr;
template <typename Arg1, typename Arg2>
class AddExpr;
template <typename Arg1, typename Arg2>
class ProductExpr;
template <typename Arg1, typename Arg2>
class QuotientExpr;
template <typename IndexType = Index>
class ValueExpr;

/** \class BaseExpr
 * \ingroup Core_Module
 * Common base class of any symbolic expressions
 */
template <typename Derived_>
class BaseExpr {
 public:
  using Derived = Derived_;
  constexpr const Derived& derived() const { return *static_cast<const Derived*>(this); }

  /** Evaluate the expression given the \a values of the symbols.
   *
   * \param values defines the values of the symbols, as constructed by SymbolExpr::operator= operator.
   *
   */
  template <typename... Tags, typename... Types>
  constexpr Index eval(const SymbolValue<Tags, Types>&... values) const {
    return derived().eval_impl(values...);
  }

  /** Evaluate the expression at compile time given the \a values of the symbols.
   *
   * If a value is not known at compile-time, returns Eigen::Undefined.
   *
   */
  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time(const SymbolValue<Tags, Types>&...) {
    return Derived::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
  }

  constexpr NegateExpr<Derived> operator-() const { return NegateExpr<Derived>(derived()); }

  constexpr AddExpr<Derived, ValueExpr<>> operator+(Index b) const {
    return AddExpr<Derived, ValueExpr<>>(derived(), b);
  }
  constexpr AddExpr<Derived, ValueExpr<>> operator-(Index a) const {
    return AddExpr<Derived, ValueExpr<>>(derived(), -a);
  }
  constexpr ProductExpr<Derived, ValueExpr<>> operator*(Index a) const {
    return ProductExpr<Derived, ValueExpr<>>(derived(), a);
  }
  constexpr QuotientExpr<Derived, ValueExpr<>> operator/(Index a) const {
    return QuotientExpr<Derived, ValueExpr<>>(derived(), a);
  }

  friend constexpr AddExpr<Derived, ValueExpr<>> operator+(Index a, const BaseExpr& b) {
    return AddExpr<Derived, ValueExpr<>>(b.derived(), a);
  }
  friend constexpr AddExpr<NegateExpr<Derived>, ValueExpr<>> operator-(Index a, const BaseExpr& b) {
    return AddExpr<NegateExpr<Derived>, ValueExpr<>>(-b.derived(), a);
  }
  friend constexpr ProductExpr<ValueExpr<>, Derived> operator*(Index a, const BaseExpr& b) {
    return ProductExpr<ValueExpr<>, Derived>(a, b.derived());
  }
  friend constexpr QuotientExpr<ValueExpr<>, Derived> operator/(Index a, const BaseExpr& b) {
    return QuotientExpr<ValueExpr<>, Derived>(a, b.derived());
  }

  template <int N>
  constexpr AddExpr<Derived, ValueExpr<internal::FixedInt<N>>> operator+(internal::FixedInt<N>) const {
    return AddExpr<Derived, ValueExpr<internal::FixedInt<N>>>(derived(), ValueExpr<internal::FixedInt<N>>());
  }
  template <int N>
  constexpr AddExpr<Derived, ValueExpr<internal::FixedInt<-N>>> operator-(internal::FixedInt<N>) const {
    return AddExpr<Derived, ValueExpr<internal::FixedInt<-N>>>(derived(), ValueExpr<internal::FixedInt<-N>>());
  }
  template <int N>
  constexpr ProductExpr<Derived, ValueExpr<internal::FixedInt<N>>> operator*(internal::FixedInt<N>) const {
    return ProductExpr<Derived, ValueExpr<internal::FixedInt<N>>>(derived(), ValueExpr<internal::FixedInt<N>>());
  }
  template <int N>
  constexpr QuotientExpr<Derived, ValueExpr<internal::FixedInt<N>>> operator/(internal::FixedInt<N>) const {
    return QuotientExpr<Derived, ValueExpr<internal::FixedInt<N>>>(derived(), ValueExpr<internal::FixedInt<N>>());
  }

  template <int N>
  friend constexpr AddExpr<Derived, ValueExpr<internal::FixedInt<N>>> operator+(internal::FixedInt<N>,
                                                                                const BaseExpr& b) {
    return AddExpr<Derived, ValueExpr<internal::FixedInt<N>>>(b.derived(), ValueExpr<internal::FixedInt<N>>());
  }
  template <int N>
  friend constexpr AddExpr<NegateExpr<Derived>, ValueExpr<internal::FixedInt<N>>> operator-(internal::FixedInt<N>,
                                                                                            const BaseExpr& b) {
    return AddExpr<NegateExpr<Derived>, ValueExpr<internal::FixedInt<N>>>(-b.derived(),
                                                                          ValueExpr<internal::FixedInt<N>>());
  }
  template <int N>
  friend constexpr ProductExpr<ValueExpr<internal::FixedInt<N>>, Derived> operator*(internal::FixedInt<N>,
                                                                                    const BaseExpr& b) {
    return ProductExpr<ValueExpr<internal::FixedInt<N>>, Derived>(ValueExpr<internal::FixedInt<N>>(), b.derived());
  }
  template <int N>
  friend constexpr QuotientExpr<ValueExpr<internal::FixedInt<N>>, Derived> operator/(internal::FixedInt<N>,
                                                                                     const BaseExpr& b) {
    return QuotientExpr<ValueExpr<internal::FixedInt<N>>, Derived>(ValueExpr<internal::FixedInt<N>>(), b.derived());
  }

  template <typename OtherDerived>
  constexpr AddExpr<Derived, OtherDerived> operator+(const BaseExpr<OtherDerived>& b) const {
    return AddExpr<Derived, OtherDerived>(derived(), b.derived());
  }

  template <typename OtherDerived>
  constexpr AddExpr<Derived, NegateExpr<OtherDerived>> operator-(const BaseExpr<OtherDerived>& b) const {
    return AddExpr<Derived, NegateExpr<OtherDerived>>(derived(), -b.derived());
  }

  template <typename OtherDerived>
  constexpr ProductExpr<Derived, OtherDerived> operator*(const BaseExpr<OtherDerived>& b) const {
    return ProductExpr<Derived, OtherDerived>(derived(), b.derived());
  }

  template <typename OtherDerived>
  constexpr QuotientExpr<Derived, OtherDerived> operator/(const BaseExpr<OtherDerived>& b) const {
    return QuotientExpr<Derived, OtherDerived>(derived(), b.derived());
  }
};

template <typename T>
struct is_symbolic {
  // BaseExpr has no conversion ctor, so we only have to check whether T can be statically cast to its base class
  // BaseExpr<T>.
  enum { value = internal::is_convertible<T, BaseExpr<T>>::value };
};

// A simple wrapper around an integral value to provide the eval method.
// We could also use a free-function symbolic_eval...
template <typename IndexType>
class ValueExpr : BaseExpr<ValueExpr<IndexType>> {
 public:
  constexpr ValueExpr() = default;
  constexpr ValueExpr(IndexType val) : value_(val) {}
  template <typename... Tags, typename... Types>
  constexpr IndexType eval_impl(const SymbolValue<Tags, Types>&...) const {
    return value_;
  }
  template <typename... Tags, typename... Types>
  static constexpr IndexType eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    return IndexType(Undefined);
  }

 protected:
  IndexType value_;
};

// Specialization for compile-time value,
// It is similar to ValueExpr(N) but this version helps the compiler to generate better code.
template <int N>
class ValueExpr<internal::FixedInt<N>> : public BaseExpr<ValueExpr<internal::FixedInt<N>>> {
 public:
  constexpr ValueExpr() = default;
  constexpr ValueExpr(internal::FixedInt<N>) {}
  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&...) const {
    return Index(N);
  }
  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    return Index(N);
  }
};

/** Represents the actual value of a symbol identified by its tag
 *
 * It is the return type of SymbolValue::operator=, and most of the time this is only way it is used.
 */
template <typename Tag, typename Type>
class SymbolValue : public BaseExpr<SymbolValue<Tag, Type>> {};

template <typename Tag>
class SymbolValue<Tag, Index> : public BaseExpr<SymbolValue<Tag, Index>> {
 public:
  constexpr SymbolValue() = default;

  /** Default constructor from the value \a val */
  constexpr SymbolValue(Index val) : value_(val) {}

  /** \returns the stored value of the symbol */
  constexpr Index value() const { return value_; }

  /** \returns the stored value of the symbol at compile time, or Undefined if not known. */
  static constexpr Index value_at_compile_time() { return Index(Undefined); }

  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&...) const {
    return value();
  }

  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    return value_at_compile_time();
  }

 protected:
  Index value_;
};

template <typename Tag, int N>
class SymbolValue<Tag, internal::FixedInt<N>> : public BaseExpr<SymbolValue<Tag, internal::FixedInt<N>>> {
 public:
  constexpr SymbolValue() = default;

  /** Default constructor from the value \a val */
  constexpr SymbolValue(internal::FixedInt<N>) {}

  /** \returns the stored value of the symbol */
  constexpr Index value() const { return static_cast<Index>(N); }

  /** \returns the stored value of the symbol at compile time, or Undefined if not known. */
  static constexpr Index value_at_compile_time() { return static_cast<Index>(N); }

  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&...) const {
    return value();
  }

  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    return value_at_compile_time();
  }
};

// Find and return a symbol value based on the tag.
template <typename Tag, typename... Types>
struct EvalSymbolValueHelper;

// Empty base case, symbol not found.
template <typename Tag>
struct EvalSymbolValueHelper<Tag> {
  static constexpr Index eval_impl() {
    eigen_assert(false && "Symbol not found.");
    return Index(Undefined);
  }
  static constexpr Index eval_at_compile_time_impl() { return Index(Undefined); }
};

// We found a symbol value matching the provided Tag!
template <typename Tag, typename Type, typename... OtherTypes>
struct EvalSymbolValueHelper<Tag, SymbolValue<Tag, Type>, OtherTypes...> {
  static constexpr Index eval_impl(const SymbolValue<Tag, Type>& symbol, const OtherTypes&...) {
    return symbol.value();
  }
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tag, Type>& symbol, const OtherTypes&...) {
    return symbol.value_at_compile_time();
  }
};

// No symbol value in first value, recursive search starting with next.
template <typename Tag, typename T1, typename... OtherTypes>
struct EvalSymbolValueHelper<Tag, T1, OtherTypes...> {
  static constexpr Index eval_impl(const T1&, const OtherTypes&... values) {
    return EvalSymbolValueHelper<Tag, OtherTypes...>::eval_impl(values...);
  }
  static constexpr Index eval_at_compile_time_impl(const T1&, const OtherTypes&...) {
    return EvalSymbolValueHelper<Tag, OtherTypes...>::eval_at_compile_time_impl(OtherTypes{}...);
  }
};

/** Expression of a symbol uniquely identified by the template parameter type \c tag */
template <typename tag>
class SymbolExpr : public BaseExpr<SymbolExpr<tag>> {
 public:
  /** Alias to the template parameter \c tag */
  typedef tag Tag;

  constexpr SymbolExpr() = default;

  /** Associate the value \a val to the given symbol \c *this, uniquely identified by its \c Tag.
   *
   * The returned object should be passed to ExprBase::eval() to evaluate a given expression with this specified
   * runtime-time value.
   */
  constexpr SymbolValue<Tag, Index> operator=(Index val) const { return SymbolValue<Tag, Index>(val); }

  template <int N>
  constexpr SymbolValue<Tag, internal::FixedInt<N>> operator=(internal::FixedInt<N>) const {
    return SymbolValue<Tag, internal::FixedInt<N>>{internal::FixedInt<N>{}};
  }

  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&... values) const {
    return EvalSymbolValueHelper<Tag, SymbolValue<Tags, Types>...>::eval_impl(values...);
  }

  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    return EvalSymbolValueHelper<Tag, SymbolValue<Tags, Types>...>::eval_at_compile_time_impl(
        SymbolValue<Tags, Types>{}...);
  }
};

template <typename Arg0>
class NegateExpr : public BaseExpr<NegateExpr<Arg0>> {
 public:
  constexpr NegateExpr() = default;
  constexpr NegateExpr(const Arg0& arg0) : m_arg0(arg0) {}

  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&... values) const {
    return -m_arg0.eval_impl(values...);
  }

  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    constexpr Index v = Arg0::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
    return (v == Undefined) ? Undefined : -v;
  }

 protected:
  Arg0 m_arg0;
};

template <typename Arg0, typename Arg1>
class AddExpr : public BaseExpr<AddExpr<Arg0, Arg1>> {
 public:
  constexpr AddExpr() = default;
  constexpr AddExpr(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&... values) const {
    return m_arg0.eval_impl(values...) + m_arg1.eval_impl(values...);
  }

  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    constexpr Index v0 = Arg0::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
    constexpr Index v1 = Arg1::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
    return (v0 == Undefined || v1 == Undefined) ? Undefined : v0 + v1;
  }

 protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

template <typename Arg0, typename Arg1>
class ProductExpr : public BaseExpr<ProductExpr<Arg0, Arg1>> {
 public:
  constexpr ProductExpr() = default;
  constexpr ProductExpr(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&... values) const {
    return m_arg0.eval_impl(values...) * m_arg1.eval_impl(values...);
  }

  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    constexpr Index v0 = Arg0::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
    constexpr Index v1 = Arg1::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
    return (v0 == Undefined || v1 == Undefined) ? Undefined : v0 * v1;
  }

 protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

template <typename Arg0, typename Arg1>
class QuotientExpr : public BaseExpr<QuotientExpr<Arg0, Arg1>> {
 public:
  constexpr QuotientExpr() = default;
  constexpr QuotientExpr(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template <typename... Tags, typename... Types>
  constexpr Index eval_impl(const SymbolValue<Tags, Types>&... values) const {
    return m_arg0.eval_impl(values...) / m_arg1.eval_impl(values...);
  }

  template <typename... Tags, typename... Types>
  static constexpr Index eval_at_compile_time_impl(const SymbolValue<Tags, Types>&...) {
    constexpr Index v0 = Arg0::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
    constexpr Index v1 = Arg1::eval_at_compile_time_impl(SymbolValue<Tags, Types>{}...);
    return (v0 == Undefined || v1 == Undefined) ? Undefined : v0 / v1;
  }

 protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

}  // end namespace symbolic

}  // end namespace Eigen

#endif  // EIGEN_SYMBOLIC_INDEX_H
