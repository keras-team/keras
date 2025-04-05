// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 The Eigen Team.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COHERENT_PAD_OP_H
#define EIGEN_COHERENT_PAD_OP_H

#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Pads a vector with zeros to a given size.
template <typename XprType, int SizeAtCompileTime_>
struct CoherentPadOp;

template <typename XprType, int SizeAtCompileTime_>
struct traits<CoherentPadOp<XprType, SizeAtCompileTime_>> : public traits<XprType> {
  typedef typename internal::remove_all<XprType>::type PlainXprType;
  typedef typename internal::ref_selector<XprType>::type XprNested;
  typedef typename std::remove_reference_t<XprNested> XprNested_;
  enum : int {
    IsRowMajor = traits<PlainXprType>::Flags & RowMajorBit,
    SizeAtCompileTime = SizeAtCompileTime_,
    RowsAtCompileTime = IsRowMajor ? 1 : SizeAtCompileTime,
    ColsAtCompileTime = IsRowMajor ? SizeAtCompileTime : 1,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,
    Flags = traits<XprType>::Flags & ~NestByRefBit,
  };
};

// Pads a vector with zeros to a given size.
template <typename XprType, int SizeAtCompileTime_>
struct CoherentPadOp : public dense_xpr_base<CoherentPadOp<XprType, SizeAtCompileTime_>>::type {
  typedef typename internal::generic_xpr_base<CoherentPadOp<XprType, SizeAtCompileTime_>>::type Base;
  EIGEN_GENERIC_PUBLIC_INTERFACE(CoherentPadOp)

  using XprNested = typename traits<CoherentPadOp>::XprNested;
  using XprNested_ = typename traits<CoherentPadOp>::XprNested_;
  using NestedExpression = XprNested_;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoherentPadOp() = delete;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoherentPadOp(const CoherentPadOp&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoherentPadOp(CoherentPadOp&& other) = default;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoherentPadOp(const XprType& xpr, Index size) : xpr_(xpr), size_(size) {
    static_assert(XprNested_::IsVectorAtCompileTime, "input type must be a vector");
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const XprNested_& nestedExpression() const { return xpr_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return size_.value(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index rows() const {
    return traits<CoherentPadOp>::IsRowMajor ? Index(1) : size();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index cols() const {
    return traits<CoherentPadOp>::IsRowMajor ? size() : Index(1);
  }

 private:
  XprNested xpr_;
  const internal::variable_if_dynamic<Index, SizeAtCompileTime> size_;
};

// Adapted from the Replicate evaluator.
template <typename ArgType, int SizeAtCompileTime>
struct unary_evaluator<CoherentPadOp<ArgType, SizeAtCompileTime>>
    : evaluator_base<CoherentPadOp<ArgType, SizeAtCompileTime>> {
  typedef CoherentPadOp<ArgType, SizeAtCompileTime> XprType;
  typedef typename internal::remove_all_t<typename XprType::CoeffReturnType> CoeffReturnType;
  typedef typename internal::nested_eval<ArgType, 1>::type ArgTypeNested;
  typedef internal::remove_all_t<ArgTypeNested> ArgTypeNestedCleaned;

  enum {
    CoeffReadCost = evaluator<ArgTypeNestedCleaned>::CoeffReadCost,
    LinearAccessMask = XprType::IsVectorAtCompileTime ? LinearAccessBit : 0,
    Flags = evaluator<ArgTypeNestedCleaned>::Flags & (HereditaryBits | LinearAccessMask | RowMajorBit),
    Alignment = evaluator<ArgTypeNestedCleaned>::Alignment
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit unary_evaluator(const XprType& pad)
      : m_arg(pad.nestedExpression()), m_argImpl(m_arg), m_size(pad.nestedExpression().size()) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index row, Index col) const {
    EIGEN_IF_CONSTEXPR(XprType::IsRowMajor) {
      if (col < m_size.value()) {
        return m_argImpl.coeff(1, col);
      }
    }
    else {
      if (row < m_size.value()) {
        return m_argImpl.coeff(row, 1);
      }
    }
    return CoeffReturnType(0);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    if (index < m_size.value()) {
      return m_argImpl.coeff(index);
    }
    return CoeffReturnType(0);
  }

  template <int LoadMode, typename PacketType>
  EIGEN_STRONG_INLINE PacketType packet(Index row, Index col) const {
    // AutoDiff scalar's derivative must be a vector, which is enforced by static assert.
    // Defer to linear access for simplicity.
    EIGEN_IF_CONSTEXPR(XprType::IsRowMajor) { return packet(col); }
    return packet(row);
  }

  template <int LoadMode, typename PacketType>
  EIGEN_STRONG_INLINE PacketType packet(Index index) const {
    constexpr int kPacketSize = unpacket_traits<PacketType>::size;
    if (index + kPacketSize <= m_size.value()) {
      return m_argImpl.template packet<LoadMode, PacketType>(index);
    } else if (index < m_size.value()) {
      // Partial packet.
      EIGEN_ALIGN_MAX std::remove_const_t<CoeffReturnType> values[kPacketSize];
      const int partial = m_size.value() - index;
      for (int i = 0; i < partial && i < kPacketSize; ++i) {
        values[i] = m_argImpl.coeff(index + i);
      }
      for (int i = partial; i < kPacketSize; ++i) {
        values[i] = CoeffReturnType(0);
      }
      return pload<PacketType>(values);
    }
    return pset1<PacketType>(CoeffReturnType(0));
  }

 protected:
  const ArgTypeNested m_arg;
  evaluator<ArgTypeNestedCleaned> m_argImpl;
  const variable_if_dynamic<Index, ArgTypeNestedCleaned::SizeAtCompileTime> m_size;
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_CWISE_BINARY_OP_H
