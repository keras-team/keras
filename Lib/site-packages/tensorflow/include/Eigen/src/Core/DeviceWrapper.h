// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DEVICEWRAPPER_H
#define EIGEN_DEVICEWRAPPER_H

namespace Eigen {
template <typename Derived, typename Device>
struct DeviceWrapper {
  using Base = EigenBase<internal::remove_all_t<Derived>>;
  using Scalar = typename Derived::Scalar;

  EIGEN_DEVICE_FUNC DeviceWrapper(Base& xpr, Device& device) : m_xpr(xpr.derived()), m_device(device) {}
  EIGEN_DEVICE_FUNC DeviceWrapper(const Base& xpr, Device& device) : m_xpr(xpr.derived()), m_device(device) {}

  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator=(const EigenBase<OtherDerived>& other) {
    using AssignOp = internal::assign_op<Scalar, typename OtherDerived::Scalar>;
    internal::call_assignment(*this, other.derived(), AssignOp());
    return m_xpr;
  }
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator+=(const EigenBase<OtherDerived>& other) {
    using AddAssignOp = internal::add_assign_op<Scalar, typename OtherDerived::Scalar>;
    internal::call_assignment(*this, other.derived(), AddAssignOp());
    return m_xpr;
  }
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator-=(const EigenBase<OtherDerived>& other) {
    using SubAssignOp = internal::sub_assign_op<Scalar, typename OtherDerived::Scalar>;
    internal::call_assignment(*this, other.derived(), SubAssignOp());
    return m_xpr;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& derived() { return m_xpr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Device& device() { return m_device; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE NoAlias<DeviceWrapper, EigenBase> noalias() {
    return NoAlias<DeviceWrapper, EigenBase>(*this);
  }

  Derived& m_xpr;
  Device& m_device;
};

namespace internal {

// this is where we differentiate between lazy assignment and specialized kernels (e.g. matrix products)
template <typename DstXprType, typename SrcXprType, typename Functor, typename Device,
          typename Kind = typename AssignmentKind<typename evaluator_traits<DstXprType>::Shape,
                                                  typename evaluator_traits<SrcXprType>::Shape>::Kind,
          typename EnableIf = void>
struct AssignmentWithDevice;

// unless otherwise specified, use the default product implementation
template <typename DstXprType, typename Lhs, typename Rhs, int Options, typename Functor, typename Device,
          typename Weak>
struct AssignmentWithDevice<DstXprType, Product<Lhs, Rhs, Options>, Functor, Device, Dense2Dense, Weak> {
  using SrcXprType = Product<Lhs, Rhs, Options>;
  using Base = Assignment<DstXprType, SrcXprType, Functor>;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(DstXprType& dst, const SrcXprType& src, const Functor& func,
                                                        Device&) {
    Base::run(dst, src, func);
  };
};

// specialization for coeffcient-wise assignment
template <typename DstXprType, typename SrcXprType, typename Functor, typename Device, typename Weak>
struct AssignmentWithDevice<DstXprType, SrcXprType, Functor, Device, Dense2Dense, Weak> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(DstXprType& dst, const SrcXprType& src, const Functor& func,
                                                        Device& device) {
#ifndef EIGEN_NO_DEBUG
    internal::check_for_aliasing(dst, src);
#endif

    call_dense_assignment_loop(dst, src, func, device);
  }
};

// this allows us to use the default evaulation scheme if it is not specialized for the device
template <typename Kernel, typename Device, int Traversal = Kernel::AssignmentTraits::Traversal,
          int Unrolling = Kernel::AssignmentTraits::Unrolling>
struct dense_assignment_loop_with_device {
  using Base = dense_assignment_loop<Kernel, Traversal, Unrolling>;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void run(Kernel& kernel, Device&) { Base::run(kernel); }
};

// entry point for a generic expression with device
template <typename Dst, typename Src, typename Func, typename Device>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment_no_alias(DeviceWrapper<Dst, Device> dst,
                                                                                    const Src& src, const Func& func) {
  enum {
    NeedToTranspose = ((int(Dst::RowsAtCompileTime) == 1 && int(Src::ColsAtCompileTime) == 1) ||
                       (int(Dst::ColsAtCompileTime) == 1 && int(Src::RowsAtCompileTime) == 1)) &&
                      int(Dst::SizeAtCompileTime) != 1
  };

  using ActualDstTypeCleaned = std::conditional_t<NeedToTranspose, Transpose<Dst>, Dst>;
  using ActualDstType = std::conditional_t<NeedToTranspose, Transpose<Dst>, Dst&>;
  ActualDstType actualDst(dst.derived());

  // TODO check whether this is the right place to perform these checks:
  EIGEN_STATIC_ASSERT_LVALUE(Dst)
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(ActualDstTypeCleaned, Src)
  EIGEN_CHECK_BINARY_COMPATIBILIY(Func, typename ActualDstTypeCleaned::Scalar, typename Src::Scalar);

  // this provides a mechanism for specializing simple assignments, matrix products, etc
  AssignmentWithDevice<ActualDstTypeCleaned, Src, Func, Device>::run(actualDst, src, func, dst.device());
}

// copy and pasted from AssignEvaluator except forward device to kernel
template <typename DstXprType, typename SrcXprType, typename Functor, typename Device>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_dense_assignment_loop(DstXprType& dst,
                                                                                      const SrcXprType& src,
                                                                                      const Functor& func,
                                                                                      Device& device) {
  using DstEvaluatorType = evaluator<DstXprType>;
  using SrcEvaluatorType = evaluator<SrcXprType>;

  SrcEvaluatorType srcEvaluator(src);

  // NOTE To properly handle A = (A*A.transpose())/s with A rectangular,
  // we need to resize the destination after the source evaluator has been created.
  resize_if_allowed(dst, src, func);

  DstEvaluatorType dstEvaluator(dst);

  using Kernel = generic_dense_assignment_kernel<DstEvaluatorType, SrcEvaluatorType, Functor>;

  Kernel kernel(dstEvaluator, srcEvaluator, func, dst.const_cast_derived());

  dense_assignment_loop_with_device<Kernel, Device>::run(kernel, device);
}

}  // namespace internal

template <typename Derived>
template <typename Device>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceWrapper<Derived, Device> EigenBase<Derived>::device(Device& device) {
  return DeviceWrapper<Derived, Device>(derived(), device);
}

template <typename Derived>
template <typename Device>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceWrapper<const Derived, Device> EigenBase<Derived>::device(
    Device& device) const {
  return DeviceWrapper<const Derived, Device>(derived(), device);
}
}  // namespace Eigen
#endif