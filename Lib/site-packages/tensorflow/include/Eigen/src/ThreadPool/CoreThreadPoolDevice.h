// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CORE_THREAD_POOL_DEVICE_H
#define EIGEN_CORE_THREAD_POOL_DEVICE_H

namespace Eigen {

// CoreThreadPoolDevice provides an easy-to-understand Device for parallelizing Eigen Core expressions with
// Threadpool. Expressions are recursively split evenly until the evaluation cost is less than the threshold for
// delegating the task to a thread.
/*
                 a
                / \
               /   \
              /     \
             /       \
            /         \
           /           \
          /             \
         a               e
        / \             / \
       /   \           /   \
      /     \         /     \
     a       c       e       g
    / \     / \     / \     / \
   /   \   /   \   /   \   /   \
  a     b c     d e     f g     h
*/
// Each task descends the binary tree to the left, delegates the right task to a new thread, and continues to the
// left. This ensures that work is evenly distributed to the thread pool as quickly as possible and minimizes the number
// of tasks created during the evaluation. Consider an expression that is divided into 8 chunks. The
// primary task 'a' creates tasks 'e' 'c' and 'b', and executes its portion of the expression at the bottom of the
// tree. Likewise, task 'e' creates tasks 'g' and 'f', and executes its portion of the expression.

struct CoreThreadPoolDevice {
  using Task = std::function<void()>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoreThreadPoolDevice(ThreadPool& pool, float threadCostThreshold = 3e-5f)
      : m_pool(pool) {
    eigen_assert(threadCostThreshold >= 0.0f && "threadCostThreshold must be non-negative");
    m_costFactor = threadCostThreshold;
  }

  template <int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int calculateLevels(Index size, float cost) const {
    eigen_assert(cost >= 0.0f && "cost must be non-negative");
    Index numOps = size / PacketSize;
    int actualThreads = numOps < m_pool.NumThreads() ? static_cast<int>(numOps) : m_pool.NumThreads();
    float totalCost = static_cast<float>(numOps) * cost;
    float idealThreads = totalCost * m_costFactor;
    if (idealThreads < static_cast<float>(actualThreads)) {
      idealThreads = numext::maxi(idealThreads, 1.0f);
      actualThreads = numext::mini(actualThreads, static_cast<int>(idealThreads));
    }
    int maxLevel = internal::log2_ceil(actualThreads);
    return maxLevel;
  }

// MSVC does not like inlining parallelForImpl
#if EIGEN_COMP_MSVC && !EIGEN_COMP_CLANG
#define EIGEN_PARALLEL_FOR_INLINE
#else
#define EIGEN_PARALLEL_FOR_INLINE EIGEN_STRONG_INLINE
#endif

  template <typename UnaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_PARALLEL_FOR_INLINE void parallelForImpl(Index begin, Index end, UnaryFunctor& f,
                                                                   Barrier& barrier, int level) {
    while (level > 0) {
      level--;
      Index size = end - begin;
      eigen_assert(size % PacketSize == 0 && "this function assumes size is a multiple of PacketSize");
      Index mid = begin + numext::round_down(size >> 1, PacketSize);
      Task right = [this, mid, end, &f, &barrier, level]() {
        parallelForImpl<UnaryFunctor, PacketSize>(mid, end, f, barrier, level);
      };
      m_pool.Schedule(std::move(right));
      end = mid;
    }
    for (Index i = begin; i < end; i += PacketSize) f(i);
    barrier.Notify();
  }

  template <typename BinaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_PARALLEL_FOR_INLINE void parallelForImpl(Index outerBegin, Index outerEnd, Index innerBegin,
                                                                   Index innerEnd, BinaryFunctor& f, Barrier& barrier,
                                                                   int level) {
    while (level > 0) {
      level--;
      Index outerSize = outerEnd - outerBegin;
      if (outerSize > 1) {
        Index outerMid = outerBegin + (outerSize >> 1);
        Task right = [this, &f, &barrier, outerMid, outerEnd, innerBegin, innerEnd, level]() {
          parallelForImpl<BinaryFunctor, PacketSize>(outerMid, outerEnd, innerBegin, innerEnd, f, barrier, level);
        };
        m_pool.Schedule(std::move(right));
        outerEnd = outerMid;
      } else {
        Index innerSize = innerEnd - innerBegin;
        eigen_assert(innerSize % PacketSize == 0 && "this function assumes innerSize is a multiple of PacketSize");
        Index innerMid = innerBegin + numext::round_down(innerSize >> 1, PacketSize);
        Task right = [this, &f, &barrier, outerBegin, outerEnd, innerMid, innerEnd, level]() {
          parallelForImpl<BinaryFunctor, PacketSize>(outerBegin, outerEnd, innerMid, innerEnd, f, barrier, level);
        };
        m_pool.Schedule(std::move(right));
        innerEnd = innerMid;
      }
    }
    for (Index outer = outerBegin; outer < outerEnd; outer++)
      for (Index inner = innerBegin; inner < innerEnd; inner += PacketSize) f(outer, inner);
    barrier.Notify();
  }

#undef EIGEN_PARALLEL_FOR_INLINE

  template <typename UnaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void parallelFor(Index begin, Index end, UnaryFunctor& f, float cost) {
    Index size = end - begin;
    int maxLevel = calculateLevels<PacketSize>(size, cost);
    Barrier barrier(1 << maxLevel);
    parallelForImpl<UnaryFunctor, PacketSize>(begin, end, f, barrier, maxLevel);
    barrier.Wait();
  }

  template <typename BinaryFunctor, int PacketSize>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void parallelFor(Index outerBegin, Index outerEnd, Index innerBegin,
                                                         Index innerEnd, BinaryFunctor& f, float cost) {
    Index outerSize = outerEnd - outerBegin;
    Index innerSize = innerEnd - innerBegin;
    Index size = outerSize * innerSize;
    int maxLevel = calculateLevels<PacketSize>(size, cost);
    Barrier barrier(1 << maxLevel);
    parallelForImpl<BinaryFunctor, PacketSize>(outerBegin, outerEnd, innerBegin, innerEnd, f, barrier, maxLevel);
    barrier.Wait();
  }

  ThreadPool& m_pool;
  // costFactor is the cost of delegating a task to a thread
  // the inverse is used to avoid a floating point division
  float m_costFactor;
};

// specialization of coefficient-wise assignment loops for CoreThreadPoolDevice

namespace internal {

template <typename Kernel>
struct cost_helper {
  using SrcEvaluatorType = typename Kernel::SrcEvaluatorType;
  using DstEvaluatorType = typename Kernel::DstEvaluatorType;
  using SrcXprType = typename SrcEvaluatorType::XprType;
  using DstXprType = typename DstEvaluatorType::XprType;
  static constexpr Index Cost = functor_cost<SrcXprType>::Cost + functor_cost<DstXprType>::Cost;
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, CoreThreadPoolDevice, DefaultTraversal, NoUnrolling> {
  static constexpr Index XprEvaluationCost = cost_helper<Kernel>::Cost;
  struct AssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer, Index inner) {
      this->assignCoeffByOuterInner(outer, inner);
    }
  };

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, CoreThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, 1>(0, outerSize, 0, innerSize, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, CoreThreadPoolDevice, DefaultTraversal, InnerUnrolling> {
  using DstXprType = typename Kernel::DstEvaluatorType::XprType;
  static constexpr Index XprEvaluationCost = cost_helper<Kernel>::Cost, InnerSize = DstXprType::InnerSizeAtCompileTime;
  struct AssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer) {
      copy_using_evaluator_DefaultTraversal_InnerUnrolling<Kernel, 0, InnerSize>::run(*this, outer);
    }
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, CoreThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    AssignmentFunctor functor(kernel);
    constexpr float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(InnerSize);
    device.template parallelFor<AssignmentFunctor, 1>(0, outerSize, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, CoreThreadPoolDevice, InnerVectorizedTraversal, NoUnrolling> {
  using PacketType = typename Kernel::PacketType;
  static constexpr Index XprEvaluationCost = cost_helper<Kernel>::Cost, PacketSize = unpacket_traits<PacketType>::size,
                         SrcAlignment = Kernel::AssignmentTraits::SrcAlignment,
                         DstAlignment = Kernel::AssignmentTraits::DstAlignment;
  struct AssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer, Index inner) {
      this->template assignPacketByOuterInner<Unaligned, Unaligned, PacketType>(outer, inner);
    }
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, CoreThreadPoolDevice& device) {
    const Index innerSize = kernel.innerSize();
    const Index outerSize = kernel.outerSize();
    const float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(innerSize);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, PacketSize>(0, outerSize, 0, innerSize, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, CoreThreadPoolDevice, InnerVectorizedTraversal, InnerUnrolling> {
  using PacketType = typename Kernel::PacketType;
  using DstXprType = typename Kernel::DstEvaluatorType::XprType;
  static constexpr Index XprEvaluationCost = cost_helper<Kernel>::Cost, PacketSize = unpacket_traits<PacketType>::size,
                         SrcAlignment = Kernel::AssignmentTraits::SrcAlignment,
                         DstAlignment = Kernel::AssignmentTraits::DstAlignment,
                         InnerSize = DstXprType::InnerSizeAtCompileTime;
  struct AssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer) {
      copy_using_evaluator_innervec_InnerUnrolling<Kernel, 0, InnerSize, SrcAlignment, DstAlignment>::run(*this, outer);
    }
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, CoreThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    constexpr float cost = static_cast<float>(XprEvaluationCost) * static_cast<float>(InnerSize);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, PacketSize>(0, outerSize, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, CoreThreadPoolDevice, SliceVectorizedTraversal, NoUnrolling> {
  using Scalar = typename Kernel::Scalar;
  using PacketType = typename Kernel::PacketType;
  static constexpr Index XprEvaluationCost = cost_helper<Kernel>::Cost, PacketSize = unpacket_traits<PacketType>::size;
  struct PacketAssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketAssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer, Index inner) {
      this->template assignPacketByOuterInner<Unaligned, Unaligned, PacketType>(outer, inner);
    }
  };
  struct ScalarAssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ScalarAssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index outer) {
      const Index innerSize = this->innerSize();
      const Index packetAccessSize = numext::round_down(innerSize, PacketSize);
      for (Index inner = packetAccessSize; inner < innerSize; inner++) this->assignCoeffByOuterInner(outer, inner);
    }
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, CoreThreadPoolDevice& device) {
    const Index outerSize = kernel.outerSize();
    const Index innerSize = kernel.innerSize();
    const Index packetAccessSize = numext::round_down(innerSize, PacketSize);
    constexpr float packetCost = static_cast<float>(XprEvaluationCost);
    const float scalarCost = static_cast<float>(XprEvaluationCost) * static_cast<float>(innerSize - packetAccessSize);
    PacketAssignmentFunctor packetFunctor(kernel);
    ScalarAssignmentFunctor scalarFunctor(kernel);
    device.template parallelFor<PacketAssignmentFunctor, PacketSize>(0, outerSize, 0, packetAccessSize, packetFunctor,
                                                                     packetCost);
    device.template parallelFor<ScalarAssignmentFunctor, 1>(0, outerSize, scalarFunctor, scalarCost);
  };
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, CoreThreadPoolDevice, LinearTraversal, NoUnrolling> {
  static constexpr Index XprEvaluationCost = cost_helper<Kernel>::Cost;
  struct AssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index index) { this->assignCoeff(index); }
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, CoreThreadPoolDevice& device) {
    const Index size = kernel.size();
    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, 1>(0, size, functor, cost);
  }
};

template <typename Kernel>
struct dense_assignment_loop_with_device<Kernel, CoreThreadPoolDevice, LinearVectorizedTraversal, NoUnrolling> {
  using Scalar = typename Kernel::Scalar;
  using PacketType = typename Kernel::PacketType;
  static constexpr Index XprEvaluationCost = cost_helper<Kernel>::Cost,
                         RequestedAlignment = Kernel::AssignmentTraits::LinearRequiredAlignment,
                         PacketSize = unpacket_traits<PacketType>::size,
                         DstIsAligned = Kernel::AssignmentTraits::DstAlignment >= RequestedAlignment,
                         DstAlignment = packet_traits<Scalar>::AlignedOnScalar ? RequestedAlignment
                                                                               : Kernel::AssignmentTraits::DstAlignment,
                         SrcAlignment = Kernel::AssignmentTraits::JointAlignment;
  struct AssignmentFunctor : public Kernel {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AssignmentFunctor(Kernel& kernel) : Kernel(kernel) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(Index index) {
      this->template assignPacket<DstAlignment, SrcAlignment, PacketType>(index);
    }
  };
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Kernel& kernel, CoreThreadPoolDevice& device) {
    const Index size = kernel.size();
    const Index alignedStart =
        DstIsAligned ? 0 : internal::first_aligned<RequestedAlignment>(kernel.dstDataPtr(), size);
    const Index alignedEnd = alignedStart + numext::round_down(size - alignedStart, PacketSize);

    unaligned_dense_assignment_loop<DstIsAligned != 0>::run(kernel, 0, alignedStart);

    constexpr float cost = static_cast<float>(XprEvaluationCost);
    AssignmentFunctor functor(kernel);
    device.template parallelFor<AssignmentFunctor, PacketSize>(alignedStart, alignedEnd, functor, cost);

    unaligned_dense_assignment_loop<>::run(kernel, alignedEnd, size);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_CORE_THREAD_POOL_DEVICE_H
