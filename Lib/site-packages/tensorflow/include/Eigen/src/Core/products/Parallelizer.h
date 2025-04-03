// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARALLELIZER_H
#define EIGEN_PARALLELIZER_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// Note that in the following, there are 3 different uses of the concept
// "number of threads":
//  1. Max number of threads used by OpenMP or ThreadPool.
//     * For OpenMP this is typically the value set by the OMP_NUM_THREADS
//       environment variable, or by a call to omp_set_num_threads() prior to
//       calling Eigen.
//     * For ThreadPool, this is the number of threads in the ThreadPool.
//  2. Max number of threads currently allowed to be used by parallel Eigen
//     operations. This is set by setNbThreads(), and cannot exceed the value
//     in 1.
//  3. The actual number of threads used for a given parallel Eigen operation.
//     This is typically computed on the fly using a cost model and cannot exceed
//     the value in 2.
//     * For OpenMP, this is typically the number of threads specified in individual
//       "omp parallel" pragmas associated with an Eigen operation.
//     * For ThreadPool, it is the number of concurrent tasks scheduled in the
//       threadpool for a given Eigen operation. Notice that since the threadpool
//       uses task stealing, there is no way to limit the number of concurrently
//       executing tasks to below the number in 1. except by limiting the total
//       number of tasks in flight.

#if defined(EIGEN_HAS_OPENMP) && defined(EIGEN_GEMM_THREADPOOL)
#error "EIGEN_HAS_OPENMP and EIGEN_GEMM_THREADPOOL may not both be defined."
#endif

namespace Eigen {

namespace internal {
inline void manage_multi_threading(Action action, int* v);
}

// Public APIs.

/** Must be call first when calling Eigen from multiple threads */
EIGEN_DEPRECATED inline void initParallel() {}

/** \returns the max number of threads reserved for Eigen
 * \sa setNbThreads */
inline int nbThreads() {
  int ret;
  internal::manage_multi_threading(GetAction, &ret);
  return ret;
}

/** Sets the max number of threads reserved for Eigen
 * \sa nbThreads */
inline void setNbThreads(int v) { internal::manage_multi_threading(SetAction, &v); }

#ifdef EIGEN_GEMM_THREADPOOL
// Sets the ThreadPool used by Eigen parallel Gemm.
//
// NOTICE: This function has a known race condition with
// parallelize_gemm below, and should not be called while
// an instance of that function is running.
//
// TODO(rmlarsen): Make the device API available instead of
// storing a local static pointer variable to avoid this issue.
inline ThreadPool* setGemmThreadPool(ThreadPool* new_pool) {
  static ThreadPool* pool;
  if (new_pool != nullptr) {
    // This will wait for work in all threads in *pool to finish,
    // then destroy the old ThreadPool, and then replace it with new_pool.
    pool = new_pool;
    // Reset the number of threads to the number of threads on the new pool.
    setNbThreads(pool->NumThreads());
  }
  return pool;
}

// Gets the ThreadPool used by Eigen parallel Gemm.
inline ThreadPool* getGemmThreadPool() { return setGemmThreadPool(nullptr); }
#endif

namespace internal {

// Implementation.

#if defined(EIGEN_USE_BLAS) || (!defined(EIGEN_HAS_OPENMP) && !defined(EIGEN_GEMM_THREADPOOL))

inline void manage_multi_threading(Action action, int* v) {
  if (action == SetAction) {
    eigen_internal_assert(v != nullptr);
  } else if (action == GetAction) {
    eigen_internal_assert(v != nullptr);
    *v = 1;
  } else {
    eigen_internal_assert(false);
  }
}
template <typename Index>
struct GemmParallelInfo {};
template <bool Condition, typename Functor, typename Index>
EIGEN_STRONG_INLINE void parallelize_gemm(const Functor& func, Index rows, Index cols, Index /*unused*/,
                                          bool /*unused*/) {
  func(0, rows, 0, cols);
}

#else

template <typename Index>
struct GemmParallelTaskInfo {
  GemmParallelTaskInfo() : sync(-1), users(0), lhs_start(0), lhs_length(0) {}
  std::atomic<Index> sync;
  std::atomic<int> users;
  Index lhs_start;
  Index lhs_length;
};

template <typename Index>
struct GemmParallelInfo {
  const int logical_thread_id;
  const int num_threads;
  GemmParallelTaskInfo<Index>* task_info;

  GemmParallelInfo(int logical_thread_id_, int num_threads_, GemmParallelTaskInfo<Index>* task_info_)
      : logical_thread_id(logical_thread_id_), num_threads(num_threads_), task_info(task_info_) {}
};

inline void manage_multi_threading(Action action, int* v) {
  static int m_maxThreads = -1;
  if (action == SetAction) {
    eigen_internal_assert(v != nullptr);
#if defined(EIGEN_HAS_OPENMP)
    // Calling action == SetAction and *v = 0 means
    // restoring m_maxThreads to the maximum number of threads specified
    // for OpenMP.
    eigen_internal_assert(*v >= 0);
    int omp_threads = omp_get_max_threads();
    m_maxThreads = (*v == 0 ? omp_threads : std::min(*v, omp_threads));
#elif defined(EIGEN_GEMM_THREADPOOL)
    // Calling action == SetAction and *v = 0 means
    // restoring m_maxThreads to the number of threads in the ThreadPool,
    // which defaults to 1 if no pool was provided.
    eigen_internal_assert(*v >= 0);
    ThreadPool* pool = getGemmThreadPool();
    int pool_threads = pool != nullptr ? pool->NumThreads() : 1;
    m_maxThreads = (*v == 0 ? pool_threads : numext::mini(pool_threads, *v));
#endif
  } else if (action == GetAction) {
    eigen_internal_assert(v != nullptr);
    *v = m_maxThreads;
  } else {
    eigen_internal_assert(false);
  }
}

template <bool Condition, typename Functor, typename Index>
EIGEN_STRONG_INLINE void parallelize_gemm(const Functor& func, Index rows, Index cols, Index depth, bool transpose) {
  // Dynamically check whether we should even try to execute in parallel.
  // The conditions are:
  // - the max number of threads we can create is greater than 1
  // - we are not already in a parallel code
  // - the sizes are large enough

  // compute the maximal number of threads from the size of the product:
  // This first heuristic takes into account that the product kernel is fully optimized when working with nr columns at
  // once.
  Index size = transpose ? rows : cols;
  Index pb_max_threads = std::max<Index>(1, size / Functor::Traits::nr);

  // compute the maximal number of threads from the total amount of work:
  double work = static_cast<double>(rows) * static_cast<double>(cols) * static_cast<double>(depth);
  double kMinTaskSize = 50000;  // FIXME improve this heuristic.
  pb_max_threads = std::max<Index>(1, std::min<Index>(pb_max_threads, static_cast<Index>(work / kMinTaskSize)));

  // compute the number of threads we are going to use
  int threads = std::min<int>(nbThreads(), static_cast<int>(pb_max_threads));

  // if multi-threading is explicitly disabled, not useful, or if we already are
  // inside a parallel session, then abort multi-threading
  bool dont_parallelize = (!Condition) || (threads <= 1);
#if defined(EIGEN_HAS_OPENMP)
  // don't parallelize if we are executing in a parallel context already.
  dont_parallelize |= omp_get_num_threads() > 1;
#elif defined(EIGEN_GEMM_THREADPOOL)
  // don't parallelize if we have a trivial threadpool or the current thread id
  // is != -1, indicating that we are already executing on a thread inside the pool.
  // In other words, we do not allow nested parallelism, since this would lead to
  // deadlocks due to the workstealing nature of the threadpool.
  ThreadPool* pool = getGemmThreadPool();
  dont_parallelize |= (pool == nullptr || pool->CurrentThreadId() != -1);
#endif
  if (dont_parallelize) return func(0, rows, 0, cols);

  func.initParallelSession(threads);

  if (transpose) std::swap(rows, cols);

  ei_declare_aligned_stack_constructed_variable(GemmParallelTaskInfo<Index>, task_info, threads, 0);

#if defined(EIGEN_HAS_OPENMP)
#pragma omp parallel num_threads(threads)
  {
    Index i = omp_get_thread_num();
    // Note that the actual number of threads might be lower than the number of
    // requested ones
    Index actual_threads = omp_get_num_threads();
    GemmParallelInfo<Index> info(i, static_cast<int>(actual_threads), task_info);

    Index blockCols = (cols / actual_threads) & ~Index(0x3);
    Index blockRows = (rows / actual_threads);
    blockRows = (blockRows / Functor::Traits::mr) * Functor::Traits::mr;

    Index r0 = i * blockRows;
    Index actualBlockRows = (i + 1 == actual_threads) ? rows - r0 : blockRows;

    Index c0 = i * blockCols;
    Index actualBlockCols = (i + 1 == actual_threads) ? cols - c0 : blockCols;

    info.task_info[i].lhs_start = r0;
    info.task_info[i].lhs_length = actualBlockRows;

    if (transpose)
      func(c0, actualBlockCols, 0, rows, &info);
    else
      func(0, rows, c0, actualBlockCols, &info);
  }

#elif defined(EIGEN_GEMM_THREADPOOL)
  ei_declare_aligned_stack_constructed_variable(GemmParallelTaskInfo<Index>, meta_info, threads, 0);
  Barrier barrier(threads);
  auto task = [=, &func, &barrier, &task_info](int i) {
    Index actual_threads = threads;
    GemmParallelInfo<Index> info(i, static_cast<int>(actual_threads), task_info);
    Index blockCols = (cols / actual_threads) & ~Index(0x3);
    Index blockRows = (rows / actual_threads);
    blockRows = (blockRows / Functor::Traits::mr) * Functor::Traits::mr;

    Index r0 = i * blockRows;
    Index actualBlockRows = (i + 1 == actual_threads) ? rows - r0 : blockRows;

    Index c0 = i * blockCols;
    Index actualBlockCols = (i + 1 == actual_threads) ? cols - c0 : blockCols;

    info.task_info[i].lhs_start = r0;
    info.task_info[i].lhs_length = actualBlockRows;

    if (transpose)
      func(c0, actualBlockCols, 0, rows, &info);
    else
      func(0, rows, c0, actualBlockCols, &info);

    barrier.Notify();
  };
  // Notice that we do not schedule more than "threads" tasks, which allows us to
  // limit number of running threads, even if the threadpool itself was constructed
  // with a larger number of threads.
  for (int i = 0; i < threads - 1; ++i) {
    pool->Schedule([=, task = std::move(task)] { task(i); });
  }
  task(threads - 1);
  barrier.Wait();
#endif
}

#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_PARALLELIZER_H
