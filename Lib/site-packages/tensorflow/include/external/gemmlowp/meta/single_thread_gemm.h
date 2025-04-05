// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GEMMLOWP_META_SINGLE_THREAD_GEMM_H_
#define GEMMLOWP_META_SINGLE_THREAD_GEMM_H_

#include <iostream>
#include "base.h"

namespace gemmlowp {
namespace meta {

template <typename Executor, typename Params, int kernel_m, int kernel_n,
          int kernel_k>
void Gemm(const Params& params);

class GemmExecutorPackRHS {
 public:
  template <typename P>
  static int EstimateScratchSize(const P& params, int kernel_m, int kernel_n,
                                 int kernel_k) {
    const int lhs_scratch =
        StreamUtil<typename P::InType, typename P::LeftStream>::Scratch(
            params.left_stream, kernel_m, kernel_k);
    const int rhs_chunks = ((params.n + kernel_n - 1) / kernel_n);
    const int rhs_scratch =
        rhs_chunks *
        StreamUtil<typename P::InType, typename P::RightStream>::Scratch(
            params.right_stream, kernel_n, kernel_k);
    return AlignTo<64 * 1024>(lhs_scratch + rhs_scratch);
  }

  template <typename P, int m, int n, int k, int m_leftovers, int n_leftovers,
            int k_leftovers>
  static void ExecuteDispatch3D(const P& params) {
    // Shorthand typedefs for streams and multiply kernels.
    typedef typename P::InType InType;
    typedef typename P::OutType OutType;

    typedef Stream<typename P::InType, m, k, k_leftovers,
                   typename P::LeftStream>
        LeftStreamF;
    typedef Stream<typename P::InType, m_leftovers, k, k_leftovers,
                   typename P::LeftStream>
        LeftStreamL;

    typedef Stream<typename P::InType, n, k, k_leftovers,
                   typename P::RightStream>
        RightStreamF;
    typedef Stream<typename P::InType, n_leftovers, k, k_leftovers,
                   typename P::RightStream>
        RightStreamL;

    typedef Stream<typename P::OutType, m, n, 0, typename P::OutputStream>
        OutputStreamFF;
    typedef Stream<typename P::OutType, m_leftovers, n, 0,
                   typename P::OutputStream>
        OutputStreamLF;

    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m, n, k>
        KernelFF;
    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m,
                      n_leftovers, k>
        KernelFL;
    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m_leftovers,
                      n, k>
        KernelLF;
    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m_leftovers,
                      n_leftovers, k>
        KernelLL;

#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "GemmExecutor(" << typeid(P).name() << "): " << m << "x" << n
              << "x" << k << " -- " << m_leftovers << "x" << n_leftovers << "x"
              << k_leftovers << " -- " << params.m << "x" << params.n << "x"
              << params.k << std::endl;
    LeftStreamF::Debug(params.left_stream);
    LeftStreamL::Debug(params.left_stream);

    RightStreamF::Debug(params.right_stream);
    RightStreamL::Debug(params.right_stream);

    OutputStreamFF::Debug(params.fused_kernel.output_stream);
    OutputStreamLF::Debug(params.fused_kernel.output_stream);

    KernelFF::Debug(params.fused_kernel);
    KernelFL::Debug(params.fused_kernel);
    KernelLF::Debug(params.fused_kernel);
    KernelLL::Debug(params.fused_kernel);
#endif
#endif

    int lhs_chunks = params.m / m;
    int rhs_chunks = params.n / n;

    // Scratch memory for packed LHS & RHS chunks.

    std::uint8_t* packed_lhs = params.scratch;
    std::uint8_t* packed_rhs =
        params.scratch + LeftStreamF::Scratch(params.left_stream);

    // Pack full RHS first.

    std::uint8_t* packed_rhs_chunk = packed_rhs;
    const int packed_rhs_chunk_size =
        RightStreamF::PackedStride(params.right_stream);

    {
      const std::uint8_t* rhs_chunk =
          reinterpret_cast<const std::uint8_t*>(params.rhs);
      const int rhs_chunk_size =
          RightStreamF::UnpackedStride(params.right_stream);

      for (int i = 0; i < rhs_chunks; ++i) {
        RightStreamF::Pack(reinterpret_cast<const InType*>(rhs_chunk),
                           params.right_stream,
                           reinterpret_cast<InType*>(packed_rhs_chunk));

        rhs_chunk += rhs_chunk_size;
        packed_rhs_chunk += packed_rhs_chunk_size;
      }

      RightStreamL::Pack(reinterpret_cast<const InType*>(rhs_chunk),
                         params.right_stream,
                         reinterpret_cast<InType*>(packed_rhs_chunk));
    }

    // Multiply RHS by LHS one LHS chunk at a time.

    const std::uint8_t* lhs_chunk =
        reinterpret_cast<const std::uint8_t*>(params.lhs);
    std::uint8_t* result_strip = reinterpret_cast<std::uint8_t*>(params.result);
    std::uint8_t* result_chunk = result_strip;

    {
      const int lhs_chunk_size =
          LeftStreamF::UnpackedStride(params.left_stream);
      const int result_strip_size =
          OutputStreamFF::UnpackedStride(params.fused_kernel.output_stream);
      const int result_chunk_size =
          OutputStreamFF::UnpackedAdvance(params.fused_kernel.output_stream);

      for (int i = 0; i < lhs_chunks; ++i) {
        LeftStreamF::Pack(reinterpret_cast<const InType*>(lhs_chunk),
                          params.left_stream,
                          reinterpret_cast<InType*>(packed_lhs));

        result_chunk = result_strip;
        packed_rhs_chunk = packed_rhs;

        for (int j = 0; j < rhs_chunks; ++j) {
          KernelFF::Multiply(reinterpret_cast<const InType*>(packed_lhs),
                             reinterpret_cast<const InType*>(packed_rhs_chunk),
                             params.fused_kernel,
                             reinterpret_cast<OutType*>(result_chunk));

          result_chunk += result_chunk_size;
          packed_rhs_chunk += packed_rhs_chunk_size;
        }

        KernelFL::Multiply(reinterpret_cast<const InType*>(packed_lhs),
                           reinterpret_cast<const InType*>(packed_rhs_chunk),
                           params.fused_kernel,
                           reinterpret_cast<OutType*>(result_chunk));

        lhs_chunk += lhs_chunk_size;
        result_strip += result_strip_size;
      }
    }

    // Leftover LHS chunk.
    if (m_leftovers > 0) {  // static if
      const int result_chunk_size =
          OutputStreamLF::UnpackedAdvance(params.fused_kernel.output_stream);

      LeftStreamL::Pack(reinterpret_cast<const InType*>(lhs_chunk),
                        params.left_stream,
                        reinterpret_cast<InType*>(packed_lhs));

      result_chunk = result_strip;
      packed_rhs_chunk = packed_rhs;

      for (int i = 0; i < rhs_chunks; ++i) {
        KernelLF::Multiply(reinterpret_cast<const InType*>(packed_lhs),
                           reinterpret_cast<const InType*>(packed_rhs_chunk),
                           params.fused_kernel,
                           reinterpret_cast<OutType*>(result_chunk));

        result_chunk += result_chunk_size;
        packed_rhs_chunk += packed_rhs_chunk_size;
      }

      KernelLL::Multiply(reinterpret_cast<const InType*>(packed_lhs),
                         reinterpret_cast<const InType*>(packed_rhs_chunk),
                         params.fused_kernel,
                         reinterpret_cast<OutType*>(result_chunk));
    }
  }
};

class GemmExecutorPackLHS {
 public:
  template <typename P>
  static int EstimateScratchSize(const P& params, int kernel_m, int kernel_n,
                                 int kernel_k) {
    const int lhs_chunks = ((params.m + kernel_m - 1) / kernel_m);
    const int lhs_scratch =
        lhs_chunks *
        StreamUtil<typename P::InType, typename P::LeftStream>::Scratch(
            params.left_stream, kernel_m, kernel_k);
    const int rhs_scratch =
        StreamUtil<typename P::InType, typename P::RightStream>::Scratch(
            params.right_stream, kernel_n, kernel_k);
    return AlignTo<64 * 1024>(lhs_scratch + rhs_scratch);
  }

  template <typename P, int m, int n, int k, int m_leftovers, int n_leftovers,
            int k_leftovers>
  static void ExecuteDispatch3D(const P& params) {
    // Shorthand typedefs for streams and multiply kernels.
    typedef typename P::InType InType;
    typedef typename P::OutType OutType;

    typedef Stream<typename P::InType, m, k, k_leftovers,
                   typename P::LeftStream>
        LeftStreamF;
    typedef Stream<typename P::InType, m_leftovers, k, k_leftovers,
                   typename P::LeftStream>
        LeftStreamL;

    typedef Stream<typename P::InType, n, k, k_leftovers,
                   typename P::RightStream>
        RightStreamF;
    typedef Stream<typename P::InType, n_leftovers, k, k_leftovers,
                   typename P::RightStream>
        RightStreamL;

    typedef Stream<typename P::OutType, m, n, 0, typename P::OutputStream>
        OutputStreamFF;
    typedef Stream<typename P::OutType, m, n_leftovers, 0,
                   typename P::OutputStream>
        OutputStreamFL;

    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m, n, k>
        KernelFF;
    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m,
                      n_leftovers, k>
        KernelFL;
    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m_leftovers,
                      n, k>
        KernelLF;
    typedef MulKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, typename P::OutputStream, m_leftovers,
                      n_leftovers, k>
        KernelLL;
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "GemmExecutor(" << typeid(P).name() << "): " << m << "x" << n
              << "x" << k << " -- " << m_leftovers << "x" << n_leftovers << "x"
              << k_leftovers << " -- " << params.m << "x" << params.n << "x"
              << params.k << std::endl;
    LeftStreamF::Debug(params.left_stream);
    LeftStreamL::Debug(params.left_stream);

    RightStreamF::Debug(params.right_stream);
    RightStreamL::Debug(params.right_stream);

    OutputStreamFF::Debug(params.fused_kernel.output_stream);
    OutputStreamFL::Debug(params.fused_kernel.output_stream);

    KernelFF::Debug(params.fused_kernel);
    KernelFL::Debug(params.fused_kernel);
    KernelLF::Debug(params.fused_kernel);
    KernelLL::Debug(params.fused_kernel);
#endif
#endif

    int lhs_chunks = params.m / m;
    int rhs_chunks = params.n / n;

    // Scratch memory for packed LHS & RHS chunks.
    std::uint8_t* packed_rhs = params.scratch;
    std::uint8_t* packed_lhs =
        params.scratch + RightStreamF::Scratch(params.right_stream);

    // Pack full LHS first.

    std::uint8_t* packed_lhs_chunk = packed_lhs;
    const int packed_lhs_chunk_size =
        LeftStreamF::PackedStride(params.left_stream);

    {
      const std::uint8_t* lhs_chunk =
          reinterpret_cast<const std::uint8_t*>(params.lhs);
      const int lhs_chunk_size =
          LeftStreamF::UnpackedStride(params.left_stream);

      for (int i = 0; i < lhs_chunks; ++i) {
        LeftStreamF::Pack(reinterpret_cast<const InType*>(lhs_chunk),
                          params.left_stream,
                          reinterpret_cast<InType*>(packed_lhs_chunk));

        lhs_chunk += lhs_chunk_size;
        packed_lhs_chunk += packed_lhs_chunk_size;
      }

      LeftStreamL::Pack(reinterpret_cast<const InType*>(lhs_chunk),
                        params.left_stream,
                        reinterpret_cast<InType*>(packed_lhs_chunk));
    }

    // Multiply RHS by LHS one RHS chunk at a time.

    const std::uint8_t* rhs_chunk =
        reinterpret_cast<const std::uint8_t*>(params.rhs);
    std::uint8_t* result_strip = reinterpret_cast<std::uint8_t*>(params.result);
    std::uint8_t* result_chunk = result_strip;

    {
      const int rhs_chunk_size =
          RightStreamF::UnpackedStride(params.right_stream);
      const int result_strip_size =
          OutputStreamFF::UnpackedAdvance(params.fused_kernel.output_stream);
      const int result_chunk_size =
          OutputStreamFF::UnpackedStride(params.fused_kernel.output_stream);

      for (int i = 0; i < rhs_chunks; ++i) {
        RightStreamF::Pack(reinterpret_cast<const InType*>(rhs_chunk),
                           params.right_stream,
                           reinterpret_cast<InType*>(packed_rhs));

        result_chunk = result_strip;
        packed_lhs_chunk = packed_lhs;

        for (int j = 0; j < lhs_chunks; ++j) {
          KernelFF::Multiply(reinterpret_cast<const InType*>(packed_lhs_chunk),
                             reinterpret_cast<const InType*>(packed_rhs),
                             params.fused_kernel,
                             reinterpret_cast<OutType*>(result_chunk));

          result_chunk += result_chunk_size;
          packed_lhs_chunk += packed_lhs_chunk_size;
        }

        KernelLF::Multiply(reinterpret_cast<const InType*>(packed_lhs_chunk),
                           reinterpret_cast<const InType*>(packed_rhs),
                           params.fused_kernel,
                           reinterpret_cast<OutType*>(result_chunk));

        rhs_chunk += rhs_chunk_size;
        result_strip += result_strip_size;
      }
    }

    // Leftover RHS chunk.
    if (n_leftovers > 0) {  // static if
      const int result_chunk_size =
          OutputStreamFL::UnpackedStride(params.fused_kernel.output_stream);

      RightStreamL::Pack(reinterpret_cast<const InType*>(rhs_chunk),
                         params.right_stream,
                         reinterpret_cast<InType*>(packed_rhs));

      result_chunk = result_strip;
      packed_lhs_chunk = packed_lhs;

      for (int i = 0; i < lhs_chunks; ++i) {
        KernelFL::Multiply(reinterpret_cast<const InType*>(packed_lhs_chunk),
                           reinterpret_cast<const InType*>(packed_rhs),
                           params.fused_kernel,
                           reinterpret_cast<OutType*>(result_chunk));

        result_chunk += result_chunk_size;
        packed_lhs_chunk += packed_lhs_chunk_size;
      }

      KernelLL::Multiply(reinterpret_cast<const InType*>(packed_lhs_chunk),
                         reinterpret_cast<const InType*>(packed_rhs),
                         params.fused_kernel,
                         reinterpret_cast<OutType*>(result_chunk));
    }
  }
};

namespace internal {

inline int CalculateCacheFriendlyTasksCount(int cache_size, int constant_memory,
                                            int per_chunk_memory, int total_dim,
                                            int chunk_dim) {
  assert(constant_memory + per_chunk_memory < cache_size);
  const int available_cache = cache_size - constant_memory;
  const int available_chunks = available_cache / per_chunk_memory;
  const int chunks_count = (total_dim + chunk_dim - 1) / chunk_dim;
  return (chunks_count + available_chunks - 1) / available_chunks;
}

template <typename Params>
inline void UpdateCacheFriendlyTask(int m_offset, int m, int n_offset, int n,
                                    const Params& params, Params* task_params) {
  task_params->m = m;
  task_params->lhs =
      StreamUtil<typename Params::InType, typename Params::LeftStream>::Offset(
          params.left_stream, params.lhs, m_offset, 0);

  task_params->n = n;
  task_params->rhs =
      StreamUtil<typename Params::InType, typename Params::RightStream>::Offset(
          params.right_stream, params.rhs, n_offset, 0);

  task_params->result =
      StreamUtil<typename Params::OutType, typename Params::OutputStream>::
          Offset(params.fused_kernel.output_stream, params.result, m_offset,
                 n_offset);
}

}  // namespace internal

template <int cache_size = 256 * 1024>
class GemmExecutorPackRHSCacheFriendly {
 public:
  template <typename P>
  static int EstimateScratchSize(const P& params, int kernel_m, int kernel_n,
                                 int kernel_k) {
    return cache_size;
  }

  template <typename P, int m, int n, int k, int m_leftovers, int n_leftovers,
            int k_leftovers>
  static void ExecuteDispatch3D(const P& params) {
    typedef Stream<typename P::InType, m, k, k_leftovers,
                   typename P::LeftStream>
        LeftStream;

    typedef Stream<typename P::InType, n, k, k_leftovers,
                   typename P::RightStream>
        RightStream;

    const int lhs_scratch = LeftStream::Scratch(params.left_stream);
    const int rhs_scratch = RightStream::Scratch(params.right_stream);

    const int cache_friendly_tasks_count =
        internal::CalculateCacheFriendlyTasksCount(cache_size, lhs_scratch,
                                                   rhs_scratch, params.n, n);

    if (cache_friendly_tasks_count == 1) {
      GemmExecutorPackRHS::ExecuteDispatch3D<P, m, n, k, m_leftovers,
                                             n_leftovers, k_leftovers>(params);
      return;
    }

    const int cache_friendly_dim = params.n / cache_friendly_tasks_count;

    P task_params = params;
    for (int i = 0; i < cache_friendly_tasks_count - 1; ++i) {
      internal::UpdateCacheFriendlyTask(0, params.m, i * cache_friendly_dim,
                                        cache_friendly_dim, params,
                                        &task_params);
      Gemm<GemmExecutorPackRHS, P, m, n, k>(task_params);
    }
    const int dim_sum = (cache_friendly_tasks_count - 1) * cache_friendly_dim;
    internal::UpdateCacheFriendlyTask(0, params.m, dim_sum, params.n - dim_sum,
                                      params, &task_params);
    Gemm<GemmExecutorPackRHS, P, m, n, k>(task_params);
  }
};

template <int cache_size = 256 * 1024>
class GemmExecutorPackLHSCacheFriendly {
 public:
  template <typename P>
  static int EstimateScratchSize(const P& params, int kernel_m, int kernel_n,
                                 int kernel_k) {
    return cache_size;
  }

  template <typename P, int m, int n, int k, int m_leftovers, int n_leftovers,
            int k_leftovers>
  static void ExecuteDispatch3D(const P& params) {
    typedef Stream<typename P::InType, m, k, k_leftovers,
                   typename P::LeftStream>
        LeftStream;

    typedef Stream<typename P::InType, n, k, k_leftovers,
                   typename P::RightStream>
        RightStream;

    const int lhs_scratch = LeftStream::Scratch(params.left_stream);
    const int rhs_scratch = RightStream::Scratch(params.right_stream);

    const int cache_friendly_tasks_count =
        internal::CalculateCacheFriendlyTasksCount(cache_size, rhs_scratch,
                                                   lhs_scratch, params.m, m);

    if (cache_friendly_tasks_count == 1) {
      GemmExecutorPackLHS::ExecuteDispatch3D<P, m, n, k, m_leftovers,
                                             n_leftovers, k_leftovers>(params);
      return;
    }

    const int cache_friendly_dim = params.m / cache_friendly_tasks_count;

    P task_params = params;
    for (int i = 0; i < cache_friendly_tasks_count - 1; ++i) {
      internal::UpdateCacheFriendlyTask(i * cache_friendly_dim,
                                        cache_friendly_dim, 0, params.n, params,
                                        &task_params);
      Gemm<GemmExecutorPackLHS, P, m, n, k>(task_params);
    }
    const int dim_sum = (cache_friendly_tasks_count - 1) * cache_friendly_dim;
    internal::UpdateCacheFriendlyTask(dim_sum, params.m - dim_sum, 0, params.n,
                                      params, &task_params);
    Gemm<GemmExecutorPackLHS, P, m, n, k>(task_params);
  }
};

namespace internal {

// Stage 3.

template <typename E, typename P, int dim_m, int dim_n, int dim_k, int fixed_m,
          int fixed_n, int variable_k>
struct Dispatch3DStage3 {
  static void Execute(const P& params, int k) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(3): " << dim_m << "x" << dim_n << "x" << dim_k
              << " : " << fixed_m << "x" << fixed_n << "x" << variable_k
              << std::endl
              << std::flush;
#endif
#endif
    if (k == variable_k) {
      E::template ExecuteDispatch3D<P, dim_m, dim_n, dim_k, fixed_m, fixed_n,
                                    variable_k>(params);
    } else {
      Dispatch3DStage3<E, P, dim_m, dim_n, dim_k, fixed_m, fixed_n,
                       variable_k - 1>::Execute(params, k);
    }
  }
};

template <typename E, typename P, int dim_m, int dim_n, int dim_k, int fixed_m,
          int fixed_n>
struct Dispatch3DStage3<E, P, dim_m, dim_n, dim_k, fixed_m, fixed_n, 0> {
  static void Execute(const P& params, int k) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(3): " << dim_m << "x" << dim_n << "x" << dim_k
              << " : " << fixed_m << "x" << fixed_n << "x" << 0 << std::endl
              << std::flush;
#endif
#endif
    if (k == 0) {
      E::template ExecuteDispatch3D<P, dim_m, dim_n, dim_k, fixed_m, fixed_n,
                                    0>(params);
    } else {
      std::cerr << "FATAL: dispatch3DStage3 failed: ran out of cases."
                << std::endl
                << std::flush;
      std::exit(1);
    }
  }
};

// Stage 2.

template <typename E, typename P, int dim_m, int dim_n, int dim_k, int fixed_m,
          int variable_n>
struct Dispatch3DStage2 {
  static void Execute(const P& params, int n, int k) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(2): " << dim_m << "x" << dim_n << "x" << dim_k
              << " : " << fixed_m << "x" << variable_n << std::endl
              << std::flush;
#endif
#endif
    if (n == variable_n) {
      Dispatch3DStage3<E, P, dim_m, dim_n, dim_k, fixed_m, variable_n,
                       dim_k - 1>::Execute(params, k);
    } else {
      Dispatch3DStage2<E, P, dim_m, dim_n, dim_k, fixed_m,
                       variable_n - 1>::Execute(params, n, k);
    }
  }
};

template <typename E, typename P, int dim_m, int dim_n, int dim_k, int fixed_m>
struct Dispatch3DStage2<E, P, dim_m, dim_n, dim_k, fixed_m, 0> {
  static void Execute(const P& params, int n, int k) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(2): " << dim_m << "x" << dim_n << "x" << dim_k
              << " : " << fixed_m << "x" << 0 << std::endl
              << std::flush;
#endif
#endif
    if (n == 0) {
      Dispatch3DStage3<E, P, dim_m, dim_n, dim_k, fixed_m, 0,
                       dim_k - 1>::Execute(params, k);
    } else {
      std::cerr << "FATAL: dispatch3DStage2 failed: ran out of cases."
                << std::endl
                << std::flush;
      std::exit(1);
    }
  }
};

// Stage 1.

template <typename E, typename P, int dim_m, int dim_n, int dim_k,
          int variable_m>
struct Dispatch3DStage1 {
  static void Execute(const P& params, int m, int n, int k) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(1): " << dim_m << "x" << dim_n << "x" << dim_k
              << " : " << variable_m << std::endl
              << std::flush;
#endif
#endif
    if (m == variable_m) {
      Dispatch3DStage2<E, P, dim_m, dim_n, dim_k, variable_m,
                       dim_n - 1>::Execute(params, n, k);
    } else {
      Dispatch3DStage1<E, P, dim_m, dim_n, dim_k, variable_m - 1>::Execute(
          params, m, n, k);
    }
  }
};

template <typename E, typename P, int dim_m, int dim_n, int dim_k>
struct Dispatch3DStage1<E, P, dim_m, dim_n, dim_k, 0> {
  static void Execute(const P& params, int m, int n, int k) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(1): " << dim_m << "x" << dim_n << "x" << dim_k
              << " : " << 0 << std::endl
              << std::flush;
#endif
#endif
    if (m == 0) {
      Dispatch3DStage2<E, P, dim_m, dim_n, dim_k, 0, dim_n - 1>::Execute(params,
                                                                         n, k);
    } else {
      std::cerr << "FATAL: dispatch3DStage1 failed: ran out of cases."
                << std::endl
                << std::flush;
      std::exit(1);
    }
  }
};

}  // namespace internal

template <typename Executor, typename Params, int kernel_m, int kernel_n,
          int kernel_k>
inline void Gemm(const Params& params) {
  internal::Dispatch3DStage1<Executor, Params, kernel_m, kernel_n, kernel_k,
                             kernel_m - 1>::Execute(params, params.m % kernel_m,
                                                    params.n % kernel_n,
                                                    params.k % kernel_k);
}

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_SINGLE_THREAD_GEMM_H_
