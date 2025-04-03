// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Intel Corporation
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CORE_ARCH_AVX512_GEMM_KERNEL_H
#define EIGEN_CORE_ARCH_AVX512_GEMM_KERNEL_H

#if EIGEN_COMP_MSVC
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>
#include <type_traits>

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

#if !defined(EIGEN_USE_AVX512_GEMM_KERNELS)
#define EIGEN_USE_AVX512_GEMM_KERNELS 1
#endif

#define SECOND_FETCH (32)
#if (EIGEN_COMP_GNUC_STRICT != 0) && !defined(EIGEN_ARCH_AVX512_GEMM_KERNEL_USE_LESS_A_REGS)
// Use less registers to load A elements to workaround compiler spills. Loose a
// bit of performance (less than ~2%).
#define EIGEN_ARCH_AVX512_GEMM_KERNEL_USE_LESS_A_REGS
#endif

namespace Eigen {
namespace internal {

template <typename Scalar, bool is_unit_inc>
class gemm_class {
  using vec = typename packet_traits<Scalar>::type;
  using vec_ymm = typename unpacket_traits<vec>::half;
  using vec_xmm = typename unpacket_traits<vec_ymm>::half;
  using umask_t = typename unpacket_traits<vec>::mask_t;

  static constexpr bool is_f32 = sizeof(Scalar) == sizeof(float);
  static constexpr bool is_f64 = sizeof(Scalar) == sizeof(double);

#ifndef EIGEN_ARCH_AVX512_GEMM_KERNEL_USE_LESS_A_REGS
  static constexpr bool use_less_a_regs = !is_unit_inc;
#else
  static constexpr bool use_less_a_regs = true;
#endif
#ifndef EIGEN_ARCH_AVX512_GEMM_KERNEL_USE_LESS_B_REGS
  static constexpr bool use_less_b_regs = !is_unit_inc;
#else
  static constexpr bool use_less_b_regs = true;
#endif

  static constexpr int a_regs[] = {0, 1, 2, use_less_a_regs ? 0 : 3, use_less_a_regs ? 1 : 4, use_less_a_regs ? 2 : 5};
  static constexpr int b_regs[] = {6, use_less_b_regs ? 6 : 7};
  static constexpr int c_regs[] = {
      8, 16, 24, 9, 17, 25, 10, 18, 26, 11, 19, 27, 12, 20, 28, 13, 21, 29, 14, 22, 30, 15, 23, 31,
  };

  static constexpr int alpha_load_reg = 0;
  static constexpr int c_load_regs[] = {1, 2, 6};

  static constexpr int a_shift = 128;
  static constexpr int b_shift = 128;

  static constexpr int nelems_in_cache_line = is_f32 ? 16 : 8;
  static constexpr int a_prefetch_size = nelems_in_cache_line * 2;
  static constexpr int b_prefetch_size = nelems_in_cache_line * 8;

  vec zmm[32];
  umask_t mask;

  // gemm arguments.
  Index m;
  const Index n, k, ldc;
  const Index inc;
  const Scalar *alpha;

  const Scalar *a, *b;
  Scalar *c;

  const bool is_alpha1;
  const bool is_beta0;

  const Index a_stride, b_stride;
  const Index a_off, b_off;

  EIGEN_ALWAYS_INLINE void prefetch_a(const Scalar *a_addr) {
    _mm_prefetch((char *)(a_prefetch_size + a_addr - a_shift), _MM_HINT_T0);
  }

  EIGEN_ALWAYS_INLINE void prefetch_b(const Scalar *b_addr) {
    _mm_prefetch((char *)(b_prefetch_size + b_addr - b_shift), _MM_HINT_T0);
  }

  EIGEN_ALWAYS_INLINE void prefetch_x(const Scalar *x_addr) { _mm_prefetch((char *)(x_addr - a_shift), _MM_HINT_T2); }

  EIGEN_ALWAYS_INLINE void prefetch_c(const Scalar *c_addr) {
#if defined(__PRFCHW__) && __PRFCHW__ == 1
    _m_prefetchw((void *)c_addr);
#else
    _mm_prefetch((char *)c_addr, _MM_HINT_T0);
#endif
  }

  template <int nelems>
  EIGEN_ALWAYS_INLINE void a_load(vec &a_reg, const Scalar *a_addr) {
    switch (nelems * sizeof(*a_addr) * 8) {
      default:
      case 512 * 3:
        a_reg = ploadu<vec>(a_addr);
        break;
      case 512 * 2:
        a_reg = ploadu<vec>(a_addr);
        break;
      case 512 * 1:
        a_reg = ploadu<vec>(a_addr);
        break;
      case 256 * 1:
        a_reg = preinterpret<vec>(_mm512_broadcast_f64x4(ploadu<Packet4d>(reinterpret_cast<const double *>(a_addr))));
        break;
      case 128 * 1:
        a_reg = preinterpret<vec>(_mm512_broadcast_f32x4(ploadu<Packet4f>(reinterpret_cast<const float *>(a_addr))));
        break;
      case 64 * 1:
        a_reg = preinterpret<vec>(pload1<Packet8d>(reinterpret_cast<const double *>(a_addr)));
        break;
      case 32 * 1:
        a_reg = pload1<vec>(a_addr);
        break;
    }
  }

  EIGEN_ALWAYS_INLINE void b_load(vec &b_reg, const Scalar *b_addr) { b_reg = pload1<vec>(b_addr); }

  template <int nelems>
  EIGEN_ALWAYS_INLINE void c_store(Scalar *mem, vec &src) {
    if (is_unit_inc) {
      switch (nelems * sizeof(*mem) * 8) {
        default:
        case 512 * 3:
          pstoreu(mem, src);
          break;
        case 512 * 2:
          pstoreu(mem, src);
          break;
        case 512 * 1:
          pstoreu(mem, src);
          break;
        case 256 * 1:
          pstoreu(mem, preinterpret<vec_ymm>(src));
          break;
        case 128 * 1:
          pstoreu(mem, preinterpret<vec_xmm>(src));
          break;
        case 64 * 1:
          pstorel(mem, preinterpret<vec_xmm>(src));
          break;
        case 32 * 1:
          pstores(mem, preinterpret<vec_xmm>(src));
          break;
      }
    } else {
      switch (nelems * sizeof(*mem) * 8) {
        default:
        case 512 * 3:
          pscatter(mem, src, inc);
          break;
        case 512 * 2:
          pscatter(mem, src, inc);
          break;
        case 512 * 1:
          pscatter(mem, src, inc);
          break;
        case 256 * 1:
          pscatter(mem, src, inc, mask);
          break;
        case 128 * 1:
          pscatter(mem, src, inc, mask);
          break;
        case 64 * 1:
          pscatter(mem, src, inc, mask);
          break;
        case 32 * 1:
          pscatter(mem, src, inc, mask);
          break;
      }
    }
  }

  template <int nelems>
  EIGEN_ALWAYS_INLINE void vaddm(vec &dst, const Scalar *mem, vec &src, vec &reg) {
    if (is_unit_inc) {
      switch (nelems * sizeof(*mem) * 8) {
        default:
        case 512 * 3:
          dst = padd(src, ploadu<vec>(mem));
          break;
        case 512 * 2:
          dst = padd(src, ploadu<vec>(mem));
          break;
        case 512 * 1:
          dst = padd(src, ploadu<vec>(mem));
          break;
        case 256 * 1:
          dst = preinterpret<vec>(padd(preinterpret<vec_ymm>(src), ploadu<vec_ymm>(mem)));
          break;
        case 128 * 1:
          dst = preinterpret<vec>(padd(preinterpret<vec_xmm>(src), ploadu<vec_xmm>(mem)));
          break;
        case 64 * 1:
          dst = preinterpret<vec>(padd(preinterpret<vec_xmm>(src), ploadl<vec_xmm>(mem)));
          break;
        case 32 * 1:
          dst = preinterpret<vec>(padds(preinterpret<vec_xmm>(src), ploads<vec_xmm>(mem)));
          break;
      }
    } else {
      // Zero out scratch register
      reg = pzero(reg);

      switch (nelems * sizeof(*mem) * 8) {
        default:
        case 512 * 3:
          reg = pgather<Scalar, vec>(mem, inc);
          dst = padd(src, reg);
          break;
        case 512 * 2:
          reg = pgather<Scalar, vec>(mem, inc);
          dst = padd(src, reg);
          break;
        case 512 * 1:
          reg = pgather<Scalar, vec>(mem, inc);
          dst = padd(src, reg);
          break;
        case 256 * 1:
          reg = preinterpret<vec>(pgather<Scalar, vec_ymm>(mem, inc));
          dst = preinterpret<vec>(padd(preinterpret<vec_ymm>(src), preinterpret<vec_ymm>(reg)));
          break;
        case 128 * 1:
          reg = preinterpret<vec>(pgather<Scalar, vec_xmm>(mem, inc));
          dst = preinterpret<vec>(padd(preinterpret<vec_xmm>(src), preinterpret<vec_xmm>(reg)));
          break;
        case 64 * 1:
          if (is_f32) {
            reg = pgather(reg, mem, inc, mask);
            dst = preinterpret<vec>(padd(preinterpret<vec_xmm>(src), preinterpret<vec_xmm>(reg)));
          } else {
            dst = preinterpret<vec>(padd(preinterpret<vec_xmm>(src), ploadl<vec_xmm>(mem)));
          }
          break;
        case 32 * 1:
          dst = preinterpret<vec>(padds(preinterpret<vec_xmm>(src), ploads<vec_xmm>(mem)));
          break;
      }
    }
  }

  EIGEN_STRONG_INLINE void vfmadd(vec &dst, const vec &src1, const vec &src2) {
    dst = pmadd(src1, src2, dst);

#if (EIGEN_COMP_GNUC != 0) || (EIGEN_COMP_CLANG != 0)
    // Workaround register spills for gcc and clang
    __asm__("#" : [dst] "+v"(dst) : [src1] "%v"(src1), [src2] "v"(src2));
#endif
  }

  template <int nelems>
  EIGEN_ALWAYS_INLINE void vfmaddm(vec &dst, const Scalar *mem, vec &src, vec &scale, vec &reg) {
    if (is_unit_inc) {
      switch (nelems * sizeof(*mem) * 8) {
        default:
        case 512 * 3:
          dst = pmadd(scale, src, ploadu<vec>(mem));
          break;
        case 512 * 2:
          dst = pmadd(scale, src, ploadu<vec>(mem));
          break;
        case 512 * 1:
          dst = pmadd(scale, src, ploadu<vec>(mem));
          break;
        case 256 * 1:
          dst =
              preinterpret<vec>(pmadd(preinterpret<vec_ymm>(scale), preinterpret<vec_ymm>(src), ploadu<vec_ymm>(mem)));
          break;
        case 128 * 1:
          dst =
              preinterpret<vec>(pmadd(preinterpret<vec_xmm>(scale), preinterpret<vec_xmm>(src), ploadu<vec_xmm>(mem)));
          break;
        case 64 * 1:
          dst =
              preinterpret<vec>(pmadd(preinterpret<vec_xmm>(scale), preinterpret<vec_xmm>(src), ploadl<vec_xmm>(mem)));
          break;
        case 32 * 1:
          dst =
              preinterpret<vec>(pmadds(preinterpret<vec_xmm>(scale), preinterpret<vec_xmm>(src), ploads<vec_xmm>(mem)));
          break;
      }
    } else {
      // Zero out scratch register
      reg = pzero(reg);

      switch (nelems * sizeof(*mem) * 8) {
        default:
        case 512 * 3:
          reg = pgather<Scalar, vec>(mem, inc);
          dst = pmadd(scale, src, reg);
          break;
        case 512 * 2:
          reg = pgather<Scalar, vec>(mem, inc);
          dst = pmadd(scale, src, reg);
          break;
        case 512 * 1:
          reg = pgather<Scalar, vec>(mem, inc);
          dst = pmadd(scale, src, reg);
          break;
        case 256 * 1:
          reg = preinterpret<vec>(pgather<Scalar, vec_ymm>(mem, inc));
          dst = preinterpret<vec>(
              pmadd(preinterpret<vec_ymm>(scale), preinterpret<vec_ymm>(src), preinterpret<vec_ymm>(reg)));
          break;
        case 128 * 1:
          reg = preinterpret<vec>(pgather<Scalar, vec_xmm>(mem, inc));
          dst = preinterpret<vec>(
              pmadd(preinterpret<vec_xmm>(scale), preinterpret<vec_xmm>(src), preinterpret<vec_xmm>(reg)));
          break;
        case 64 * 1:
          if (is_f32) {
            reg = pgather(reg, mem, inc, mask);
            dst = preinterpret<vec>(
                pmadd(preinterpret<vec_xmm>(scale), preinterpret<vec_xmm>(src), preinterpret<vec_xmm>(reg)));
          } else {
            dst = preinterpret<vec>(
                pmadd(preinterpret<vec_xmm>(scale), preinterpret<vec_xmm>(src), ploadl<vec_xmm>(mem)));
          }
          break;
        case 32 * 1:
          dst =
              preinterpret<vec>(pmadds(preinterpret<vec_xmm>(scale), preinterpret<vec_xmm>(src), ploads<vec_xmm>(mem)));
          break;
      }
    }
  }

  template <int j, int endX, int i, int endY, int nelems>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(j > endX) || (i > endY)> a_loads(const Scalar *ao) {
    EIGEN_UNUSED_VARIABLE(ao);
  }

  template <int j, int endX, int i, int endY, int nelems>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(j <= endX) && (i <= endY)> a_loads(const Scalar *ao) {
    if (j < endX) {
      if (i < endY) {
        auto &a_reg = zmm[a_regs[i + (j % 2) * 3]];
        const Scalar *a_addr = ao + nelems * j + nelems_in_cache_line * i - a_shift;
        a_load<nelems>(a_reg, a_addr);

        a_loads<j, endX, i + 1, endY, nelems>(ao);
      } else {
        a_loads<j + 1, endX, 0, endY, nelems>(ao);
      }
    }
  }

  template <int un, int max_b_unroll, int i, int um_vecs, int a_unroll, int b_unroll>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(un > max_b_unroll) || (i > um_vecs)> prefetch_cs(const Scalar *co1,
                                                                                         const Scalar *co2) {
    EIGEN_UNUSED_VARIABLE(co1);
    EIGEN_UNUSED_VARIABLE(co2);
  }

  /* C prefetch loop structure.
   * for (int un = 0; un < 8; un++) {
   *     if (b_unroll >= un + 1) {
   *         if (un == 4) co2 = co1 + 4 * ldc;
   *
   *         for (int i = 0; i < um_vecs; i++) {
   *             Scalar *co = (un + 1 <= 4) ? co1 : co2;
   *             auto co_off = (un % 4) * ldc + a_unroll - 1 + i * nelems_in_cache_line * sizeof *co;
   *             prefetch_c(co + co_off);
   *         }
   *     }
   * }
   */

  template <int un, int max_b_unroll, int i, int um_vecs, int a_unroll, int b_unroll>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(un <= max_b_unroll) && (i <= um_vecs)> prefetch_cs(Scalar *&co1, Scalar *&co2) {
    if (un < max_b_unroll) {
      if (b_unroll >= un + 1) {
        if (un == 4 && i == 0) co2 = co1 + 4 * ldc;

        if (i < um_vecs) {
          Scalar *co = (un + 1 <= 4) ? co1 : co2;
          auto co_off = (un % 4) * ldc + a_unroll - 1 + i * nelems_in_cache_line * sizeof *co;
          prefetch_c(co + co_off);

          prefetch_cs<un, max_b_unroll, i + 1, um_vecs, a_unroll, b_unroll>(co1, co2);
        } else {
          prefetch_cs<un + 1, max_b_unroll, 0, um_vecs, a_unroll, b_unroll>(co1, co2);
        }

      } else {
        prefetch_cs<un + 1, max_b_unroll, 0, um_vecs, a_unroll, b_unroll>(co1, co2);
      }
    }
  }

  // load_c
  template <int i, int um_vecs, int idx, int nelems>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(i > um_vecs)> scale_load_c(const Scalar *cox, vec &alpha_reg) {
    EIGEN_UNUSED_VARIABLE(cox);
    EIGEN_UNUSED_VARIABLE(alpha_reg);
  }

  template <int i, int um_vecs, int idx, int nelems>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(i <= um_vecs)> scale_load_c(const Scalar *cox, vec &alpha_reg) {
    if (i < um_vecs) {
      auto &c_reg = zmm[c_regs[i + idx * 3]];
      auto &c_load_reg = zmm[c_load_regs[i % 3]];
      auto c_mem = cox;
      if (is_unit_inc)
        c_mem += i * nelems_in_cache_line;
      else
        c_mem += i * nelems_in_cache_line * inc;

      if (!is_beta0 && is_alpha1)
        vaddm<nelems>(c_reg, c_mem, c_reg, c_load_reg);
      else if (!is_beta0 && !is_alpha1)
        vfmaddm<nelems>(c_reg, c_mem, c_reg, alpha_reg, c_load_reg);
      else if (is_beta0 && !is_alpha1)
        c_reg = pmul(alpha_reg, c_reg);

      scale_load_c<i + 1, um_vecs, idx, nelems>(cox, alpha_reg);
    }
  }

  // store_c
  template <int i, int um_vecs, int idx, int nelems>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(i > um_vecs)> write_c(Scalar *cox) {
    EIGEN_UNUSED_VARIABLE(cox);
  }

  template <int i, int um_vecs, int idx, int nelems>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(i <= um_vecs)> write_c(Scalar *cox) {
    if (i < um_vecs) {
      auto &c_reg = zmm[c_regs[i + idx * 3]];
      auto c_mem = cox;
      if (is_unit_inc)
        c_mem += i * nelems_in_cache_line;
      else
        c_mem += i * nelems_in_cache_line * inc;

      c_store<nelems>(c_mem, c_reg);
      c_reg = pzero(c_reg);

      write_c<i + 1, um_vecs, idx, nelems>(cox);
    }
  }

  /*  C update loop structure.
   *  co2 = co1 + ldc;
   *
   *  auto &alpha_reg = zmm[alpha_load_reg];
   *  if (!is_alpha1) alpha_reg = pload1<vec>(alpha);
   *
   *  int idx = 0;
   *  for (pow = 1; pow <= 8; pow <<= 1) {
   *
   *      if (b_unroll >= pow) {
   *          for (count = 1; count < (pow + 1) / 2 + 1;  count++) {
   *              if (pow >= 4) co2 += ldc;
   *
   *              const Scalar *cox = (idx == 0) ? co1 : co2;
   *
   *              const int um_vecs = numext::div_ceil(a_unroll, nelems_in_cache_line);
   *              scale_load_c<0, um_vecs, idx, a_unroll>(cox, alpha_reg);
   *              write_c<0, um_vecs, idx, a_unroll>(cox);
   *
   *              idx++;
   *          }
   *      }
   *  }
   *
   *  if (b_unroll == 1)
   *      co1 += ldc;
   *  else
   *      co1 = co2 + ldc;
   */

  template <int pow, int a_unroll, int idx>
  EIGEN_ALWAYS_INLINE void c_update_1count(Scalar *&cox) {
    if (pow >= 4) cox += ldc;

    const int um_vecs = numext::div_ceil(a_unroll, nelems_in_cache_line);
    auto &alpha_reg = zmm[alpha_load_reg];

    scale_load_c<0, um_vecs, idx, a_unroll>(cox, alpha_reg);
    write_c<0, um_vecs, idx, a_unroll>(cox);
  }

  template <int pow, int a_unroll>
  EIGEN_ALWAYS_INLINE void c_update_1pow(Scalar *&co1, Scalar *&co2) {
    constexpr int idx = pow / 2;
    Scalar *&cox = idx == 0 ? co1 : co2;

    constexpr int max_count = (pow + 1) / 2;
    static_assert(max_count <= 4, "Unsupported max_count.");

    if (1 <= max_count) c_update_1count<pow, a_unroll, idx + 0>(cox);
    if (2 <= max_count) c_update_1count<pow, a_unroll, idx + 1>(cox);
    if (3 <= max_count) c_update_1count<pow, a_unroll, idx + 2>(cox);
    if (4 <= max_count) c_update_1count<pow, a_unroll, idx + 3>(cox);
  }

  template <int max_b_unroll, int a_unroll, int b_unroll>
  EIGEN_ALWAYS_INLINE void c_update(Scalar *&co1, Scalar *&co2) {
    auto &alpha_reg = zmm[alpha_load_reg];

    co2 = co1 + ldc;
    if (!is_alpha1) alpha_reg = pload1<vec>(alpha);
    if (!is_unit_inc && a_unroll < nelems_in_cache_line) mask = static_cast<umask_t>((1ull << a_unroll) - 1);

    static_assert(max_b_unroll <= 8, "Unsupported max_b_unroll");

    if (1 <= max_b_unroll && 1 <= b_unroll) c_update_1pow<1, a_unroll>(co1, co2);
    if (2 <= max_b_unroll && 2 <= b_unroll) c_update_1pow<2, a_unroll>(co1, co2);
    if (4 <= max_b_unroll && 4 <= b_unroll) c_update_1pow<4, a_unroll>(co1, co2);
    if (8 <= max_b_unroll && 8 <= b_unroll) c_update_1pow<8, a_unroll>(co1, co2);

    if (b_unroll == 1)
      co1 += ldc;
    else
      co1 = co2 + ldc;
  }

  // compute
  template <int um, int um_vecs, int idx, int uk, bool fetch_x, bool ktail>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(um > um_vecs)> compute(const Scalar *ao, const Scalar *bo, int &fetchA_idx,
                                                               int &fetchB_idx, vec &b_reg) {
    EIGEN_UNUSED_VARIABLE(ao);
    EIGEN_UNUSED_VARIABLE(bo);
    EIGEN_UNUSED_VARIABLE(fetchA_idx);
    EIGEN_UNUSED_VARIABLE(fetchB_idx);
    EIGEN_UNUSED_VARIABLE(b_reg);
  }

  template <int um, int um_vecs, int idx, int uk, bool fetch_x, bool ktail>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(um <= um_vecs)> compute(const Scalar *ao, const Scalar *bo, int &fetchA_idx,
                                                                int &fetchB_idx, vec &b_reg) {
    if (um < um_vecs) {
      auto &c_reg = zmm[c_regs[um + idx * 3]];
      auto &a_reg = zmm[a_regs[um + (uk % 2) * 3]];

      vfmadd(c_reg, a_reg, b_reg);

      if (!fetch_x && um == 0 &&
          (((idx == 0 || idx == 6) && (uk % 2 == 0 || is_f64 || ktail)) ||
           (idx == 3 && (uk % 2 == 1 || is_f64 || ktail)))) {
        prefetch_a(ao + nelems_in_cache_line * fetchA_idx);
        fetchA_idx++;
      }

      if (um == 0 && idx == 1 && (uk % 2 == 0 || is_f64 || ktail)) {
        prefetch_b(bo + nelems_in_cache_line * fetchB_idx);
        fetchB_idx++;
      }

      compute<um + 1, um_vecs, idx, uk, fetch_x, ktail>(ao, bo, fetchA_idx, fetchB_idx, b_reg);
    }
  }

  // load_a
  template <int um, int um_vecs, int uk, int nelems, bool ktail>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(um > um_vecs)> load_a(const Scalar *ao) {
    EIGEN_UNUSED_VARIABLE(ao);
  }

  template <int um, int um_vecs, int uk, int nelems, bool ktail>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(um <= um_vecs)> load_a(const Scalar *ao) {
    if (um < um_vecs) {
      auto &a_reg = zmm[a_regs[um + (uk % 2) * 3]];
      const Scalar *a_addr = ao + nelems * (1 + !ktail * !use_less_a_regs + uk) + nelems_in_cache_line * um - a_shift;
      a_load<nelems>(a_reg, a_addr);

      load_a<um + 1, um_vecs, uk, nelems, ktail>(ao);
    }
  }
  template <int uk, int pow, int count, int um_vecs, int b_unroll, bool ktail, bool fetch_x, bool c_fetch>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(count > (pow + 1) / 2)> innerkernel_1pow(const Scalar *&aa,
                                                                                 const Scalar *const &ao,
                                                                                 const Scalar *const &bo, Scalar *&co2,
                                                                                 int &fetchA_idx, int &fetchB_idx) {
    EIGEN_UNUSED_VARIABLE(aa);
    EIGEN_UNUSED_VARIABLE(ao);
    EIGEN_UNUSED_VARIABLE(bo);
    EIGEN_UNUSED_VARIABLE(co2);
    EIGEN_UNUSED_VARIABLE(fetchA_idx);
    EIGEN_UNUSED_VARIABLE(fetchB_idx);
  }

  template <int uk, int pow, int count, int um_vecs, int b_unroll, bool ktail, bool fetch_x, bool c_fetch>
  EIGEN_ALWAYS_INLINE std::enable_if_t<(count <= (pow + 1) / 2)> innerkernel_1pow(const Scalar *&aa,
                                                                                  const Scalar *const &ao,
                                                                                  const Scalar *const &bo, Scalar *&co2,
                                                                                  int &fetchA_idx, int &fetchB_idx) {
    const int idx = (pow / 2) + count;

    if (count < (pow + 1) / 2) {
      auto &b_reg = zmm[b_regs[idx % 2]];

      if (fetch_x && uk == 3 && idx == 0) prefetch_x(aa);
      if (fetch_x && uk == 3 && idx == 4) aa += 8;

      if (b_unroll >= pow) {
        compute<0, um_vecs, idx, uk, fetch_x, ktail>(ao, bo, fetchA_idx, fetchB_idx, b_reg);

        const Scalar *b_addr = bo + b_unroll * uk + idx + 1 + (b_unroll > 1) * !use_less_b_regs - b_shift;
        b_load(b_reg, b_addr);
      }

      // Go to the next count.
      innerkernel_1pow<uk, pow, count + 1, um_vecs, b_unroll, ktail, fetch_x, c_fetch>(aa, ao, bo, co2, fetchA_idx,
                                                                                       fetchB_idx);

    } else {
      // Maybe prefetch C data after count-loop.
      if (pow == 2 && c_fetch) {
        if (uk % 3 == 0 && uk > 0) {
          co2 += ldc;
        } else {
          prefetch_c(co2 + (uk % 3) * nelems_in_cache_line);
        }
      }
    }
  }

  template <int uk, int max_b_unroll, int a_unroll, int b_unroll, bool ktail, bool fetch_x, bool c_fetch,
            bool no_a_preload = false>
  EIGEN_ALWAYS_INLINE void innerkernel_1uk(const Scalar *&aa, const Scalar *const &ao, const Scalar *const &bo,
                                           Scalar *&co2, int &fetchA_idx, int &fetchB_idx) {
    const int um_vecs = numext::div_ceil(a_unroll, nelems_in_cache_line);

    if (max_b_unroll >= 1)
      innerkernel_1pow<uk, 1, 0, um_vecs, b_unroll, ktail, fetch_x, c_fetch>(aa, ao, bo, co2, fetchA_idx, fetchB_idx);
    if (max_b_unroll >= 2)
      innerkernel_1pow<uk, 2, 0, um_vecs, b_unroll, ktail, fetch_x, c_fetch>(aa, ao, bo, co2, fetchA_idx, fetchB_idx);
    if (max_b_unroll >= 4)
      innerkernel_1pow<uk, 4, 0, um_vecs, b_unroll, ktail, fetch_x, c_fetch>(aa, ao, bo, co2, fetchA_idx, fetchB_idx);
    if (max_b_unroll >= 8)
      innerkernel_1pow<uk, 8, 0, um_vecs, b_unroll, ktail, fetch_x, c_fetch>(aa, ao, bo, co2, fetchA_idx, fetchB_idx);

    // Load A after pow-loop. Skip this at the end to prevent running over the buffer
    if (!no_a_preload) load_a<0, um_vecs, uk, a_unroll, ktail>(ao);
  }

  /*  Inner kernel loop structure.
   *  for (int uk = 0; uk < kfactor; uk++) {
   *      int idx = 0;
   *
   *      for (pow = 1; pow < max_b_unroll << 1; pow <<= 1) {
   *          for (int count = 0; count < (pow + 1) / 2; count++) {
   *              auto &b_reg = zmm[b_regs[idx % 2]];
   *
   *              if (fetch_x && uk == 3 && idx == 0) prefetch_x(aa);
   *              if (fetch_x && uk == 3 && idx == 4) aa += 8;
   *
   *              if (b_unroll >= pow) {
   *                  compute<0, um_vecs, idx, uk, fetchx, ktail>(ao, bo, fetchA_idx, fetchB_idx, b_reg);
   *
   *                  const Scalar *b_addr = bo + b_unroll * uk + idx + 1 + (b_unroll > 1) - b_shift ;
   *                  b_load(b_reg, b_addr);
   *              }
   *              idx++;
   *          }
   *
   *          Maybe prefetch C data.
   *          if (pow == 2 && c_fetch) {
   *              if (uk % 3 == 0 && uk > 0) {
   *                  co2 += ldc;
   *              } else {
   *                  prefetch_c(co2 + (uk % 3) * nelems_in_cache_line);
   *              }
   *          }
   *      }
   *
   *      Load A.
   *      load_a<0, um_vecs, uk, ktail, a_unroll>(ao);
   *  }
   *
   *  Advance A/B pointers after uk-loop.
   *  ao += a_unroll * kfactor;
   *  bo += b_unroll * kfactor;
   */

  template <int a_unroll, int b_unroll, int k_factor, int max_b_unroll, int max_k_factor, bool c_fetch,
            bool no_a_preload = false>
  EIGEN_ALWAYS_INLINE void innerkernel(const Scalar *&aa, const Scalar *&ao, const Scalar *&bo, Scalar *&co2) {
    int fetchA_idx = 0;
    int fetchB_idx = 0;

    const bool fetch_x = k_factor == max_k_factor;
    const bool ktail = k_factor == 1;

    static_assert(k_factor <= 4 && k_factor > 0, "innerkernel maximum k_factor supported is 4");
    static_assert(no_a_preload == false || (no_a_preload == true && k_factor == 1),
                  "skipping a preload only allowed when k unroll is 1");

    if (k_factor > 0)
      innerkernel_1uk<0, max_b_unroll, a_unroll, b_unroll, ktail, fetch_x, c_fetch, no_a_preload>(
          aa, ao, bo, co2, fetchA_idx, fetchB_idx);
    if (k_factor > 1)
      innerkernel_1uk<1, max_b_unroll, a_unroll, b_unroll, ktail, fetch_x, c_fetch, no_a_preload>(
          aa, ao, bo, co2, fetchA_idx, fetchB_idx);
    if (k_factor > 2)
      innerkernel_1uk<2, max_b_unroll, a_unroll, b_unroll, ktail, fetch_x, c_fetch, no_a_preload>(
          aa, ao, bo, co2, fetchA_idx, fetchB_idx);
    if (k_factor > 3)
      innerkernel_1uk<3, max_b_unroll, a_unroll, b_unroll, ktail, fetch_x, c_fetch, no_a_preload>(
          aa, ao, bo, co2, fetchA_idx, fetchB_idx);

    // Advance A/B pointers after uk-loop.
    ao += a_unroll * k_factor;
    bo += b_unroll * k_factor;
  }

  template <int a_unroll, int b_unroll, int max_b_unroll>
  EIGEN_ALWAYS_INLINE void kloop(const Scalar *&aa, const Scalar *&ao, const Scalar *&bo, Scalar *&co1, Scalar *&co2) {
    const int um_vecs = numext::div_ceil(a_unroll, nelems_in_cache_line);
    if (!use_less_a_regs && k > 1)
      a_loads<0, 2, 0, um_vecs, a_unroll>(ao);
    else
      a_loads<0, 1, 0, um_vecs, a_unroll>(ao);

    b_load(zmm[b_regs[0]], bo - b_shift + 0);
    if (!use_less_b_regs) b_load(zmm[b_regs[1]], bo - b_shift + 1);

#ifndef SECOND_FETCH
    prefetch_cs<0, max_b_unroll, 0, um_vecs, a_unroll, b_unroll>(co1, co2);
#endif  // SECOND_FETCH

    // Unrolling k-loop by a factor of 4.
    const int max_k_factor = 4;
    Index kRem = k % max_k_factor;
    Index k_ = k - kRem;
    if (k_ >= max_k_factor) {
      k_ -= max_k_factor;
      kRem += max_k_factor;
    }
    Index loop_count = k_ / max_k_factor;

    if (loop_count > 0) {
#ifdef SECOND_FETCH
      loop_count -= SECOND_FETCH;
#endif
      while (loop_count > 0) {
        innerkernel<a_unroll, b_unroll, max_k_factor, max_b_unroll, max_k_factor, 0>(aa, ao, bo, co2);
        loop_count--;
      }
#ifdef SECOND_FETCH
      co2 = co1 + nelems_in_cache_line - 1;

      loop_count += b_unroll;
      while (loop_count > 0) {
        innerkernel<a_unroll, b_unroll, max_k_factor, max_b_unroll, max_k_factor, 1>(aa, ao, bo, co2);
        loop_count--;
      }

      loop_count += SECOND_FETCH - b_unroll;
      while (loop_count > 0) {
        innerkernel<a_unroll, b_unroll, max_k_factor, max_b_unroll, max_k_factor, 0>(aa, ao, bo, co2);
        loop_count--;
      }
#endif
    }

    // k-loop remainder handling.
    loop_count = kRem;
    while (loop_count > 1) {
      innerkernel<a_unroll, b_unroll, 1, max_b_unroll, max_k_factor, 0>(aa, ao, bo, co2);
      loop_count--;
    }
    if (loop_count > 0) {
      innerkernel<a_unroll, b_unroll, 1, max_b_unroll, max_k_factor, 0, true>(aa, ao, bo, co2);
    }

    // Update C matrix.
    c_update<max_b_unroll, a_unroll, b_unroll>(co1, co2);
  }

  template <int a_unroll, int b_unroll, int max_b_unroll>
  EIGEN_ALWAYS_INLINE void nloop(const Scalar *&aa, const Scalar *&ao, const Scalar *&bo, Scalar *&co1, Scalar *&co2) {
    // Set A matrix pointer.
    ao = a + a_off * a_unroll;

    // Set B matrix pointer if needed.
    bo += b_unroll * b_off;

    kloop<a_unroll, b_unroll, max_b_unroll>(aa, ao, bo, co1, co2);

    // Advance B matrix pointer if needed.
    bo += b_unroll * (b_stride - k - b_off);

    // Advance prefetch A pointer.
    aa += 16;
  }

  template <int a_unroll, int max_a_unroll, int max_b_unroll>
  EIGEN_ALWAYS_INLINE void mloop(const Scalar *&ao, const Scalar *&bo, Scalar *&co1, Scalar *&co2) {
    // Set prefetch A pointers.
    const Scalar *aa = a + a_unroll * a_stride;

    // Set C matrix pointers.
    co1 = c;
    if (a_unroll >= max_a_unroll) co2 = c + 2 * ldc;
    if (is_unit_inc)
      c += a_unroll;
    else
      c += a_unroll * inc;

    // Set B matrix pointer.
    bo = b;

    // Main n-loop.
    for (Index i = n / max_b_unroll; i > 0; i--) nloop<a_unroll, max_b_unroll, max_b_unroll>(aa, ao, bo, co1, co2);

    // n-remainders.
    if (n & 4 && max_b_unroll > 4) nloop<a_unroll, 4, max_b_unroll>(aa, ao, bo, co1, co2);
#if 0
        if (n & 2 && max_b_unroll > 2) nloop<a_unroll, 2, max_b_unroll>(aa, ao, bo, co1, co2);
        if (n & 1 && max_b_unroll > 1) nloop<a_unroll, 1, max_b_unroll>(aa, ao, bo, co1, co2);
#else
    // Copy kernels don't support tails of n = 2 for single/double precision.
    // Loop over ones.
    int n_rem = 2 * ((n & 2) != 0) + 1 * ((n & 1) != 0);
    while (n_rem > 0) {
      nloop<a_unroll, 1, max_b_unroll>(aa, ao, bo, co1, co2);
      n_rem--;
    }
#endif

    // Advance A matrix pointer.
    a = ao + a_unroll * (a_stride - k - a_off);
  }

 public:
  // Compute kernel unrolling C matrix by max_a_unroll x max_b_unroll.
  template <int max_a_unroll, int max_b_unroll>
  EIGEN_ALWAYS_INLINE void compute_kern() {
    a -= -a_shift;
    b -= -b_shift;

    const Scalar *ao = nullptr;
    const Scalar *bo = nullptr;
    Scalar *co1 = nullptr;
    Scalar *co2 = nullptr;

    // Main m-loop.
    for (; m >= max_a_unroll; m -= max_a_unroll) mloop<max_a_unroll, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);

    // m-remainders.
    if (m & 32 && max_a_unroll > 32) mloop<32, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);
    if (m & 16 && max_a_unroll > 16) mloop<16, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);
    if (m & 8 && max_a_unroll > 8) mloop<8, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);
    if (m & 4 && max_a_unroll > 4) mloop<4, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);
    if (m & 2 && max_a_unroll > 2 && is_f64) mloop<2, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);
    if (m & 1 && max_a_unroll > 1 && is_f64) mloop<1, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);

    // Copy kernels don't support tails of m = 2 for single precision.
    // Loop over ones.
    if (is_f32) {
      int m_rem = 2 * ((m & 2) != 0) + 1 * ((m & 1) != 0);
      while (m_rem > 0) {
        mloop<1, max_a_unroll, max_b_unroll>(ao, bo, co1, co2);
        m_rem--;
      }
    }
  }

  gemm_class(Index m_, Index n_, Index k_, Index ldc_, Index inc_, const Scalar *alpha_, const Scalar *a_,
             const Scalar *b_, Scalar *c_, bool is_alpha1_, bool is_beta0_, Index a_stride_, Index b_stride_,
             Index a_off_, Index b_off_)
      : m(m_),
        n(n_),
        k(k_),
        ldc(ldc_),
        inc(inc_),
        alpha(alpha_),
        a(a_),
        b(b_),
        c(c_),
        is_alpha1(is_alpha1_),
        is_beta0(is_beta0_),
        a_stride(a_stride_),
        b_stride(b_stride_),
        a_off(a_off_),
        b_off(b_off_) {
    // Zero out all accumulation registers.
    zmm[8] = pzero(zmm[8]);
    zmm[9] = pzero(zmm[9]);
    zmm[10] = pzero(zmm[10]);
    zmm[11] = pzero(zmm[11]);
    zmm[12] = pzero(zmm[12]);
    zmm[13] = pzero(zmm[13]);
    zmm[14] = pzero(zmm[14]);
    zmm[15] = pzero(zmm[15]);
    zmm[16] = pzero(zmm[16]);
    zmm[17] = pzero(zmm[17]);
    zmm[18] = pzero(zmm[18]);
    zmm[19] = pzero(zmm[19]);
    zmm[20] = pzero(zmm[20]);
    zmm[21] = pzero(zmm[21]);
    zmm[22] = pzero(zmm[22]);
    zmm[23] = pzero(zmm[23]);
    zmm[24] = pzero(zmm[24]);
    zmm[25] = pzero(zmm[25]);
    zmm[26] = pzero(zmm[26]);
    zmm[27] = pzero(zmm[27]);
    zmm[28] = pzero(zmm[28]);
    zmm[29] = pzero(zmm[29]);
    zmm[30] = pzero(zmm[30]);
    zmm[31] = pzero(zmm[31]);
  }
};

// Compute kernel with max unroll support of:
//   Single precision:
//     max_a_unroll: 48, 32, 16, 8, 4, 2, 1
//     max_b_unroll: 8, 4, 2, 1
//   Double precision:
//     max_a_unroll: 24, 16, 8, 4, 2, 1
//     max_b_unroll: 8, 4, 2, 1
template <typename Scalar, int max_a_unroll, int max_b_unroll, bool is_alpha1, bool is_beta0, bool is_unit_inc>
EIGEN_DONT_INLINE void gemm_kern_avx512(Index m, Index n, Index k, Scalar *alpha, const Scalar *a, const Scalar *b,
                                        Scalar *c, Index ldc, Index inc = 1, Index a_stride = -1, Index b_stride = -1,
                                        Index a_off = 0, Index b_off = 0) {
  if (a_stride == -1) a_stride = k;
  if (b_stride == -1) b_stride = k;

  gemm_class<Scalar, is_unit_inc> g(m, n, k, ldc, inc, alpha, a, b, c, is_alpha1, is_beta0, a_stride, b_stride, a_off,
                                    b_off);
  g.template compute_kern<max_a_unroll, max_b_unroll>();
}

// Template specializations of GEBP kernels with nr = 8.
#if EIGEN_USE_AVX512_GEMM_KERNELS
template <bool ConjLhs_, bool ConjRhs_, int PacketSize_>
class gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Target, PacketSize_>
    : public gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_> {
  using Base = gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_>;

 public:
  enum { nr = Base::Vectorizable ? 8 : 4 };
};

template <bool ConjLhs_, bool ConjRhs_, int PacketSize_>
class gebp_traits<double, double, ConjLhs_, ConjRhs_, Architecture::Target, PacketSize_>
    : public gebp_traits<double, double, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_> {
  using Base = gebp_traits<double, double, ConjLhs_, ConjRhs_, Architecture::Generic, PacketSize_>;

 public:
  enum { nr = Base::Vectorizable ? 8 : 4 };
};

template <typename Scalar, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, DataMapper, 8, ColMajor, Conjugate, PanelMode> {
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename DataMapper::LinearMapper LinearMapper;
  enum { PacketSize = packet_traits<Scalar>::size };
  EIGEN_DONT_INLINE void operator()(Scalar *blockB, const DataMapper &rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0);
};

template <typename Scalar, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, DataMapper, 8, ColMajor, Conjugate, PanelMode>::operator()(
    Scalar *blockB, const DataMapper &rhs, Index depth, Index cols, Index stride, Index offset) {
  constexpr int nr = 8;
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS COLMAJOR");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride == 0 && offset == 0) || (PanelMode && stride >= depth && offset <= stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols8 = nr >= 8 ? (cols / 8) * 8 : 0;
  Index packet_cols4 = nr >= 4 ? (cols / 4) * 4 : 0;
  Index count = 0;
  const Index peeled_k = (depth / PacketSize) * PacketSize;
  if (nr >= 8) {
    for (Index j2 = 0; j2 < packet_cols8; j2 += 8) {
      // skip what we have before
      if (PanelMode) count += 8 * offset;
      const LinearMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const LinearMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const LinearMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const LinearMapper dm3 = rhs.getLinearMapper(0, j2 + 3);
      const LinearMapper dm4 = rhs.getLinearMapper(0, j2 + 4);
      const LinearMapper dm5 = rhs.getLinearMapper(0, j2 + 5);
      const LinearMapper dm6 = rhs.getLinearMapper(0, j2 + 6);
      const LinearMapper dm7 = rhs.getLinearMapper(0, j2 + 7);
      Index k = 0;
      if ((PacketSize % 8) == 0)  // TODO enable vectorized transposition for PacketSize==4
      {
        for (; k < peeled_k; k += PacketSize) {
          PacketBlock<Packet, (PacketSize % 8) == 0 ? 8 : PacketSize> kernel;

          kernel.packet[0] = dm0.template loadPacket<Packet>(k);
          kernel.packet[1] = dm1.template loadPacket<Packet>(k);
          kernel.packet[2] = dm2.template loadPacket<Packet>(k);
          kernel.packet[3] = dm3.template loadPacket<Packet>(k);
          kernel.packet[4] = dm4.template loadPacket<Packet>(k);
          kernel.packet[5] = dm5.template loadPacket<Packet>(k);
          kernel.packet[6] = dm6.template loadPacket<Packet>(k);
          kernel.packet[7] = dm7.template loadPacket<Packet>(k);

          ptranspose(kernel);

          pstoreu(blockB + count + 0 * PacketSize, cj.pconj(kernel.packet[0]));
          pstoreu(blockB + count + 1 * PacketSize, cj.pconj(kernel.packet[1 % PacketSize]));
          pstoreu(blockB + count + 2 * PacketSize, cj.pconj(kernel.packet[2 % PacketSize]));
          pstoreu(blockB + count + 3 * PacketSize, cj.pconj(kernel.packet[3 % PacketSize]));
          pstoreu(blockB + count + 4 * PacketSize, cj.pconj(kernel.packet[4 % PacketSize]));
          pstoreu(blockB + count + 5 * PacketSize, cj.pconj(kernel.packet[5 % PacketSize]));
          pstoreu(blockB + count + 6 * PacketSize, cj.pconj(kernel.packet[6 % PacketSize]));
          pstoreu(blockB + count + 7 * PacketSize, cj.pconj(kernel.packet[7 % PacketSize]));
          count += 8 * PacketSize;
        }
      }
      for (; k < depth; k++) {
        blockB[count + 0] = cj(dm0(k));
        blockB[count + 1] = cj(dm1(k));
        blockB[count + 2] = cj(dm2(k));
        blockB[count + 3] = cj(dm3(k));
        blockB[count + 4] = cj(dm4(k));
        blockB[count + 5] = cj(dm5(k));
        blockB[count + 6] = cj(dm6(k));
        blockB[count + 7] = cj(dm7(k));
        count += 8;
      }
      // skip what we have after
      if (PanelMode) count += 8 * (stride - offset - depth);
    }
  }

  if (nr >= 4) {
    for (Index j2 = packet_cols8; j2 < packet_cols4; j2 += 4) {
      // skip what we have before
      if (PanelMode) count += 4 * offset;
      const LinearMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const LinearMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const LinearMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const LinearMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k = 0;
      if ((PacketSize % 4) == 0)  // TODO enable vectorized transposition for PacketSize==2 ??
      {
        for (; k < peeled_k; k += PacketSize) {
          PacketBlock<Packet, (PacketSize % 4) == 0 ? 4 : PacketSize> kernel;
          kernel.packet[0] = dm0.template loadPacket<Packet>(k);
          kernel.packet[1 % PacketSize] = dm1.template loadPacket<Packet>(k);
          kernel.packet[2 % PacketSize] = dm2.template loadPacket<Packet>(k);
          kernel.packet[3 % PacketSize] = dm3.template loadPacket<Packet>(k);
          ptranspose(kernel);
          pstoreu(blockB + count + 0 * PacketSize, cj.pconj(kernel.packet[0]));
          pstoreu(blockB + count + 1 * PacketSize, cj.pconj(kernel.packet[1 % PacketSize]));
          pstoreu(blockB + count + 2 * PacketSize, cj.pconj(kernel.packet[2 % PacketSize]));
          pstoreu(blockB + count + 3 * PacketSize, cj.pconj(kernel.packet[3 % PacketSize]));
          count += 4 * PacketSize;
        }
      }
      for (; k < depth; k++) {
        blockB[count + 0] = cj(dm0(k));
        blockB[count + 1] = cj(dm1(k));
        blockB[count + 2] = cj(dm2(k));
        blockB[count + 3] = cj(dm3(k));
        count += 4;
      }
      // skip what we have after
      if (PanelMode) count += 4 * (stride - offset - depth);
    }
  }

  // copy the remaining columns one at a time (nr==1)
  for (Index j2 = packet_cols4; j2 < cols; ++j2) {
    if (PanelMode) count += offset;
    const LinearMapper dm0 = rhs.getLinearMapper(0, j2);
    for (Index k = 0; k < depth; k++) {
      blockB[count] = cj(dm0(k));
      count += 1;
    }
    if (PanelMode) count += (stride - offset - depth);
  }
}

template <typename Scalar, typename Index, typename DataMapper, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, DataMapper, 8, RowMajor, Conjugate, PanelMode> {
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename unpacket_traits<Packet>::half HalfPacket;
  typedef typename unpacket_traits<typename unpacket_traits<Packet>::half>::half QuarterPacket;
  typedef typename DataMapper::LinearMapper LinearMapper;
  enum {
    PacketSize = packet_traits<Scalar>::size,
    HalfPacketSize = unpacket_traits<HalfPacket>::size,
    QuarterPacketSize = unpacket_traits<QuarterPacket>::size
  };
  EIGEN_DONT_INLINE void operator()(Scalar *blockB, const DataMapper &rhs, Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) {
    constexpr int nr = 8;
    EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS ROWMAJOR");
    EIGEN_UNUSED_VARIABLE(stride);
    EIGEN_UNUSED_VARIABLE(offset);
    eigen_assert(((!PanelMode) && stride == 0 && offset == 0) || (PanelMode && stride >= depth && offset <= stride));
    const bool HasHalf = (int)HalfPacketSize < (int)PacketSize;
    const bool HasQuarter = (int)QuarterPacketSize < (int)HalfPacketSize;
    conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
    Index packet_cols8 = nr >= 8 ? (cols / 8) * 8 : 0;
    Index packet_cols4 = nr >= 4 ? (cols / 4) * 4 : 0;
    Index count = 0;

    if (nr >= 8) {
      for (Index j2 = 0; j2 < packet_cols8; j2 += 8) {
        // skip what we have before
        if (PanelMode) count += 8 * offset;
        for (Index k = 0; k < depth; k++) {
          if (PacketSize == 8) {
            // Packet A = ploadu<Packet>(&rhs.data()[k*rhs.stride() + j2]);
            Packet A = rhs.template loadPacket<Packet>(k, j2);
            pstoreu(blockB + count, cj.pconj(A));
          } else if (HasHalf && HalfPacketSize == 8) {
            HalfPacket A = rhs.template loadPacket<HalfPacket>(k, j2);
            pstoreu(blockB + count, cj.pconj(A));
          } else if (HasQuarter && QuarterPacketSize == 8) {
            QuarterPacket A = rhs.template loadPacket<QuarterPacket>(k, j2);
            pstoreu(blockB + count, cj.pconj(A));
          } else if (PacketSize == 4) {
            // Packet A = ploadu<Packet>(&rhs.data()[k*rhs.stride() + j2]);
            // Packet B = ploadu<Packet>(&rhs.data()[k*rhs.stride() + j2 + PacketSize]);
            Packet A = rhs.template loadPacket<Packet>(k, j2);
            Packet B = rhs.template loadPacket<Packet>(k, j2 + PacketSize);
            pstoreu(blockB + count, cj.pconj(A));
            pstoreu(blockB + count + PacketSize, cj.pconj(B));
          } else {
            // const Scalar* b0 = &rhs.data()[k*rhs.stride() + j2];
            const LinearMapper dm0 = rhs.getLinearMapper(k, j2);
            blockB[count + 0] = cj(dm0(0));
            blockB[count + 1] = cj(dm0(1));
            blockB[count + 2] = cj(dm0(2));
            blockB[count + 3] = cj(dm0(3));
            blockB[count + 4] = cj(dm0(4));
            blockB[count + 5] = cj(dm0(5));
            blockB[count + 6] = cj(dm0(6));
            blockB[count + 7] = cj(dm0(7));
          }
          count += 8;
        }
        // skip what we have after
        if (PanelMode) count += 8 * (stride - offset - depth);
      }
    }

    if (nr >= 4) {
      for (Index j2 = packet_cols8; j2 < packet_cols4; j2 += 4) {
        // skip what we have before
        if (PanelMode) count += 4 * offset;
        for (Index k = 0; k < depth; k++) {
          if (PacketSize == 4) {
            Packet A = rhs.template loadPacket<Packet>(k, j2);
            pstoreu(blockB + count, cj.pconj(A));
            count += PacketSize;
          } else if (HasHalf && HalfPacketSize == 4) {
            HalfPacket A = rhs.template loadPacket<HalfPacket>(k, j2);
            pstoreu(blockB + count, cj.pconj(A));
            count += HalfPacketSize;
          } else if (HasQuarter && QuarterPacketSize == 4) {
            QuarterPacket A = rhs.template loadPacket<QuarterPacket>(k, j2);
            pstoreu(blockB + count, cj.pconj(A));
            count += QuarterPacketSize;
          } else {
            const LinearMapper dm0 = rhs.getLinearMapper(k, j2);
            blockB[count + 0] = cj(dm0(0));
            blockB[count + 1] = cj(dm0(1));
            blockB[count + 2] = cj(dm0(2));
            blockB[count + 3] = cj(dm0(3));
            count += 4;
          }
        }
        // skip what we have after
        if (PanelMode) count += 4 * (stride - offset - depth);
      }
    }
    // copy the remaining columns one at a time (nr==1)
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      if (PanelMode) count += offset;
      for (Index k = 0; k < depth; k++) {
        blockB[count] = cj(rhs(k, j2));
        count += 1;
      }
      if (PanelMode) count += stride - offset - depth;
    }
  }
};

template <typename Scalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<Scalar, Scalar, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs> {
  EIGEN_ALWAYS_INLINE void operator()(const DataMapper &res, const Scalar *blockA, const Scalar *blockB, Index rows,
                                      Index depth, Index cols, Scalar alpha, Index strideA = -1, Index strideB = -1,
                                      Index offsetA = 0, Index offsetB = 0);
};

template <typename Scalar, typename Index, typename DataMapper, int mr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_ALWAYS_INLINE void gebp_kernel<Scalar, Scalar, Index, DataMapper, mr, 8, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper &res, const Scalar *blockA, const Scalar *blockB, Index rows, Index depth, Index cols,
    Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  if (res.incr() == 1) {
    if (alpha == 1) {
      gemm_kern_avx512<Scalar, mr, 8, true, false, true>(rows, cols, depth, &alpha, blockA, blockB,
                                                         (Scalar *)res.data(), res.stride(), res.incr(), strideA,
                                                         strideB, offsetA, offsetB);
    } else {
      gemm_kern_avx512<Scalar, mr, 8, false, false, true>(rows, cols, depth, &alpha, blockA, blockB,
                                                          (Scalar *)res.data(), res.stride(), res.incr(), strideA,
                                                          strideB, offsetA, offsetB);
    }
  } else {
    if (alpha == 1) {
      gemm_kern_avx512<Scalar, mr, 8, true, false, false>(rows, cols, depth, &alpha, blockA, blockB,
                                                          (Scalar *)res.data(), res.stride(), res.incr(), strideA,
                                                          strideB, offsetA, offsetB);
    } else {
      gemm_kern_avx512<Scalar, mr, 8, false, false, false>(rows, cols, depth, &alpha, blockA, blockB,
                                                           (Scalar *)res.data(), res.stride(), res.incr(), strideA,
                                                           strideB, offsetA, offsetB);
    }
  }
}
#endif  // EIGEN_USE_AVX512_GEMM_KERNELS

}  // namespace internal
}  // namespace Eigen

#undef SECOND_FETCH

#endif  // EIGEN_CORE_ARCH_AVX512_GEMM_KERNEL_H
