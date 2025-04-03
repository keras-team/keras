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

#ifndef GEMMLOWP_META_QUANTIZED_MUL_KERNELS_ARM_32_H_
#define GEMMLOWP_META_QUANTIZED_MUL_KERNELS_ARM_32_H_

#ifdef GEMMLOWP_NEON_32

#include <cassert>
#include <cstdint>

namespace gemmlowp {
namespace meta {

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 1,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 1, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d2}, [%[lhs]:64]!\n"
      "vld1.32 {d3}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q2, d3, d2\n"
      "vpadal.u16 q0, q2\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[multiplicative_offset]\n"
      "vdup.32 q7, %[rounding_offset]\n"
      "vdup.32 q8, %[shift]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vmul.i32 q0, q0, q6\n"
      "vadd.i32 q0, q0, q7\n"
      "vshl.s32 q0, q0, q8\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.8 {d0[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11", "d12",
        "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 2,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 2, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d4}, [%[lhs]:64]!\n"
      "vld1.32 {d5, d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d5, d4\n"
      "vmull.u8 q5, d6, d4\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[multiplicative_offset]\n"
      "vdup.32 q7, %[rounding_offset]\n"
      "vdup.32 q8, %[shift]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vmul.i32 q0, q0, q6\n"
      "vadd.i32 q0, q0, q7\n"
      "vshl.s32 q0, q0, q8\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.16 {d0[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 3,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 3, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d6}, [%[lhs]:64]!\n"
      "vld1.32 {d7, d8, d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d7, d6\n"
      "vmull.u8 q6, d8, d6\n"
      "vmull.u8 q7, d9, d6\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[multiplicative_offset]\n"
      "vdup.32 q7, %[rounding_offset]\n"
      "vdup.32 q8, %[shift]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vmul.i32 q0, q0, q6\n"
      "vadd.i32 q0, q0, q7\n"
      "vshl.s32 q0, q0, q8\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.16 {d0[0]}, [%[result]]!\n"
      "vst1.8 {d0[2]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 4,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 4, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d8}, [%[lhs]:64]!\n"
      "vld1.32 {d9, d10, d11, d12}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q7, d9, d8\n"
      "vmull.u8 q8, d10, d8\n"
      "vmull.u8 q9, d11, d8\n"
      "vmull.u8 q10, d12, d8\n"
      "vpadal.u16 q0, q7\n"
      "vpadal.u16 q1, q8\n"
      "vpadal.u16 q2, q9\n"
      "vpadal.u16 q3, q10\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[multiplicative_offset]\n"
      "vdup.32 q7, %[rounding_offset]\n"
      "vdup.32 q8, %[shift]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vmul.i32 q0, q0, q6\n"
      "vadd.i32 q0, q0, q7\n"
      "vshl.s32 q0, q0, q8\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 5,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 5, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d10, d11, d12, d13}, [%[rhs]:64]!\n"
      "vld1.32 {d14}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q8, d10, d14\n"
      "vmull.u8 q9, d11, d14\n"
      "vmull.u8 q10, d12, d14\n"
      "vmull.u8 q11, d13, d14\n"
      "vld1.32 {d10}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q8\n"
      "vpadal.u16 q1, q9\n"
      "vpadal.u16 q2, q10\n"
      "vpadal.u16 q3, q11\n"
      "vmull.u8 q8, d10, d14\n"
      "vpadal.u16 q4, q8\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d10, d11}, [%[lhs]:64]!\n"
      "vld1.32 {d12, d13, d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, %[multiplicative_offset]\n"
      "vdup.32 q9, %[rounding_offset]\n"
      "vdup.32 q10, %[shift]\n"
      "vdup.32 q5, d10[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d8\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q1, q1, q7\n"
      "vmul.i32 q0, q0, q8\n"
      "vmul.i32 q1, q1, q8\n"
      "vadd.i32 q0, q0, q9\n"
      "vadd.i32 q1, q1, q9\n"
      "vshl.s32 q0, q0, q10\n"
      "vshl.s32 q1, q1, q10\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d1, q1\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.8 {d0[4]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 6,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 6, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13, d14, d15}, [%[rhs]:64]!\n"
      "vld1.32 {d16}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vmull.u8 q11, d14, d16\n"
      "vmull.u8 q12, d15, d16\n"
      "vld1.32 {d12, d13}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vpadal.u16 q4, q9\n"
      "vpadal.u16 q5, q10\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [%[rhs]:64]!\n"
      "vdup.32 q9, %[multiplicative_offset]\n"
      "vdup.32 q10, %[rounding_offset]\n"
      "vdup.32 q11, %[shift]\n"
      "vdup.32 q6, d12[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q1, q1, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q1, q1, q8\n"
      "vmul.i32 q0, q0, q9\n"
      "vmul.i32 q1, q1, q9\n"
      "vadd.i32 q0, q0, q10\n"
      "vadd.i32 q1, q1, q10\n"
      "vshl.s32 q0, q0, q11\n"
      "vshl.s32 q1, q1, q11\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d1, q1\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.16 {d0[2]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 7,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 7, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d14, d15, d16, d17}, [%[rhs]:64]!\n"
      "vld1.32 {d18}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d17, d18\n"
      "vld1.32 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q10\n"
      "vpadal.u16 q1, q11\n"
      "vpadal.u16 q2, q12\n"
      "vpadal.u16 q3, q13\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d14, d15}, [%[lhs]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [%[rhs]:64]!\n"
      "vdup.32 q10, %[multiplicative_offset]\n"
      "vdup.32 q11, %[rounding_offset]\n"
      "vdup.32 q12, %[shift]\n"
      "vdup.32 q7, d14[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"
      "vpadd.u32 d3, d12, d12\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q1, q1, q7\n"
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q1, q1, q9\n"
      "vmul.i32 q0, q0, q10\n"
      "vmul.i32 q1, q1, q10\n"
      "vadd.i32 q0, q0, q11\n"
      "vadd.i32 q1, q1, q11\n"
      "vshl.s32 q0, q0, q12\n"
      "vshl.s32 q1, q1, q12\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d1, q1\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.16 {d0[2]}, [%[result]]!\n"
      "vst1.8 {d0[6]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 8,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 8, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // 1x8 lanes loop.
      "1:"

      "vld1.32 {d17, d18, d19, d20}, [%[rhs]:256]!\n"
      "vld1.32 {d16}, [%[lhs]:64]!\n"
      "vmull.u8 q11, d16, d17\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d16, d19\n"
      "vmull.u8 q14, d16, d20\n"
      "vld1.32 {d17, d18, d19, d20}, [%[rhs]:256]!\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vpadal.u16 q3, q14\n"
      "pld [%[rhs], #256]\n"
      "vmull.u8 q15, d16, d17\n"
      "vmull.u8 q11, d16, d18\n"
      "vmull.u8 q12, d16, d19\n"
      "vmull.u8 q13, d16, d20\n"
      "pld [%[lhs], #32]\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vpadal.u16 q4, q15\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d16, d17}, [%[lhs]:64]!\n"
      "vld1.32 {d18, d19, d20, d21}, [%[rhs]:64]!\n"
      "vdup.32 q11, %[multiplicative_offset]\n"
      "vdup.32 q12, %[rounding_offset]\n"
      "vdup.32 q13, %[shift]\n"
      "vdup.32 q8, d16[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"
      "vpadd.u32 d3, d12, d14\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q1, q1, q8\n"
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q1, q1, q10\n"
      "vmul.i32 q0, q0, q11\n"
      "vmul.i32 q1, q1, q11\n"
      "vadd.i32 q0, q0, q12\n"
      "vadd.i32 q1, q1, q12\n"
      "vshl.s32 q0, q0, q13\n"
      "vshl.s32 q1, q1, q13\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d1, q1\n"
      "vqmovun.s16 d0, q0\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 1,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 1, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d4, d5}, [%[lhs]:64]!\n"
      "vld1.32 {d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d6, d4\n"
      "vmull.u8 q5, d6, d5\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[multiplicative_offset]\n"
      "vdup.32 q7, %[rounding_offset]\n"
      "vdup.32 q8, %[shift]\n"
      "vdup.32 q2, d8[0]\n"
      "vdup.32 q4, d8[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d2, d2, d2\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q2\n"
      "vadd.s32 q1, q1, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vmul.i32 q0, q0, q6\n"
      "vmul.i32 q1, q1, q6\n"
      "vadd.i32 q0, q0, q7\n"
      "vadd.i32 q1, q1, q7\n"
      "vshl.s32 q0, q0, q8\n"
      "vshl.s32 q1, q1, q8\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d2, q1\n"
      "vqmovun.s16 d0, q0\n"
      "vqmovun.s16 d2, q1\n"

      // RowMajorOutput::Output
      "vst1.8 {d0[0]}, [%[result]]!\n"
      "vst1.8 {d2[0]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 2,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 2, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q6, d10, d8\n"
      "vmull.u8 q7, d11, d8\n"
      "vmull.u8 q8, d10, d9\n"
      "vmull.u8 q9, d11, d9\n"
      "vpadal.u16 q0, q6\n"
      "vpadal.u16 q1, q7\n"
      "vpadal.u16 q2, q8\n"
      "vpadal.u16 q3, q9\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[multiplicative_offset]\n"
      "vdup.32 q7, %[rounding_offset]\n"
      "vdup.32 q8, %[shift]\n"
      "vdup.32 q9, d8[0]\n"
      "vdup.32 q4, d8[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d4, d4, d6\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q2, q2, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q2, q2, q5\n"
      "vmul.i32 q0, q0, q6\n"
      "vmul.i32 q2, q2, q6\n"
      "vadd.i32 q0, q0, q7\n"
      "vadd.i32 q2, q2, q7\n"
      "vshl.s32 q0, q0, q8\n"
      "vshl.s32 q2, q2, q8\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d4, q2\n"
      "vqmovun.s16 d0, q0\n"
      "vqmovun.s16 d4, q2\n"

      // RowMajorOutput::Output
      "vst1.16 {d0[0]}, [%[result]]!\n"
      "vst1.16 {d4[0]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 3,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 3, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d14, d12\n"
      "vmull.u8 q10, d15, d12\n"
      "vmull.u8 q11, d16, d12\n"
      "vmull.u8 q12, d14, d13\n"
      "vmull.u8 q13, d15, d13\n"
      "vmull.u8 q14, d16, d13\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, %[multiplicative_offset]\n"
      "vdup.32 q9, %[rounding_offset]\n"
      "vdup.32 q10, %[shift]\n"
      "vdup.32 q11, d12[0]\n"
      "vdup.32 q6, d12[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q11\n"
      "vadd.s32 q3, q3, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q3, q3, q7\n"
      "vmul.i32 q0, q0, q8\n"
      "vmul.i32 q3, q3, q8\n"
      "vadd.i32 q0, q0, q9\n"
      "vadd.i32 q3, q3, q9\n"
      "vshl.s32 q0, q0, q10\n"
      "vshl.s32 q3, q3, q10\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d6, q3\n"
      "vqmovun.s16 d0, q0\n"
      "vqmovun.s16 d6, q3\n"

      // RowMajorOutput::Output
      "vst1.16 {d0[0]}, [%[result]]!\n"
      "vst1.8 {d0[2]}, [%[result]]!\n"
      "vst1.16 {d6[0]}, [r0]!\n"
      "vst1.8 {d6[2]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 4,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 4, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // 2x4 lanes loop.
      "1:"

      "vld1.8 {d18, d19, d20, d21}, [%[rhs]:256]!\n"
      "vld1.8 {d16}, [%[lhs]:64]!\n"
      "vmull.u8 q11, d16, d18\n"
      "vld1.8 {d17}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d16, d19\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q13, d16, d20\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q14, d16, d21\n"
      "vmull.u8 q15, d17, d18\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vmull.u8 q11, d17, d19\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d17, d21\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vpadal.u16 q3, q14\n"
      "vpadal.u16 q4, q15\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d16, d17}, [%[lhs]:64]!\n"
      "vld1.32 {d18, d19}, [%[rhs]:64]!\n"
      "vdup.32 q10, %[multiplicative_offset]\n"
      "vdup.32 q11, %[rounding_offset]\n"
      "vdup.32 q12, %[shift]\n"
      "vdup.32 q13, d16[0]\n"
      "vdup.32 q8, d16[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d8, d8, d10\n"
      "vpadd.u32 d9, d12, d14\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q13\n"
      "vadd.s32 q4, q4, q8\n"
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q4, q4, q9\n"
      "vmul.i32 q0, q0, q10\n"
      "vmul.i32 q4, q4, q10\n"
      "vadd.i32 q0, q0, q11\n"
      "vadd.i32 q4, q4, q11\n"
      "vshl.s32 q0, q0, q12\n"
      "vshl.s32 q4, q4, q12\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d0, q0\n"
      "vqmovun.s16 d8, q4\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.32 {d8[0]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 3, 1,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 3, 1, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d6, d7, d8}, [%[lhs]:64]!\n"
      "vld1.32 {d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d9, d6\n"
      "vmull.u8 q6, d9, d7\n"
      "vmull.u8 q7, d9, d8\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[multiplicative_offset]\n"
      "vdup.32 q7, %[rounding_offset]\n"
      "vdup.32 q8, %[shift]\n"
      "vdup.32 q3, d8[0]\n"
      "vdup.32 q9, d8[1]\n"
      "vdup.32 q4, d9[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d2, d2, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d4, d4, d4\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q3\n"
      "vadd.s32 q1, q1, q9\n"
      "vadd.s32 q2, q2, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vadd.s32 q2, q2, q5\n"
      "vmul.i32 q0, q0, q6\n"
      "vmul.i32 q1, q1, q6\n"
      "vmul.i32 q2, q2, q6\n"
      "vadd.i32 q0, q0, q7\n"
      "vadd.i32 q1, q1, q7\n"
      "vadd.i32 q2, q2, q7\n"
      "vshl.s32 q0, q0, q8\n"
      "vshl.s32 q1, q1, q8\n"
      "vshl.s32 q2, q2, q8\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d2, q1\n"
      "vqmovn.s32 d4, q2\n"
      "vqmovun.s16 d0, q0\n"
      "vqmovun.s16 d2, q1\n"
      "vqmovun.s16 d4, q2\n"

      // RowMajorOutput::Output
      "vst1.8 {d0[0]}, [%[result]]!\n"
      "vst1.8 {d2[0]}, [r0]!\n"
      "vst1.8 {d4[0]}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 3, 2,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 3, 2, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13, d14}, [%[lhs]:64]!\n"
      "vld1.32 {d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d15, d12\n"
      "vmull.u8 q10, d16, d12\n"
      "vmull.u8 q11, d15, d13\n"
      "vmull.u8 q12, d16, d13\n"
      "vmull.u8 q13, d15, d14\n"
      "vmull.u8 q14, d16, d14\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, %[multiplicative_offset]\n"
      "vdup.32 q9, %[rounding_offset]\n"
      "vdup.32 q10, %[shift]\n"
      "vdup.32 q11, d12[0]\n"
      "vdup.32 q12, d12[1]\n"
      "vdup.32 q6, d13[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d4, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d8, d8, d10\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q11\n"
      "vadd.s32 q2, q2, q12\n"
      "vadd.s32 q4, q4, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q2, q2, q7\n"
      "vadd.s32 q4, q4, q7\n"
      "vmul.i32 q0, q0, q8\n"
      "vmul.i32 q2, q2, q8\n"
      "vmul.i32 q4, q4, q8\n"
      "vadd.i32 q0, q0, q9\n"
      "vadd.i32 q2, q2, q9\n"
      "vadd.i32 q4, q4, q9\n"
      "vshl.s32 q0, q0, q10\n"
      "vshl.s32 q2, q2, q10\n"
      "vshl.s32 q4, q4, q10\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d0, q0\n"
      "vqmovun.s16 d4, q2\n"
      "vqmovun.s16 d8, q4\n"

      // RowMajorOutput::Output
      "vst1.16 {d0[0]}, [%[result]]!\n"
      "vst1.16 {d4[0]}, [r0]!\n"
      "vst1.16 {d8[0]}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 3, 3,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 3, 3, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // 3x3 lanes loop.
      "1:"

      "vld1.8 {d21, d22, d23}, [%[rhs]:64]!\n"
      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d18, d21\n"
      "vld1.8 {d19}, [%[lhs]:64]!\n"
      "vmull.u8 q13, d18, d22\n"
      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vmull.u8 q14, d18, d23\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q15, d19, d21\n"
      "pld [%[rhs], #64]\n"
      "vpadal.u16 q0, q12\n"
      "vpadal.u16 q1, q13\n"
      "vpadal.u16 q2, q14\n"
      "vpadal.u16 q3, q15\n"
      "vmull.u8 q12, d19, d22\n"
      "vmull.u8 q13, d19, d23\n"
      "vmull.u8 q14, d20, d21\n"
      "vmull.u8 q15, d20, d22\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vmull.u8 q9, d20, d23\n"
      "vpadal.u16 q4, q12\n"
      "vpadal.u16 q5, q13\n"
      "vpadal.u16 q6, q14\n"
      "vpadal.u16 q7, q15\n"
      "vpadal.u16 q8, q9\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "vld1.32 {d18, d19}, [%[lhs]:64]!\n"
      "vld1.32 {d20, d21}, [%[rhs]:64]!\n"
      "vdup.32 q11, %[multiplicative_offset]\n"
      "vdup.32 q12, %[rounding_offset]\n"
      "vdup.32 q13, %[shift]\n"
      "vdup.32 q14, d18[0]\n"
      "vdup.32 q15, d18[1]\n"
      "vdup.32 q9, d19[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d12, d12, d14\n"
      "vpadd.u32 d13, d16, d16\n"

      // StaticQuantization::Transform
      "vadd.s32 q0, q0, q14\n"
      "vadd.s32 q3, q3, q15\n"
      "vadd.s32 q6, q6, q9\n"
      "vadd.s32 q0, q0, q10\n"
      "vadd.s32 q3, q3, q10\n"
      "vadd.s32 q6, q6, q10\n"
      "vmul.i32 q0, q0, q11\n"
      "vmul.i32 q3, q3, q11\n"
      "vmul.i32 q6, q6, q11\n"
      "vadd.i32 q0, q0, q12\n"
      "vadd.i32 q3, q3, q12\n"
      "vadd.i32 q6, q6, q12\n"
      "vshl.s32 q0, q0, q13\n"
      "vshl.s32 q3, q3, q13\n"
      "vshl.s32 q6, q6, q13\n"
      "vqmovn.s32 d0, q0\n"
      "vqmovn.s32 d6, q3\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovun.s16 d0, q0\n"
      "vqmovun.s16 d6, q3\n"
      "vqmovun.s16 d12, q6\n"

      // RowMajorOutput::Output
      "vst1.16 {d0[0]}, [%[result]]!\n"
      "vst1.8 {d0[2]}, [%[result]]!\n"
      "vst1.16 {d6[0]}, [r0]!\n"
      "vst1.8 {d6[2]}, [r0]!\n"
      "vst1.16 {d12[0]}, [r1]!\n"
      "vst1.8 {d12[2]}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "d30", "d31", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d2}, [%[lhs]:64]!\n"
      "vld1.32 {d3}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q2, d3, d2\n"
      "vpadal.u16 q0, q2\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d4}, [%[lhs]:64]!\n"
      "vld1.32 {d5, d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d5, d4\n"
      "vmull.u8 q5, d6, d4\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d6}, [%[lhs]:64]!\n"
      "vld1.32 {d7, d8, d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d7, d6\n"
      "vmull.u8 q6, d8, d6\n"
      "vmull.u8 q7, d9, d6\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d8}, [%[lhs]:64]!\n"
      "vld1.32 {d9, d10, d11, d12}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q7, d9, d8\n"
      "vmull.u8 q8, d10, d8\n"
      "vmull.u8 q9, d11, d8\n"
      "vmull.u8 q10, d12, d8\n"
      "vpadal.u16 q0, q7\n"
      "vpadal.u16 q1, q8\n"
      "vpadal.u16 q2, q9\n"
      "vpadal.u16 q3, q10\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
        "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 5,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 5, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d10, d11, d12, d13}, [%[rhs]:64]!\n"
      "vld1.32 {d14}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q8, d10, d14\n"
      "vmull.u8 q9, d11, d14\n"
      "vmull.u8 q10, d12, d14\n"
      "vmull.u8 q11, d13, d14\n"
      "vld1.32 {d10}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q8\n"
      "vpadal.u16 q1, q9\n"
      "vpadal.u16 q2, q10\n"
      "vpadal.u16 q3, q11\n"
      "vmull.u8 q8, d10, d14\n"
      "vpadal.u16 q4, q8\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d10, d11}, [%[lhs]:64]!\n"
      "vld1.32 {d12, d13, d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q5, d10[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d8\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q1, q1, q7\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1}, [%[result]]!\n"
      "vst1.32 {d2[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 6,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 6, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13, d14, d15}, [%[rhs]:64]!\n"
      "vld1.32 {d16}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vmull.u8 q11, d14, d16\n"
      "vmull.u8 q12, d15, d16\n"
      "vld1.32 {d12, d13}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vpadal.u16 q4, q9\n"
      "vpadal.u16 q5, q10\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [%[rhs]:64]!\n"
      "vdup.32 q6, d12[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q1, q1, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q1, q1, q8\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1, d2}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 7,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 7, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d14, d15, d16, d17}, [%[rhs]:64]!\n"
      "vld1.32 {d18}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d17, d18\n"
      "vld1.32 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q10\n"
      "vpadal.u16 q1, q11\n"
      "vpadal.u16 q2, q12\n"
      "vpadal.u16 q3, q13\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d14, d15}, [%[lhs]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [%[rhs]:64]!\n"
      "vdup.32 q7, d14[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"
      "vpadd.u32 d3, d12, d12\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q1, q1, q7\n"
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q1, q1, q9\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1, d2}, [%[result]]!\n"
      "vst1.32 {d3[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 8,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 8, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // 1x8 lanes loop.
      "1:"

      "vld1.32 {d17, d18, d19, d20}, [%[rhs]:256]!\n"
      "vld1.32 {d16}, [%[lhs]:64]!\n"
      "vmull.u8 q11, d16, d17\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d16, d19\n"
      "vmull.u8 q14, d16, d20\n"
      "vld1.32 {d17, d18, d19, d20}, [%[rhs]:256]!\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vpadal.u16 q3, q14\n"
      "pld [%[rhs], #256]\n"
      "vmull.u8 q15, d16, d17\n"
      "vmull.u8 q11, d16, d18\n"
      "vmull.u8 q12, d16, d19\n"
      "vmull.u8 q13, d16, d20\n"
      "pld [%[lhs], #32]\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vpadal.u16 q4, q15\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d16, d17}, [%[lhs]:64]!\n"
      "vld1.32 {d18, d19, d20, d21}, [%[rhs]:64]!\n"
      "vdup.32 q8, d16[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"
      "vpadd.u32 d3, d12, d14\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q1, q1, q8\n"
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q1, q1, q10\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1, d2, d3}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d4, d5}, [%[lhs]:64]!\n"
      "vld1.32 {d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d6, d4\n"
      "vmull.u8 q5, d6, d5\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q2, d8[0]\n"
      "vdup.32 q4, d8[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d2, d2, d2\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q2\n"
      "vadd.s32 q1, q1, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.32 {d2[0]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q6, d10, d8\n"
      "vmull.u8 q7, d11, d8\n"
      "vmull.u8 q8, d10, d9\n"
      "vmull.u8 q9, d11, d9\n"
      "vpadal.u16 q0, q6\n"
      "vpadal.u16 q1, q7\n"
      "vpadal.u16 q2, q8\n"
      "vpadal.u16 q3, q9\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, d8[0]\n"
      "vdup.32 q4, d8[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d4, d4, d6\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q2, q2, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q2, q2, q5\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d4}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d14, d12\n"
      "vmull.u8 q10, d15, d12\n"
      "vmull.u8 q11, d16, d12\n"
      "vmull.u8 q12, d14, d13\n"
      "vmull.u8 q13, d15, d13\n"
      "vmull.u8 q14, d16, d13\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, d12[0]\n"
      "vdup.32 q6, d12[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q3, q3, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q3, q3, q7\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]]!\n"
      "vst1.32 {d6}, [r0]!\n"
      "vst1.32 {d7[0]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // 2x4 lanes loop.
      "1:"

      "vld1.8 {d18, d19, d20, d21}, [%[rhs]:256]!\n"
      "vld1.8 {d16}, [%[lhs]:64]!\n"
      "vmull.u8 q11, d16, d18\n"
      "vld1.8 {d17}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d16, d19\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q13, d16, d20\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q14, d16, d21\n"
      "vmull.u8 q15, d17, d18\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vmull.u8 q11, d17, d19\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d17, d21\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vpadal.u16 q3, q14\n"
      "vpadal.u16 q4, q15\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d16, d17}, [%[lhs]:64]!\n"
      "vld1.32 {d18, d19}, [%[rhs]:64]!\n"
      "vdup.32 q10, d16[0]\n"
      "vdup.32 q8, d16[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d8, d8, d10\n"
      "vpadd.u32 d9, d12, d14\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q10\n"
      "vadd.s32 q4, q4, q8\n"
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q4, q4, q9\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1}, [%[result]]!\n"
      "vst1.32 {d8, d9}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d6, d7, d8}, [%[lhs]:64]!\n"
      "vld1.32 {d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d9, d6\n"
      "vmull.u8 q6, d9, d7\n"
      "vmull.u8 q7, d9, d8\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q3, d8[0]\n"
      "vdup.32 q6, d8[1]\n"
      "vdup.32 q4, d9[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d2, d2, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d4, d4, d4\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q3\n"
      "vadd.s32 q1, q1, q6\n"
      "vadd.s32 q2, q2, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vadd.s32 q2, q2, q5\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.32 {d2[0]}, [r0]!\n"
      "vst1.32 {d4[0]}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13, d14}, [%[lhs]:64]!\n"
      "vld1.32 {d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d15, d12\n"
      "vmull.u8 q10, d16, d12\n"
      "vmull.u8 q11, d15, d13\n"
      "vmull.u8 q12, d16, d13\n"
      "vmull.u8 q13, d15, d14\n"
      "vmull.u8 q14, d16, d14\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, d12[0]\n"
      "vdup.32 q9, d12[1]\n"
      "vdup.32 q6, d13[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d4, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d8, d8, d10\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q2, q2, q9\n"
      "vadd.s32 q4, q4, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q2, q2, q7\n"
      "vadd.s32 q4, q4, q7\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d4}, [r0]!\n"
      "vst1.32 {d8}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // 3x3 lanes loop.
      "1:"

      "vld1.8 {d21, d22, d23}, [%[rhs]:64]!\n"
      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d18, d21\n"
      "vld1.8 {d19}, [%[lhs]:64]!\n"
      "vmull.u8 q13, d18, d22\n"
      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vmull.u8 q14, d18, d23\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q15, d19, d21\n"
      "pld [%[rhs], #64]\n"
      "vpadal.u16 q0, q12\n"
      "vpadal.u16 q1, q13\n"
      "vpadal.u16 q2, q14\n"
      "vpadal.u16 q3, q15\n"
      "vmull.u8 q12, d19, d22\n"
      "vmull.u8 q13, d19, d23\n"
      "vmull.u8 q14, d20, d21\n"
      "vmull.u8 q15, d20, d22\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vmull.u8 q9, d20, d23\n"
      "vpadal.u16 q4, q12\n"
      "vpadal.u16 q5, q13\n"
      "vpadal.u16 q6, q14\n"
      "vpadal.u16 q7, q15\n"
      "vpadal.u16 q8, q9\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "vld1.32 {d18, d19}, [%[lhs]:64]!\n"
      "vld1.32 {d20, d21}, [%[rhs]:64]!\n"
      "vdup.32 q11, d18[0]\n"
      "vdup.32 q12, d18[1]\n"
      "vdup.32 q9, d19[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d12, d12, d14\n"
      "vpadd.u32 d13, d16, d16\n"

      // StaticQuantizationInt32::Transform
      "vadd.s32 q0, q0, q11\n"
      "vadd.s32 q3, q3, q12\n"
      "vadd.s32 q6, q6, q9\n"
      "vadd.s32 q0, q0, q10\n"
      "vadd.s32 q3, q3, q10\n"
      "vadd.s32 q6, q6, q10\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]]!\n"
      "vst1.32 {d6}, [r0]!\n"
      "vst1.32 {d7[0]}, [r0]!\n"
      "vst1.32 {d12}, [r1]!\n"
      "vst1.32 {d13[0]}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "d30", "d31", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d2}, [%[lhs]:64]!\n"
      "vld1.32 {d3}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q2, d3, d2\n"
      "vpadal.u16 q0, q2\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[scale]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vcvt.f32.s32 q0, q0\n"
      "vmul.f32 q0, q0, q6\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11", "d12",
        "d13", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d4}, [%[lhs]:64]!\n"
      "vld1.32 {d5, d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d5, d4\n"
      "vmull.u8 q5, d6, d4\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[scale]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vcvt.f32.s32 q0, q0\n"
      "vmul.f32 q0, q0, q6\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "d12", "d13", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d6}, [%[lhs]:64]!\n"
      "vld1.32 {d7, d8, d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d7, d6\n"
      "vmull.u8 q6, d8, d6\n"
      "vmull.u8 q7, d9, d6\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[scale]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vcvt.f32.s32 q0, q0\n"
      "vmul.f32 q0, q0, q6\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d8}, [%[lhs]:64]!\n"
      "vld1.32 {d9, d10, d11, d12}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q7, d9, d8\n"
      "vmull.u8 q8, d10, d8\n"
      "vmull.u8 q9, d11, d8\n"
      "vmull.u8 q10, d12, d8\n"
      "vpadal.u16 q0, q7\n"
      "vpadal.u16 q1, q8\n"
      "vpadal.u16 q2, q9\n"
      "vpadal.u16 q3, q10\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[scale]\n"
      "vdup.32 q4, d8[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vcvt.f32.s32 q0, q0\n"
      "vmul.f32 q0, q0, q6\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 5,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 5, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d10, d11, d12, d13}, [%[rhs]:64]!\n"
      "vld1.32 {d14}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q8, d10, d14\n"
      "vmull.u8 q9, d11, d14\n"
      "vmull.u8 q10, d12, d14\n"
      "vmull.u8 q11, d13, d14\n"
      "vld1.32 {d10}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q8\n"
      "vpadal.u16 q1, q9\n"
      "vpadal.u16 q2, q10\n"
      "vpadal.u16 q3, q11\n"
      "vmull.u8 q8, d10, d14\n"
      "vpadal.u16 q4, q8\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d10, d11}, [%[lhs]:64]!\n"
      "vld1.32 {d12, d13, d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, %[scale]\n"
      "vdup.32 q5, d10[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d8\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q1, q1, q7\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q1, q1\n"
      "vmul.f32 q0, q0, q8\n"
      "vmul.f32 q1, q1, q8\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1}, [%[result]]!\n"
      "vst1.32 {d2[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 6,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 6, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13, d14, d15}, [%[rhs]:64]!\n"
      "vld1.32 {d16}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vmull.u8 q11, d14, d16\n"
      "vmull.u8 q12, d15, d16\n"
      "vld1.32 {d12, d13}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vpadal.u16 q4, q9\n"
      "vpadal.u16 q5, q10\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [%[rhs]:64]!\n"
      "vdup.32 q9, %[scale]\n"
      "vdup.32 q6, d12[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q1, q1, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q1, q1, q8\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q1, q1\n"
      "vmul.f32 q0, q0, q9\n"
      "vmul.f32 q1, q1, q9\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1, d2}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 7,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 7, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d14, d15, d16, d17}, [%[rhs]:64]!\n"
      "vld1.32 {d18}, [%[lhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d17, d18\n"
      "vld1.32 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[rhs], #128]\n"
      "vpadal.u16 q0, q10\n"
      "vpadal.u16 q1, q11\n"
      "vpadal.u16 q2, q12\n"
      "vpadal.u16 q3, q13\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d14, d15}, [%[lhs]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [%[rhs]:64]!\n"
      "vdup.32 q10, %[scale]\n"
      "vdup.32 q7, d14[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"
      "vpadd.u32 d3, d12, d12\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q1, q1, q7\n"
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q1, q1, q9\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q1, q1\n"
      "vmul.f32 q0, q0, q10\n"
      "vmul.f32 q1, q1, q10\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1, d2}, [%[result]]!\n"
      "vst1.32 {d3[0]}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 8,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 8, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // 1x8 lanes loop.
      "1:"

      "vld1.32 {d17, d18, d19, d20}, [%[rhs]:256]!\n"
      "vld1.32 {d16}, [%[lhs]:64]!\n"
      "vmull.u8 q11, d16, d17\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d16, d19\n"
      "vmull.u8 q14, d16, d20\n"
      "vld1.32 {d17, d18, d19, d20}, [%[rhs]:256]!\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vpadal.u16 q3, q14\n"
      "pld [%[rhs], #256]\n"
      "vmull.u8 q15, d16, d17\n"
      "vmull.u8 q11, d16, d18\n"
      "vmull.u8 q12, d16, d19\n"
      "vmull.u8 q13, d16, d20\n"
      "pld [%[lhs], #32]\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vpadal.u16 q4, q15\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d16, d17}, [%[lhs]:64]!\n"
      "vld1.32 {d18, d19, d20, d21}, [%[rhs]:64]!\n"
      "vdup.32 q11, %[scale]\n"
      "vdup.32 q8, d16[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d2, d8, d10\n"
      "vpadd.u32 d3, d12, d14\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q8\n"
      "vadd.s32 q1, q1, q8\n"
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q1, q1, q10\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q1, q1\n"
      "vmul.f32 q0, q0, q11\n"
      "vmul.f32 q1, q1, q11\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1, d2, d3}, [%[result]]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d4, d5}, [%[lhs]:64]!\n"
      "vld1.32 {d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d6, d4\n"
      "vmull.u8 q5, d6, d5\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[scale]\n"
      "vdup.32 q2, d8[0]\n"
      "vdup.32 q4, d8[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d2, d2, d2\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q2\n"
      "vadd.s32 q1, q1, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q1, q1\n"
      "vmul.f32 q0, q0, q6\n"
      "vmul.f32 q1, q1, q6\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.32 {d2[0]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q6, d10, d8\n"
      "vmull.u8 q7, d11, d8\n"
      "vmull.u8 q8, d10, d9\n"
      "vmull.u8 q9, d11, d9\n"
      "vpadal.u16 q0, q6\n"
      "vpadal.u16 q1, q7\n"
      "vpadal.u16 q2, q8\n"
      "vpadal.u16 q3, q9\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[scale]\n"
      "vdup.32 q7, d8[0]\n"
      "vdup.32 q4, d8[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d4, d4, d6\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q2, q2, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q2, q2, q5\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q2, q2\n"
      "vmul.f32 q0, q0, q6\n"
      "vmul.f32 q2, q2, q6\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d4}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d14, d12\n"
      "vmull.u8 q10, d15, d12\n"
      "vmull.u8 q11, d16, d12\n"
      "vmull.u8 q12, d14, d13\n"
      "vmull.u8 q13, d15, d13\n"
      "vmull.u8 q14, d16, d13\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, %[scale]\n"
      "vdup.32 q9, d12[0]\n"
      "vdup.32 q6, d12[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q3, q3, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q3, q3, q7\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q3, q3\n"
      "vmul.f32 q0, q0, q8\n"
      "vmul.f32 q3, q3, q8\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]]!\n"
      "vst1.32 {d6}, [r0]!\n"
      "vst1.32 {d7[0]}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // 2x4 lanes loop.
      "1:"

      "vld1.8 {d18, d19, d20, d21}, [%[rhs]:256]!\n"
      "vld1.8 {d16}, [%[lhs]:64]!\n"
      "vmull.u8 q11, d16, d18\n"
      "vld1.8 {d17}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d16, d19\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q13, d16, d20\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q14, d16, d21\n"
      "vmull.u8 q15, d17, d18\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vmull.u8 q11, d17, d19\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d17, d21\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vpadal.u16 q3, q14\n"
      "vpadal.u16 q4, q15\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d16, d17}, [%[lhs]:64]!\n"
      "vld1.32 {d18, d19}, [%[rhs]:64]!\n"
      "vdup.32 q10, %[scale]\n"
      "vdup.32 q11, d16[0]\n"
      "vdup.32 q8, d16[1]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d8, d8, d10\n"
      "vpadd.u32 d9, d12, d14\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q11\n"
      "vadd.s32 q4, q4, q8\n"
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q4, q4, q9\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q4, q4\n"
      "vmul.f32 q0, q0, q10\n"
      "vmul.f32 q4, q4, q10\n"

      // RowMajorOutput::Output
      "vst1.32 {d0, d1}, [%[result]]!\n"
      "vst1.32 {d8, d9}, [r0]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d6, d7, d8}, [%[lhs]:64]!\n"
      "vld1.32 {d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d9, d6\n"
      "vmull.u8 q6, d9, d7\n"
      "vmull.u8 q7, d9, d8\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.32 {d10, d11}, [%[rhs]:64]!\n"
      "vdup.32 q6, %[scale]\n"
      "vdup.32 q3, d8[0]\n"
      "vdup.32 q7, d8[1]\n"
      "vdup.32 q4, d9[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d0, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d2, d2, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d4, d4, d4\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q3\n"
      "vadd.s32 q1, q1, q7\n"
      "vadd.s32 q2, q2, q4\n"
      "vadd.s32 q0, q0, q5\n"
      "vadd.s32 q1, q1, q5\n"
      "vadd.s32 q2, q2, q5\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q1, q1\n"
      "vcvt.f32.s32 q2, q2\n"
      "vmul.f32 q0, q0, q6\n"
      "vmul.f32 q1, q1, q6\n"
      "vmul.f32 q2, q2, q6\n"

      // RowMajorOutput::Output
      "vst1.32 {d0[0]}, [%[result]]!\n"
      "vst1.32 {d2[0]}, [r0]!\n"
      "vst1.32 {d4[0]}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.32 {d12, d13, d14}, [%[lhs]:64]!\n"
      "vld1.32 {d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d15, d12\n"
      "vmull.u8 q10, d16, d12\n"
      "vmull.u8 q11, d15, d13\n"
      "vmull.u8 q12, d16, d13\n"
      "vmull.u8 q13, d15, d14\n"
      "vmull.u8 q14, d16, d14\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.32 {d14, d15}, [%[rhs]:64]!\n"
      "vdup.32 q8, %[scale]\n"
      "vdup.32 q9, d12[0]\n"
      "vdup.32 q10, d12[1]\n"
      "vdup.32 q6, d13[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d4, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d8, d8, d10\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q2, q2, q10\n"
      "vadd.s32 q4, q4, q6\n"
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q2, q2, q7\n"
      "vadd.s32 q4, q4, q7\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q2, q2\n"
      "vcvt.f32.s32 q4, q4\n"
      "vmul.f32 q0, q0, q8\n"
      "vmul.f32 q2, q2, q8\n"
      "vmul.f32 q4, q4, q8\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d4}, [r0]!\n"
      "vst1.32 {d8}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // 3x3 lanes loop.
      "1:"

      "vld1.8 {d21, d22, d23}, [%[rhs]:64]!\n"
      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d18, d21\n"
      "vld1.8 {d19}, [%[lhs]:64]!\n"
      "vmull.u8 q13, d18, d22\n"
      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vmull.u8 q14, d18, d23\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q15, d19, d21\n"
      "pld [%[rhs], #64]\n"
      "vpadal.u16 q0, q12\n"
      "vpadal.u16 q1, q13\n"
      "vpadal.u16 q2, q14\n"
      "vpadal.u16 q3, q15\n"
      "vmull.u8 q12, d19, d22\n"
      "vmull.u8 q13, d19, d23\n"
      "vmull.u8 q14, d20, d21\n"
      "vmull.u8 q15, d20, d22\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vmull.u8 q9, d20, d23\n"
      "vpadal.u16 q4, q12\n"
      "vpadal.u16 q5, q13\n"
      "vpadal.u16 q6, q14\n"
      "vpadal.u16 q7, q15\n"
      "vpadal.u16 q8, q9\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "vld1.32 {d18, d19}, [%[lhs]:64]!\n"
      "vld1.32 {d20, d21}, [%[rhs]:64]!\n"
      "vdup.32 q11, %[scale]\n"
      "vdup.32 q12, d18[0]\n"
      "vdup.32 q13, d18[1]\n"
      "vdup.32 q9, d19[0]\n"

      // RowMajorOutput::Prepare
      "add r0, %[result], %[stride]\n"
      "add r1, r0, %[stride]\n"

      // Reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d12, d12, d14\n"
      "vpadd.u32 d13, d16, d16\n"

      // StaticQuantizationFloat::Transform
      "vadd.s32 q0, q0, q12\n"
      "vadd.s32 q3, q3, q13\n"
      "vadd.s32 q6, q6, q9\n"
      "vadd.s32 q0, q0, q10\n"
      "vadd.s32 q3, q3, q10\n"
      "vadd.s32 q6, q6, q10\n"
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q3, q3\n"
      "vcvt.f32.s32 q6, q6\n"
      "vmul.f32 q0, q0, q11\n"
      "vmul.f32 q3, q3, q11\n"
      "vmul.f32 q6, q6, q11\n"

      // RowMajorOutput::Output
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]]!\n"
      "vst1.32 {d6}, [r0]!\n"
      "vst1.32 {d7[0]}, [r0]!\n"
      "vst1.32 {d12}, [r1]!\n"
      "vst1.32 {d13[0]}, [r1]!\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "d30", "d31", "cc", "memory");
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm for arm32 requires: GEMMLOWP_NEON_32!"
#endif

#endif  // GEMMLOWP_META_QUANTIZED_MUL_KERNELS_ARM_32_H_
