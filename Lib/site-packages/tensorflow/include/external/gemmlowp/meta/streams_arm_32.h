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

#ifndef GEMMLOWP_META_STREAMS_ARM_32_H_
#define GEMMLOWP_META_STREAMS_ARM_32_H_

#ifdef GEMMLOWP_NEON_32

#include <cassert>
#include <cstdint>

namespace gemmlowp {
namespace meta {

template <>
inline void Stream<uint8_t, 1, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x1.
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x2.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x3.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x4.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x5.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x6.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 1x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x7.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x1.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x2.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x3.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x4.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x5.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x6.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 2x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x7.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x1.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x2.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x3.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x4.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x5.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x6.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 3x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x7.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x1.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vld1.8 {d3[0]}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x2.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x3.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[2]}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x4.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x5.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[4]}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x6.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 4x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x7.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.8 {d3[6]}, [r2]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x1.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vld1.8 {d3[0]}, [r2]!\n"
      "vld1.8 {d4[0]}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x2.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x3.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[2]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[2]}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x4.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x5.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[4]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[4]}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x6.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 5x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x7.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.8 {d3[6]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vld1.8 {d4[6]}, [r3]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x1.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vld1.8 {d3[0]}, [r2]!\n"
      "vld1.8 {d4[0]}, [r3]!\n"
      "vld1.8 {d5[0]}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x2.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vld1.16 {d5[0]}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x3.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[2]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[2]}, [r3]!\n"
      "vld1.16 {d5[0]}, [r4]!\n"
      "vld1.8 {d5[2]}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x4.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x5.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[4]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[4]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.8 {d5[4]}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x6.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.16 {d5[2]}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 6x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x7.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.8 {d3[6]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vld1.8 {d4[6]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.16 {d5[2]}, [r4]!\n"
      "vld1.8 {d5[6]}, [r4]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4", "d5", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x1.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vld1.8 {d3[0]}, [r2]!\n"
      "vld1.8 {d4[0]}, [r3]!\n"
      "vld1.8 {d5[0]}, [r4]!\n"
      "vld1.8 {d6[0]}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x2.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vld1.16 {d5[0]}, [r4]!\n"
      "vld1.16 {d6[0]}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x3.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[2]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[2]}, [r3]!\n"
      "vld1.16 {d5[0]}, [r4]!\n"
      "vld1.8 {d5[2]}, [r4]!\n"
      "vld1.16 {d6[0]}, [r5]!\n"
      "vld1.8 {d6[2]}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x4.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x5.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[4]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[4]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.8 {d5[4]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vld1.8 {d6[4]}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x6.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.16 {d5[2]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vld1.16 {d6[2]}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 7x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x7.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.8 {d3[6]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vld1.8 {d4[6]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.16 {d5[2]}, [r4]!\n"
      "vld1.8 {d5[6]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vld1.16 {d6[2]}, [r5]!\n"
      "vld1.8 {d6[6]}, [r5]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
        "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x1.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.8 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vld1.8 {d3[0]}, [r2]!\n"
      "vld1.8 {d4[0]}, [r3]!\n"
      "vld1.8 {d5[0]}, [r4]!\n"
      "vld1.8 {d6[0]}, [r5]!\n"
      "vld1.8 {d7[0]}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x2.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vld1.16 {d5[0]}, [r4]!\n"
      "vld1.16 {d6[0]}, [r5]!\n"
      "vld1.16 {d7[0]}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x3.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.16 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[2]}, [%[in]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[2]}, [r2]!\n"
      "vld1.16 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[2]}, [r3]!\n"
      "vld1.16 {d5[0]}, [r4]!\n"
      "vld1.8 {d5[2]}, [r4]!\n"
      "vld1.16 {d6[0]}, [r5]!\n"
      "vld1.8 {d6[2]}, [r5]!\n"
      "vld1.16 {d7[0]}, [r6]!\n"
      "vld1.8 {d7[2]}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x4.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vld1.32 {d7[0]}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x5.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d0[4]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[4]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.8 {d4[4]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.8 {d5[4]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vld1.8 {d6[4]}, [r5]!\n"
      "vld1.32 {d7[0]}, [r6]!\n"
      "vld1.8 {d7[4]}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x6.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.16 {d5[2]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vld1.16 {d6[2]}, [r5]!\n"
      "vld1.32 {d7[0]}, [r6]!\n"
      "vld1.16 {d7[2]}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add r0, %[in], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "add r3, r2, %[stride]\n"
      "add r4, r3, %[stride]\n"
      "add r5, r4, %[stride]\n"
      "add r6, r5, %[stride]\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store: 8x8.
      "vld1.32 {d0}, [%[in]]!\n"
      "vld1.32 {d1}, [r0]!\n"
      "vld1.32 {d2}, [r1]!\n"
      "vld1.32 {d3}, [r2]!\n"
      "vld1.32 {d4}, [r3]!\n"
      "vld1.32 {d5}, [r4]!\n"
      "vld1.32 {d6}, [r5]!\n"
      "vld1.32 {d7}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x7.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d0[2]}, [%[in]]!\n"
      "vld1.8 {d0[6]}, [%[in]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.8 {d3[6]}, [r2]!\n"
      "vld1.32 {d4[0]}, [r3]!\n"
      "vld1.16 {d4[2]}, [r3]!\n"
      "vld1.8 {d4[6]}, [r3]!\n"
      "vld1.32 {d5[0]}, [r4]!\n"
      "vld1.16 {d5[2]}, [r4]!\n"
      "vld1.8 {d5[6]}, [r4]!\n"
      "vld1.32 {d6[0]}, [r5]!\n"
      "vld1.16 {d6[2]}, [r5]!\n"
      "vld1.8 {d6[6]}, [r5]!\n"
      "vld1.32 {d7[0]}, [r6]!\n"
      "vld1.16 {d7[2]}, [r6]!\n"
      "vld1.8 {d7[6]}, [r6]!\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "ldr r0, %[multiplicative_sum_offset]\n"
      "ldr r1, %[additive_sum_offset]\n"
      "vmov.32 d0[0], r0\n"
      "vdup.32 q1, r1\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "d0", "d1", "d2", "d3", "d4",
        "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x1
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x2
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x3
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x4
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x5
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x6
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x7
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[4]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[5]}, [%[in]], %[stride]\n"
      "vld1.8 {d0[6]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vst1.32 {d0}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d16, d16, d16\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d2", "d3", "d16", "d17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x1
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x2
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x3
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x4
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x5
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x6
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x7
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[2]}, [%[in]], %[stride]\n"
      "vld1.16 {d0[3]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.16 {d1[2]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vuzp.8 d0, d1\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vst1.32 {d0, d1}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d16, d16, d18\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x1
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x2
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x3
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x4
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x5
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x6
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[7], d1[7], d2[7]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x7
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld3.8 {d0[0], d1[0], d2[0]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[1], d1[1], d2[1]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[2], d1[2], d2[2]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[3], d1[3], d2[3]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[4], d1[4], d2[4]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[5], d1[5], d2[5]}, [%[in]], %[stride]\n"
      "vld3.8 {d0[6], d1[6], d2[6]}, [%[in]], %[stride]\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vst1.32 {d0, d1, d2}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d20\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "cc",
        "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x1
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x2
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x3
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x4
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x5
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x6
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x7
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vst1.32 {d16, d17}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
        "d23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x1
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x2
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x3
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x4
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x5
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x6
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.8 {d4[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x7
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.8 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.8 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.8 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.8 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.8 {d4[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.8 {d4[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.8 {d4[6]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d24\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x1
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x2
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x3
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x4
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x5
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x6
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld1.16 {d5[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x7
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld1.16 {d4[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld1.16 {d4[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld1.16 {d4[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld1.16 {d4[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld1.16 {d5[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld1.16 {d5[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld1.16 {d5[2]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vuzp.8 d4, d5\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:128]!\n"
      "vst1.32 {d4, d5}, [%[out]:128]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:128]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x1
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x2
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x3
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x4
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x5
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x6
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %[stride], %[stride], #4\n"
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[1]}, [%[in]]!\n"
      "vld3.8 {d4[7], d5[7], d6[7]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x7
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vld1.32 {d0[0]}, [%[in]]!\n"
      "vld3.8 {d4[0], d5[0], d6[0]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[0]}, [%[in]]!\n"
      "vld3.8 {d4[1], d5[1], d6[1]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[0]}, [%[in]]!\n"
      "vld3.8 {d4[2], d5[2], d6[2]}, [%[in]], %[stride]\n"
      "vld1.32 {d3[0]}, [%[in]]!\n"
      "vld3.8 {d4[3], d5[3], d6[3]}, [%[in]], %[stride]\n"
      "vld1.32 {d0[1]}, [%[in]]!\n"
      "vld3.8 {d4[4], d5[4], d6[4]}, [%[in]], %[stride]\n"
      "vld1.32 {d1[1]}, [%[in]]!\n"
      "vld3.8 {d4[5], d5[5], d6[5]}, [%[in]], %[stride]\n"
      "vld1.32 {d2[1]}, [%[in]]!\n"
      "vld3.8 {d4[6], d5[6], d6[6]}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:64]!\n"
      "vst1.32 {d4, d5, d6}, [%[out]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d28\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:64]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d16", "d17", "d18", "d19",
        "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x1
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x2
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x3
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x4
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x5
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x6
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "vmov.i16 q8, #0\n"
      "vmov.i16 q9, #0\n"
      "vmov.i16 q10, #0\n"
      "vmov.i16 q11, #0\n"
      "vmov.i16 q12, #0\n"
      "vmov.i16 q13, #0\n"
      "vmov.i16 q14, #0\n"
      "vmov.i16 q15, #0\n"

      // Reduce count by leftovers.
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "vld1.32 {d7}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x7
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vmov.i8 d4, #0\n"
      "vmov.i8 d5, #0\n"
      "vmov.i8 d6, #0\n"
      "vmov.i8 d7, #0\n"
      "vld1.32 {d0}, [%[in]], %[stride]\n"
      "vld1.32 {d1}, [%[in]], %[stride]\n"
      "vld1.32 {d2}, [%[in]], %[stride]\n"
      "vld1.32 {d3}, [%[in]], %[stride]\n"
      "vld1.32 {d4}, [%[in]], %[stride]\n"
      "vld1.32 {d5}, [%[in]], %[stride]\n"
      "vld1.32 {d6}, [%[in]], %[stride]\n"
      "pld [%[in]]\n"
      "vtrn.8 d0, d1\n"
      "vtrn.8 d2, d3\n"
      "vtrn.8 d4, d5\n"
      "vtrn.8 d6, d7\n"
      "vtrn.16 d0, d2\n"
      "vtrn.16 d1, d3\n"
      "vtrn.16 d4, d6\n"
      "vtrn.16 d5, d7\n"
      "vtrn.32 d0, d4\n"
      "vtrn.32 d1, d5\n"
      "vtrn.32 d2, d6\n"
      "vtrn.32 d3, d7\n"
      "vaddw.u8 q8, q8, d0\n"
      "vaddw.u8 q9, q9, d1\n"
      "vaddw.u8 q10, q10, d2\n"
      "vaddw.u8 q11, q11, d3\n"
      "vaddw.u8 q12, q12, d4\n"
      "vaddw.u8 q13, q13, d5\n"
      "vaddw.u8 q14, q14, d6\n"
      "vaddw.u8 q15, q15, d7\n"
      "vst1.32 {d0, d1, d2, d3}, [%[out]:256]!\n"
      "vst1.32 {d4, d5, d6, d7}, [%[out]:256]!\n"

      // Aggregator Reduction.
      "vmov.32 d0[0], %[multiplicative_sum_offset]\n"
      "vdup.32 q1, %[additive_sum_offset]\n"
      "vpaddl.u16 q8, q8\n"
      "vpaddl.u16 q9, q9\n"
      "vpaddl.u16 q10, q10\n"
      "vpaddl.u16 q11, q11\n"
      "vpaddl.u16 q12, q12\n"
      "vpaddl.u16 q13, q13\n"
      "vpaddl.u16 q14, q14\n"
      "vpaddl.u16 q15, q15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"
      "vpadd.u32 d24, d24, d25\n"
      "vpadd.u32 d26, d26, d27\n"
      "vpadd.u32 d28, d28, d29\n"
      "vpadd.u32 d30, d30, d31\n"
      "vpadd.u32 d16, d16, d18\n"
      "vpadd.u32 d17, d20, d22\n"
      "vpadd.u32 d18, d24, d26\n"
      "vpadd.u32 d19, d28, d30\n"
      "vmul.i32 q8, q8, d0[0]\n"
      "vmul.i32 q9, q9, d0[0]\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vst1.32 {d16, d17, d18, d19}, [%[out]:256]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18",
        "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
        "d29", "d30", "d31", "cc", "memory");
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm for arm32 requires: GEMMLOWP_NEON_32!"
#endif

#endif  // GEMMLOWP_META_STREAMS_ARM_32_H_
