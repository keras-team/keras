// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// kernel_neon.h: a collection of NEON optimized kernels.
// Check in kernel_default.h which one(s) are actually used by default.
// Others are mere experiments; they are still covered by tests
// in case they might be useful some day.

#ifndef GEMMLOWP_INTERNAL_KERNEL_NEON_H_
#define GEMMLOWP_INTERNAL_KERNEL_NEON_H_

#include "kernel.h"

#include <arm_neon.h>
#include <cassert>

namespace gemmlowp {

// The kernels here are specifically arm 32bit assembly, not arm 64bit.
#ifdef GEMMLOWP_NEON_32

// Our main GEMM kernel.
struct NEON_32_Kernel12x4Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2>, 3>,
                       KernelSideFormat<CellFormat<4, 2>, 1> >
      Format;

  const char* Name() const override { return "NEON, 12x4, depth 2"; }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel (NEON 12x4)");

// For iOS assembler, the %= style of local labels cause compilation errors,
//  so use numerical ones instead. See
// http://stackoverflow.com/questions/3898435/labels-in-gcc-inline-assembly
// If you add any labels, remember to undef them at the end.
#define GEMMLOWP_LABEL_CLEAR_ACCUMULATORS "1"
#define GEMMLOWP_LABEL_BEFORE_LOOP "2"
#define GEMMLOWP_LABEL_LOOP "3"
#define GEMMLOWP_LABEL_AFTER_LOOP "4"

    assert(dst_row_stride == 1);
    (void)dst_row_stride;
    asm volatile(
        // Overview of register layout:
        //
        // A 2x4 cell of Rhs is stored in 16bit in d0--d1 (q0).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in d2--d7
        // (q1--q3).
        // A 12x4 block of accumulators is stored in 32bit in q4--q15.
        //
        //                   +-----+-----+-----+-----+
        //                   |d0[0]|d0[1]|d0[2]|d0[3]|
        //              Rhs  +-----+-----+-----+-----+
        //                   |d1[0]|d1[1]|d1[2]|d1[3]|
        //                   +-----+-----+-----+-----+
        //
        //                   |     |     |     |     |
        //
        //    Lhs            |     |     |     |     |
        //
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //
        //                            Accumulator

        // Load 1 Rhs cell of size 2x4
        "vld1.8 {d0}, [%[rhs_ptr]]!\n"
        // Load 3 Lhs cells of size 4x2 each
        "vld1.8 {d2}, [%[lhs_ptr]]!\n"
        "vld1.8 {d4}, [%[lhs_ptr]]!\n"
        "vld1.8 {d6}, [%[lhs_ptr]]!\n"

        // Check if start_depth==0 to decide whether we will clear
        // accumulators or load existing accumulators.
        "cmp %[start_depth], #0\n"

        // Multiply dst_col_stride by 4 == sizeof(int32) to use
        // it as a byte offset below.
        "lsl %[dst_col_stride], #2\n"

        "beq " GEMMLOWP_LABEL_CLEAR_ACCUMULATORS
        "f\n"

        // Load accumulators (start_depth != 0)
        "mov r1, %[dst_ptr]\n"
        "subs %[run_depth], #2\n"
        "mov r0, r1\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "add r1, %[dst_col_stride]\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]\n"
        "mov r0, r1\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "add r1, %[dst_col_stride]\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]\n"
        "mov r0, r1\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "add r1, %[dst_col_stride]\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]\n"
        "mov r0, r1\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]\n"

        "b " GEMMLOWP_LABEL_BEFORE_LOOP "f\n"

        GEMMLOWP_LABEL_CLEAR_ACCUMULATORS
        ":\n"

        // Clear accumulators (start_depth == 0)
        "vmov.s32 q4, #0\n"
        "subs %[run_depth], #2\n"
        "vmov.s32 q8, q4\n"
        "vmov.s32 q12, q4\n"
        "vmov.s32 q5, q4\n"
        "vmov.s32 q9, q4\n"
        "vmov.s32 q13, q4\n"
        "vmov.s32 q6, q4\n"
        "vmov.s32 q10, q4\n"
        "vmov.s32 q14, q4\n"
        "vmov.s32 q7, q4\n"
        "vmov.s32 q11, q4\n"
        "vmov.s32 q15, q4\n"

        GEMMLOWP_LABEL_BEFORE_LOOP
        ":\n"

        // If there are only two levels of depth, skip the loop.
        "beq " GEMMLOWP_LABEL_AFTER_LOOP "f\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"
        // Expand Lhs/Rhs cells to 16 bit.
        // Note: moving theses vmovls further down to allow for
        // longer data pipelining helps a little on A57 but is
        // harmful on A53 --- It looks as if A53 doesn't like
        // interleaving vmovl's into the vmlal's.
        "vmovl.u8 q0, d0\n"
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q4, d2, d0[0]\n"
        "vmlal.u16 q5, d2, d0[1]\n"
        "vmlal.u16 q6, d2, d0[2]\n"
        "vmlal.u16 q7, d2, d0[3]\n"
        "vldr d2, [%[lhs_ptr]]\n"
        "vmlal.u16 q8, d4, d0[0]\n"
        "vmlal.u16 q9, d4, d0[1]\n"
        "vmlal.u16 q10, d4, d0[2]\n"
        "vmlal.u16 q11, d4, d0[3]\n"
        "vldr d4, [%[lhs_ptr], #8]\n"
        "vmlal.u16 q12, d6, d0[0]\n"
        "vmlal.u16 q13, d6, d0[1]\n"
        "vmlal.u16 q14, d6, d0[2]\n"
        "vmlal.u16 q15, d6, d0[3]\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vldr d0, [%[rhs_ptr]]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q4, d3, d1[0]\n"
        "vmlal.u16 q5, d3, d1[1]\n"
        "add %[lhs_ptr], #24\n"
        "vmlal.u16 q6, d3, d1[2]\n"
        "vmlal.u16 q7, d3, d1[3]\n"
        "add %[rhs_ptr], #8\n"
        "vmlal.u16 q8, d5, d1[0]\n"
        "vmlal.u16 q9, d5, d1[1]\n"
        "subs %[run_depth], #2\n"
        "vmlal.u16 q10, d5, d1[2]\n"
        "vmlal.u16 q11, d5, d1[3]\n"
        "vmlal.u16 q12, d7, d1[0]\n"
        "vmlal.u16 q13, d7, d1[1]\n"
        "vmlal.u16 q14, d7, d1[2]\n"
        "vmlal.u16 q15, d7, d1[3]\n"

        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Do remaining arithmetic for the last 2 levels of depth.

        // Expand Lhs/Rhs cells to 16 bit.
        "vmovl.u8 q0, d0\n"
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q4, d2, d0[0]\n"
        "vmlal.u16 q5, d2, d0[1]\n"
        "vmlal.u16 q6, d2, d0[2]\n"
        "vmlal.u16 q7, d2, d0[3]\n"
        "vmlal.u16 q8, d4, d0[0]\n"
        "vmlal.u16 q9, d4, d0[1]\n"
        "vmlal.u16 q10, d4, d0[2]\n"
        "vmlal.u16 q11, d4, d0[3]\n"
        "vmlal.u16 q12, d6, d0[0]\n"
        "vmlal.u16 q13, d6, d0[1]\n"
        "vmlal.u16 q14, d6, d0[2]\n"
        "vmlal.u16 q15, d6, d0[3]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q4, d3, d1[0]\n"
        "vmlal.u16 q5, d3, d1[1]\n"
        "vmlal.u16 q6, d3, d1[2]\n"
        "vmlal.u16 q7, d3, d1[3]\n"
        "vmlal.u16 q8, d5, d1[0]\n"
        "vmlal.u16 q9, d5, d1[1]\n"
        "vmlal.u16 q10, d5, d1[2]\n"
        "vmlal.u16 q11, d5, d1[3]\n"
        "vmlal.u16 q12, d7, d1[0]\n"
        "vmlal.u16 q13, d7, d1[1]\n"
        "vmlal.u16 q14, d7, d1[2]\n"
        "vmlal.u16 q15, d7, d1[3]\n"

        // Store accumulators
        "mov r1, %[dst_ptr]\n"
        "mov r0, r1\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "add r1, %[dst_col_stride]\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]\n"
        "mov r0, r1\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "add r1, %[dst_col_stride]\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]\n"
        "mov r0, r1\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "add r1, %[dst_col_stride]\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]\n"
        "mov r0, r1\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride] "r"(dst_col_stride)
        :  // clobbers
        "cc", "memory", "r0", "r1",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
#undef GEMMLOWP_LABEL_CLEAR_ACCUMULATORS
#undef GEMMLOWP_LABEL_BEFORE_LOOP
#undef GEMMLOWP_LABEL_LOOP
#undef GEMMLOWP_LABEL_AFTER_LOOP
  }
};

struct NEON_32_Kernel12x4Depth2Assuming12BitProducts : KernelBase {
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 1> >
      Format;

  const char* Name() const override {
    return "NEON, 12x4, depth 2, assuming 12-bit products";
  }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label(
        "optimized kernel (NEON 12x4, assuming 12-bit products)");
    assert(dst_row_stride == 1);
    (void)dst_row_stride;

// See comments above for why we need local numerical labels in our asm.
#define GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS "1"
#define GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT "2"
#define GEMMLOWP_LABEL_32 "3"
#define GEMMLOWP_LABEL_24 "4"
#define GEMMLOWP_LABEL_16 "5"
#define GEMMLOWP_LABEL_8 "6"
#define GEMMLOWP_LABEL_2 "7"

    // This kernel is special in that it uses local 16-bit accumulators.
    // Because it assumes that each product fits in 12 bits, it can accumulate
    // 16 products into a local 16-bit accumulator without risking overflow.
    // At that point, it must accumulate these local 16-bit accumulators back
    // into global 32-bit accumulators, which have to be stored in memory for
    // lack of register space.
    // This 12x4 block of global accumulators is laid out as 3 cells of size 4x4
    // stored in diagonal-major order like this for the first 4x4 cell:
    //
    //   0   4   8  12
    //  13   1   5   9
    //  10  14   2   6
    //   7  11  15   3
    //
    // and likewise for the 2nd  cell (16--31) and 3rd cell (32--47)
    std::int32_t global_accumulators[3 * 4 * 4];
    asm volatile(
        // Compute stride between consecutive columns, in bytes
        "mov r0, #4\n"  // multiply by 4 = sizeof(int32)
        "mul %[dst_col_stride], r0\n"

        "cmp %[start_depth], #0\n"
        "bne"
        " " GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT
        "f\n"

        // If start_depth==0, we need to clear our global accumulators
        "mov r0, %[global_accumulators]\n"
        "vmov.s32 q8, #0\n"
        "vmov.s32 q9, q8\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "b " GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
        "f\n"

        // If start_depth!=0, we need to load our existing global accumulators
        GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT
        ":\n"
        // Load global accumulators from destination matrix, column-major
        "mov r1, %[dst_ptr]\n"
        "mov r0, %[dst_col_stride]\n"
        "sub r0, #32\n"
        "vld1.32 {d0,d1}, [r1]!\n"
        "vld1.32 {d8,d9}, [r1]!\n"
        "vld1.32 {d16,d17}, [r1], r0\n"
        "vld1.32 {d2,d3}, [r1]!\n"
        "vld1.32 {d10,d11}, [r1]!\n"
        "vld1.32 {d18,d19}, [r1], r0\n"
        "vld1.32 {d4,d5}, [r1]!\n"
        "vld1.32 {d12,d13}, [r1]!\n"
        "vld1.32 {d20,d21}, [r1], r0\n"
        "vld1.32 {d6,d7}, [r1]!\n"
        "vld1.32 {d14,d15}, [r1]!\n"
        "vld1.32 {d22,d23}, [r1], r0\n"
        // Now we need to convert the global accumulator registers to
        // 4x4-block-wise diagonal-major order. What we effectively want to do
        // is to rotate the rows, however the accumulators are stored in
        // column-major order in registers. So we achieve this by
        // transposing, rotating the registers, and transposing again each
        // 4x4 block.
        //
        // Transpose 3 4x4 blocks separately
        "vtrn.32 q0, q1\n"
        "vtrn.32 q2, q3\n"
        "vswp d1, d4\n"
        "vswp d3, d6\n"
        "vtrn.32 q4, q5\n"
        "vtrn.32 q6, q7\n"
        "vswp d9, d12\n"
        "vswp d11, d14\n"
        "vtrn.32 q8, q9\n"
        "vtrn.32 q10, q11\n"
        "vswp d17, d20\n"
        "vswp d19, d22\n"
        // Rotate the registers
        "vext.32 q1, q1, q1, #1\n"
        "vext.32 q2, q2, q2, #2\n"
        "vext.32 q3, q3, q3, #3\n"
        "vext.32 q5, q5, q5, #1\n"
        "vext.32 q6, q6, q6, #2\n"
        "vext.32 q7, q7, q7, #3\n"
        "vext.32 q9, q9, q9, #1\n"
        "vext.32 q10, q10, q10, #2\n"
        "vext.32 q11, q11, q11, #3\n"
        // Transpose again and store into our global accumulators
        // buffer. These two operations are done at once using vst4.
        "mov r0, %[global_accumulators]\n"
        "vst4.32 {d0,d2,d4,d6}, [r0]!\n"
        "vst4.32 {d1,d3,d5,d7}, [r0]!\n"
        "vst4.32 {d8,d10,d12,d14}, [r0]!\n"
        "vst4.32 {d9,d11,d13,d15}, [r0]!\n"
        "vst4.32 {d16,d18,d20,d22}, [r0]!\n"
        "vst4.32 {d17,d19,d21,d23}, [r0]!\n"

        /* Main loop */

        GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
        ":\n"

    // Overview of register layout:
    //
    // Registers q4--q16 are the local 16-bit accumulators.
    // However, each entry in the result matrix is represented
    // by *two* local 16-bit accumulators: one for even levels
    // of depth and one for odd levels of depth. These correspond
    // to the scalars at even and odd indices within each q-register.
    // Thus we effectively use 32 bits of register space for each
    // entry in the result matrix. The accumulators register layout
    // is the same as was described above for the global 32-bit
    // accumulators (3 cells of size 4x4 in diagonal-major order)
    // with the only difference that instead of 32bit values we have
    // pairs of 16bit values.
    //
    // A 2x4 cell of Rhs is stored in 8bit in d0.
    // A 12x2 block of 3 4x2 cells Lhs is stored in 8bit in d1--d3.
    //
    //                      +--------+--------+--------+--------+
    //                      |d0[0]   |d0[2]   |d0[4]   |d0[6]   |
    //                 Rhs  +--------+--------+--------+--------+
    //                      |d0[1]   |d0[3]   |d0[5]   |d0[7]   |
    //                      +--------+--------+--------+--------+
    //
    //                      |        |        |        |        |
    //
    //    Lhs               |        |        |        |        |
    //
    //  +-----+-----+ - - - +--------+--------+--------+--------+
    //  |d1[0]|d1[1]|       |q4[0,1] |q5[0,1] |q6[0,1] |q7[0,1] |
    //  |d1[2]|d1[3]|       |q7[2,3] |q4[2,3] |q5[2,3] |q6[2,3] |
    //  |d1[4]|d1[5]|       |q6[4,5] |q7[4,5] |q4[4,5] |q5[4,5] |
    //  |d1[6]|d1[7]|       |q5[6,7] |q6[6,7] |q7[6,7] |q4[6,7] |
    //  +-----+-----+ - - - +--------+--------+--------+--------+
    //  |d2[0]|d2[1]|       |q8[0,1] |q8[0,1] |q8[0,1] |q8[0,1] |
    //  |d2[2]|d2[3]|       |q9[2,3] |q9[2,3] |q9[2,3] |q9[2,3] |
    //  |d2[4]|d2[5]|       |q10[4,5]|q10[4,5]|q10[4,5]|q10[4,5]|
    //  |d2[6]|d2[7]|       |q11[6,7]|q11[6,7]|q11[6,7]|q11[6,7]|
    //  +-----+-----+ - - - +--------+--------+--------+--------+
    //  |d3[0]|d3[1]|       |q12[0,1]|q12[0,1]|q12[0,1]|q12[0,1]|
    //  |d3[2]|d3[3]|       |q13[2,3]|q13[2,3]|q13[2,3]|q13[2,3]|
    //  |d3[4]|d3[5]|       |q14[4,5]|q14[4,5]|q14[4,5]|q14[4,5]|
    //  |d3[6]|d3[7]|       |q15[6,7]|q15[6,7]|q15[6,7]|q15[6,7]|
    //  +-----+-----+ - - - +--------+--------+--------+--------+
    //
    //                            Local 16-bit accumulators
    //                         Note: 2 scalars per matrix entry

#define GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH \
  /* Load 3 Lhs cells of size 4x2 */          \
  "vld1.8 {d1,d2,d3}, [%[lhs_ptr]:64]!\n"     \
                                              \
  /* Load 1 Rhs cell of size 2x4 */           \
  "vld1.8 {d0}, [%[rhs_ptr]:64]!\n"           \
                                              \
  /* Multiply-accumulate */                   \
  "vmlal.u8 q4, d1, d0\n"                     \
  "vmlal.u8 q8, d2, d0\n"                     \
  "vmlal.u8 q12, d3, d0\n"                    \
  "vext.8 d0, d0, d0, #2\n"                   \
  "vmlal.u8 q5, d1, d0\n"                     \
  "vmlal.u8 q9, d2, d0\n"                     \
  "vmlal.u8 q13, d3, d0\n"                    \
  "vext.8 d0, d0, d0, #2\n"                   \
  "vmlal.u8 q6, d1, d0\n"                     \
  "vmlal.u8 q10, d2, d0\n"                    \
  "vmlal.u8 q14, d3, d0\n"                    \
  "vext.8 d0, d0, d0, #2\n"                   \
  "vmlal.u8 q7, d1, d0\n"                     \
  "vmlal.u8 q11, d2, d0\n"                    \
  "vmlal.u8 q15, d3, d0\n"                    \
                                              \
  "sub %[run_depth], #2\n"

#define GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH       \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH       \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH       \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH

        // Clear local 16-bit accumulators
        "vmov.s32 q4, #0\n"
        "vmov.s32 q5, q4\n"
        "vmov.s32 q6, q4\n"
        "vmov.s32 q7, q4\n"
        "vmov.s32 q8, q4\n"
        "vmov.s32 q9, q4\n"
        "vmov.s32 q10, q4\n"
        "vmov.s32 q11, q4\n"
        "vmov.s32 q12, q4\n"
        "vmov.s32 q13, q4\n"
        "vmov.s32 q14, q4\n"
        "vmov.s32 q15, q4\n"

        // Select a suitable number of depth levels
        // to process at this iteration. TODO (benoitjacob) I guess that
        // someone who really knows asm should make this a jump table.
        "cmp %[run_depth], #32\n"
        "bge " GEMMLOWP_LABEL_32
        "f\n"
        "cmp %[run_depth], #24\n"
        "bge " GEMMLOWP_LABEL_24
        "f\n"
        "cmp %[run_depth], #16\n"
        "bge " GEMMLOWP_LABEL_16
        "f\n"
        "cmp %[run_depth], #8\n"
        "bge " GEMMLOWP_LABEL_8
        "f\n"
        "b " GEMMLOWP_LABEL_2 "f\n"

        GEMMLOWP_LABEL_32
        ":\n" GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH GEMMLOWP_LABEL_24
        ":\n" GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH GEMMLOWP_LABEL_16
        ":\n" GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH GEMMLOWP_LABEL_8
        ":\n" GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH
            GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH
                GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH GEMMLOWP_LABEL_2
        ":\n" GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH

        // Accumulate the local accumulators into the global accumulators.
        // This is about summing adjacent pairs of 16-bit scalars into
        // single 32-bit scalars, so we use pairwise long addition (vpadal).
        "mov r0, %[global_accumulators]\n"
        "mov r1, %[global_accumulators]\n"
        "vld1.32 {d0,d1,d2,d3}, [r0]!\n"
        "vld1.32 {d4,d5,d6,d7}, [r0]!\n"
        "vpadal.u16 q0, q4\n"
        "vpadal.u16 q1, q5\n"
        "vpadal.u16 q2, q6\n"
        "vpadal.u16 q3, q7\n"
        "vst1.32 {d0,d1,d2,d3}, [r1]!\n"
        "vst1.32 {d4,d5,d6,d7}, [r1]!\n"
        "vld1.32 {d0,d1,d2,d3}, [r0]!\n"
        "vld1.32 {d4,d5,d6,d7}, [r0]!\n"
        "vpadal.u16 q0, q8\n"
        "vpadal.u16 q1, q9\n"
        "vpadal.u16 q2, q10\n"
        "vpadal.u16 q3, q11\n"
        "vst1.32 {d0,d1,d2,d3}, [r1]!\n"
        "vst1.32 {d4,d5,d6,d7}, [r1]!\n"
        "vld1.32 {d0,d1,d2,d3}, [r0]!\n"
        "vld1.32 {d4,d5,d6,d7}, [r0]!\n"
        "vpadal.u16 q0, q12\n"
        "vpadal.u16 q1, q13\n"
        "vpadal.u16 q2, q14\n"
        "vpadal.u16 q3, q15\n"
        "vst1.32 {d0,d1,d2,d3}, [r1]!\n"
        "vst1.32 {d4,d5,d6,d7}, [r1]!\n"

        // Loop.
        "cmp %[run_depth], #0\n"
        "bne " GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
        "b\n"

#undef GEMMLOWP_CLEAR_LOCAL_ACCUMULATORS
#undef GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH
#undef GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH
#undef GEMMLOWP_ADD_TO_GLOBAL_ACCUMULATORS

        /* end of main loop */

        // Store the global accumulators to the destination matrix
        // (column-major)
        // This is the reverse of the steps that we followed at the beginning
        // when we load the global accumulators from the destination matrix.
        // The problem is the same: how to convert 4x4 blocks
        // between column-major and diagonal-major orders.
        // Like above, we do this by rotating rows, and we achieve that by
        // tranposing, rotating columns, and transposing again.
        //
        // Load and transpose 4x4 blocks of global accumulators
        // These two steps are done at once by the vld4 instruction.
        "mov r0, %[global_accumulators]\n"
        "vld4.32 {d0,d2,d4,d6}, [r0]!\n"
        "vld4.32 {d1,d3,d5,d7}, [r0]!\n"
        "vld4.32 {d8,d10,d12,d14}, [r0]!\n"
        "vld4.32 {d9,d11,d13,d15}, [r0]!\n"
        "vld4.32 {d16,d18,d20,d22}, [r0]!\n"
        "vld4.32 {d17,d19,d21,d23}, [r0]!\n"
        // Rotate the rows of each 4x4 block
        "vext.32 q1, q1, q1, #3\n"
        "vext.32 q2, q2, q2, #2\n"
        "vext.32 q3, q3, q3, #1\n"
        "vext.32 q5, q5, q5, #3\n"
        "vext.32 q6, q6, q6, #2\n"
        "vext.32 q7, q7, q7, #1\n"
        "vext.32 q9, q9, q9, #3\n"
        "vext.32 q10, q10, q10, #2\n"
        "vext.32 q11, q11, q11, #1\n"
        // Transpose again each 4x4 block
        "vtrn.32 q0, q1\n"
        "vtrn.32 q2, q3\n"
        "vswp d1, d4\n"
        "vswp d3, d6\n"
        "vtrn.32 q4, q5\n"
        "vtrn.32 q6, q7\n"
        "vswp d9, d12\n"
        "vswp d11, d14\n"
        "vtrn.32 q8, q9\n"
        "vtrn.32 q10, q11\n"
        "vswp d17, d20\n"
        "vswp d19, d22\n"
        // Store into the column-major destination matrix
        "mov r1, %[dst_ptr]\n"
        "mov r0, %[dst_col_stride]\n"
        "sub r0, #32\n"
        "vst1.32 {d0,d1}, [r1]!\n"
        "vst1.32 {d8,d9}, [r1]!\n"
        "vst1.32 {d16,d17}, [r1], r0\n"
        "vst1.32 {d2,d3}, [r1]!\n"
        "vst1.32 {d10,d11}, [r1]!\n"
        "vst1.32 {d18,d19}, [r1], r0\n"
        "vst1.32 {d4,d5}, [r1]!\n"
        "vst1.32 {d12,d13}, [r1]!\n"
        "vst1.32 {d20,d21}, [r1], r0\n"
        "vst1.32 {d6,d7}, [r1]!\n"
        "vst1.32 {d14,d15}, [r1]!\n"
        "vst1.32 {d22,d23}, [r1], r0\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth), [dst_col_stride] "r"(dst_col_stride),
        [global_accumulators] "r"(&global_accumulators[0])
        :  // clobbers
        "cc", "memory", "r0", "r1",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
#undef GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
#undef GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT
#undef GEMMLOWP_LABEL_32
#undef GEMMLOWP_LABEL_24
#undef GEMMLOWP_LABEL_16
#undef GEMMLOWP_LABEL_8
#undef GEMMLOWP_LABEL_2
  }
};

struct NEON_32bit_GEMM_Int8Operands_LhsNonzero : KernelBase {
  typedef KernelFormat<
      KernelSideFormatInt8<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormatInt8<CellFormat<2, 16, CellOrder::WidthMajor>, 1> >
      Format;
  const char* Name() const override {
    return "NEON, 4x2, depth 16, accumulating two within signed int16";
  }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    (void)dst_row_stride;
#define GEMMLOWP_LABEL_AFTER_LOOP "1"
#define GEMMLOWP_LABEL_LOOP "2"
#define GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES "3"
#define GEMMLOWP_LABEL_STORE "4"
    asm volatile(
        // Multiply dst_col_stride by 4 == sizeof(int32) to use
        // it as a byte offset below.
        "lsl %[dst_col_stride], %[dst_col_stride], #2\n"

        // Overview of register layout:
        //
        // A 2x16 block of Rhs is stored in 8 bit in d0--d3.
        // A 4x16 block of Lhs is stored in 8 bit in d4--d7. That is only
        // half of the register space required, so we loop over these registers
        // twice. Only half of it, a 2x16 block, is stored in d4--d7 at
        // any given time.
        //
        // A 4x2 block of accumulators is stored in q8--q15 (as 4x32 bit
        // components which need to be horizontally-added at the end)
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a widening
        // multiply over the 8 first levels of depth, producing int16x8
        // vectors of products for each position in the accumulator matrix.
        // Here comes the special trick: since the operands are signed int8,
        // their range being [ -2^7 , 2^7 ), their products are in range
        // [ -2^14 , 2^14 - 1 ), meaning that we can add two such values
        // without any risk of overflowing int16.
        // We thus proceed with the 8 next levels of depth, multiplying
        // again Lhs by Rhs, accumulating into this existing int16x8 vector.
        //
        // Only then, having processed 16 levels of depth, do we need to
        // horizontally add these int16x8 accumulators into the final
        // int32x4 accumulators.
        //
        // As we do not have enough registers to store all 16 int16x8
        // temporary-16bit-accumulators, we have them cycle through q4--q7.
        //
        //
        // Register layout (ignoring the q4--q7 temporary 16bit accumulators):
        //
        //                               +----+----+
        //                               | d0 | d2 |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               | .  | .  |
        //                       Rhs     +----+----+
        //                               | d1 | d3 |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               +----+----+
        //
        //                               |    |    |
        //
        //    Lhs                        |    |    |
        //
        //  +--------+--------+ - - - -  +----+----+
        //  | d4 ... | d5 ... |          | q8 | q9 |
        //  | d6 ... | d7 ... |          | q10| q11|
        //  | d4 ... | d5 ... |          | q12| q13|
        //  | d6 ... | d7 ... |          | q14| q15|
        //  +--------+--------+ - - - -  +----+----+
        //
        //                               Accumulator
        //

        // Clear accumulators, and, interleaved with it,
        // initial loads of the first loop iteration,
        // taken out of the loop so that in the loop itself we have
        // optimal streaming of data from memory.
        "vldr d0, [%[rhs_ptr], #0]\n"
        "vmov.i32 q8, #0\n"
        "vldr d4, [%[lhs_ptr], #0]\n"
        "vmov.i32 q9, #0\n"
        "vldr d2, [%[rhs_ptr], #16]\n"
        "vmov.i32 q10, q8\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vmov.i32 q11, q8\n"
        "vldr d1, [%[rhs_ptr], #8]\n"
        "vmov.i32 q12, q8\n"
        "vldr d5, [%[lhs_ptr], #8]\n"
        "vmov.i32 q13, q8\n"
        "vldr d3, [%[rhs_ptr], #24]\n"
        "vmov.i32 q14, q8\n"
        "vldr d7, [%[lhs_ptr], #24]\n"
        "vmov.i32 q15, q8\n"

        // General loop.
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Multiply 8 first levels of depth.
        "vmull.s8    q4,  d0,  d4\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vldr d4, [%[lhs_ptr], #32]\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vmull.s8    q7,  d2,  d6\n"
        "vldr d6, [%[lhs_ptr], #48]\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vldr d5, [%[lhs_ptr], #40]\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vmlal.s8    q7,  d3,  d7\n"
        "vldr d7, [%[lhs_ptr], #56]\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q8,  q4\n"
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "vpadal.s16   q9,  q5\n"
        "subs %[run_depth], %[run_depth], #16\n"
        "vpadal.s16   q10, q6\n"
        "vpadal.s16   q11, q7\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP
        "f\n"

        // Multiply first half.
        "vmull.s8    q4,  d0,  d4\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vldr d4, [%[lhs_ptr], #0]\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vldr d0, [%[rhs_ptr], #0]\n"
        "vmull.s8    q7,  d2,  d6\n"
        "vldr d2, [%[rhs_ptr], #16]\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vldr d5, [%[lhs_ptr], #8]\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vldr d1, [%[rhs_ptr], #8]\n"
        "vmlal.s8    q7,  d3,  d7\n"
        "vldr d3, [%[rhs_ptr], #24]\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q12, q4\n"
        "vldr d7, [%[lhs_ptr], #24]\n"
        "vpadal.s16   q13, q5\n"
        "vpadal.s16   q14, q6\n"
        "vpadal.s16   q15, q7\n"

        "b " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Multiply first half.
        "vmull.s8    q4,  d0,  d4\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vmull.s8    q7,  d2,  d6\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vmlal.s8    q7,  d3,  d7\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q12, q4\n"
        "vpadal.s16   q13, q5\n"
        "vpadal.s16   q14, q6\n"
        "vpadal.s16   q15, q7\n"
        "cmp %[start_depth], #0\n"

        // Reduce 32bit accumulators horizontally.
        "vpadd.s32 d0, d16, d17\n"
        "vpadd.s32 d1, d18, d19\n"
        "vpadd.s32 d2, d20, d21\n"
        "vpadd.s32 d3, d22, d23\n"
        "vpadd.s32 d4, d24, d25\n"
        "vpadd.s32 d5, d26, d27\n"
        "vpadd.s32 d6, d28, d29\n"
        "vpadd.s32 d7, d30, d31\n"

        "bne " GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        "f\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "vpadd.s32 d8, d0, d2\n"
        "vpadd.s32 d9, d4, d6\n"
        "vpadd.s32 d10, d1, d3\n"
        "vpadd.s32 d11, d5, d7\n"

        "b " GEMMLOWP_LABEL_STORE "f\n"

        GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        ":\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise),
        // and load destination values from memory.
        "mov r0, %[dst_ptr]\n"
        "vld1.32 {d16, d17}, [r0], %[dst_col_stride]\n"
        "vpadd.s32 d8, d0, d2\n"
        "vpadd.s32 d9, d4, d6\n"
        "vld1.32 {d18, d19}, [r0]\n"
        "vpadd.s32 d10, d1, d3\n"
        "vpadd.s32 d11, d5, d7\n"

        // Add horizontally-reduced accumulators into
        // the values loaded from memory
        "vadd.s32 q4, q8, q4\n"
        "vadd.s32 q5, q9, q5\n"

        GEMMLOWP_LABEL_STORE
        ":\n"
        // Store back into memory
        "mov r0, %[dst_ptr]\n"
        "vst1.32 {d8, d9}, [r0], %[dst_col_stride]\n"
        "vst1.32 {d10, d11}, [r0]\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr), [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride] "r"(dst_col_stride)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
#undef GEMMLOWP_LABEL_LOOP
#undef GEMMLOWP_LABEL_AFTER_LOOP
#undef GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
#undef GEMMLOWP_LABEL_STORE
  }
};

// Same as NEON_32bit_GEMM_Int8Operands_LhsNonzero, but uses a side format that
// requires that user inputs were originally int8. This avoids the uint8->int8
// conversion in the pack step.
struct NEON_32bit_GEMM_Int8Operands_LhsNonzero_Int8Inputs
    : NEON_32bit_GEMM_Int8Operands_LhsNonzero {
  typedef KernelFormat<
      KernelSideFormatInt8Inputs<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormatInt8Inputs<CellFormat<2, 16, CellOrder::WidthMajor>, 1> >
      Format;
};

#endif  // GEMMLOWP_NEON_32

// The kernels here are specifically arm 64bit assembly, not arm 32bit.
#ifdef GEMMLOWP_NEON_64

struct NEON_64bit_GEMM_Int8Operands_LhsNonzero : KernelBase {
  typedef KernelFormat<
      KernelSideFormatInt8<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormatInt8<CellFormat<4, 16, CellOrder::WidthMajor>, 1> >
      Format;
  const char* Name() const override {
    return "NEON, 4x4, depth 16, accumulating two within signed int16";
  }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    (void)dst_row_stride;
#define GEMMLOWP_LABEL_AFTER_LOOP_LAST16 "1"
#define GEMMLOWP_LABEL_LOOP "2"
#define GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES "3"
#define GEMMLOWP_LABEL_STORE "4"
    asm volatile(
        // Clear accumulators, and, interleaved with it,
        // initial loads of the first loop iteration,
        // taken out of the loop so that in the loop itself we have
        // optimal streaming of data from memory.
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"
        "dup v16.4s, wzr\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "dup v17.4s, wzr\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"
        "dup v18.4s, wzr\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "dup v19.4s, wzr\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "dup v20.4s, wzr\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"
        "dup v21.4s, wzr\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "dup v22.4s, wzr\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"
        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"
        "dup v31.4s, wzr\n"

        // Multiply dst_col_stride by 4 == sizeof(int32) to use
        // it as a byte offset below.
        "lsl %[dst_col_stride], %[dst_col_stride], #2\n"

        // Initial arithmetic of the first loop iteration,
        // taken out of the loop so that in the loop itself we have
        // optimal streaming of data from memory.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        "subs %[run_depth], %[run_depth], #16\n"

        // If the loop depth is only 16, then we can skip the general loop
        // and go straight to the final part of the code.
        "beq " GEMMLOWP_LABEL_AFTER_LOOP_LAST16 "f\n"

        // General loop.
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Overview of register layout:
        //
        // A 4x16 block of Rhs is stored in 8 bit in v0--v3.
        // A 4x16 block of Lhs is stored in 8 bit in v4--v7.
        //
        // A 4x4 block of accumulators is stored in v16-v31 (as 4x32 bit
        // components which need to be horizontally-added at the end)
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a widening
        // multiply over the 8 first levels of depth, producing int16x8
        // vectors of products for each position in the accumulator matrix.
        // Here comes the special trick: since the operands are signed int8,
        // their range being [ -2^7 , 2^7 ), their products are in range
        // [ -2^14 , 2^14 - 1 ), meaning that we can add two such values
        // without any risk of overflowing int16.
        // We thus proceed with the 8 next levels of depth, multiplying
        // again Lhs by Rhs, accumulating into this existing int16x8 vector.
        //
        // Only then, having processed 16 levels of depth, do we need to
        // horizontally add these int16x8 accumulators into the final
        // int32x4 accumulators.
        //
        // As we do not have enough registers to store all 16 int16x8
        // temporary-16bit-accumulators, we have them cycle through v8--v15.
        //
        //
        // Register layout (ignoring the v8--v15 temporary 16bit accumulators):
        //
        //                               +--------+--------+--------+--------+
        //                               |v0.b[0] |v1.b[0] |v2.b[0] |v3.b[0] |
        //                          Rhs  +--------+--------+--------+--------+
        //                               |  ...   |  ...   |  ...   |  ...   |
        //                               +--------+--------+--------+--------|
        //                               |v0.b[15]|v1.b[15]|v2.b[15]|v3.b[15]|
        //                               +--------+--------+--------+--------+
        //
        //                               |        |        |        |        |
        //
        //    Lhs                        |        |        |        |        |
        //
        //  +-------+-----+--------+ - - +--------+--------+--------+--------+
        //  |v4.b[0]| ... |v4.b[15]|     | v16.4s | v17.4s | v18.4s | v19.4s |
        //  |v5.b[0]| ... |v5.b[15]|     | v20.4s | v21.4s | v22.4s | v23.4s |
        //  |v6.b[0]| ... |v6.b[15]|     | v24.4s | v25.4s | v26.4s | v27.4s |
        //  |v7.b[0]| ... |v7.b[15]|     | v28.4s | v29.4s | v30.4s | v31.4s |
        //  +-------+--------------+ - - +--------+--------+--------+--------+
        //
        //                                                Accumulator
        //

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        "sadalp  v24.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "sadalp  v25.4s, v9.8h\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "sadalp  v26.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "sadalp  v27.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "sadalp  v28.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "sadalp  v29.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "sadalp  v30.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v31.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        // Loop. Decrement loop index (depth) by 16, since we just handled
        // 16 levels of depth.  Do this subs a bit before the end of the loop
        // for better dispatch on A57.
        "subs %[run_depth], %[run_depth], #16\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        // Final code for the last 16 levels of depth.
        // There is nothing to load anymore, only some arithmetic to finish.
        GEMMLOWP_LABEL_AFTER_LOOP_LAST16
        ":\n"

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"
        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"

        "sadalp  v24.4s, v8.8h\n"
        "sadalp  v25.4s, v9.8h\n"
        "sadalp  v26.4s, v10.8h\n"
        "sadalp  v27.4s, v11.8h\n"
        "sadalp  v28.4s, v12.8h\n"
        "sadalp  v29.4s, v13.8h\n"
        "sadalp  v30.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        // Reduce 32bit accumulators horizontally.
        "addp v0.4s, v16.4s, v20.4s\n"
        "addp v2.4s, v17.4s, v21.4s\n"
        "addp v4.4s, v18.4s, v22.4s\n"
        "addp v6.4s, v19.4s, v23.4s\n"
        "addp v1.4s, v24.4s, v28.4s\n"
        "addp v3.4s, v25.4s, v29.4s\n"
        "addp v5.4s, v26.4s, v30.4s\n"
        "addp v7.4s, v27.4s, v31.4s\n"

        "cmp %[start_depth], #0\n"
        "bne " GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        "f\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "addp v12.4s, v0.4s, v1.4s\n"
        "addp v13.4s, v2.4s, v3.4s\n"
        "addp v14.4s, v4.4s, v5.4s\n"
        "addp v15.4s, v6.4s, v7.4s\n"

        "b " GEMMLOWP_LABEL_STORE "f\n"

        GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        ":\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise),
        // and load destination values from memory.
        "mov x0, %[dst_ptr]\n"
        "ld1 {v12.16b}, [x0], %[dst_col_stride]\n"
        "addp v8.4s, v0.4s, v1.4s\n"
        "ld1 {v13.16b}, [x0], %[dst_col_stride]\n"
        "addp v9.4s, v2.4s, v3.4s\n"
        "ld1 {v14.16b}, [x0], %[dst_col_stride]\n"
        "addp v10.4s, v4.4s, v5.4s\n"
        "ld1 {v15.16b}, [x0]\n"
        "addp v11.4s, v6.4s, v7.4s\n"

        // Add horizontally-reduced accumulators into
        // the values loaded from memory
        "add v12.4s, v12.4s, v8.4s\n"
        "add v13.4s, v13.4s, v9.4s\n"
        "add v14.4s, v14.4s, v10.4s\n"
        "add v15.4s, v15.4s, v11.4s\n"

        GEMMLOWP_LABEL_STORE
        ":\n"
        // Store back into memory
        "mov x0, %[dst_ptr]\n"
        "st1 {v12.16b}, [x0], %[dst_col_stride]\n"
        "st1 {v13.16b}, [x0], %[dst_col_stride]\n"
        "st1 {v14.16b}, [x0], %[dst_col_stride]\n"
        "st1 {v15.16b}, [x0]\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr), [run_depth] "+r"(run_depth),
        [dst_col_stride] "+r"(dst_col_stride)
        :  // inputs
        [start_depth] "r"(start_depth)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
#undef GEMMLOWP_LABEL_LOOP
#undef GEMMLOWP_LABEL_AFTER_LOOP_LAST16
#undef GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
#undef GEMMLOWP_LABEL_STORE
  }
};

// Same as NEON_32bit_GEMM_Int8Operands_LhsNonzero, but uses a side format that
// requires that user inputs were originally int8. This avoids the uint8->int8
// conversion in the pack step.
struct NEON_64bit_GEMM_Int8Operands_LhsNonzero_Int8Inputs
    : NEON_64bit_GEMM_Int8Operands_LhsNonzero {
  typedef KernelFormat<
      KernelSideFormatInt8Inputs<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormatInt8Inputs<CellFormat<4, 16, CellOrder::WidthMajor>, 1> >
      Format;
};

// Our main GEMM kernel.
struct NEON_64_Kernel12x8Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2>, 3>,
                       KernelSideFormat<CellFormat<4, 2>, 2> >
      Format;

  const char* Name() const override { return "NEON, 12x8, depth 2"; }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    (void)dst_row_stride;
    ScopedProfilingLabel label("optimized kernel (NEON 12x8)");
// See comments above for why we need local numerical labels in our asm.
#define GEMMLOWP_LABEL_CLEAR_ACCUMULATORS "1"
#define GEMMLOWP_LABEL_BEFORE_LOOP "2"
#define GEMMLOWP_LABEL_LOOP "3"
#define GEMMLOWP_LABEL_AFTER_LOOP "4"

    assert(dst_row_stride == 1);
    asm volatile(
        // Load 1 Rhs cell of size 2x8
        "ld1 {v5.8b}, [%[rhs_ptr]], #8\n"
        "ld1 {v6.8b}, [%[rhs_ptr]], #8\n"

        // Load 3 Lhs cells of size 4x2 each
        "ld1 {v2.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v3.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v4.8b}, [%[lhs_ptr]], #8\n"

        // Multiply dst_col_stride by 4 == sizeof(int32) to use
        // it as a byte offset below.
        "lsl %[dst_col_stride], %[dst_col_stride], #2\n"

        "cmp %[start_depth], #0\n"
        "beq " GEMMLOWP_LABEL_CLEAR_ACCUMULATORS
        "f\n"

        // Load accumulators
        "mov x1, %[dst_ptr]\n"
        "mov x0, x1\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "subs %[run_depth], %[run_depth], #2\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v24.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0]\n"

        "b " GEMMLOWP_LABEL_BEFORE_LOOP "f\n"

        GEMMLOWP_LABEL_CLEAR_ACCUMULATORS
        ":\n"

        // Clear accumulator registers (see layout below)
        "dup v8.4s, wzr\n"
        "subs %[run_depth], %[run_depth], #2\n"
        "dup v9.4s, wzr\n"
        "dup v10.4s, wzr\n"
        "dup v11.4s, wzr\n"
        "dup v12.4s, wzr\n"
        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"
        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"
        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"
        "dup v31.4s, wzr\n"

        GEMMLOWP_LABEL_BEFORE_LOOP
        ":\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP "f\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Overview of register layout:
        //
        // A 2x8 block of 2 2x4 cells of Rhs is stored in 16bit in v0--v1.
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in v2--v4.
        // A 12x8 block of accumulators is stored in 32bit in v8--v31.
        //
        //                         +--------+--------+-----+--------+--------+
        //                         |v0.h[0] |v0.h[1] | ... |v1.h[2] |v1.h[3] |
        //                    Rhs  +--------+--------+-----+--------+--------+
        //                         |v0.h[4] |v0.h[5] | ... |v1.h[6] |v1.h[7] |
        //                         +--------+--------+-----+--------+--------+
        //
        //                         |        |        |     |        |        |
        //
        //    Lhs                  |        |        |     |        |        |
        //
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v2.h[0]|v2.h[4]|      |v8.s[0] |v9.s[0] | ... |v14.s[0]|v15.s[0]|
        //  |v2.h[1]|v2.h[5]|      |v8.s[1] |v9.s[1] | ... |v14.s[1]|v15.s[1]|
        //  |v2.h[2]|v2.h[6]|      |v8.s[2] |v9.s[2] | ... |v14.s[2]|v15.s[2]|
        //  |v2.h[3]|v2.h[7]|      |v8.s[3] |v9.s[3] | ... |v14.s[3]|v15.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v3.h[0]|v3.h[4]|      |v16.s[0]|v17.s[0]| ... |v22.s[0]|v23.s[0]|
        //  |v3.h[1]|v3.h[5]|      |v16.s[1]|v17.s[1]| ... |v22.s[1]|v23.s[1]|
        //  |v3.h[2]|v3.h[6]|      |v16.s[2]|v17.s[2]| ... |v22.s[2]|v23.s[2]|
        //  |v3.h[3]|v3.h[7]|      |v16.s[3]|v17.s[3]| ... |v22.s[3]|v23.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v4.h[0]|v4.h[4]|      |v24.s[0]|v25.s[0]| ... |v30.s[0]|v31.s[0]|
        //  |v4.h[1]|v4.h[5]|      |v24.s[1]|v25.s[1]| ... |v30.s[1]|v31.s[1]|
        //  |v4.h[2]|v4.h[6]|      |v24.s[2]|v25.s[2]| ... |v30.s[2]|v31.s[2]|
        //  |v4.h[3]|v4.h[7]|      |v24.s[3]|v25.s[3]| ... |v30.s[3]|v31.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //
        //                            Accumulator

        // Expand Lhs/Rhs cells to 16 bit.
        "uxtl v0.8h, v5.8b\n"
        "ld1 {v5.8b}, [%[rhs_ptr]], #8\n"
        "uxtl v1.8h, v6.8b\n"
        "ld1 {v6.8b}, [%[rhs_ptr]], #8\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v4.8h, v4.8b\n"

        // Multiply-accumulate, top third
        "umlal v8.4s, v2.4h, v0.h[0]\n"
        "umlal v9.4s, v2.4h, v0.h[1]\n"
        "umlal v10.4s, v2.4h, v0.h[2]\n"
        "umlal v11.4s, v2.4h, v0.h[3]\n"
        "umlal v12.4s, v2.4h, v1.h[0]\n"
        "umlal v13.4s, v2.4h, v1.h[1]\n"
        "umlal v14.4s, v2.4h, v1.h[2]\n"
        "umlal v15.4s, v2.4h, v1.h[3]\n"
        "umlal2 v8.4s, v2.8h, v0.h[4]\n"
        "umlal2 v9.4s, v2.8h, v0.h[5]\n"
        "umlal2 v10.4s, v2.8h, v0.h[6]\n"
        "umlal2 v11.4s, v2.8h, v0.h[7]\n"
        "umlal2 v12.4s, v2.8h, v1.h[4]\n"
        "umlal2 v13.4s, v2.8h, v1.h[5]\n"
        "umlal2 v14.4s, v2.8h, v1.h[6]\n"
        "umlal2 v15.4s, v2.8h, v1.h[7]\n"
        "ld1 {v2.8b}, [%[lhs_ptr]], #8\n"

        // Multiply-accumulate, middle third
        "umlal v16.4s, v3.4h, v0.h[0]\n"
        "umlal v17.4s, v3.4h, v0.h[1]\n"
        "umlal v18.4s, v3.4h, v0.h[2]\n"
        "umlal v19.4s, v3.4h, v0.h[3]\n"
        "umlal v20.4s, v3.4h, v1.h[0]\n"
        "umlal v21.4s, v3.4h, v1.h[1]\n"
        "umlal v22.4s, v3.4h, v1.h[2]\n"
        "umlal v23.4s, v3.4h, v1.h[3]\n"
        "umlal2 v16.4s, v3.8h, v0.h[4]\n"
        "umlal2 v17.4s, v3.8h, v0.h[5]\n"
        "umlal2 v18.4s, v3.8h, v0.h[6]\n"
        "umlal2 v19.4s, v3.8h, v0.h[7]\n"
        "umlal2 v20.4s, v3.8h, v1.h[4]\n"
        "umlal2 v21.4s, v3.8h, v1.h[5]\n"
        "umlal2 v22.4s, v3.8h, v1.h[6]\n"
        "umlal2 v23.4s, v3.8h, v1.h[7]\n"
        "ld1 {v3.8b}, [%[lhs_ptr]], #8\n"

        "subs %[run_depth], %[run_depth], #2\n"

        // Multiply-accumulate, bottom third
        "umlal v24.4s, v4.4h, v0.h[0]\n"
        "umlal v25.4s, v4.4h, v0.h[1]\n"
        "umlal v26.4s, v4.4h, v0.h[2]\n"
        "umlal v27.4s, v4.4h, v0.h[3]\n"
        "umlal v28.4s, v4.4h, v1.h[0]\n"
        "umlal v29.4s, v4.4h, v1.h[1]\n"
        "umlal v30.4s, v4.4h, v1.h[2]\n"
        "umlal v31.4s, v4.4h, v1.h[3]\n"
        "umlal2 v24.4s, v4.8h, v0.h[4]\n"
        "umlal2 v25.4s, v4.8h, v0.h[5]\n"
        "umlal2 v26.4s, v4.8h, v0.h[6]\n"
        "umlal2 v27.4s, v4.8h, v0.h[7]\n"
        "umlal2 v28.4s, v4.8h, v1.h[4]\n"
        "umlal2 v29.4s, v4.8h, v1.h[5]\n"
        "umlal2 v30.4s, v4.8h, v1.h[6]\n"
        "umlal2 v31.4s, v4.8h, v1.h[7]\n"
        "ld1 {v4.8b}, [%[lhs_ptr]], #8\n"

        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "uxtl v0.8h, v5.8b\n"
        "uxtl v1.8h, v6.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v4.8h, v4.8b\n"

        // Multiply-accumulate, level of depth 0
        "umlal v8.4s, v2.4h, v0.h[0]\n"
        "umlal v9.4s, v2.4h, v0.h[1]\n"
        "umlal v10.4s, v2.4h, v0.h[2]\n"
        "umlal v11.4s, v2.4h, v0.h[3]\n"
        "umlal v12.4s, v2.4h, v1.h[0]\n"
        "umlal v13.4s, v2.4h, v1.h[1]\n"
        "umlal v14.4s, v2.4h, v1.h[2]\n"
        "umlal v15.4s, v2.4h, v1.h[3]\n"
        "umlal v16.4s, v3.4h, v0.h[0]\n"
        "umlal v17.4s, v3.4h, v0.h[1]\n"
        "umlal v18.4s, v3.4h, v0.h[2]\n"
        "umlal v19.4s, v3.4h, v0.h[3]\n"
        "umlal v20.4s, v3.4h, v1.h[0]\n"
        "umlal v21.4s, v3.4h, v1.h[1]\n"
        "umlal v22.4s, v3.4h, v1.h[2]\n"
        "umlal v23.4s, v3.4h, v1.h[3]\n"
        "umlal v24.4s, v4.4h, v0.h[0]\n"
        "umlal v25.4s, v4.4h, v0.h[1]\n"
        "umlal v26.4s, v4.4h, v0.h[2]\n"
        "umlal v27.4s, v4.4h, v0.h[3]\n"
        "umlal v28.4s, v4.4h, v1.h[0]\n"
        "umlal v29.4s, v4.4h, v1.h[1]\n"
        "umlal v30.4s, v4.4h, v1.h[2]\n"
        "umlal v31.4s, v4.4h, v1.h[3]\n"

        // Multiply-accumulate, level of depth 1
        "umlal2 v8.4s, v2.8h, v0.h[4]\n"
        "umlal2 v9.4s, v2.8h, v0.h[5]\n"
        "umlal2 v10.4s, v2.8h, v0.h[6]\n"
        "umlal2 v11.4s, v2.8h, v0.h[7]\n"
        "umlal2 v12.4s, v2.8h, v1.h[4]\n"
        "umlal2 v13.4s, v2.8h, v1.h[5]\n"
        "umlal2 v14.4s, v2.8h, v1.h[6]\n"
        "umlal2 v15.4s, v2.8h, v1.h[7]\n"
        "umlal2 v16.4s, v3.8h, v0.h[4]\n"
        "umlal2 v17.4s, v3.8h, v0.h[5]\n"
        "umlal2 v18.4s, v3.8h, v0.h[6]\n"
        "umlal2 v19.4s, v3.8h, v0.h[7]\n"
        "umlal2 v20.4s, v3.8h, v1.h[4]\n"
        "umlal2 v21.4s, v3.8h, v1.h[5]\n"
        "umlal2 v22.4s, v3.8h, v1.h[6]\n"
        "umlal2 v23.4s, v3.8h, v1.h[7]\n"
        "umlal2 v24.4s, v4.8h, v0.h[4]\n"
        "umlal2 v25.4s, v4.8h, v0.h[5]\n"
        "umlal2 v26.4s, v4.8h, v0.h[6]\n"
        "umlal2 v27.4s, v4.8h, v0.h[7]\n"
        "umlal2 v28.4s, v4.8h, v1.h[4]\n"
        "umlal2 v29.4s, v4.8h, v1.h[5]\n"
        "umlal2 v30.4s, v4.8h, v1.h[6]\n"
        "umlal2 v31.4s, v4.8h, v1.h[7]\n"

        // Store accumulators
        "mov x1, %[dst_ptr]\n"
        "mov x0, x1\n"
        "st1 {v8.16b}, [x0], #16\n"
        "subs %[run_depth], %[run_depth], #2\n"
        "st1 {v16.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v24.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v9.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v10.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v11.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v12.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v13.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v14.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0]\n"
#undef GEMMLOWP_LABEL_CLEAR_ACCUMULATORS
#undef GEMMLOWP_LABEL_BEFORE_LOOP
#undef GEMMLOWP_LABEL_LOOP
#undef GEMMLOWP_LABEL_AFTER_LOOP
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride] "r"(dst_col_stride)
        :  // clobbers
        "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
  }
};

#ifdef GEMMLOWP_DOTPROD_KERNEL
#ifndef __ARM_FEATURE_DOTPROD
#error This kernel requires ARM dot-product instructions. Enable them by \
  adding '+dotprod' to a compiler flag, e.g. -march=armv8.2-a+dotprod . \
  Note that Clang up to version 7 fails to define the corresponding \
  preprocessor token __ARM_FEATURE_DOTPROD, so you will still have to define \
  it manually.
#endif
// Kernels utilizing the Armv8.2 Dot Product extension.
//
// The dot product instructions work by taking 4 consecutive 8-bit depth
// values from each operand, multiplying the 4 pairs together and
// accumulating all the results into the corresponding 32-bit accumulator
// lane.  As such, the operation is identical to a 32-bit instruction (like
// FMLA used in SGEMM), except that 4 depth values are processed at a time
// instead of 1.

// Thus, this first kernel is a carbon copy of
// "NEON_64bit_GEMM_Float32_WithScalar_A57" (which should provide good
// performance for most processors) below with the opcode (fmla -> udot) and
// types (float32 -> uint8/uint32) changed.
//
// A signed version of this kernel could be produced by replacing "udot"
// with "sdot" - performance should be identical to this udot kernel.
struct NEON_64_Kernel12x8Depth4_dotprod : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 4, CellOrder::WidthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 4, CellOrder::WidthMajor>, 2> >
      Format;

  const char* Name() const override { return "NEON, 12x8, depth 4, dotprod"; }

  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride, std::size_t dst_col_stride,
           const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t depth) const override {
    (void)dst_row_stride;
    ScopedProfilingLabel label("optimized kernel (NEON 12x8, depth 4, dotprod)");
// See comments above for why we need local numerical labels in our asm.
#define GEMMLOWP_LABEL_CLEAR_ACCUMULATORS "1"
#define GEMMLOWP_LABEL_BEFORE_LOOP "2"
#define GEMMLOWP_LABEL_LOOP "3"
#define GEMMLOWP_LABEL_AFTER_LOOP "4"

    assert(dst_row_stride == 1);
    asm volatile(
        // Multiply dst_col_stride by 4 == sizeof(int32) to use
        // it as a byte offset below.
        "lsl %[dst_col_stride], %[dst_col_stride], #2\n"

        "cmp %[start_depth], #0\n"
        "beq " GEMMLOWP_LABEL_CLEAR_ACCUMULATORS "f\n"

        // Load accumulators
        "mov x1, %[dst_ptr]\n"
        "mov x0, x1\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v24.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0]\n"
        "mov x0, x1\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0]\n"

        "b " GEMMLOWP_LABEL_BEFORE_LOOP "f\n"

        GEMMLOWP_LABEL_CLEAR_ACCUMULATORS ":\n"

        // Clear accumulator registers (see layout below)
        "dup v8.4s, wzr\n"
        "dup v9.4s, wzr\n"
        "dup v10.4s, wzr\n"
        "dup v11.4s, wzr\n"
        "dup v12.4s, wzr\n"
        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"
        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"
        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"
        "dup v31.4s, wzr\n"

        GEMMLOWP_LABEL_BEFORE_LOOP ":\n"

        "subs %w[depth], %w[depth], #4\n"

        // The start of the loop assumes first Rhs cell is already loaded, so
        // do it here for first iteration.
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"

        // And the same for the first Lhs cell.
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP "f\n"

        GEMMLOWP_LABEL_LOOP ":\n"

        // Start the MACs at the head of the loop - 1st cell from each side
        // already loaded.
        ".word 0x6f80e048  // udot v8.4s, v2.16b, v0.4b[0]\n"
        ".word 0x6fa0e049  // udot v9.4s, v2.16b, v0.4b[1]\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"  // Load second Rhs cell.
        ".word 0x6f80e84a  // udot v10.4s, v2.16b, v0.4b[2]\n"
        ".word 0x6fa0e84b  // udot v11.4s, v2.16b, v0.4b[3]\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"  // Load second Lhs cell.
        ".word 0x6f81e04c  // udot v12.4s, v2.16b, v1.4b[0]\n"
        ".word 0x6fa1e04d  // udot v13.4s, v2.16b, v1.4b[1]\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"  // Load third Lhs cell.
        ".word 0x6f81e84e  // udot v14.4s, v2.16b, v1.4b[2]\n"
        ".word 0x6fa1e84f  // udot v15.4s, v2.16b, v1.4b[3]\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"  // Done with first Lhs cell - load
        // for the next iteration early.
        ".word 0x6f80e070  // udot v16.4s, v3.16b, v0.4b[0]\n"
        ".word 0x6fa0e071  // udot v17.4s, v3.16b, v0.4b[1]\n"
        ".word 0x6f80e872  // udot v18.4s, v3.16b, v0.4b[2]\n"
        ".word 0x6fa0e873  // udot v19.4s, v3.16b, v0.4b[3]\n"
        ".word 0x6f81e074  // udot v20.4s, v3.16b, v1.4b[0]\n"
        ".word 0x6fa1e075  // udot v21.4s, v3.16b, v1.4b[1]\n"
        ".word 0x6f81e876  // udot v22.4s, v3.16b, v1.4b[2]\n"
        ".word 0x6fa1e877  // udot v23.4s, v3.16b, v1.4b[3]\n"
        ".word 0x6f80e098  // udot v24.4s, v4.16b, v0.4b[0]\n"
        ".word 0x6fa0e099  // udot v25.4s, v4.16b, v0.4b[1]\n"
        ".word 0x6f80e89a  // udot v26.4s, v4.16b, v0.4b[2]\n"
        ".word 0x6fa0e89b  // udot v27.4s, v4.16b, v0.4b[3]\n"
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"  // Done with the first Rhs cell -
        // load for the next iteration early.
        ".word 0x6f81e09c  // udot v28.4s, v4.16b, v1.4b[0]\n"
        ".word 0x6fa1e09d  // udot v29.4s, v4.16b, v1.4b[1]\n"

        // Loop.  Decrement loop index (depth) by 4 as udot processes 4
        // depth values.
        "subs %w[depth], %w[depth], #4\n"
        ".word 0x6f81e89e  // udot v30.4s, v4.16b, v1.4b[2]\n"
        ".word 0x6fa1e89f  // udot v31.4s, v4.16b, v1.4b[3]\n"

        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP ":\n"

        // Final iteration. v0 and v2 were already loaded, don't load
        // them again, don't read past the end of buffers.
        ".word 0x6f80e048  // udot v8.4s, v2.16b, v0.4b[0]\n"
        ".word 0x6fa0e049  // udot v9.4s, v2.16b, v0.4b[1]\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"  // Load second Rhs cell.
        ".word 0x6f80e84a  // udot v10.4s, v2.16b, v0.4b[2]\n"
        ".word 0x6fa0e84b  // udot v11.4s, v2.16b, v0.4b[3]\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"  // Load second Lhs cell.
        ".word 0x6f81e04c  // udot v12.4s, v2.16b, v1.4b[0]\n"
        ".word 0x6fa1e04d  // udot v13.4s, v2.16b, v1.4b[1]\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"  // Load third Lhs cell.
        ".word 0x6f81e84e  // udot v14.4s, v2.16b, v1.4b[2]\n"
        ".word 0x6fa1e84f  // udot v15.4s, v2.16b, v1.4b[3]\n"
        ".word 0x6f80e070  // udot v16.4s, v3.16b, v0.4b[0]\n"
        ".word 0x6fa0e071  // udot v17.4s, v3.16b, v0.4b[1]\n"
        ".word 0x6f80e872  // udot v18.4s, v3.16b, v0.4b[2]\n"
        ".word 0x6fa0e873  // udot v19.4s, v3.16b, v0.4b[3]\n"
        ".word 0x6f81e074  // udot v20.4s, v3.16b, v1.4b[0]\n"
        ".word 0x6fa1e075  // udot v21.4s, v3.16b, v1.4b[1]\n"
        ".word 0x6f81e876  // udot v22.4s, v3.16b, v1.4b[2]\n"
        ".word 0x6fa1e877  // udot v23.4s, v3.16b, v1.4b[3]\n"
        ".word 0x6f80e098  // udot v24.4s, v4.16b, v0.4b[0]\n"
        ".word 0x6fa0e099  // udot v25.4s, v4.16b, v0.4b[1]\n"
        ".word 0x6f80e89a  // udot v26.4s, v4.16b, v0.4b[2]\n"
        ".word 0x6fa0e89b  // udot v27.4s, v4.16b, v0.4b[3]\n"
        ".word 0x6f81e09c  // udot v28.4s, v4.16b, v1.4b[0]\n"
        ".word 0x6fa1e09d  // udot v29.4s, v4.16b, v1.4b[1]\n"

        // Loop.  Decrement loop index (depth) by 4 as udot processes 4
        // depth values.
        "subs %w[depth], %w[depth], #4\n"
        ".word 0x6f81e89e  // udot v30.4s, v4.16b, v1.4b[2]\n"
        ".word 0x6fa1e89f  // udot v31.4s, v4.16b, v1.4b[3]\n"

        // Store accumulators
        "mov x1, %[dst_ptr]\n"
        "mov x0, x1\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v24.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v9.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v10.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v11.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v12.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v13.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v14.16b}, [x0], #16\n"
        "add x1, x1, %[dst_col_stride]\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0]\n"
        "mov x0, x1\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0]\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [dst_ptr] "r"(dst_ptr), [dst_col_stride] "r"(dst_col_stride), [start_depth] "r"(start_depth)
        :  // clobbers
        "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
        "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
  }
};
#endif  // GEMMLOWP_DOTPROD_KERNEL

#endif  // GEMMLOWP_NEON_64

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_NEON_H_
