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

// kernel_SSE.h: a collection of Intel SSE optimized kernels.
// Check in kernel_default.h which one(s) are actually used by default.
// Others are mere experiments; they are still covered by tests
// in case they might be useful some day.
//

#ifndef GEMMLOWP_INTERNAL_KERNEL_SSE_H_
#define GEMMLOWP_INTERNAL_KERNEL_SSE_H_

#include "kernel.h"

#include <string.h>
#include <cassert>

namespace gemmlowp {

#ifdef GEMMLOWP_SSE4_32
struct SSE4_32_Kernel4x4Depth2 : KernelBase {
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 1> >
      Format;

  const char* Name() const override { return "SSE, 4x4, depth 2"; }

  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel");
    assert(dst_row_stride == 1);
    (void)dst_row_stride;
    std::int32_t run_depth_cells = run_depth / Format::kDepth;
    /* Main loop */

    // A 2x4 cell of Rhs is stored in 16bit in xmm1 .
    // A 4x2 block Lhs is stored in 16bit in xmm0.
    // A 4x4 block of accumulators is stored in 32bit in xmm4--xmm7.
    //
    //                   +-------+-------+-------+-------+
    //                   |xmm1[0]|xmm1[2]|xmm1[4]|xmm1[6]|
    //              Rhs  +-------+---------------+-------+
    //                   |xmm1[1]|xmm1[3]|xmm1[5]|xmm1[7]|
    //                   +-------+-------+-------+-------+
    //
    //                   |       |       |       |       |
    //
    //    Lhs            |       |       |       |       |
    //
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //  |xmm0 |          | xmm4  | xmm5  | xmm6  | xmm7  |
    //  |xmm0 | (Iter1)  | xmm4  | xmm5  | xmm6  | xmm7  |
    //  |xmm0 |          | xmm4  | xmm5  | xmm6  | xmm7  |
    //  |xmm0 |          | xmm4  | xmm5  | xmm6  | xmm7  |
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //
    //                              Accumulator

    asm volatile(

        // set accumulators to zero.
        "pxor %%xmm4  , %%xmm4 \n\t"
        "pxor %%xmm5  , %%xmm5 \n\t"
        "pxor %%xmm6  , %%xmm6 \n\t"
        "pxor %%xmm7  , %%xmm7 \n\t"

        "movl  %[run_depth_cells], %%eax\n\t"
        "subl $2, %%eax\n\t"
        "js outerLoop1%=\n\t"

        // Loop for K unrolled by 4
        "outerLoop2%=:\n\t"

        // K = 1,2
        // RHS cell to xmm1
        "pmovzxbw (%[rhs_ptr]), %%xmm1\n\t"

        // LHS cell
        "pmovzxbw 0x00(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm4           \n\t"
        "paddd %%xmm3, %%xmm5           \n\t"

        "prefetcht0 0x80(%[lhs_ptr]) \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"

        "prefetcht0 0x80(%[rhs_ptr]) \n\t"

        // K = 3,4
        // RHS cell to xmm1
        "pmovzxbw 0x08(%[rhs_ptr]), %%xmm1\n\t"

        "paddd %%xmm2, %%xmm6           \n\t"
        "paddd %%xmm3, %%xmm7           \n\t"

        // LHS cell
        "pmovzxbw 0x08(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm4           \n\t"
        "paddd %%xmm3, %%xmm5           \n\t"
        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"

        "addl $0x10, %[lhs_ptr]         \n\t"
        "addl $0x10, %[rhs_ptr]         \n\t"

        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm3, %%xmm7           \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "paddd %%xmm2, %%xmm6           \n\t"

        "subl $2, %[run_depth_cells]\n\t"
        "ja outerLoop2%=\n\t"

        "movl %[run_depth_cells], %%eax\n\t"
        "decl %%eax\n\t"
        "js finish%=\n\t"

        // Loop for K unrolled by 2
        "outerLoop1%=:\n\t"

        // RHS cell to xmm1
        "pmovzxbw (%[rhs_ptr]), %%xmm1\n\t"

        // LHS cell
        "pmovzxbw 0x00(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "paddd %%xmm2, %%xmm4           \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm3, %%xmm5           \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "paddd %%xmm2, %%xmm6           \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm3, %%xmm7           \n\t"

        "addl $0x08, %[lhs_ptr]\n\t"
        "addl $0x08, %[rhs_ptr]\n\t"

        "decl %[run_depth_cells]\n\t"
        "jnz outerLoop1%=\n\t"

        "finish%=:\n\t"

        "movl  %[dst_col_stride], %%eax\n\t"
        "shll $2, %%eax\n\t"

        "movl  %[start_depth], %%ecx\n\t"
        "test %%ecx, %%ecx\n\t"
        "jz storeDst%=\n\t"

        "leal (%%eax,%%eax,0x2), %%ecx\n\t"
        "paddd 0x00(%[dst_ptr])           , %%xmm4 \n\t"
        "paddd 0x00(%[dst_ptr], %%eax, 1) , %%xmm5 \n\t"
        "paddd 0x00(%[dst_ptr], %%eax, 2) , %%xmm6 \n\t"
        "paddd 0x00(%[dst_ptr], %%ecx, 1) , %%xmm7 \n\t"

        "storeDst%=:\n\t"

        "leal (%%eax,%%eax,0x2), %%ecx\n\t"
        "movdqu %%xmm4  , 0x00(%[dst_ptr])          \n\t"
        "movdqu %%xmm5  , 0x00(%[dst_ptr], %%eax, 1)\n\t"
        "movdqu %%xmm6  , 0x00(%[dst_ptr], %%eax, 2)\n\t"
        "movdqu %%xmm7  , 0x00(%[dst_ptr], %%ecx, 1)\n\t"

        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr)
        :  // inputs
        [start_depth] "g"(start_depth), [dst_col_stride] "g"(dst_col_stride),
        [run_depth_cells] "g"(run_depth_cells)
        :  // clobbers
        "cc", "memory", "%xmm0", "%xmm1", "%xmm3", "%xmm2", "%xmm4", "%xmm5",
        "%xmm6", "%xmm7", "%eax", "%ecx");
  }
};
#endif
#ifdef GEMMLOWP_SSE4_64
struct SSE4_64_Kernel12x4Depth2 : KernelBase {
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 1> >
      Format;

  const char* Name() const override { return "SSE, 12x4, depth 2"; }

  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel");
    assert(dst_row_stride == 1);
    (void)dst_row_stride;
    const std::int64_t run_depth_cells = run_depth / Format::kDepth;
    const std::int64_t dst_col_stride_q = dst_col_stride;

    /* Main loop */

    // A 2x4 cell of Rhs is stored in 16bit in xmm1 .
    // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in xmm0, replaced
    // every Iteration.
    // A 12x4 block of accumulators is stored in 32bit in xmm4--xmm15.
    //
    //                   +-------+-------+-------+-------+
    //                   |xmm1[0]|xmm1[2]|xmm1[4]|xmm1[6]|
    //              Rhs  +-------+---------------+-------+
    //                   |xmm1[1]|xmm1[3]|xmm1[5]|xmm1[7]|
    //                   +-------+-------+-------+-------+
    //
    //                   |       |       |       |       |
    //
    //    Lhs            |       |       |       |       |
    //
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //  |xmm0 |          | xmm4  | xmm5  | xmm6  | xmm7  |
    //  |xmm0 | (Iter1)  | xmm4  | xmm5  | xmm6  | xmm7  |
    //  |xmm0 |          | xmm4  | xmm5  | xmm6  | xmm7  |
    //  |xmm0 |          | xmm4  | xmm5  | xmm6  | xmm7  |
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //  |xmm0 |          | xmm8  | xmm9  | xmm10 | xmm11 |
    //  |xmm0 | (Iter2)  | xmm8  | xmm9  | xmm10 | xmm11 |
    //  |xmm0 |          | xmm8  | xmm9  | xmm10 | xmm11 |
    //  |xmm0 |          | xmm8  | xmm9  | xmm10 | xmm11 |
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //  |xmm0 |          | xmm12 | xmm13 | xmm14 | xmm15 |
    //  |xmm0 | (Iter3)  | xmm12 | xmm13 | xmm14 | xmm15 |
    //  |xmm0 |          | xmm12 | xmm13 | xmm14 | xmm15 |
    //  |xmm0 |          | xmm12 | xmm13 | xmm14 | xmm15 |
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //
    //                              Accumulator

    asm volatile(

        // Set registers for destination
        "movq  %[dst_col_stride_q], %%r12\n\t"
        "shlq $2, %%r12\n\t"
        "leaq (%%r12,%%r12,0x2), %%r13\n\t"

        // Set accumulators to zero.
        "pxor %%xmm4  , %%xmm4 \n\t"
        "pxor %%xmm5  , %%xmm5 \n\t"
        "pxor %%xmm6  , %%xmm6 \n\t"
        "pxor %%xmm7  , %%xmm7 \n\t"
        "pxor %%xmm8  , %%xmm8 \n\t"
        "pxor %%xmm9  , %%xmm9 \n\t"
        "pxor %%xmm10 , %%xmm10\n\t"
        "pxor %%xmm11 , %%xmm11\n\t"
        "pxor %%xmm12 , %%xmm12\n\t"
        "pxor %%xmm13 , %%xmm13\n\t"
        "pxor %%xmm14 , %%xmm14\n\t"
        "pxor %%xmm15 , %%xmm15\n\t"

        "movq  %[run_depth_cells], %%r14\n\t"
        "subq $2, %%r14\n\t"
        "js outerLoop1%=\n\t"

        // Loop for K unrolled by 4
        "outerLoop2%=:\n\t"

        // K = 1,2
        // RHS cell to xmm1

        "pmovzxbw (%[rhs_ptr]), %%xmm1\n\t"

        // LHS cell
        "pmovzxbw 0x00(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm4           \n\t"
        "paddd %%xmm3, %%xmm5           \n\t"

        "prefetcht0 0x80(%[lhs_ptr]) \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"

        // next LHS cell
        "pmovzxbw 0x08(%[lhs_ptr]), %%xmm0\n\t"

        "paddd %%xmm2, %%xmm6           \n\t"
        "paddd %%xmm3, %%xmm7           \n\t"

        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm8           \n\t"
        "paddd %%xmm3, %%xmm9           \n\t"

        "prefetcht0 0x80(%[rhs_ptr]) \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm10          \n\t"
        "paddd %%xmm3, %%xmm11          \n\t"

        // next LHS cell
        "pmovzxbw 0x10(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm12          \n\t"
        "paddd %%xmm3, %%xmm13          \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm14          \n\t"
        "paddd %%xmm3, %%xmm15          \n\t"

        // K = 3,4
        // RHS cell to xmm1
        "pmovzxbw 0x08(%[rhs_ptr]), %%xmm1\n\t"

        // LHS cell
        "pmovzxbw 0x18(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm4           \n\t"
        "paddd %%xmm3, %%xmm5           \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm6           \n\t"
        "paddd %%xmm3, %%xmm7           \n\t"

        // next LHS cell
        "pmovzxbw 0x20(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm8           \n\t"
        "paddd %%xmm3, %%xmm9           \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm10          \n\t"
        "paddd %%xmm3, %%xmm11          \n\t"

        // next LHS cell
        "pmovzxbw 0x28(%[lhs_ptr]), %%xmm0\n\t"

        "addq $0x30, %[lhs_ptr]         \n\t"
        "addq $0x10, %[rhs_ptr]         \n\t"

        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm12          \n\t"
        "paddd %%xmm3, %%xmm13          \n\t"

        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm14          \n\t"
        "paddd %%xmm3, %%xmm15          \n\t"

        "subq $2, %[run_depth_cells]\n\t"
        "ja outerLoop2%=\n\t"

        "movq %[run_depth_cells], %%r14\n\t"
        "decq %%r14\n\t"
        "js finish%=\n\t"

        // Loop for K unrolled by 2
        "outerLoop1%=:\n\t"

        // RHS cell to xmm1
        "pmovzxbw (%[rhs_ptr]), %%xmm1\n\t"

        // LHS cell
        "pmovzxbw 0x00(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm4           \n\t"
        "paddd %%xmm3, %%xmm5           \n\t"
        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm6           \n\t"
        "paddd %%xmm3, %%xmm7           \n\t"

        // next LHS cell
        "pmovzxbw 0x08(%[lhs_ptr]), %%xmm0\n\t"
        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm8           \n\t"
        "paddd %%xmm3, %%xmm9           \n\t"
        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm10          \n\t"
        "paddd %%xmm3, %%xmm11          \n\t"

        // next LHS cell
        "pmovzxbw 0x10(%[lhs_ptr]), %%xmm0\n\t"

        "addq $0x18, %[lhs_ptr]         \n\t"
        "addq $0x08, %[rhs_ptr]         \n\t"

        "pshufd $0x00,%%xmm1,%%xmm2     \n\t"
        "pshufd $0x55,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm12          \n\t"
        "paddd %%xmm3, %%xmm13          \n\t"
        "pshufd $0xaa,%%xmm1,%%xmm2     \n\t"
        "pshufd $0xff,%%xmm1,%%xmm3     \n\t"
        "pmaddwd %%xmm0, %%xmm2         \n\t"
        "pmaddwd %%xmm0, %%xmm3         \n\t"
        "paddd %%xmm2, %%xmm14          \n\t"
        "paddd %%xmm3, %%xmm15          \n\t"

        "decq %[run_depth_cells]\n\t"
        "jnz outerLoop1%=\n\t"

        "finish%=:\n\t"

        "test %[start_depth], %[start_depth]\n\t"
        "jz storeDst%=\n\t"

        "paddd 0x00(%[dst_ptr])           , %%xmm4 \n\t"
        "paddd 0x10(%[dst_ptr])           , %%xmm8 \n\t"
        "paddd 0x20(%[dst_ptr])           , %%xmm12\n\t"
        "paddd 0x00(%[dst_ptr], %%r12, 1) , %%xmm5 \n\t"
        "paddd 0x10(%[dst_ptr], %%r12, 1) , %%xmm9 \n\t"
        "paddd 0x20(%[dst_ptr], %%r12, 1) , %%xmm13\n\t"
        "paddd 0x00(%[dst_ptr], %%r12, 2) , %%xmm6 \n\t"
        "paddd 0x10(%[dst_ptr], %%r12, 2) , %%xmm10\n\t"
        "paddd 0x20(%[dst_ptr], %%r12, 2) , %%xmm14\n\t"
        "paddd 0x00(%[dst_ptr], %%r13, 1) , %%xmm7 \n\t"
        "paddd 0x10(%[dst_ptr], %%r13, 1) , %%xmm11\n\t"
        "paddd 0x20(%[dst_ptr], %%r13, 1) , %%xmm15\n\t"

        "storeDst%=:\n\t"

        "movdqu %%xmm4  , 0x00(%[dst_ptr])          \n\t"
        "movdqu %%xmm8  , 0x10(%[dst_ptr])          \n\t"
        "movdqu %%xmm12 , 0x20(%[dst_ptr])          \n\t"
        "movdqu %%xmm5  , 0x00(%[dst_ptr], %%r12, 1)\n\t"
        "movdqu %%xmm9  , 0x10(%[dst_ptr], %%r12, 1)\n\t"
        "movdqu %%xmm13 , 0x20(%[dst_ptr], %%r12, 1)\n\t"
        "movdqu %%xmm6  , 0x00(%[dst_ptr], %%r12, 2)\n\t"
        "movdqu %%xmm10 , 0x10(%[dst_ptr], %%r12, 2)\n\t"
        "movdqu %%xmm14 , 0x20(%[dst_ptr], %%r12, 2)\n\t"
        "movdqu %%xmm7  , 0x00(%[dst_ptr], %%r13, 1)\n\t"
        "movdqu %%xmm11 , 0x10(%[dst_ptr], %%r13, 1)\n\t"
        "movdqu %%xmm15 , 0x20(%[dst_ptr], %%r13, 1)\n\t"

        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride_q] "r"(dst_col_stride_q),
        [run_depth_cells] "r"(run_depth_cells)
        :  // clobbers
        "cc", "memory", "%xmm0", "%xmm1", "%xmm3", "%xmm2", "%xmm4", "%xmm5",
        "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%r12", "%r13", "%r14",
        "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15");
  }
};
#endif

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_SSE_H_
