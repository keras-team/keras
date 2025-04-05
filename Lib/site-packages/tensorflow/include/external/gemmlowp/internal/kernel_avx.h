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

#ifndef GEMMLOWP_INTERNAL_KERNEL_AVX_H_
#define GEMMLOWP_INTERNAL_KERNEL_AVX_H_

#include "kernel.h"

#include <string.h>
#include <cassert>

namespace gemmlowp {

#ifdef GEMMLOWP_AVX2_64
struct AVX2_64_Kernel24x8Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<8, 2, CellOrder::WidthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 1>>
      Format;

  const char *Name() const override { return "AVX, 24x8, depth 2"; }

  void Run(std::int32_t *dst_ptr, std::size_t dst_row_stride, std::size_t dst_col_stride,
           const std::uint8_t *lhs_ptr, const std::uint8_t *rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel");
    assert(dst_row_stride == 1);
    const std::int64_t run_depth_cells = run_depth / Format::kDepth;
    const std::int64_t dst_col_stride_q = dst_col_stride;

    /* Main loop */

    // A 2x8 cell of Rhs is stored in 16bit in ymm1 .
    // A 24x2 block of 3 8x2 cells Lhs is stored in 16bit in ymm0, replaced
    // every Iteration.
    // A 8x8 block of accumulators is stored in 32bit in xmm4--xmm15.
    //
    //                   +-------+-------+-------+-------+
    //                   |ymm1[0]        |ymm2[2]        |
    //              Rhs  +-------+---------------+-------+
    //                   |ymm1[1]        |ymm1[4]        |
    //                   +-------+-------+-------+-------+
    //
    //                   |       |       |       |       |
    //
    //    Lhs            |       |       |       |       |
    //
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //  |ymm0 |          | ymm4  | ymm5  | ymm6  | ymm7  |
    //  |ymm0 | (Iter1)  | ymm4  | ymm5  | ymm6  | ymm7  |
    //  |ymm0 |          | ymm4  | ymm5  | ymm6  | ymm7  |
    //  |ymm0 |          | ymm4  | ymm5  | ymm6  | ymm7  |
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //  |ymm0 |          | ymm8  | ymm9  | ymm10 | ymm11 |
    //  |ymm0 | (Iter2)  | ymm8  | ymm9  | ymm10 | ymm11 |
    //  |ymm0 |          | ymm8  | ymm9  | ymm10 | ymm11 |
    //  |ymm0 |          | ymm8  | ymm9  | ymm10 | ymm11 |
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //  |ymm0 |          | ymm12 | ymm13 | ymm14 | ymm15 |
    //  |ymm0 | (Iter3)  | ymm12 | ymm13 | ymm14 | ymm15 |
    //  |ymm0 |          | ymm12 | ymm13 | ymm14 | ymm15 |
    //  |ymm0 |          | ymm12 | ymm13 | ymm14 | ymm15 |
    //  +--+--+ - - - -  +-------+-------+-------+-------+
    //
    //                              Accumulator

    asm volatile(
        // Set registers for destination
        "movq  %[dst_col_stride_q], %%r12\n\t"  // stride is r12
        "shlq $2, %%r12\n\t"                    // set stride dword
        "leaq (%%r12,%%r12,0x2), %%r13\n\t"     // load stride aligned r13

        // Set accumulators to zero.
        "vpxor %%ymm4, %%ymm4, %%ymm4 \n\t"    // zero accumulators
        "vpxor %%ymm5, %%ymm5, %%ymm5 \n\t"    // zero accumulators
        "vpxor %%ymm6, %%ymm6, %%ymm6 \n\t"    // zero accumulators
        "vpxor %%ymm7, %%ymm7, %%ymm7 \n\t"    // zero accumulators
        "vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"    // zero accumulators
        "vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"    // zero accumulators
        "vpxor %%ymm10, %%ymm10, %%ymm10\n\t"  // zero accumulators
        "vpxor %%ymm11, %%ymm11, %%ymm11\n\t"  // zero accumulators
        "vpxor %%ymm12, %%ymm12, %%ymm12\n\t"  // zero accumulators
        "vpxor %%ymm13, %%ymm13, %%ymm13\n\t"  // zero accumulators
        "vpxor %%ymm14, %%ymm14, %%ymm14\n\t"  // zero accumulators
        "vpxor %%ymm15, %%ymm15, %%ymm15\n\t"  // zero accumulators

        "movq  %[run_depth_cells], %%r14 \n\t"  // load cell depth r14
        "subq $2, %%r14 \n\t"                   // cell depth is 2
        "js outerLoop1%= \n\t"                  // outerloop for matrix

        // Loop for K unrolled by 4
        "outerLoop2%=: \n\t"  // outer loop unroll

        // K = 0,1,2,3
        // RHS cell to ymm1

        // lower half
        "vpmovzxbw (%[rhs_ptr]), %%ymm1 \n\t"  // mov rhs to ymm1
        "vpermq $0x44,%%ymm1, %%ymm1 \n\t"
        // LHS cell elements 0 and 1
        "vpmovzxbw 0x00(%[lhs_ptr]), %%ymm0\n\t"  // mov lhs to ymm0
        "vpshufd $0x00,%%ymm1,%%ymm2     \n\t"    // move rhs 0 element to all ymm2
        "vpshufd $0x55,%%ymm1,%%ymm3     \n\t"    // move rhs 1 element to all ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2 \n\t"    // mul add lhs rhs0 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3 \n\t"    // mul add lhs rhs1 into ymm3
        "vpaddd %%ymm2, %%ymm4, %%ymm4   \n\t"    // add muladd lhs + rhs0 into ymm4
        "vpaddd %%ymm3, %%ymm5, %%ymm5   \n\t"    // add muladd lhs + rhs1 into ymm5
        // LHS cell elements 2 and 3
        "vpshufd $0xaa, %%ymm1, %%ymm2   \n\t"  // move rhs 2 element to all ymm2
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2 \n\t"  // mul add lhs rh3 into ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3     \n\t"  // mov rhs 3 element into all ymm3
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3 \n\t"  // mul add lhs rh4 into ymm3
        "vpaddd %%ymm2, %%ymm6, %%ymm6   \n\t"  // add muladd lhs + rhs2 into ymm6
        "vpaddd %%ymm3, %%ymm7, %%ymm7   \n\t"  // add muladd lhs + rhs3 into ymm7

        // cache prefect lhs //see if it works better?
        //"prefetcht0 0x80(%[lhs_ptr]) \n\t" //prefetch cache lines
        "vpmovzxbw (%[rhs_ptr]), %%ymm1 \n\t"  // mov rhs to ymm1
        "vpermq $0x44,%%ymm1, %%ymm1 \n\t"

        // K = 5,6,7,8
        // next LHS cell elements 0 and 1
        "vpmovzxbw 0x10(%[lhs_ptr]), %%ymm0 \n\t"  // mov lhs to ymm0
        "vpshufd $0x00,%%ymm1,%%ymm2        \n\t"  // mov rhs 0 element to all ymm2
        "vpshufd $0x55,%%ymm1,%%ymm3        \n\t"  // mov rhs 1 element to all ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // mul add lhs rhs0 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // mul add lhs rhs1 into ymm3
        "vpaddd %%ymm2, %%ymm8, %%ymm8      \n\t"  // add muladd lhs + rhs0 into ymm8
        "vpaddd %%ymm3, %%ymm9, %%ymm9      \n\t"  // add muladd lhs + rhs1 into ymm9
        // next LHS cell elements 2 and 3
        "vpshufd $0xaa,%%ymm1,%%ymm2        \n\t"  // mov rhs 2 element to all ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3        \n\t"  // mov rhs 3 element to all ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // mul add lhs rhs2 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // mul add lhs rhs3 into ymm3
        "vpaddd %%ymm2, %%ymm10, %%ymm10    \n\t"  // add muladd lhs + rhs2 into ymm10
        "vpaddd %%ymm3, %%ymm11, %%ymm11    \n\t"  // add muladd lhs + rhs3 into ymm11

        // rhs lower half
        "vpmovzxbw (%[rhs_ptr]), %%ymm1 \n\t"  // mov rhs to ymm1
        "vpermq $0x44,%%ymm1, %%ymm1 \n\t"     // duplcate lower 16

        // next LHS cell elements 0 and 1
        "vpmovzxbw 0x20(%[lhs_ptr]), %%ymm0 \n\t"    // mov lhs to ymm0
        "vpshufd $0x00,%%ymm1,%%ymm2        \n\t"    // mov rhs 0 element to all ymm2
        "vpshufd $0x55,%%ymm1,%%ymm3        \n\t"    // mov rhs 1 element to all ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"    // mul add lhs rhs0 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"    // mul add lhs rhs1 into ymm3
        "vpaddd %%ymm2, %%ymm12, %%ymm12      \n\t"  // add muladd lhs + rhs0 into ymm8
        "vpaddd %%ymm3, %%ymm13, %%ymm13      \n\t"  // add muladd lhs + rhs1 into ymm9

        // cache prefetch rhs //see if it works better?
        //"prefetcht0 0x80(%[rhs_ptr]) \n\t"

        // next LHS cell elements 2 and 3
        "vpshufd $0xaa,%%ymm1,%%ymm2        \n\t"  // mov rhs 2 element to all ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3        \n\t"  // mov rhs 3 element to all ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // mul add lhs rhs2 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // mul add lhs rhs3 into ymm3
        "vpaddd %%ymm2, %%ymm14, %%ymm14    \n\t"  // add muladd lhs + rhs2 into ymm10
        "vpaddd %%ymm3, %%ymm15, %%ymm15    \n\t"  // add muladd lhs + rhs3 into ymm11

        // current result in ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10 ymm11 ymm12 ymm13 ymm14 ymm15

        // rhs+10 lower half
        "vpmovzxbw 0x08(%[rhs_ptr]), %%ymm1 \n\t"  // mov rhs to ymm1
        "vpermq $0x44,%%ymm1, %%ymm1 \n\t"
        // next LHS cell elements 0 and 1
        "vpmovzxbw 0x30(%[lhs_ptr]), %%ymm0 \n\t"  // mov lhs to ymm0
        "vpshufd $0x00,%%ymm1,%%ymm2        \n\t"  // move rhs 0 element to ymm2
        "vpshufd $0x55,%%ymm1,%%ymm3        \n\t"  // move rhs 1 element to ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // muladd lhs rhs0 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // muladd lhs rhs1 into ymm3
        "vpaddd %%ymm2, %%ymm4, %%ymm4      \n\t"  // accumulate to ymm4
        "vpaddd %%ymm3, %%ymm5, %%ymm5      \n\t"  // accumulate to ymm5
        // next LHS cell elements 2 and 3
        "vpshufd $0xaa,%%ymm1,%%ymm2        \n\t"  // mov rhs 2 element to ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3        \n\t"  // mov rhs 3 element to ymm2
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // mul add lhs rhs2 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // mull add lhs rhs3 into ymm3
        "vpaddd %%ymm2, %%ymm6, %%ymm6      \n\t"  // add lhs rhs2 to ymm6
        "vpaddd %%ymm3, %%ymm7, %%ymm7      \n\t"  // add lhs rhs3 to ymm7

        // rhs+10 lower half
        "vpmovzxbw 0x08(%[rhs_ptr]), %%ymm1 \n\t"  // mov rhs to ymm1
        "vpermq $0x44,%%ymm1, %%ymm1 \n\t"

        // next LHS cell elements 4 and 5
        "vpmovzxbw 0x40(%[lhs_ptr]), %%ymm0 \n\t"  // mov lhs to ymm0
        "vpshufd $0x00,%%ymm1,%%ymm2        \n\t"  // move rhs 0 element to ymm2
        "vpshufd $0x55,%%ymm1,%%ymm3        \n\t"  // move rhs 1 element to ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // muladd lhs rhs0 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // muladd lhs rhs1 into ymm3
        "vpaddd %%ymm2, %%ymm8, %%ymm8      \n\t"  // accumulate to ymm8
        "vpaddd %%ymm3, %%ymm9, %%ymm9      \n\t"  // accumulate to ymm9
        // next LHS cell elements 6 and 7
        "vpshufd $0xaa,%%ymm1,%%ymm2        \n\t"  // mov rhs 2 element to ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3        \n\t"  // mov rhs 3 element to ymm2
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // mul add lhs rhs2 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // mull add lhs rhs3 into ymm3
        "vpaddd %%ymm2, %%ymm10, %%ymm10    \n\t"  // add lhs rhs2 to ymm10
        "vpaddd %%ymm3, %%ymm11, %%ymm11    \n\t"  // add lhs rhs3 to ymm11

        "vpmovzxbw 0x08(%[rhs_ptr]), %%ymm1 \n\t"  // mov rhs to ymm1
        "vpermq $0x44,%%ymm1, %%ymm1 \n\t"
        // next LHS cell elements 9 and 10
        "vpmovzxbw 0x50(%[lhs_ptr]), %%ymm0 \n\t"  // mov lhs to ymm0
        "vpshufd $0x00,%%ymm1,%%ymm2        \n\t"  // move rhs 0 element to ymm2
        "vpshufd $0x55,%%ymm1,%%ymm3        \n\t"  // move rhs 1 element to ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // muladd lhs rhs0 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // muladd lhs rhs1 into ymm3
        "vpaddd %%ymm2, %%ymm12, %%ymm12    \n\t"  // accumulate to ymm12
        "vpaddd %%ymm3, %%ymm13, %%ymm13    \n\t"  // accumulate to ymm13

        // next LHS cell elements 11 and 12
        "vpshufd $0xaa,%%ymm1,%%ymm2        \n\t"  // mov rhs 2 element to ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3        \n\t"  // mov rhs 3 element to ymm2
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2    \n\t"  // mul add lhs rhs2 into ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3    \n\t"  // mull add lhs rhs3 into ymm3
        "vpaddd %%ymm2, %%ymm14, %%ymm14    \n\t"  // add lhs rhs2 to ymm14
        "vpaddd %%ymm3, %%ymm15, %%ymm15    \n\t"  // add lhs rhs3 to ymm15

        // completed rhs+10
        "addq $0x60, %[lhs_ptr]             \n\t"  // increment stride lhs
        "addq $0x10, %[rhs_ptr]             \n\t"  // increment stride rhs

        "subq $2, %[run_depth_cells] \n\t"
        "ja outerLoop2%= \n\t"

        "movq %[run_depth_cells], %%r14 \n\t"
        "decq %%r14 \n\t"
        "js finish%= \n\t"

        // Loop for K unrolled by 2
        "outerLoop1%=: \n\t"

        // rhs lower
        "vpmovzxbw (%[rhs_ptr]), %%ymm1 \n\t"  // get rhs into ymm1
        "vpermq $0x44,%%ymm1, %%ymm1 \n\t"

        // LHS cell
        "vpmovzxbw (%[lhs_ptr]), %%ymm0  \n\t"      // lhs in into ymm0
        "vpshufd $0x00,%%ymm1,%%ymm2         \n\t"  // rhs element 0 into ymm2
        "vpshufd $0x55,%%ymm1,%%ymm3         \n\t"  // rhs element 1 into ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2     \n\t"  // muladd lhs rhs element 0 ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3     \n\t"  // muladd lhs rhs element 1 ymm3
        "vpaddd %%ymm2, %%ymm4, %%ymm4       \n\t"  // acc element 0 ymm4
        "vpaddd %%ymm3, %%ymm5, %%ymm5       \n\t"  // acc element 1 ymm5
        "vpshufd $0xaa,%%ymm1,%%ymm2         \n\t"  // rhs element 2 into ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3         \n\t"  // rhs element 3 into ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2     \n\t"  // muladd lhs rhs element 2 ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3     \n\t"  // muladd lhs rhs element 3 ymm3
        "vpaddd %%ymm2, %%ymm6, %%ymm6       \n\t"  // acc element 2 into ymm6
        "vpaddd %%ymm3, %%ymm7, %%ymm7       \n\t"  // acc element 3 into ymm7

        // lhs+10
        "vpmovzxbw 0x10(%[lhs_ptr]), %%ymm0  \n\t"  // lhs in into ymm0
        "vpshufd $0x00, %%ymm1, %%ymm2       \n\t"  // rhs element 0 into ymm2
        "vpshufd $0x55, %%ymm1, %%ymm3       \n\t"  // rhs element 1 into ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2     \n\t"  // muladd lhs rhs element 0 ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3     \n\t"  // muladd lhs rhs element 1 ymm3
        "vpaddd %%ymm2, %%ymm8, %%ymm8       \n\t"  // acc element 0 ymm8
        "vpaddd %%ymm3, %%ymm9, %%ymm9       \n\t"  // acc element 1 ymm9
        "vpshufd $0xaa,%%ymm1,%%ymm2         \n\t"  // rhs element 2 into ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3         \n\t"  // rhs element 3 into ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2     \n\t"  // muladd lhs rhs element 2 ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3     \n\t"  // muladd lhs rhs element 3 ymm3
        "vpaddd %%ymm2, %%ymm10, %%ymm10     \n\t"  // acc element 2 into ymm10
        "vpaddd %%ymm3, %%ymm11, %%ymm11     \n\t"  // acc element 3 into ymm11

        "vpmovzxbw 0x20(%[lhs_ptr]), %%ymm0  \n\t"
        "vpshufd $0x00, %%ymm1, %%ymm2       \n\t"  // rhs element 0 into ymm2
        "vpshufd $0x55, %%ymm1, %%ymm3       \n\t"  // rhs element 1 into ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2     \n\t"  // muladd lhs rhs element 0 ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3     \n\t"  // muladd lhs rhs element 1 ymm3
        "vpaddd %%ymm2, %%ymm12, %%ymm12     \n\t"  // acc element 0 ymm12
        "vpaddd %%ymm3, %%ymm13, %%ymm13     \n\t"  // acc element 1 ymm13
        "vpshufd $0xaa,%%ymm1,%%ymm2         \n\t"  // rhs element 2 into ymm2
        "vpshufd $0xff,%%ymm1,%%ymm3         \n\t"  // rhs element 3 into ymm3
        "vpmaddwd %%ymm0, %%ymm2, %%ymm2     \n\t"  // muladd lhs rhs element 2 ymm2
        "vpmaddwd %%ymm0, %%ymm3, %%ymm3     \n\t"  // muladd lhs rhs element 3 ymm3
        "vpaddd %%ymm2, %%ymm14, %%ymm14     \n\t"  // acc element 2 into ymm14
        "vpaddd %%ymm3, %%ymm15, %%ymm15     \n\t"  // acc element 3 into ymm15

        // update matrix pointers
        "addq $0x30, %[lhs_ptr]              \n\t"
        "addq $0x08, %[rhs_ptr]              \n\t"

        "decq %[run_depth_cells]             \n\t"
        "jnz outerLoop1%=                    \n\t"

        "finish%=:\n\t"

        "test %[start_depth], %[start_depth] \n\t"
        "jz storeDst%= \n\t"

        "vpaddd 0x00(%[dst_ptr]), %%ymm4, %%ymm4 \n\t"    // rhs0
        "vpaddd 0x20(%[dst_ptr]), %%ymm8, %%ymm8 \n\t"    // rhs0
        "vpaddd 0x40(%[dst_ptr]), %%ymm12, %%ymm12 \n\t"  // rhs0

        "vpaddd 0x00(%[dst_ptr], %%r12, 1) , %%ymm5, %%ymm5   \n\t"  // rhs1
        "vpaddd 0x20(%[dst_ptr], %%r12, 1) , %%ymm9, %%ymm9   \n\t"  // rhs1
        "vpaddd 0x40(%[dst_ptr], %%r12, 1) , %%ymm13, %%ymm13 \n\t"  // rhs1

        "vpaddd 0x00(%[dst_ptr], %%r12, 2) , %%ymm6, %%ymm6   \n\t"  // rhs2
        "vpaddd 0x20(%[dst_ptr], %%r12, 2) , %%ymm10, %%ymm10 \n\t"  // rhs2
        "vpaddd 0x40(%[dst_ptr], %%r12, 2) , %%ymm14, %%ymm14 \n\t"  // rhs2

        "vpaddd 0x00(%[dst_ptr], %%r13, 1) , %%ymm7, %%ymm7   \n\t"  // rhs3
        "vpaddd 0x20(%[dst_ptr], %%r13, 1) , %%ymm11, %%ymm11 \n\t"  // rhs3
        "vpaddd 0x40(%[dst_ptr], %%r13, 1) , %%ymm15, %%ymm15 \n\t"  // rhs3

        "storeDst%=:\n\t"

        "vmovdqu %%ymm4, 0x00(%[dst_ptr])            \n\t"  // rhs0
        "vmovdqu %%ymm8, 0x20(%[dst_ptr])            \n\t"  // rhs0
        "vmovdqu %%ymm12, 0x40(%[dst_ptr])           \n\t"  // rhs0

        "vmovdqu %%ymm5, 0x00(%[dst_ptr], %%r12, 1)  \n\t"  // rhs1
        "vmovdqu %%ymm9, 0x20(%[dst_ptr], %%r12, 1)  \n\t"  // rhs1
        "vmovdqu %%ymm13, 0x40(%[dst_ptr], %%r12, 1) \n\t"  // rhs1

        "vmovdqu %%ymm6, 0x00(%[dst_ptr], %%r12, 2)  \n\t"  // rhs2
        "vmovdqu %%ymm10, 0x20(%[dst_ptr], %%r12, 2) \n\t"  // rhs2
        "vmovdqu %%ymm14, 0x40(%[dst_ptr], %%r12, 2) \n\t"  // rhs2

        "vmovdqu %%ymm7, 0x00(%[dst_ptr], %%r13, 1)  \n\t"  // rhs3
        "vmovdqu %%ymm11, 0x20(%[dst_ptr], %%r13, 1) \n\t"  // rhs3
        "vmovdqu %%ymm15, 0x40(%[dst_ptr], %%r13, 1) \n\t"  // rhs3

        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr)
        :  // inputs
        [start_depth] "r"(start_depth), [dst_col_stride_q] "r"(dst_col_stride_q),
        [run_depth_cells] "r"(run_depth_cells)
        :  // clobbers
        "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
        "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%r12",
        "%r13", "%r14");
  }
};
#endif

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_AVX_H_
