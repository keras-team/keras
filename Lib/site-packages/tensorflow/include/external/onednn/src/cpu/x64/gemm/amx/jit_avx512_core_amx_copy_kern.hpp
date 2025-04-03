/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_X64_GEMM_AMX_JIT_AVX512_CORE_AMX_COPY_KERN_HPP
#define CPU_X64_GEMM_AMX_JIT_AVX512_CORE_AMX_COPY_KERN_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_avx512_core_amx_copy_kern : public jit_generator {
public:
    jit_avx512_core_amx_copy_kern(bool is_a, bool is_trans, int isize);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_copy_kern);

protected:
    bool is_a_;
    bool is_trans_;
    int lscale_, lsize_;

    int size_;
    int isize_;

    int unroll_k_;

    void load(const Xbyak::Xmm &dst, const Xbyak::Address &src, bool corner);
    void store(const Xbyak::Address &dst, const Xbyak::Xmm &src);

    void transpose(int s, const Xbyak::Ymm &dst1, const Xbyak::Ymm &dst2,
            const Xbyak::Ymm &src1, const Xbyak::Ymm &src2);

    void amxtrans8(const Xbyak::Ymm &dst1, const Xbyak::Ymm &dst2,
            const Xbyak::Ymm &src1, const Xbyak::Ymm &src2,
            const Xbyak::Ymm &src3, const Xbyak::Ymm &src4);
    void amxtrans16(const Xbyak::Ymm &dst1, const Xbyak::Ymm &dst2,
            const Xbyak::Ymm &src1, const Xbyak::Ymm &src2);

    void kernel(int unroll_x, int unroll_y, int step, Xbyak::Reg64 A,
            Xbyak::Reg64 B, bool corner);
    void kernel_AN(int unroll_x, int unroll_y, int step, Xbyak::Reg64 A,
            Xbyak::Reg64 B, bool corner);
    void kernel_BN(int unroll_x, int unroll_y, int step, Xbyak::Reg64 A,
            Xbyak::Reg64 B, bool corner);
    void kernel_AT(int unroll_x, int unroll_y, int step, Xbyak::Reg64 A,
            Xbyak::Reg64 B, bool corner);
    void kernel_BT(int unroll_x, int unroll_y, int step, Xbyak::Reg64 A,
            Xbyak::Reg64 B, bool corner);

    void copy_m(int unroll_m, int unroll_n);
    void copy_n(int unroll_n, Xbyak::Label &epilogue);
    void copy_ns(int unroll_n, Xbyak::Label &epilogue);

    void generate() override ATTRIBUTE_OPTIMIZE;

private:
    static const int offset_a_ = 0, offset_b_ = 0;

    static const int ntiles_m_ = 2;
    static const int unroll_m_ = 16 * ntiles_m_;
    static const int max_unroll_ = 32;
    static const int nstages_ = 4;

    const int idx_[nstages_][2][9] = {
            {{16, 0, 1, 2, 3, 4, 5, 6, 7}, {17, 8, 9, 10, 11, 12, 13, 14, 15}},
            {{7, 16, 0, 1, 2, 17, 8, 9, 10}, {15, 3, 4, 5, 6, 11, 12, 13, 14}},
            {{10, 7, 16, 15, 3, 2, 17, 6, 11}, {14, 0, 1, 4, 5, 8, 9, 12, 13}},
            {{11, 10, 14, 16, 1, 3, 5, 17, 9}, {13, 7, 0, 15, 4, 2, 8, 6, 12}}};

    // Integer register assignments
    Xbyak::Reg64 M_, N_, B_, A_, LDA_;
    Xbyak::Reg64 B1_, BB_, A1_, A2_, LDA3_;
    Xbyak::Reg64 I_, T_, STRIDE_;

    // Vector register assignments
    Xbyak::Ymm src_[16], vecs_[nstages_][2][8 + 1];
    Xbyak::Ymm tmp1_, tmp2_;

    // Stack variable assignments
    int stack_alloc_size_;
    Xbyak::Address arg_b_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_AMX_JIT_AVX512_CORE_AMX_COPY_KERN_HPP
