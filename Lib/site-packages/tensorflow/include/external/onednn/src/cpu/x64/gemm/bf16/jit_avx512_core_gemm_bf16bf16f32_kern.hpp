/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_X64_GEMM_BF16_JIT_AVX512_CORE_GEMM_BF16BF16F32_KERN_HPP
#define CPU_X64_GEMM_BF16_JIT_AVX512_CORE_GEMM_BF16BF16F32_KERN_HPP

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_avx512_core_gemm_bf16bf16f32_kern : public jit_generator {
public:
    jit_avx512_core_gemm_bf16bf16f32_kern(
            bool beta_zero, bool alpha_one, bool use_zmm);
    ~jit_avx512_core_gemm_bf16bf16f32_kern();
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_gemm_bf16bf16f32_kern);

protected:
    bool beta_zero_;
    bool alpha_one_;
    bool bfloat16_;

    int UNROLL_M_;

    int a_nelems_;
    int b_nelems_;
    int c_nelems_;

    void prefetch_a(const Xbyak::Address &src) { prefetcht0(src); }
    void prefetch_b(const Xbyak::Address &src) { prefetcht0(src); }
    void prefetch_c(const Xbyak::Address &src) { prefetchw(src); }
    void prefetch_x(const Xbyak::Address &src) { prefetcht0(src); }

    void c_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    void c_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);

    void dot_product(const Xbyak::Xmm &dst, const Xbyak::Xmm &src1,
            const Xbyak::Xmm &src2);
    void kernel_loop(int unroll_m, int unroll_n, bool cfetch);
    void remainder_kernel(int unroll_m, int unroll_n, int unroll_k, int bwidth);
    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y, Xbyak::Label *&outerloop_label);

    void generate() override ATTRIBUTE_OPTIMIZE;

private:
    static const int UNROLL_N_ = 8;

    static const int isize_ = 2;
    static const int size_ = 4;

    // Prefetch configuration
    static const int prefetch_size_a_ = 32 * 5;
    static const int prefetch_size_b_ = 32 * 4;

    static const int offset_a_ = 256, offset_b_ = 256;
    static const int max_unroll_m_ = 48, max_unroll_n_ = 8;

    // Integer register assignments
    Xbyak::Reg64 M_, N_, K_, ALPHA_, A_, B_, C_, LDC_, I_, J_, LoopCount_;
    Xbyak::Reg64 AO_, BO_, CO1_, CO2_, AA_;

    // Vector register assignments
    Xbyak::Xmm alpha_, a_regs_[max_unroll_m_ >> 4], b_regs_[2];
    Xbyak::Xmm c_regs_[max_unroll_m_ >> 4][max_unroll_n_];

    // Kind of vector register (Zmm or Ymm)
    Xbyak::Operand::Kind kind_;

    // Stack variable assignments
    int stack_alloc_size_;
    Xbyak::Address arg_a_, arg_b_, arg_c_, arg_ldc_, arg_coffset_c_,
            arg_coffset_r_;

    // For bfloat16 emulation on avx512 and avx512_vnni ISAs
    bf16_emulation_t *bf16_emu_;
    Xbyak::Reg64 scratch_;
    Xbyak::Zmm one_;
    Xbyak::Zmm even_;
    Xbyak::Zmm selector_;
    Xbyak::Zmm zmm_tmp0_;
    Xbyak::Zmm zmm_tmp1_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_BF16_JIT_AVX512_CORE_GEMM_BF16BF16F32_KERN_HPP
