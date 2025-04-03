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

#ifndef CPU_X64_GEMM_BF16_JIT_AVX512_CORE_GEMV_BF16BF16F32_KERN_HPP
#define CPU_X64_GEMM_BF16_JIT_AVX512_CORE_GEMV_BF16BF16F32_KERN_HPP

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_avx512_core_gemv_bf16bf16f32_kern : public jit_generator {
public:
    jit_avx512_core_gemv_bf16bf16f32_kern(bool trans);
    ~jit_avx512_core_gemv_bf16bf16f32_kern();
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_gemv_bf16bf16f32_kern);

protected:
    bool trans_;
    bool bfloat16_;

    void prefetch_a(const Xbyak::Address &src) { prefetcht0(src); }
    void prefetch_x(const Xbyak::Address &src) { prefetcht0(src); }
    void prefetch_y(const Xbyak::Address &src) { prefetcht0(src); }

    void y_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    void y_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);
    void v_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);

    void dot_product(const Xbyak::Xmm &dst, const Xbyak::Xmm &src1,
            const Xbyak::Xmm &src2);
    void kernel_loop_t(int unroll_m, int unroll_n, bool fetch, bool last);
    void kernel_loop_n(int unroll_m, int unroll_n, bool cfetch, bool last);
    void innerloop_t(int unroll_n);
    void innerloop_n(int unroll_n);
    void outerloop(int unroll_y, Xbyak::Label *&cur_outerloop_label,
            Xbyak::Label *&outerloop_end_label);

    void generate() override ATTRIBUTE_OPTIMIZE;

private:
    static const int UNROLL_M_ = 64;
    static const int UNROLL_N_ = 8;

    static const int size_a_ = 2;
    static const int size_x_ = 2;
    static const int size_y_ = 4;

    // Prefetch configuration
    static const int prefetch_size_a_ = 32;
    static const int prefetch_size_x_ = 32;
    static const int prefetch_size_y_ = 32;

    static const int offset_a_ = 32;
    static const int offset_x_ = 32;
    static const int offset_y_ = 32;

    // Integer register assignments
    Xbyak::Reg64 M_, N_, ALPHA_, A_, LDA_, X_, INCX_, Y_, INCY_, I_;
    Xbyak::Reg64 A1_, A2_, X1_, Y1_, LDA3_;

    // Vector register assignments
    Xbyak::Zmm alpha_, a_[UNROLL_N_], a_pack_[2];
    Xbyak::Zmm x_[2], x_pack_[UNROLL_N_ >> 1];
    Xbyak::Zmm y_[UNROLL_M_ >> 4], acc_[UNROLL_N_];
    Xbyak::Zmm scratch_[4];
    // Stack variable assignments
    Xbyak::Address arg_lda_, arg_x_, arg_incx_, arg_y_, arg_incy_;

    // For bfloat16 emulation on avx512 and avx512_vnni ISAs
    bf16_emulation_t *bf16_emu_;
    Xbyak::Reg64 gpr_;
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

#endif // CPU_X64_GEMM_BF16_JIT_AVX512_CORE_GEMV_BF16BF16F32_KERN_HPP
