/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_GEMM_F32_JIT_SSE41_GEMV_N_F32_KERN_HPP
#define CPU_X64_GEMM_F32_JIT_SSE41_GEMV_N_F32_KERN_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_sse41_gemv_n_f32_kern : public jit_generator {
public:
    jit_sse41_gemv_n_f32_kern();
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_gemv_n_f32_kern);

protected:
    bool has_avx512_;
    bool has_avx2_;
    bool has_avx_;
    bool has_sse41_;

    int unroll_m_, unroll_n_;

    int v_nelems_; // Max number of elements in a vector register.

    bool fetch_;
    void prefetch_a(const Xbyak::Address &src) {
        if (fetch_) prefetcht0(src);
    }
    void prefetch_x(const Xbyak::Address &src) {
        if (fetch_) prefetcht0(src);
    }
    void prefetch_y(const Xbyak::Address &src) {
        if (fetch_) prefetcht0(src);
    }

    void v_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    void v_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);
    void dot_product(const Xbyak::Xmm &dst, const Xbyak::Xmm &src1,
            const Xbyak::Xmm &src2);

    void kernel_loop(int unroll_m, int unroll_n, bool fetch, bool last);
    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y,
            Xbyak::Label *&cur_outerloop_label,
            Xbyak::Label *&outerloop_end_label);

    void generate() override ATTRIBUTE_OPTIMIZE;

private:
    static const int max_um_vecs_ = 16;
    static const int max_unroll_n_ = 8;

    static const int size_ = 4;

    static const int offset_a_ = 32;
    static const int offset_x_ = 32;
    static const int offset_y_ = 32;

    // Prefetch configuration
    static const int prefetch_size_a_ = 16;
    static const int prefetch_size_x_ = 16;
    static const int prefetch_size_y_ = 32;

    // Integer register assignments
    Xbyak::Reg64 M_, N_, A_, LDA_, X_, INCX_, Y_, ALPHA_;
    Xbyak::Reg64 A1_, A2_, Y1_, LDA3_, I_;

    // Vector register assignments
    Xbyak::Xmm scratch_, alpha_;
    Xbyak::Xmm a_, x_[max_unroll_n_], y_[max_um_vecs_];
    Xbyak::Xmm acc_[max_um_vecs_];

    // Kind of vector register (Zmm, Ymm, Xmm)
    Xbyak::Operand::Kind kind_;

    // Stack variable assignments
    Xbyak::Address arg_lda_, arg_x_, arg_incx_, arg_y_, arg_incy_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_F32_JIT_SSE41_GEMV_N_F32_KERN_HPP
