/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef CPU_X64_GEMM_S8X8S32_JIT_AVX512_CORE_KERNEL_GEMV_S8X8S32_KERN_HPP
#define CPU_X64_GEMM_S8X8S32_JIT_AVX512_CORE_KERNEL_GEMV_S8X8S32_KERN_HPP

#include <cstdint>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/gemm_info.hpp"

#include "cpu/x64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

enum class ver_t { undef, s8s8, s8u8, u8s8 };

class jit_avx512_core_gemv_s8x8s32_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_gemv_s8x8s32_kern);

    enum class vnni_op_t { add, sub };

    void vnni(Xbyak::Zmm acc, Xbyak::Zmm b, Xbyak::Zmm a, vnni_op_t op);
    void n_loop_body(int nreg_acc, Xbyak::Reg64 A, Xbyak::Reg64 lda,
            Xbyak::Reg64 X, int use_mask, Xbyak::Opmask mask_n);
    void shuffle_and_add(
            Xbyak::Zmm, Xbyak::Zmm, Xbyak::Zmm, Xbyak::Zmm, Xbyak::Zmm);
    void update_c(int, Xbyak::Reg64, int, Xbyak::Opmask);
    void generate() override ATTRIBUTE_OPTIMIZE;

    cpu_isa_t isa = isa_undef;
    ver_t ver = ver_t::undef;

    // Assumes unroll_{m,n} are a power of 2.
    static constexpr unsigned int unroll_m_ = 4; // Unrolling is 2^unroll_m_.
    static constexpr unsigned int unroll_n_ = 6; // Unrolling is 2^unroll_n_.

    enum {
        zmm_a_idx_start = 5,
        zmm_a_idx_count = 1 << (unroll_m_ - 1), // nreg_A
        zmm_acc_idx_start = zmm_a_idx_start + zmm_a_idx_count,
        zmm_acc_idx_count = 1 << unroll_m_, // nreg_acc
    };

    Xbyak::Zmm zmm_tmp = Xbyak::Zmm(0);
    Xbyak::Xmm xmm_beta = Xbyak::Xmm(1);

    Xbyak::Zmm zmm_1_s16 = Xbyak::Zmm(2); // avx512_core
    Xbyak::Zmm zmm_1_u1 = Xbyak::Zmm(2); // s8s8, avx512_core_vnni
    Xbyak::Zmm zmm_128_u8 = Xbyak::Zmm(3); // s8s8

    Xbyak::Zmm zmm_b = Xbyak::Zmm(4); // x-vector
    Xbyak::Zmm zmm_a(int idx) { // matrix
        assert(idx < zmm_a_idx_count);
        return Xbyak::Zmm(zmm_a_idx_start + idx);
    }
    Xbyak::Zmm zmm_acc(int idx) { // y-vector
        assert(idx < zmm_acc_idx_count);
        return Xbyak::Zmm(zmm_acc_idx_start + idx);
    }

public:
    jit_avx512_core_gemv_s8x8s32_kern(ver_t ver)
        : jit_generator(jit_name(),
                mayiuse(avx512_core_vnni) ? avx512_core_vnni : avx512_core)
        , ver(ver) {}
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_S8X8S32_JIT_AVX512_CORE_KERNEL_GEMV_S8X8S32_KERN_HPP
