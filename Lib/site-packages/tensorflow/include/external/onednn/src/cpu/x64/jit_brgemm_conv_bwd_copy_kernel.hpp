/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_BWD_COPY_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_CONV_BWD_COPY_KERNEL_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_avx512_core_brgemm_conv_bwd_copy_kernel {
struct jit_brgemm_conv_bwd_copy_kernel_call_s {
    const void *src;
    const void *dst;
    size_t num_ic;
};

template <typename Vmm>
struct jit_avx512_core_brgemm_conv_bwd_copy_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_brgemm_conv_bwd_copy_kernel_t)

    using reg64_t = const Xbyak::Reg64;

    jit_avx512_core_brgemm_conv_bwd_copy_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp);

protected:
    static constexpr bool is_zmm_ = std::is_same<Vmm, Xbyak::Zmm>::value;
    const jit_brgemm_conv_conf_t &jcp;
    const reg64_t inp_ptr = r15;
    const reg64_t dst_ptr = r14;
    const reg64_t reg_num_ic = r10;
    const reg64_t reg_tmp = rsi;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask kblock_tail_mask = Xbyak::Opmask(3);

    const Vmm vmm_tmp = Vmm(0);
    void load(
            const Vmm &x, const Xbyak::Address &addr, const int load_size = 0);

    void store(
            const Xbyak::Address &addr, const Vmm &x, const int store_size = 0);
    void generate() override;
};

} // namespace jit_avx512_core_brgemm_conv_bwd_copy_kernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
