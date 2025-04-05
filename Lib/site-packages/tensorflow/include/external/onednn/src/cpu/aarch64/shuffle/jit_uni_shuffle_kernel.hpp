/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_SHUFFLE_KERNEL_HPP
#define CPU_AARCH64_JIT_UNI_SHUFFLE_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_shuffle_pd.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/shuffle/jit_uni_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
struct jit_uni_shuffle_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_shuffle_kernel_t)

    jit_uni_shuffle_kernel_t(const jit_shuffle_conf_t conf);

    using TReg = typename utils::conditional<isa == asimd, VReg, ZReg>::type;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    constexpr int vmm_idx(int idx) const {
        return (cpu_isa_traits<isa>::n_vregs - 1) - idx;
    }

    /*
     * Prepare the mask to be used during tail processing.
     * if it is sve_512 at least then k_tail_mask_ is filled.
     */
    void prepare_mask();

    /*
     * Emulates the behavior of vgatherdps for architectures
     * that do not support this instruction.
     */
    void emu_gather_data(const XReg &reg_src_addr, const int indices_idx,
            const int data_idx, const bool is_tail = false);

    void gather_data(const XReg &reg_src_addr, const int indices_idx,
            const int data_idx, const bool is_tail = false);

    void store_data(const int data_idx, const XReg &reg_dst_addr,
            const int offset = 0, const bool is_tail = false);

    void shuffle_blocked_format();

    void append_zero_padding(
            const XReg &reg_dst_addr, const bool zero_extend_write);

    void generate() override;

    const TReg vmm_full_mask_ = TReg(1);
    const TReg vmm_src_ = TReg(2);
    const TReg vmm_tmp_ = TReg(3);
    const TReg vmm_indices_ = TReg(4);
    const TReg vmm_zero_ = TReg(11);

    const PReg k_tail_mask_ = p1;
    const PReg k_full_mask_ = p2;

    const XReg &reg_tmp_ = x7;
    const XReg &reg_dst_ = x3;
    const XReg &reg_indices_ = x1;
    const XReg &reg_work_ = x2;
    // Always mimic the Unix ABI
    const XReg &reg_param = x0;
    const XReg &reg_src_ = x6;
    const XReg &reg_tmp1_ = x8;
    const XReg &reg_tmp2_ = x9;
    const XReg &reg_tmp3_ = x10;
    const XReg &reg_tmp4_ = x11;
    const XReg &reg_tmp5_ = x12;
    const XReg &reg_tmp6_ = x13;
    const XReg &reg_padded_block = x14; //WReg(x14.getIdx());

    const jit_shuffle_conf_t conf_;
    const size_t padding_size_;
    const uint64_t isa_sveLen
            = is_superset(isa, sve_128) ? cpu_isa_traits<isa>::vlen : 0;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
