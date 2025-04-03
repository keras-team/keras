/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_FWD_BASE_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_FWD_BASE_HPP

#include <functional>
#include <memory>
#include "common/c_types_map.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using acc_data_t = float;
using acc_data_bf16_t = uint16_t;

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;
using namespace Xbyak;
using namespace Xbyak::util;

template <data_type_t d_type>
class jit_avx512_common_lrn_kernel_fwd_t : public jit_generator {
public:
    jit_avx512_common_lrn_kernel_fwd_t(prop_kind_t prop_kind, float alpha,
            float beta, float k, int local_size, const char *name = jit_name());

    using data_t = typename prec_traits<d_type>::type;

    struct jit_args_fwd_t {
        jit_args_fwd_t();
        const data_t *src;
        data_t *dst, *ws0, *ws1;
        static const int32_t mask[48];
        const int32_t *mask_ptr;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_fwd_t);

protected:
    Zmm zreg(int irb, int i) const;
    Ymm yreg(int irb, int i) const;
    Xmm xreg(int irb, int i) const;

    void store_data(const Address addr, Zmm zr, Ymm yr);
    void load_tail(int tail_value, Reg64 src, int src_mem_offset,
            int dst_stack_offset, int tmp_load_to_stack_idx_tail);
    void load_data(Xmm reg, const Address p, bool from_stack = false);
    void store_tail(int tail_value, Zmm src, Reg64 dst, int dst_mem_offset,
            int tmp_stack_offset, int tmp_idx);

    prop_kind_t pk_;
    float alpha_, beta_, k_;
    static constexpr int xmm_size_ = 4 * sizeof(acc_data_t);
    static constexpr int zmm_size_ = 64;
    const Reg64 imm_addr64_ = rbx;
    const Reg16 imm_addr16_ = bx;

    const Xmm xalpha_ = xmm0;
    const Zmm zalpha_ = zmm0;
    const Zmm zk_ = zmm1;
    const Xmm xk_ = xmm1;
    const Reg64 src_ = rax;
    const Reg64 dst_ = r8;
    const Reg64 ws0_ = rdx;
    const Reg64 ws1_ = rsi;
    const Reg64 param_ = abi_param1;

    const int local_size_;

    static constexpr int zc_ = 2;
    const std::vector<int> z_prev_;
    const std::vector<int> z_next_;

    const int zsum_;
    static constexpr int zsrc_ = 2;
    static constexpr int zdst_ = 3;
    static constexpr int zsum2_ = 5;
    static constexpr int zbase_ = 4;

    const Zmm bf16_emu_reserv_1_ = zmm28;
    const Zmm bf16_emu_reserv_2_ = zmm29;
    const Reg64 bf16_emu_scratch_ = rax;
    const Zmm bf16_emu_reserv_3_ = zmm30;
    const Zmm bf16_emu_reserv_4_ = zmm31;

    const bool emulateBfloat_;
    const int regs_used_per_block_;
    const int reg_block_;
    static constexpr int vlen_ = utils::one_of(d_type, bf16, f16) ? 32 : 64;
    std::unique_ptr<bf16_emulation_t> bf16_emu_ = nullptr;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
