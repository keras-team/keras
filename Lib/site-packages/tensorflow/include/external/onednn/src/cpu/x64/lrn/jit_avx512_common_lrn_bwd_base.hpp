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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BASE_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BASE_HPP

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
class jit_avx512_common_lrn_kernel_bwd_t : public jit_generator {
public:
    jit_avx512_common_lrn_kernel_bwd_t(float alpha, float beta, int local_size,
            const char *name = jit_name());

    using data_t = typename prec_traits<d_type>::type;

    struct jit_args_bwd_t {
        jit_args_bwd_t();
        const data_t *src, *diff_dst, *ws0, *ws1;
        data_t *diff_src;
        static const int32_t mask[48];
        const int32_t *mask_ptr;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_bwd_t);

protected:
    Zmm zreg(int irb, int i) const;
    Ymm yreg(int irb, int i) const;
    Xmm xreg(int irb, int i) const;

    void store_data(bool non_temp_hint, const Address addr, Zmm zr);
    void load_data(Xmm reg, const Address p, bool from_stack = false);
    void load_tail(int tail_value, Reg64 src, int src_mem_offset,
            int dst_stack_offset, int tmp_load_to_stack_idx_tail);
    void store_tail(int tail_value, Zmm src, Reg64 dst, int dst_mem_offset,
            int tmp_stack_offset, int tmp_idx);

    const Reg64 src_ = rax;
    const Reg64 diffsrc_ = r8;
    const Reg64 diffdst_ = r9;
    const Reg64 workspace0_ = rdx;
    const Reg64 workspace1_ = rsi;
    const Reg64 imm_addr64_ = rbx;
    const Reg64 param_ = abi_param1;
    const Reg16 imm_addr16_ = bx;
    const Zmm znalphabeta_ = zmm0;
    const Xmm xnalphabeta_ = xmm0;

    const Zmm bf16_emu_reserv_1_ = Zmm(28);
    const Zmm bf16_emu_reserv_2_ = Zmm(29);
    const Reg64 bf16_emu_scratch_ = rax;
    const Zmm bf16_emu_reserv_3_ = Zmm(30);
    const Zmm bf16_emu_reserv_4_ = Zmm(31);
    const int local_size_;

    static constexpr int z_tmp_ = 7;

    static constexpr int zdiffdst_ = 1;
    static constexpr int zdiffsrc_ = 2;
    static constexpr int zsrc_ = 3;

    const std::vector<int> z_prev_;
    const std::vector<int> z_next_;

    static constexpr int zws0_ = 4;

    float nalphabeta_;
    const bool emulateBfloat_;
    const int regs_used_per_block_;
    const int reg_block_;
    static constexpr int vlen_ = utils::one_of(d_type, bf16, f16) ? 32 : 64;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
