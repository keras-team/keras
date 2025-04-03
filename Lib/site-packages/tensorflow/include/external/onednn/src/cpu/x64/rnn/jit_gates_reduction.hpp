/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef CPU_X64_RNN_JIT_GATES_REDUCTION_HPP
#define CPU_X64_RNN_JIT_GATES_REDUCTION_HPP

#include <vector>
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rnn_utils {
struct rnn_conf_t;
}; // namespace rnn_utils

namespace x64 {

/*
 * Used in gates reduction phase during backward rnn/lstm calculations.
 * Fused into diff weights calculations. Performing diff_bias calculations.
 *
 * diff_bias = scratch_blocked reduction over mb
 *
 * Data formats
 * scratch_blocked Oi32o(f32)/OI32o2i(bf16) (n_gates * rnn.dhc, mb)
 * diff_bias = o(n_gates * rnn.dhc)
 */
class jit_gates_reduction_t : public jit_generator {
public:
    jit_gates_reduction_t(const rnn_utils::rnn_conf_t &rnn, bool is_n_tail);

    struct call_params_t {
        const void *src = nullptr;
        void *dst = nullptr;
    };

    void operator()(jit_gates_reduction_t::call_params_t *params) const {
        jit_generator::operator()(params);
    }

private:
    std::vector<Xbyak::Zmm> reserve_acc_regs();
    void generate() override;
    void load_addresses();
    void init();
    void store_data();
    void compute_loop();
    void compute(dim_t unrolling);
    void compute_step(
            const Xbyak::Zmm &acc, const Xbyak::Address &addr, bool tail);
    size_t reserve_vmm();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_gates_reduction_t)
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_gates_reduction_t);

    static constexpr dim_t simd_w_ = 16;

    size_t number_reserved_vmms_ = 0;
    const rnn_utils::rnn_conf_t &rnn_;
    const bool is_n_tail_;
    const dim_t n_block_;
    const dim_t n_simd_w_blks_;
    const dim_t n_tail_;

    const Xbyak::Reg64 &reg_src_ = r8;
    const Xbyak::Reg64 &reg_dst_ = r9;
    const Xbyak::Reg64 &reg_tmp_ = r10;
    const Xbyak::Reg64 &reg_loop_ = r11;
    const Xbyak::Opmask &tail_mask_ = k3;
    const Xbyak::Opmask &k_f16_perm_mask = k4;
    const Xbyak::Zmm bf16_ones_;
    const Xbyak::Zmm f16_tmp_vreg_;
    const Xbyak::Zmm f16_vperm_vreg_;
    std::vector<Xbyak::Zmm> acc_regs_;
    Xbyak::Label f16_perm_table_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
