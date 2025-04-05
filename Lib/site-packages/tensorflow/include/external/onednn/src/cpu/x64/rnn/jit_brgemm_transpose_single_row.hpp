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

#ifndef CPU_X64_RNN_JIT_BRGEMM_TRANSPOSE_SINGLE_ROW_HPP
#define CPU_X64_RNN_JIT_BRGEMM_TRANSPOSE_SINGLE_ROW_HPP

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
 * Transpose generator for brgemm based rnn, optimized for cases when number of
 * input rows == 1.
 * In such case, because of perf reasons, number of output columns is extended
 * to 2.
 */
class jit_brgemm_transpose_single_row_t : public jit_generator {
public:
    jit_brgemm_transpose_single_row_t(const int m_block);

    struct call_params_t {
        const void *src = nullptr;
        void *dst = nullptr;
    };

    void operator()(
            jit_brgemm_transpose_single_row_t::call_params_t *params) const {
        jit_generator::operator()(params);
    }

private:
    std::vector<Xbyak::Zmm> reserve_acc_regs();
    void generate() override;
    void load_addresses();
    void compute_loop();
    void compute(const dim_t unrolling, const bool is_tail);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_transpose_single_row_t)
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_brgemm_transpose_single_row_t);

    static constexpr dim_t simd_w_ = 16;
    static constexpr dim_t vmms_available_ = 32;

    const int m_block_;
    const int full_loop_iters_;
    const int tail_;
    const int k_blocks_nb_;

    const Xbyak::Reg64 &reg_src_ = r8;
    const Xbyak::Reg64 &reg_dst_ = r9;

    const Xbyak::Reg64 &reg_tmp_ = r10;
    const Xbyak::Reg64 &reg_full_loop_ = r11;
    const Xbyak::Opmask &tail_mask_ = k1;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
