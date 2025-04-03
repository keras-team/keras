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

#ifndef CPU_X64_RNN_JIT_DIFF_WEIGHTS_PEEPHOLE_HPP
#define CPU_X64_RNN_JIT_DIFF_WEIGHTS_PEEPHOLE_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rnn_utils {
struct rnn_conf_t;
}; // namespace rnn_utils
namespace x64 {

class jit_diff_weights_peephole_t : public jit_generator {
public:
    jit_diff_weights_peephole_t(
            const rnn_utils::rnn_conf_t &rnn, const dim_t dhc_block);

    struct call_params_t {
        const void *c_states = nullptr;
        const void *scratch_gates = nullptr;
        void *dst = nullptr;
    };

    void operator()(jit_diff_weights_peephole_t::call_params_t *params) const {
        jit_generator::operator()(params);
    }

private:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_diff_weights_peephole_t);
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_diff_weights_peephole_t);

    void generate() override;
    void load_addresses();
    void init();
    void compute_loop();
    void compute_dst(size_t unrolling, bool tail);

    static constexpr dim_t simd_w_ = 16;
    static constexpr dim_t max_unrolling = 10;

    const data_type_t c_states_dt_;
    const data_type_t scratch_dt_;
    const data_type_t dst_dt_;

    const Xbyak::Reg64 &loop_cnt_ = rax;
    const Xbyak::Reg64 &reg_c_states_ = r8;
    const Xbyak::Reg64 &reg_scratch_gates_ = r9;
    const Xbyak::Reg64 &reg_dst_ = r10;
    const Xbyak::Reg64 &reg_tmp_ = r11;
    const Xbyak::Reg64 &reg_offset_ = r12;

    const Xbyak::Opmask &tail_opmask_ = k3;

    const dim_t compute_block_size_;
    const dim_t tail_size_;

    io::jit_io_multi_dt_helper_t<Xbyak::Zmm> io_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
