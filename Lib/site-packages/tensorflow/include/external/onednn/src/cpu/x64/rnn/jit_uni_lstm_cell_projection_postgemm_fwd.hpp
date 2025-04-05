/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_PROJECTION_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_PROJECTION_POSTGEMM_FWD_HPP

#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_lstm_cell_projection_postgemm_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_projection_postgemm_fwd)

    jit_uni_lstm_cell_projection_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    ~jit_uni_lstm_cell_projection_postgemm_fwd() {}

    status_t init(data_type_t sdt) override {
        jit_uni_rnn_postgemm::init(src_data_t);
        projection_ = true;
        return create_kernel();
    }

protected:
    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;
    static constexpr size_t qscale_dt_size = sizeof(float);
    const size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t hstate_dt_size = types::data_type_size(src_data_t);
    const size_t scratch_dt_size = types::data_type_size(scratch_data_t);

    void generate() {
        using namespace Xbyak;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;

        // Register map
        const Reg64 loop_cnt(rbx); // loop counter
        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        const Vmm in(1), tmp1_vmm(5), tmp2_vmm(6);

        const int mask = pd_->attr()->rnn_weights_projection_qparams_.mask_;
        float *weights_scales
                = pd_->attr()->rnn_weights_projection_qparams_.scales_;

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_scratch_reg = abi_param2;
        const auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
        const auto addr_states_t_l_copy_reg = r10;
        const auto addr_wcomp_reg = rdi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_wcomp_reg, ptr[base_args + 8]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        const auto addr_wcomp_reg = abi_param6;
#endif

        // initialize registers with addresses and constants
        init_regs(weights_scales, vlen);

        mov(loop_cnt, rnn_.dic * scratch_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            uni_vmovups(in, ptr[addr_scratch_reg]);
            deq_w(src_data_t, in, tmp1_vmm, tmp2_vmm, 0, mask, vlen,
                    &addr_wcomp_reg);
            to_src(ptr[addr_states_t_l_reg], in, src_data_t, vlen);

            // if states_t_l_copy is a non null ptr, we write the output to both
            // tensors
            cmp(addr_states_t_l_copy_reg, 0);
            je(vector_loop_inc_regs);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(ptr[addr_states_t_l_copy_reg], in, src_data_t, vlen, true);
            add(addr_states_t_l_copy_reg, vlen_dst);

            // increment address pointers
            L(vector_loop_inc_regs);
            add(addr_scratch_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
            inc_regs(mask, vlen);

            // increment loop counter
            sub(loop_cnt, vlen);
            cmp(loop_cnt, vlen);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use vmovss for accessing inputs
        L(rem_loop_start_label);
        {

            uni_vmovss(in, ptr[addr_scratch_reg]);
            deq_w(src_data_t, in, tmp1_vmm, tmp2_vmm, 0, mask, scratch_dt_size,
                    &addr_wcomp_reg);
            to_src(ptr[addr_states_t_l_reg], in, src_data_t, scratch_dt_size);

            // if states_t_l_copy is a non null ptr, we write the output to both
            // tensors
            cmp(addr_states_t_l_copy_reg, 0);
            je(rem_loop_inc_regs);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(ptr[addr_states_t_l_copy_reg], in, src_data_t,
                    scratch_dt_size, true);
            add(addr_states_t_l_copy_reg, hstate_dt_size);

            // increment address pointers
            L(rem_loop_inc_regs);
            add(addr_scratch_reg, scratch_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            inc_regs(mask, qscale_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();
        init_table(vlen);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
