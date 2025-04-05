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

#ifndef CPU_X64_RNN_JIT_UNI_GRU_LBR_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_GRU_LBR_CELL_POSTGEMM_FWD_HPP

#include <memory>
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_gru_lbr_cell_postgemm_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_lbr_cell_postgemm_fwd)

    using injector_t = typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_core>,
            jit_uni_eltwise_injector_f32<isa>>::type;

    jit_uni_gru_lbr_cell_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    status_t init(data_type_t sdt) override {
        CHECK(jit_uni_rnn_postgemm::init(src_data_t));
        // we use rax for both constant tables and load correspondent label
        // into it when calling correspondent injector.
        sigmoid_injector_ = utils::make_unique<injector_t>(
                this, alg_kind::eltwise_logistic, 0.0f, 0.0f, 1.0f, true, rax);
        tanh_injector_ = utils::make_unique<injector_t>(
                this, alg_kind::eltwise_tanh, 0.0f, 0.0f, 1.0f, true, rax);
        return create_kernel();
    }

protected:
    std::unique_ptr<injector_t> sigmoid_injector_;
    std::unique_ptr<injector_t> tanh_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;

    const size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t vlen_bias_ = vlen / (sizeof(float) / bias_dt_size_);
    const size_t hstate_dt_size = types::data_type_size(src_data_t);
    const size_t scratch_dt_size = types::data_type_size(scratch_data_t);
    const size_t gate_dt_size = types::data_type_size(src_data_t);
    const size_t vlen_elems = vlen / scratch_dt_size;
    const size_t loop_len = rnn_.dhc;
    const size_t loop_tail = loop_len % nstl::max(size_t(1), vlen_elems);

    void generate() override {
        using namespace Xbyak;

        const auto is_training
                = (pd_->desc()->prop_kind == prop_kind::forward_training);

        const bool is_augru = pd_->cell_kind() == alg_kind::lbr_augru;

        // Labels declaration
        Label tail_processing_or_exit_label, table_label;

        // Register map
        const Reg64 loop_cnt(r10); // loop counter
        const Reg64 table_reg(rbx); // table is used for data scale and shifts

        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        const Vmm G0(1), G1(2), G2(3), tmp1_vmm(5), tmp2_vmm(6), tmp3_vmm(7);

        // constant table map
        const Address one_addr = ptr[table_reg];

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_bias_reg = abi_param3;
        const auto addr_states_t_l_reg = abi_param4;
        const auto addr_attn_reg = r15;
#ifdef _WIN32
        const auto addr_states_t_l_copy_reg = r11;
        const auto addr_states_tm1_l_reg = r12;
        const auto addr_scratch_cell_reg = rsi;
        const auto addr_ws_h_reg = rdi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_scratch_cell_reg, ptr[base_args + 16]);
        mov(addr_ws_h_reg, ptr[base_args + 24]);
        if (is_augru) mov(addr_attn_reg, ptr[base_args + 48]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        const auto addr_states_tm1_l_reg = abi_param6;
        const auto addr_scratch_cell_reg = r11;
        const auto addr_ws_h_reg = r12;
        const auto base_args = get_stack_params_address();
        mov(addr_scratch_cell_reg, ptr[base_args]);
        mov(addr_ws_h_reg, ptr[base_args + 8]);
        if (is_augru) mov(addr_attn_reg, ptr[base_args + 32]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size];
        };
        const auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size];
        };
        const auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size_];
        };
        const auto sc_addr = [&](int i) {
            return ptr[addr_scratch_cell_reg + i * rnn_.dhc * scratch_dt_size];
        };

        auto compute_loop = [&](size_t current_vlen_elems) {
            const auto current_vlen = current_vlen_elems * scratch_dt_size;
            Label loop_start_label, loop_inc_regs_or_finish;
            L(loop_start_label);
            {
                load(G0, sg_addr(0), scratch_data_t, current_vlen);
                to_float(tmp1_vmm, B_addr(0), rnn_.bias_dt, current_vlen);
                compute_vaddps(G0, G0, tmp1_vmm, current_vlen);
                if (!rnn_.is_brgemm) {
                    load(tmp1_vmm, sc_addr(0), scratch_data_t, current_vlen);
                    compute_vaddps(G0, G0, tmp1_vmm, current_vlen);
                }
                sigmoid_injector_->load_table_addr();
                sigmoid_injector_->compute_vector(G0.getIdx());
                // if training we write back the gates
                if (is_training)
                    to_src(wg_addr(0), G0, src_data_t, current_vlen);

                // Compute gate 1
                load(G1, sg_addr(1), scratch_data_t, current_vlen);
                to_float(tmp1_vmm, B_addr(1), rnn_.bias_dt, current_vlen);
                compute_vaddps(G1, G1, tmp1_vmm, current_vlen);
                if (!rnn_.is_brgemm) {
                    load(tmp1_vmm, sc_addr(1), scratch_data_t, current_vlen);
                    compute_vaddps(G1, G1, tmp1_vmm, current_vlen);
                }
                sigmoid_injector_->load_table_addr();
                sigmoid_injector_->compute_vector(G1.getIdx());
                // if training we write back the gates
                if (is_training)
                    to_src(wg_addr(1), G1, src_data_t, current_vlen);

                // compute last gate
                const auto wh_b_addr = sc_addr(rnn_.is_brgemm ? 0 : 2);
                const auto ws_h_addr = ptr[addr_ws_h_reg];
                load(tmp1_vmm, wh_b_addr, scratch_data_t, current_vlen);
                to_float(tmp2_vmm, B_addr(3), rnn_.bias_dt, current_vlen);
                compute_vaddps(tmp1_vmm, tmp1_vmm, tmp2_vmm, current_vlen);
                if (is_training)
                    to_src(ws_h_addr, tmp1_vmm, src_data_t, current_vlen);
                load(G2, sg_addr(2), scratch_data_t, current_vlen);
                to_float(tmp2_vmm, B_addr(2), rnn_.bias_dt, current_vlen);
                compute_vaddps(G2, G2, tmp2_vmm, current_vlen);
                compute_vfmadd231ps(G2, G1, tmp1_vmm, current_vlen);
                tanh_injector_->load_table_addr();
                tanh_injector_->compute_vector(G2.getIdx());
                // if training we write back the gates
                if (is_training)
                    to_src(wg_addr(2), G2, src_data_t, current_vlen);

                if (is_augru) {
                    load(tmp1_vmm, one_addr, scratch_data_t, current_vlen);
                    // for augru there is additional step G01 = (1 - a) * G0
                    // states_t_l = states_tm1_l * G01 + (1 - G01) * G2
                    const Xmm tmp2s_vmm(tmp2_vmm.getIdx());
                    to_float(tmp2s_vmm, ptr[addr_attn_reg], src_data_t,
                            scratch_dt_size);
                    uni_vbroadcastss(tmp2_vmm, tmp2s_vmm);
                    // G01 = (1 - a) * G0
                    compute_vsubps(tmp2_vmm, tmp1_vmm, tmp2_vmm, tmp3_vmm,
                            current_vlen);
                    compute_vmulps(G0, G0, tmp2_vmm, current_vlen);
                    // tmp1 = 1 - G01
                    compute_vsubps(tmp1_vmm, tmp1_vmm, G0, current_vlen);
                    // tmp1 = G2 * tmp1
                    compute_vmulps(
                            tmp1_vmm, G2, tmp1_vmm, tmp3_vmm, current_vlen);
                    // states_t_l = G01 * states_tm1_l + tmp2
                    to_float(tmp2_vmm, ptr[addr_states_tm1_l_reg], src_data_t,
                            current_vlen);
                    compute_vfmadd213ps(G0, tmp2_vmm, tmp1_vmm, current_vlen);
                } else {
                    // states_t_l = states_tm1_l * G0 + (1 - G0) * G2
                    load(tmp1_vmm, one_addr, scratch_data_t, current_vlen);
                    compute_vsubps(tmp1_vmm, tmp1_vmm, G0, current_vlen);
                    to_float(tmp2_vmm, ptr[addr_states_tm1_l_reg], src_data_t,
                            current_vlen);
                    compute_vmulps(G0, G0, tmp2_vmm, current_vlen);
                    compute_vfmadd231ps(G0, tmp1_vmm, G2, current_vlen);
                }

                // write back the result
                to_src(ptr[addr_states_t_l_reg], G0, src_data_t, current_vlen);
                // if states_t_l_copy is a non null ptr, we write the output to
                // both tensors
                cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
                jle(loop_inc_regs_or_finish);
                // As to_src is called with write_only=true it's important for
                // xf16 src_dt to execute just after to_src method with
                // write_only=false for the same Vmm
                to_src(ptr[addr_states_t_l_copy_reg], G0, src_data_t,
                        current_vlen, true);
                // increment address pointers
                L(loop_inc_regs_or_finish);
                if (current_vlen_elems != loop_tail) {
                    const auto current_gate_size
                            = current_vlen == vlen ? vlen_dst : gate_dt_size;
                    const auto current_states_size
                            = current_vlen == vlen ? vlen_dst : hstate_dt_size;
                    add(addr_scratch_gates_reg, current_vlen);
                    add(addr_ws_h_reg, current_gate_size);
                    add(addr_bias_reg,
                            current_vlen == vlen ? vlen_bias_ : bias_dt_size_);
                    add(addr_states_t_l_reg, current_states_size);
                    add(addr_states_t_l_copy_reg, current_states_size);
                    add(addr_states_tm1_l_reg, current_states_size);
                    add(addr_scratch_cell_reg, current_vlen);
                    if (is_training) add(addr_ws_gates_reg, current_gate_size);

                    // increment loop counter
                    sub(loop_cnt, current_vlen_elems);
                    cmp(loop_cnt, current_vlen_elems);
                    jge(loop_start_label);
                }
            }
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        init_regs(vlen, loop_tail);
        if (rnn_.is_brgemm) {
#ifdef _WIN32
            mov(loop_cnt, ptr[base_args + 40]);
#else
            // Here we cannot use rbp to have initial stack pointer so we
            // use rsp and offset it with the size of pushed registers in
            // preamble
            const auto base_args = get_stack_params_address();
            mov(loop_cnt, ptr[base_args + 24]);
#endif
        } else {
            mov(loop_cnt, loop_len);
        }

        if (loop_tail > 0) {
            cmp(loop_cnt, vlen_elems);
            jl(tail_processing_or_exit_label, T_NEAR);
        }

        compute_loop(vlen_elems);

        L(tail_processing_or_exit_label);
        if (loop_tail > 0) {
            Label exit_label;
            cmp(loop_cnt, 0);
            jle(exit_label, T_NEAR);
            compute_loop(is_avx512 ? loop_tail : 1);
            L(exit_label);
        }

        postamble();

        sigmoid_injector_->prepare_table(true);
        tanh_injector_->prepare_table(true);
        init_table(vlen);

        L(table_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(1.0f));
        }
    }
}; // namespace cpu

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
