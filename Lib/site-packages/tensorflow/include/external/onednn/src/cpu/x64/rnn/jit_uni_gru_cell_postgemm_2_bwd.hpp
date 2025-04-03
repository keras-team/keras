/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef CPU_X64_RNN_JIT_UNI_GRU_CELL_POSTGEMM_2_BWD_HPP
#define CPU_X64_RNN_JIT_UNI_GRU_CELL_POSTGEMM_2_BWD_HPP

#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_gru_cell_postgemm_part2_bwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_cell_postgemm_part2_bwd)

    jit_uni_gru_cell_postgemm_part2_bwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    ~jit_uni_gru_cell_postgemm_part2_bwd() {}

    status_t init(data_type_t sdt) override {
        CHECK(jit_uni_rnn_postgemm::init(src_data_t));
        return create_kernel();
    }

protected:
    // register size in bytes
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;
    static constexpr size_t hstate_dt_size = sizeof(float);
    const size_t vlen_scratch
            = vlen / (sizeof(float) / types::data_type_size(scratch_data_t));
    const size_t gate_dt_size = types::data_type_size(scratch_data_t);
    const size_t scratch_dt_size = types::data_type_size(scratch_data_t);

    void generate() override {
        using namespace Xbyak;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;

        // Register map
        const Reg64 loop_cnt(rbx); // loop counter

        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        enum {
            dG1_idx = 1,
            dhG1_idx = 2,
            hG1_idx = 3,
            G1_idx = 4,
            dH_idx = 5,
            tmp1_idx = 6,
            h_idx = 7
        };

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        // auto addr_diff_states_t_lp1_reg = abi_param3; // not needed
        // auto addr_diff_states_tp1_l_reg = abi_param4; // not needed
#ifdef _WIN32
        const auto addr_diff_states_t_l_reg = r10;
        const auto addr_states_tm1_l_reg = r11;
        const auto addr_scratch_cell_reg = r12;
        // auto addr_ws_grid_reg = rsi; // not needed
        const auto addr_dhG1_reg = rsi;
        const auto base_args = get_stack_params_address();
        mov(addr_diff_states_t_l_reg, ptr[base_args]);
        mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_scratch_cell_reg, ptr[base_args + 16]);
        // mov(addr_ws_grid_reg, ptr[base_args + 24]);
        mov(addr_dhG1_reg, ptr[base_args + 32]);
#else
        const auto addr_diff_states_t_l_reg = abi_param5;
        const auto addr_states_tm1_l_reg = abi_param6;
        const auto addr_scratch_cell_reg = r10;
        // auto addr_ws_grid_reg = r11; // not needed
        const auto addr_dhG1_reg = r11;
        const auto base_args = get_stack_params_address();
        mov(addr_scratch_cell_reg, ptr[base_args]);
        // mov(addr_ws_grid_reg, ptr[base_args + 8]);
        mov(addr_dhG1_reg, ptr[base_args + 16]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size];
        };
        const auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size];
        };

        // initialize registers with addresses and constants
        init_regs(vlen);

        mov(loop_cnt, rnn_.dhc * scratch_dt_size);
        cmp(loop_cnt, vlen_scratch);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            const Vmm dG1(dG1_idx), dhG1(dhG1_idx), hG1(hG1_idx), G1(G1_idx),
                    dH(dH_idx), tmp1(tmp1_idx), h(h_idx);

            to_float(G1, wg_addr(1), src_data_t, vlen);
            to_float(h, ptr[addr_states_tm1_l_reg], src_data_t, vlen);

            // compute dG1
            uni_vmovups(dG1, G1);
            uni_vmovups(tmp1, G1);
            uni_vfnmadd231ps(dG1, tmp1, tmp1); // (G1 - G1^2)
            uni_vmulps(dG1, dG1, h);
            uni_vmovups(dhG1, ptr[addr_dhG1_reg]);
            uni_vmulps(dG1, dG1, dhG1); // dhG1 * h * (G0 - G0^2) * dHt

            // compute hG1
            uni_vmovups(hG1, G1);
            uni_vmulps(hG1, hG1, h);

            // compute diff_states_t_l = diff_states_t_l + dhG1 * G1
            uni_vmovups(dH, ptr[addr_diff_states_t_l_reg]);
            uni_vfmadd231ps(dH, dhG1, G1);

            // downconvert and write data
            to_src(sg_addr(1), dG1, scratch_data_t, vlen);
            to_src(ptr[addr_scratch_cell_reg], hG1, scratch_data_t, vlen);
            uni_vmovups(ptr[addr_diff_states_t_l_reg], dH);

            // increment address pointers
            add(addr_ws_gates_reg, vlen_scratch);
            add(addr_scratch_gates_reg, vlen_scratch);
            add(addr_dhG1_reg, vlen);
            add(addr_diff_states_t_l_reg, vlen);
            add(addr_states_tm1_l_reg, vlen_scratch);
            add(addr_scratch_cell_reg, vlen_scratch);
            inc_regs(vlen);

            // increment loop counter
            sub(loop_cnt, vlen_scratch);
            cmp(loop_cnt, vlen_scratch);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use movuss for accessing inputs
        // TODO: smarter handling of tails with Zmm -> Ymm -> Xmm -> scalar
        L(rem_loop_start_label);
        {
            const Xmm dG1(dG1_idx), dhG1(dhG1_idx), hG1(hG1_idx), G1(G1_idx),
                    dH(dH_idx), tmp1(tmp1_idx), h(h_idx);

            to_float(G1, wg_addr(1), src_data_t, hstate_dt_size);
            to_float(h, ptr[addr_states_tm1_l_reg], src_data_t, hstate_dt_size);

            // compute dG1
            uni_vmovss(dG1, G1);
            uni_vmovss(tmp1, G1);
            uni_vfnmadd231ps(dG1, tmp1, tmp1); // (G1 - G1^2)
            uni_vmulss(dG1, dG1, h);
            uni_vmovss(dhG1, ptr[addr_dhG1_reg]);
            uni_vmulss(dG1, dG1, dhG1); // dhG1 * h * (G0 - G0^2) * dHt

            // compute hG1
            uni_vmovss(hG1, G1);
            uni_vmulss(hG1, hG1, h);

            // compute diff_states_t_l = diff_states_t_l + dhG1 * G1
            uni_vmovss(dH, ptr[addr_diff_states_t_l_reg]);
            uni_vfmadd231ps(dH, dhG1, G1);

            // downconvert and write data
            to_src(sg_addr(1), dG1, scratch_data_t, hstate_dt_size);
            to_src(ptr[addr_scratch_cell_reg], hG1, scratch_data_t,
                    hstate_dt_size);
            uni_vmovss(ptr[addr_diff_states_t_l_reg], dH);

            // increment address pointers
            add(addr_ws_gates_reg, scratch_dt_size);
            add(addr_scratch_gates_reg, scratch_dt_size);
            add(addr_dhG1_reg, hstate_dt_size);
            add(addr_diff_states_t_l_reg, hstate_dt_size);
            add(addr_states_tm1_l_reg, scratch_dt_size);
            add(addr_scratch_cell_reg, scratch_dt_size);
            inc_regs(hstate_dt_size);

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
