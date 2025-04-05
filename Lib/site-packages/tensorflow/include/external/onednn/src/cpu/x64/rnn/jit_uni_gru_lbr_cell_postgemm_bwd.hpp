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

#ifndef CPU_X64_RNN_JIT_UNI_GRU_LBR_CELL_POSTGEMM_BWD_HPP
#define CPU_X64_RNN_JIT_UNI_GRU_LBR_CELL_POSTGEMM_BWD_HPP

#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_gru_lbr_cell_postgemm_bwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_lbr_cell_postgemm_bwd)

    jit_uni_gru_lbr_cell_postgemm_bwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    ~jit_uni_gru_lbr_cell_postgemm_bwd() {}

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

        const bool is_augru = pd_->cell_kind() == alg_kind::lbr_augru;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;
        Label table_label;

        // Register map
        const Reg64 table_reg(rbx); // used to load ones before the loop
        const Reg64 loop_cnt(
                rbx); // loop counter, can be aliased with table_reg

        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        const int dG0_idx = 1, dG1_idx = 2, dG2_idx = 3, G0_idx = 4, G1_idx = 5,
                  G2_idx = 6, h_idx = 7, dHt_idx = 8, one_idx = 9,
                  tmp1_idx = 10, tmp2_idx = 11, dattn_acc_idx = 12,
                  attn_idx = 13;
        const Vmm one_vmm(one_idx);
        const Xmm one_xmm(one_idx);

        // constant table map
        const Address one_addr = ptr[table_reg];

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_diff_states_t_lp1_reg = abi_param3;
        const auto addr_diff_states_tp1_l_reg = abi_param4;
        const auto addr_attn_reg = r14;
#ifdef _WIN32
        const auto addr_diff_states_t_l_reg = r10;
        const auto addr_states_tm1_l_reg = r11;
        const auto addr_scratch_cell_reg = r12;
        const auto addr_ws_grid_reg = rsi;
        const auto base_args = get_stack_params_address();
        mov(addr_diff_states_t_l_reg, ptr[base_args]);
        mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_scratch_cell_reg, ptr[base_args + 16]);
        mov(addr_ws_grid_reg, ptr[base_args + 24]);
        if (is_augru) mov(addr_attn_reg, ptr[base_args + 48]);
#else
        const auto addr_diff_states_t_l_reg = abi_param5;
        const auto addr_states_tm1_l_reg = abi_param6;
        const auto addr_scratch_cell_reg = r10;
        const auto addr_ws_grid_reg = r11;
        const auto base_args = get_stack_params_address();
        mov(addr_scratch_cell_reg, ptr[base_args]);
        mov(addr_ws_grid_reg, ptr[base_args + 8]);
        if (is_augru) mov(addr_attn_reg, ptr[base_args + 32]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size];
        };
        const auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size];
        };
        const auto sc_addr = [&](int i) {
            return ptr[addr_scratch_cell_reg + i * rnn_.dhc * scratch_dt_size];
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        init_regs(vlen);
        uni_vmovups(one_vmm, one_addr);

        if (is_augru) {
            uni_vpxor(
                    Vmm(dattn_acc_idx), Vmm(dattn_acc_idx), Vmm(dattn_acc_idx));
            const Xmm attn1s(attn_idx);
            to_float(attn1s, ptr[addr_attn_reg], src_data_t, hstate_dt_size);
        }

        mov(loop_cnt, rnn_.dhc * scratch_dt_size);
        cmp(loop_cnt, vlen_scratch);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        if (is_augru) {
            const Xmm attn1s(attn_idx);
            const Vmm attn(attn_idx);
            uni_vbroadcastss(attn, attn1s);
        }

        L(vector_loop_start_label);
        {
            const Vmm dG0(dG0_idx), dG1(dG1_idx), dG2(dG2_idx), G0(G0_idx),
                    G1(G1_idx), G2(G2_idx), dHt(dHt_idx), tmp1(tmp1_idx),
                    tmp2(tmp2_idx), h(h_idx), diff_attn_acc(dattn_acc_idx),
                    attn(attn_idx);

            to_float(G0, wg_addr(0), src_data_t, vlen);
            to_float(G1, wg_addr(1), src_data_t, vlen);
            to_float(G2, wg_addr(2), src_data_t, vlen);

            // compute dHt
            uni_vmovups(dHt, ptr[addr_diff_states_tp1_l_reg]);
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovups(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddps(dHt, dHt, tmp1);

            // compute dG0
            to_float(h, ptr[addr_states_tm1_l_reg], src_data_t, vlen);
            uni_vmovups(dG0, G0);
            uni_vmovups(tmp1, G0);
            uni_vfnmadd231ps(dG0, tmp1, tmp1); // (G0 - G0^2)
            uni_vsubps(h, h, G2); // (h - G2)
            uni_vmulps(dG0, dG0, h);
            uni_vmulps(dG0, dG0, dHt); // (h - G2) * (G0 - G0^2) * dHt

            if (is_augru) {
                // Compute diff_attention
                // 1. compute dAttention = -dG0 * G
                uni_vfnmadd231ps(diff_attn_acc, dG0, G0, tmp2);
                // 2. Compute dG0 *= 1 - Attention
                uni_vsubps(tmp1, one_vmm, attn, tmp2);
                uni_vmulps(dG0, dG0, tmp1);
            }
            // compute dG2
            uni_vmovups(tmp1, one_vmm);
            uni_vsubps(tmp1, tmp1, G0); // (1 - G0)
            uni_vmovups(dG2, one_vmm);
            uni_vmovups(tmp2, G2);
            uni_vfnmadd231ps(dG2, tmp2, tmp2); // (1 - G2^2)
            uni_vmulps(dG2, dG2, tmp1);
            uni_vmulps(dG2, dG2, dHt); //(1 - G0) * (1 - G2^2) * dHt

            // compute dG1
            to_float(tmp1, ptr[addr_ws_grid_reg], src_data_t, vlen);
            uni_vmovups(dG1, G1);
            uni_vmovups(tmp2, G1);
            uni_vfnmadd231ps(dG1, tmp2, tmp2); // (G1 - G1^2)
            uni_vmulps(dG1, dG1, dG2);
            uni_vmulps(dG1, dG1, tmp1); // (G1 - G1^2) * dG2 * ws_grid

            // compute diff_state_t_l
            uni_vmulps(dHt, dHt, G0);
            uni_vmovups(ptr[addr_diff_states_t_l_reg], dHt);

            // compute scratch_cell
            uni_vmovups(tmp1, dG2);
            uni_vmulps(tmp1, tmp1, G1);

            // downconvert and write data
            to_src(sc_addr(0), dG0, scratch_data_t, vlen);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(sg_addr(0), dG0, scratch_data_t, vlen, true);

            to_src(sc_addr(1), dG1, scratch_data_t, vlen);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(sg_addr(1), dG1, scratch_data_t, vlen, true);

            to_src(sc_addr(2), tmp1, scratch_data_t, vlen);
            to_src(sg_addr(2), dG2, scratch_data_t, vlen);

            // increment address pointers
            add(addr_ws_gates_reg, vlen_scratch);
            add(addr_scratch_gates_reg, vlen_scratch);
            add(addr_diff_states_t_lp1_reg, vlen);
            add(addr_diff_states_tp1_l_reg, vlen);
            add(addr_diff_states_t_l_reg, vlen);
            add(addr_states_tm1_l_reg, vlen_scratch);
            add(addr_scratch_cell_reg, vlen_scratch);
            add(addr_ws_grid_reg, vlen_scratch);
            inc_regs(vlen);

            // increment loop counter
            sub(loop_cnt, vlen_scratch);
            cmp(loop_cnt, vlen_scratch);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        // Reduce diff attention into XMM size. Otherwise accumulation
        // using XMM will zero high part of YMM/ZMM.
        if (vlen >= cpu_isa_traits<avx512_core>::vlen) {
            Zmm diff_attn_acc(dattn_acc_idx);
            Ymm diff_attn_acc_high(tmp1_idx);
            Ymm diff_attn_acc_low(dattn_acc_idx);
            vextractf32x8(diff_attn_acc_high, diff_attn_acc, 1);
            vaddps(diff_attn_acc_low, diff_attn_acc_low, diff_attn_acc_high);
        }
        if (vlen >= cpu_isa_traits<avx2>::vlen) {
            Ymm diff_attn_acc(dattn_acc_idx);
            Xmm diff_attn_acc_high(tmp1_idx);
            Xmm diff_attn_acc_low(dattn_acc_idx);
            vextractf128(diff_attn_acc_high, diff_attn_acc, 1);
            vaddps(diff_attn_acc_low, diff_attn_acc_low, diff_attn_acc_high);
        }

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        // Same code as above, we just use movuss for accessing inputs
        // TODO: smarter handling of tails with Zmm -> Ymm -> Xmm -> scalar
        L(rem_loop_start_label);
        {
            const Xmm dG0(dG0_idx), dG1(dG1_idx), dG2(dG2_idx), G0(G0_idx),
                    G1(G1_idx), G2(G2_idx), dHt(dHt_idx), tmp1(tmp1_idx),
                    tmp2(tmp2_idx), h(h_idx), diff_attn_acc(dattn_acc_idx),
                    attn(attn_idx);

            to_float(G0, wg_addr(0), src_data_t, hstate_dt_size);
            to_float(G1, wg_addr(1), src_data_t, hstate_dt_size);
            to_float(G2, wg_addr(2), src_data_t, hstate_dt_size);

            // compute dHt
            uni_vmovss(dHt, ptr[addr_diff_states_tp1_l_reg]);
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovss(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddss(dHt, dHt, tmp1);

            // compute dG0
            to_float(h, ptr[addr_states_tm1_l_reg], src_data_t, hstate_dt_size);
            uni_vmovss(dG0, G0);
            uni_vmovss(tmp1, dG0);
            uni_vfnmadd231ps(dG0, tmp1, tmp1); // (G0 - G0^2)
            uni_vsubss(h, h, G2); // (h - G2)
            uni_vmulss(dG0, dG0, h);
            uni_vmulss(dG0, dG0, dHt); // (h - G2) * (G0 - G0^2) * dHt

            if (is_augru) {
                // compute diff_attention
                // 1. compute tmp2 = dG0 * G
                uni_vmovss(tmp2, dG0);
                uni_vmulss(tmp2, tmp2, G0);
                // 2. Store dAttention
                uni_vsubss(diff_attn_acc, diff_attn_acc, tmp2);
                // 3. Compute dG0 *= 1 - attention
                uni_vmovss(tmp1, one_xmm);
                uni_vsubss(tmp1, tmp1, attn);
                uni_vmulss(dG0, dG0, tmp1);
            }

            // compute dG2
            uni_vmovss(tmp1, one_xmm);
            uni_vsubss(tmp1, tmp1, G0); // (1 - G0)

            uni_vmovss(dG2, one_xmm);
            uni_vmovss(tmp2, G2);
            uni_vfnmadd231ps(dG2, tmp2, tmp2); // (1 - G2^2)
            uni_vmulss(dG2, dG2, tmp1);
            uni_vmulss(dG2, dG2, dHt); //(1 - G0) * (1 - G2^2) * dHt

            // compute dG1
            to_float(tmp1, ptr[addr_ws_grid_reg], src_data_t, hstate_dt_size);
            uni_vmovss(dG1, G1);
            uni_vmovss(tmp2, G1);
            uni_vfnmadd231ps(dG1, tmp2, tmp2); // (G1 - G1^2)
            uni_vmulss(dG1, dG1, dG2);
            uni_vmulss(dG1, dG1, tmp1); // (G1 - G1^2) * dG2 * ws_grid

            // compute diff_state_t_l
            uni_vmulss(dHt, dHt, G0);
            uni_vmovss(ptr[addr_diff_states_t_l_reg], dHt);

            // compute scratch_cell
            uni_vmovss(tmp1, dG2);
            uni_vmulss(tmp1, tmp1, G1);

            // downconvert and write data
            to_src(sc_addr(0), dG0, scratch_data_t, hstate_dt_size);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(sg_addr(0), dG0, scratch_data_t, hstate_dt_size, true);

            to_src(sc_addr(1), dG1, scratch_data_t, hstate_dt_size);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(sg_addr(1), dG1, scratch_data_t, hstate_dt_size, true);

            to_src(sc_addr(2), tmp1, scratch_data_t, hstate_dt_size);
            to_src(sg_addr(2), dG2, scratch_data_t, hstate_dt_size);

            // increment address pointers
            add(addr_ws_gates_reg, scratch_dt_size);
            add(addr_scratch_gates_reg, scratch_dt_size);
            add(addr_diff_states_t_lp1_reg, hstate_dt_size);
            add(addr_diff_states_tp1_l_reg, hstate_dt_size);
            add(addr_diff_states_t_l_reg, hstate_dt_size);
            add(addr_states_tm1_l_reg, scratch_dt_size);
            add(addr_scratch_cell_reg, scratch_dt_size);
            add(addr_ws_grid_reg, scratch_dt_size);
            inc_regs(hstate_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        if (is_augru) {
            // Complete diff attention reduction
            Xmm diff_attn_acc(dattn_acc_idx);
            uni_vhaddps(diff_attn_acc, diff_attn_acc, diff_attn_acc);
            uni_vhaddps(diff_attn_acc, diff_attn_acc, diff_attn_acc);
            const auto base_args = get_stack_params_address();
#ifdef _WIN32
            mov(addr_attn_reg, ptr[base_args + 56]);
#else
            mov(addr_attn_reg, ptr[base_args + 40]);
#endif
            uni_vmovss(ptr[addr_attn_reg], diff_attn_acc);
        }

        postamble();

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
