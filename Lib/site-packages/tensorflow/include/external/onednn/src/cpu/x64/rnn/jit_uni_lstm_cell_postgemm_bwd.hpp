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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_BWD_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_BWD_HPP

#include <memory>
#include "common/utils.hpp"
#include "cpu/x64/rnn/jit_uni_lstm_cell_postgemm.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_lstm_cell_postgemm_bwd
    : public jit_uni_rnn_postgemm,
      public jit_uni_lstm_cell_postgemm_t<isa> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_bwd)

    jit_uni_lstm_cell_postgemm_bwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name())
        , jit_uni_lstm_cell_postgemm_t<isa>(this, 11 /*tmp_id_begin*/,
                  // usage of jit_uni_rnn_postgemm::bf16_emu_ to identify bf16
                  // emulation case is illegal here, it's created in
                  // jit_uni_rnn_postgemm::init(), not in constructor, so
                  // jit_uni_rnn_postgemm::bf16_emu_ = nullptr always on this
                  // stage
                  src_data_t == data_type::bf16 && !mayiuse(avx512_core_bf16)) {
    }
    ~jit_uni_lstm_cell_postgemm_bwd() = default;

    status_t init(data_type_t sdt) override {
        CHECK(jit_uni_rnn_postgemm::init(src_data_t));
        // we use rax for both constant tables as they use the same table
        tanh_injector_ = utils::make_unique<injector_t>(
                this, alg_kind::eltwise_tanh, 0.0f, 0.0f, 1.0f, true, rax);
        return create_kernel();
    }

protected:
    using injector_t = typename jit_uni_lstm_cell_postgemm_t<isa>::injector_t;
    using Vmm = typename jit_uni_lstm_cell_postgemm_t<isa>::Vmm;

    std::unique_ptr<injector_t> tanh_injector_;

    // register size in bytes
    static constexpr size_t vlen_ = cpu_isa_traits<isa>::vlen;
    const size_t vlen_c_states_ = vlen_ / (sizeof(float) / cstate_dt_size_);

    static constexpr size_t diff_cstate_dt_size_ = sizeof(float);
    static constexpr size_t hstate_dt_size_ = sizeof(float);
    static constexpr size_t weights_peephole_dt_size_ = sizeof(float);

    const size_t vlen_scratch_
            = vlen_ / (sizeof(float) / types::data_type_size(scratch_data_t));
    const size_t gate_dt_size_ = types::data_type_size(scratch_data_t);
    const size_t scratch_dt_size_ = types::data_type_size(scratch_data_t);

    void generate() override {
        using namespace Xbyak;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_end_label;
        Label table_label;

        // Register map
        const Reg64 table_reg(rbx); // used to load ones before the loop
        const Reg64 loop_cnt(
                rbx); // loop counter, can be aliased with table_reg
        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        const int dG0_idx = 1, dG1_idx = 2, dG2_idx = 3, dG3_idx = 4,
                  tanhCt_idx = 5, dHt_idx = 6, dCt_idx = 7, G0_idx = 8,
                  G1_idx = 9, one_idx = 10;
        const Vmm one_vmm(one_idx);
        const Xmm one_xmm(one_idx);

        // Adress maping
        const Address one_addr = ptr[table_reg];
        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_diff_states_t_lp1_reg = abi_param3;
        const auto addr_diff_states_tp1_l_reg = abi_param4;
        const auto addr_weights_peephole_reg = r12;
#ifdef _WIN32
        const auto addr_diff_c_states_t_l_reg = r10;
        const auto addr_diff_c_states_tp1_l_reg = r11;
        const auto addr_c_states_tm1_l_reg = rdi;
        const auto addr_c_states_t_l_reg = rsi;
        const auto base_args = get_stack_params_address();
        mov(addr_diff_c_states_t_l_reg, ptr[base_args]);
        mov(addr_diff_c_states_tp1_l_reg, ptr[base_args + 8]);
        mov(addr_c_states_tm1_l_reg, ptr[base_args + 16]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 24]);
        mov(addr_weights_peephole_reg, ptr[base_args + 32]);
#else
        const auto addr_diff_c_states_t_l_reg = abi_param5;
        const auto addr_diff_c_states_tp1_l_reg = abi_param6;
        const auto addr_c_states_tm1_l_reg = r10;
        const auto addr_c_states_t_l_reg = r11;
        const auto base_args = get_stack_params_address();
        mov(addr_c_states_tm1_l_reg, ptr[base_args]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 8]);
        mov(addr_weights_peephole_reg, ptr[base_args + 16]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg
                    + i * rnn_.dhc * scratch_dt_size_];
        };
        const auto weights_peephole_addr = [&](int i) {
            return ptr[addr_weights_peephole_reg
                    + i * rnn_.dhc * weights_peephole_dt_size_];
        };
        const auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size_];
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        init_regs(vlen_);
        uni_vmovups(one_vmm, one_addr);
        tanh_injector_->load_table_addr();

        mov(loop_cnt, rnn_.dhc * scratch_dt_size_);
        cmp(loop_cnt, vlen_scratch_);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            const Vmm dG0(dG0_idx), dG1(dG1_idx), dG2(dG2_idx), dG3(dG3_idx),
                    tanhCt(tanhCt_idx), dHt(dHt_idx), dCt(dCt_idx), G0(G0_idx),
                    G1(G1_idx);

            // TODO: if w_gates are bfloat, we have to convert them to float
            // datatypes summary:
            // - c states are all float
            // - h states are all src_data_t
            // - diff_* are all float
            // - scratch is src_data_t
            // - ws_gates is src_data_t

            // compute tanhCt
            to_float(tanhCt, ptr[addr_c_states_t_l_reg], rnn_.src_iter_c_dt,
                    vlen_);
            tanh_injector_->compute_vector(tanhCt.getIdx());

            // compute dHt
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovups(dHt, ptr[addr_diff_states_t_lp1_reg]);
            if (!rnn_.is_lstm_projection) {
                this->vaddps_rhs_op_mem(
                        dHt, dHt, ptr[addr_diff_states_tp1_l_reg]);
            }

            // compute dCt
            const auto tmp_dCt1 = this->get_next_tmp_vmm();
            const auto tmp_dCt2 = this->get_next_tmp_vmm();

            uni_vmovups(tmp_dCt1, one_vmm);
            uni_vmovups(tmp_dCt2, tanhCt);
            uni_vfnmadd231ps(tmp_dCt1, tmp_dCt2, tmp_dCt2);
            uni_vmulps(tmp_dCt1, tmp_dCt1, dHt);
            to_float(dG3, wg_addr(3), src_data_t, vlen_);
            uni_vmulps(tmp_dCt1, tmp_dCt1, dG3);
            uni_vmovups(dCt, ptr[addr_diff_c_states_tp1_l_reg]);
            uni_vaddps(dCt, dCt, tmp_dCt1);

            // compute dG3
            const auto tmp_dG3 = this->get_next_tmp_vmm();
            uni_vmovups(tmp_dG3, dG3);
            uni_vfnmadd231ps(dG3, tmp_dG3, tmp_dG3);
            uni_vmulps(dG3, dG3, dHt);
            uni_vmulps(dG3, dG3, tanhCt);

            // update dCt if lstm_peephole
            if (rnn_.is_lstm_peephole)
                this->vfmadd231ps_rhs_op_mem(
                        dCt, dG3, weights_peephole_addr(2));

            // compute dG0
            // we will reuse G0 and G2 later for dG2
            to_float(G0, wg_addr(0), src_data_t, vlen_);
            to_float(dG2, wg_addr(2), src_data_t, vlen_);
            uni_vmovups(dG0, G0);
            const auto tmp_g0 = this->vmm_backup(G0);
            uni_vfnmadd231ps(dG0, tmp_g0, tmp_g0);
            uni_vmulps(dG0, dG0, dCt);
            uni_vmulps(dG0, dG0, dG2);

            // compute dG1
            to_float(G1, wg_addr(1), src_data_t, vlen_);
            uni_vmovups(dG1, G1);
            const auto tmp_g1 = this->vmm_backup(G1);
            uni_vfnmadd231ps(dG1, tmp_g1, tmp_g1);
            uni_vmulps(dG1, dG1, dCt);

            const auto tmp_c_states_tm1 = this->get_next_tmp_vmm();
            to_float(tmp_c_states_tm1, ptr[addr_c_states_tm1_l_reg],
                    rnn_.src_iter_c_dt, vlen_);
            this->uni_vmulps(dG1, dG1, tmp_c_states_tm1);

            // compute dG2
            const auto tmp_dg2 = this->get_next_tmp_vmm();
            uni_vmovups(tmp_dg2, one_vmm);
            const auto tmp_g2 = this->vmm_backup(dG2);

            uni_vfnmadd231ps(tmp_dg2, tmp_g2, tmp_g2);
            uni_vmulps(G0, G0, dCt);
            uni_vmulps(tmp_dg2, tmp_dg2, G0);
            uni_vmovups(dG2, tmp_dg2);

            // compute diff_state_t_l
            uni_vmulps(dCt, dCt, G1);
            if (rnn_.is_lstm_peephole) {
                this->vfmadd231ps_rhs_op_mem(
                        dCt, dG0, weights_peephole_addr(0));
                this->vfmadd231ps_rhs_op_mem(
                        dCt, dG1, weights_peephole_addr(1));
            }
            uni_vmovups(ptr[addr_diff_c_states_t_l_reg], dCt);

            to_src(sg_addr(0), dG0, scratch_data_t, vlen_);
            to_src(sg_addr(1), dG1, scratch_data_t, vlen_);
            to_src(sg_addr(2), dG2, scratch_data_t, vlen_);
            to_src(sg_addr(3), dG3, scratch_data_t, vlen_);

            // increment address pointers
            add(addr_ws_gates_reg, vlen_scratch_);
            add(addr_scratch_gates_reg, vlen_scratch_);
            add(addr_diff_states_t_lp1_reg, vlen_);
            add(addr_diff_states_tp1_l_reg, vlen_);
            add(addr_diff_c_states_t_l_reg, vlen_);
            add(addr_diff_c_states_tp1_l_reg, vlen_);
            add(addr_c_states_tm1_l_reg, vlen_c_states_);
            add(addr_c_states_t_l_reg, vlen_c_states_);
            if (rnn_.is_lstm_peephole) add(addr_weights_peephole_reg, vlen_);
            inc_regs(vlen_);

            // increment loop counter
            sub(loop_cnt, vlen_scratch_);
            cmp(loop_cnt, vlen_scratch_);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use vmovss for accessing inputs
        this->reset_vmm_cnt();
        L(rem_loop_start_label);
        {
            const Xmm dG0(dG0_idx), dG1(dG1_idx), dG2(dG2_idx), dG3(dG3_idx),
                    tanhCt(tanhCt_idx), dHt(dHt_idx), dCt(dCt_idx), G0(G0_idx),
                    G1(G1_idx);

            // compute tanhCt
            to_float(tanhCt, ptr[addr_c_states_t_l_reg], rnn_.src_iter_c_dt,
                    sizeof(float));
            tanh_injector_->compute_vector(tanhCt.getIdx());

            // compute dHt
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovss(dHt, ptr[addr_diff_states_t_lp1_reg]);
            if (!rnn_.is_lstm_projection)
                this->vaddss_rhs_op_mem(
                        dHt, dHt, ptr[addr_diff_states_tp1_l_reg]);

            // compute dCt
            const auto tmp_dCt1 = this->get_next_tmp_xmm();
            const auto tmp_dCt2 = this->get_next_tmp_xmm();

            uni_vmovss(tmp_dCt1, one_xmm);
            // This overrides tanhCt when using Xmm
            uni_vmovss(tmp_dCt2, tanhCt);
            uni_vfnmadd231ss(tmp_dCt1, tmp_dCt2, tmp_dCt2);
            uni_vmulss(tmp_dCt1, tmp_dCt1, dHt);
            to_float(dG3, wg_addr(3), src_data_t, hstate_dt_size_);
            uni_vmulss(tmp_dCt1, tmp_dCt1, dG3);
            uni_vmovss(dCt, ptr[addr_diff_c_states_tp1_l_reg]);
            uni_vaddss(dCt, dCt, tmp_dCt1);

            // compute dG3
            const auto tmp_dG3 = this->get_next_tmp_xmm();
            uni_vmovss(tmp_dG3, dG3);
            uni_vfnmadd231ss(dG3, tmp_dG3, tmp_dG3);
            uni_vmulss(dG3, dG3, dHt);
            uni_vmulss(dG3, dG3, tanhCt);

            // update dCt if lstm_peephole
            if (rnn_.is_lstm_peephole) {
                this->vfmadd231ss_rhs_op_mem(
                        dCt, dG3, weights_peephole_addr(2));
            }

            // compute dG0
            // we will reuse G0 and G2 later for dG2
            to_float(G0, wg_addr(0), src_data_t, hstate_dt_size_);
            to_float(dG2, wg_addr(2), src_data_t, hstate_dt_size_);

            uni_vmovss(dG0, G0);
            const auto tmp_g0 = this->xmm_backup(G0);
            uni_vfnmadd231ss(dG0, tmp_g0, tmp_g0);
            uni_vmulss(dG0, dG0, dCt);
            uni_vmulss(dG0, dG0, dG2);

            // compute dG1
            to_float(G1, wg_addr(1), src_data_t, hstate_dt_size_);
            const auto tmp_g1 = this->xmm_backup(G1);
            uni_vmovss(dG1, G1);
            uni_vfnmadd231ss(dG1, tmp_g1, tmp_g1);
            uni_vmulss(dG1, dG1, dCt);

            const auto tmp_c_states_tm1 = this->get_next_tmp_xmm();
            to_float(tmp_c_states_tm1, ptr[addr_c_states_tm1_l_reg],
                    rnn_.src_iter_c_dt, sizeof(float));
            this->uni_vmulss(dG1, dG1, tmp_c_states_tm1);

            // compute dG2
            const auto tmp_dG2 = this->get_next_tmp_xmm();
            uni_vmovss(tmp_dG2, one_xmm);
            const auto tmp_g2 = this->xmm_backup(dG2);

            uni_vfnmadd231ss(tmp_dG2, tmp_g2, tmp_g2);
            uni_vmulss(G0, G0, dCt);
            uni_vmulss(tmp_dG2, tmp_dG2, G0);
            uni_vmovss(dG2, tmp_dG2);

            // compute diff_state_t_l
            uni_vmulss(dCt, dCt, G1);
            if (rnn_.is_lstm_peephole) {
                this->vfmadd231ss_rhs_op_mem(
                        dCt, dG1, weights_peephole_addr(1));
                this->vfmadd231ss_rhs_op_mem(
                        dCt, dG0, weights_peephole_addr(0));
            }
            uni_vmovss(ptr[addr_diff_c_states_t_l_reg], dCt);

            to_src(sg_addr(0), dG0, scratch_data_t, hstate_dt_size_);
            to_src(sg_addr(1), dG1, scratch_data_t, hstate_dt_size_);
            to_src(sg_addr(2), dG2, scratch_data_t, hstate_dt_size_);
            to_src(sg_addr(3), dG3, scratch_data_t, hstate_dt_size_);

            // increment address pointers
            add(addr_ws_gates_reg, scratch_dt_size_);
            add(addr_scratch_gates_reg, scratch_dt_size_);
            add(addr_diff_states_t_lp1_reg, hstate_dt_size_);
            add(addr_diff_states_tp1_l_reg, hstate_dt_size_);
            add(addr_diff_c_states_t_l_reg, diff_cstate_dt_size_);
            add(addr_diff_c_states_tp1_l_reg, diff_cstate_dt_size_);
            add(addr_c_states_tm1_l_reg, cstate_dt_size_);
            add(addr_c_states_t_l_reg, cstate_dt_size_);
            if (rnn_.is_lstm_peephole)
                add(addr_weights_peephole_reg, weights_peephole_dt_size_);
            inc_regs(hstate_dt_size_);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size_);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();

        tanh_injector_->prepare_table();
        init_table(vlen_);
        L(table_label);
        {
            for (size_t i = 0; i < vlen_ / sizeof(float); ++i)
                dd(float2int(1.0f));
        }
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
