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

#ifndef CPU_X64_RNN_JIT_UNI_GRU_CELL_POSTGEMM_2_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_GRU_CELL_POSTGEMM_2_FWD_HPP

#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_gru_cell_postgemm_part2_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_cell_postgemm_part2_fwd)

    using injector_t = typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_core>,
            jit_uni_eltwise_injector_f32<isa>>::type;

    jit_uni_gru_cell_postgemm_part2_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    status_t init(data_type_t sdt) override {
        CHECK(jit_uni_rnn_postgemm::init(src_data_t));
        // no need to save state of registers
        // (unless emulating bf16 support or using pre-avx2 isa)
        const bool save_state = (isa == sse41 || isa == avx)
                || (src_data_t == data_type::bf16
                        && !mayiuse(avx512_core_bf16));
        // we use rax for both constant tables as they use the same table
        CHECK(safe_ptr_assign(tanh_injector_,
                new injector_t(this, alg_kind::eltwise_tanh, 0.0f, 0.0f, 1.0f,
                        save_state, rax)));
        return create_kernel();
    }

protected:
    std::unique_ptr<injector_t> tanh_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;
    static constexpr size_t qscale_dt_size = sizeof(float);
    const size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t vlen_bias = vlen / (sizeof(float) / bias_dt_size_);
    const size_t hstate_dt_size = types::data_type_size(src_data_t);
    const size_t gate_dt_size = types::data_type_size(src_data_t);
    const size_t scratch_dt_size = types::data_type_size(scratch_data_t);
    const size_t vlen_qscale = vlen / qscale_dt_size;
    const size_t vlen_elems = vlen / scratch_dt_size;

    const int loop_ur_max = 4;
    // We skip vmm0 as it can be used by the injector for masks on sse4.1
    Vmm G0(int i) {
        const int idx = 1 + i;
        assert(idx < loop_ur_max + 1);
        return Vmm(idx); // max of vmm4
    }
    Vmm G2(int i) {
        const int idx = loop_ur_max + 1 + i;
        assert(idx < 2 * loop_ur_max + 1);
        return Vmm(idx); // max of vmm8
    }
    const Vmm tmp1_vmm = Vmm(9);
    const Vmm tmp2_vmm = Vmm(10);
    const Vmm tmp3_vmm = Vmm(11);

    void generate() override {
        using namespace Xbyak;
        const auto is_training
                = pd_->desc()->prop_kind == prop_kind::forward_training;

        const bool is_augru = pd_->cell_kind() == alg_kind::vanilla_augru;

        const int mask = pd_->attr()->rnn_weights_qparams_.mask_;
        float *weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;

        // Labels declaration
        Label table_label;

        // Register map
        const Reg64 loop_cnt(r10); // loop counter
        const Reg64 table_reg(rbx); // table is used for data scale and shifts

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
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
        if (is_augru) mov(addr_attn_reg, ptr[base_args + 48]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        const auto addr_states_tm1_l_reg = abi_param6;
        const auto base_args = get_stack_params_address();
        if (is_augru) mov(addr_attn_reg, ptr[base_args + 32]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i, int j) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size
                    + j * vlen];
        };
        const auto wg_addr = [&](int i, int j) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size
                    + j * vlen_dst];
        };
        const auto B_addr = [&](int i, int j) {
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size_
                    + j * vlen_bias];
        };

        const size_t loop_len = rnn_.dhc;
        const size_t loop_tail = loop_len % vlen_elems;

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        tanh_injector_->load_table_addr();
        init_regs(weights_scales, vlen, loop_tail);

        const size_t nb_loop_len = loop_len / vlen_elems;
        size_t loop_ur_val = 1;
        const bool is_brgemm = rnn_.is_brgemm && !rnn_.unfused_post_gemm;
        if (is_brgemm) {
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
            for (loop_ur_val = loop_ur_max; loop_ur_val > 1; --loop_ur_val)
                if (nb_loop_len % loop_ur_val == 0) break;

            mov(loop_cnt, loop_len);
        }
        const size_t loop_ur = loop_ur_val;

        auto compute_loop = [&](size_t current_vlen_elem,
                                    size_t current_loop_unroll) {
            const auto current_vlen = current_vlen_elem * scratch_dt_size;
            Label loop_start_label;
            L(loop_start_label);
            {
                for (size_t loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                        ++loop_ur_idx) {
                    // Compute gate 2: G2 = tanh(G2 + b2)
                    load(G2(loop_ur_idx), sg_addr(2, loop_ur_idx),
                            scratch_data_t, current_vlen);
                    // dequantize gate from s32 to f32 if needed
                    deq_w(src_data_t, G2(loop_ur_idx), tmp1_vmm, tmp2_vmm,
                            2 * rnn_.dhc + loop_ur_idx * vlen_qscale, mask,
                            current_vlen);
                    to_float(tmp1_vmm, B_addr(2, loop_ur_idx), rnn_.bias_dt,
                            current_vlen);
                    compute_vaddps(G2(loop_ur_idx), G2(loop_ur_idx), tmp1_vmm,
                            current_vlen);
                }

                // Compute tanh of unrolled G2 regs together
                // (this allows to not save any registers during eltwise)
                injector_utils::vmm_index_set_t vmm_idxs;
                for (size_t loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                        ++loop_ur_idx) {
                    vmm_idxs.emplace(G2(loop_ur_idx).getIdx());
                }
                tanh_injector_->compute_vector_range(vmm_idxs);

                for (size_t loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                        ++loop_ur_idx) {
                    // if training we write back the gates
                    if (is_training)
                        to_src(wg_addr(2, loop_ur_idx), G2(loop_ur_idx),
                                src_data_t, current_vlen);

                    load(G0(loop_ur_idx), sg_addr(0, loop_ur_idx),
                            scratch_data_t, current_vlen);
                    load(tmp1_vmm, one_addr, scratch_data_t, current_vlen);
                    if (is_augru) {
                        // for augru there is additional step G01 = (1 - a) * G0
                        // states_t_l = states_tm1_l * G01 + (1 - G01) * G2
                        const Xmm tmp2s_vmm(tmp2_vmm.getIdx());
                        to_float(tmp2s_vmm, ptr[addr_attn_reg], src_data_t,
                                scratch_dt_size);
                        uni_vbroadcastss(tmp2_vmm, tmp2s_vmm);
                        // G01 = (1 - a) * G0
                        compute_vsubps(tmp2_vmm, tmp1_vmm, tmp2_vmm, tmp3_vmm,
                                current_vlen);
                        compute_vmulps(G0(loop_ur_idx), G0(loop_ur_idx),
                                tmp2_vmm, current_vlen);
                        to_float(tmp2_vmm,
                                ptr[addr_states_tm1_l_reg
                                        + loop_ur_idx * vlen_dst],
                                src_data_t, current_vlen);
                        // tmp1 = 1 - G01
                        compute_vsubps(tmp1_vmm, tmp1_vmm, G0(loop_ur_idx),
                                current_vlen);
                        // tmp1 = G2 * tmp1
                        compute_vmulps(tmp1_vmm, G2(loop_ur_idx), tmp1_vmm,
                                tmp3_vmm, current_vlen);
                        // states_t_l = G01 * states_tm1_l + tmp1
                        compute_vfmadd213ps(G0(loop_ur_idx), tmp2_vmm, tmp1_vmm,
                                current_vlen);
                    } else {
                        // states_t_l = states_tm1_l * G0 + (1 - G0) * G2
                        compute_vsubps(tmp1_vmm, tmp1_vmm, G0(loop_ur_idx),
                                current_vlen);
                        to_float(tmp2_vmm,
                                ptr[addr_states_tm1_l_reg
                                        + loop_ur_idx * vlen_dst],
                                src_data_t, current_vlen);
                        compute_vmulps(G0(loop_ur_idx), G0(loop_ur_idx),
                                tmp2_vmm, current_vlen);
                        compute_vfmadd231ps(G0(loop_ur_idx), tmp1_vmm,
                                G2(loop_ur_idx), current_vlen);
                    }
                    to_src(ptr[addr_states_t_l_reg + loop_ur_idx * vlen_dst],
                            G0(loop_ur_idx), src_data_t, current_vlen);
                    // if states_t_l_copy is a non null ptr, we write the output
                    // to both tensors
                    Label loop_inc_regs;
                    cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
                    jle(loop_inc_regs);
                    // As to_src is called with write_only=true it's important
                    // for xf16 src_dt to execute just after to_src method with
                    // write_only=false for the same Vmm
                    to_src(ptr[addr_states_t_l_copy_reg
                                   + loop_ur_idx * vlen_dst],
                            G0(loop_ur_idx), src_data_t, current_vlen, true);
                    L(loop_inc_regs);
                }

                if (current_vlen_elem != loop_tail) {
                    // increment address pointers
                    const auto current_gate_size = current_vlen == vlen
                            ? vlen_dst * current_loop_unroll
                            : gate_dt_size;
                    const auto current_states_size = current_vlen == vlen
                            ? vlen_dst * current_loop_unroll
                            : hstate_dt_size;

                    add(addr_scratch_gates_reg,
                            current_vlen * current_loop_unroll);
                    add(addr_bias_reg,
                            current_vlen == vlen
                                    ? vlen_bias * current_loop_unroll
                                    : bias_dt_size_);
                    add(addr_states_t_l_reg, current_states_size);
                    add(addr_states_t_l_copy_reg, current_states_size);
                    add(addr_states_tm1_l_reg, current_states_size);
                    if (is_training) add(addr_ws_gates_reg, current_gate_size);
                    inc_regs(mask,
                            current_vlen == vlen
                                    ? current_vlen * current_loop_unroll
                                    : qscale_dt_size);

                    // increment loop counter
                    sub(loop_cnt, current_vlen_elem * current_loop_unroll);
                    cmp(loop_cnt, current_vlen_elem * current_loop_unroll);
                    jge(loop_start_label);
                }
            }
        };

        // vector processing
        if (loop_len >= vlen_elems) {
            Label tail_processing_or_exit_label;
            if (is_brgemm) {
                cmp(loop_cnt, vlen_elems * loop_ur);
                jl(tail_processing_or_exit_label, T_NEAR);
            }
            compute_loop(vlen_elems, loop_ur);
            L(tail_processing_or_exit_label);
        }

        // tail processing
        if (loop_tail > 0) {
            Label exit_label;
            if (is_brgemm) {
                cmp(loop_cnt, 0);
                jle(exit_label, T_NEAR);
            }

            compute_loop(is_avx512 ? loop_tail : 1, 1);
            L(exit_label);
        }

        postamble();

        tanh_injector_->prepare_table(true);
        init_table(vlen);
        L(table_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(1.0f));
        }
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
