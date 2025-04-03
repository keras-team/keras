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

#ifndef CPU_X64_RNN_JIT_UNI_GRU_CELL_POSTGEMM_1_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_GRU_CELL_POSTGEMM_1_FWD_HPP

#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_gru_cell_postgemm_part1_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_cell_postgemm_part1_fwd)

    using injector_t = typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_core>,
            jit_uni_eltwise_injector_f32<isa>>::type;

    jit_uni_gru_cell_postgemm_part1_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    status_t init(data_type_t sdt) override {
        CHECK(jit_uni_rnn_postgemm::init(src_data_t));
        // no need to save state of registers
        // (unless emulating bf16 support)
        const bool save_state
                = src_data_t == data_type::bf16 && !mayiuse(avx512_core_bf16);
        // we use rax for both constant tables as they use the same table
        CHECK(safe_ptr_assign(sigmoid_injector_,
                new injector_t(this, alg_kind::eltwise_logistic, 0.0f, 0.0f,
                        1.0f, save_state, rax)));
        return create_kernel();
    }

protected:
    std::unique_ptr<injector_t> sigmoid_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;
    static constexpr size_t qscale_dt_size = sizeof(float);
    const size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t vlen_bias_ = vlen / (sizeof(float) / bias_dt_size_);
    const size_t hstate_dt_size = types::data_type_size(src_data_t);
    const size_t gate_dt_size = types::data_type_size(src_data_t);
    const size_t scratch_dt_size = types::data_type_size(scratch_data_t);
    const size_t vlen_qscale = vlen / qscale_dt_size;
    const size_t vlen_elems = vlen / scratch_dt_size;

    const int loop_ur_max = 4;
    // We skip vmm0 as it can be used by the injector for masks on sse4.1
    int G0_idx(int i) {
        const int idx = 1 + i;
        assert(idx < loop_ur_max + 1);
        return idx;
    }
    int G1_idx(int i) {
        const int idx = loop_ur_max + 1 + i;
        assert(idx < 2 * loop_ur_max + 1);
        return idx;
    }
    const Vmm tmp1_vmm = Vmm(9);
    const Vmm tmp2_vmm = Vmm(10);

    void generate() override {
        using namespace Xbyak;
        const auto is_training
                = pd_->desc()->prop_kind == prop_kind::forward_training;

        const int mask = pd_->attr()->rnn_weights_qparams_.mask_;
        float *weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;

        // Register map
        const Reg64 loop_cnt(rbx); // loop counter

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_bias_reg = abi_param3;
        const auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
        const auto addr_states_t_l_copy_reg = r10;
        const auto addr_states_tm1_l_reg = r11;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        const auto addr_states_tm1_l_reg = abi_param6;
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
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size_ + j * vlen];
        };

        const size_t loop_len = rnn_.dhc;
        const size_t loop_tail = loop_len % vlen_elems;
        // initialize registers with addresses and constants
        init_regs(weights_scales, vlen, loop_tail);

        // both sigmoid and tanh use the same table so load address just once in rax
        sigmoid_injector_->load_table_addr();

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
                    const Vmm G0(G0_idx(loop_ur_idx));
                    const Vmm G1(G1_idx(loop_ur_idx));
                    // batch these operations in order to combine calls to injector:
                    //      Compute gate 0: G0 = sigmoid(G0 + b0)
                    //      Compute gate 1: G1 = sigmoid(G1 + b1)

                    // load gates from scratchpad
                    load(G0, sg_addr(0, loop_ur_idx), scratch_data_t,
                            current_vlen);
                    load(G1, sg_addr(1, loop_ur_idx), scratch_data_t,
                            current_vlen);

                    // dequantize gates from s32 to f32 if needed
                    deq_w(src_data_t, G0, tmp1_vmm, tmp2_vmm,
                            0 * rnn_.dhc + loop_ur_idx * vlen_qscale, mask,
                            current_vlen);
                    deq_w(src_data_t, G1, tmp1_vmm, tmp2_vmm,
                            1 * rnn_.dhc + loop_ur_idx * vlen_qscale, mask,
                            current_vlen);

                    // apply bias
                    to_float(tmp1_vmm, B_addr(0, loop_ur_idx), rnn_.bias_dt,
                            current_vlen);
                    compute_vaddps(G0, G0, tmp1_vmm, current_vlen);
                    to_float(tmp2_vmm, B_addr(1, loop_ur_idx), rnn_.bias_dt,
                            current_vlen);
                    compute_vaddps(G1, G1, tmp2_vmm, current_vlen);
                }

                // Compute sigmoid of unrolled G0 and G1 regs together
                // (this allows to not save any registers during eltwise)
                injector_utils::vmm_index_set_t vmm_idxs;
                for (size_t loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                        ++loop_ur_idx) {
                    vmm_idxs.emplace(G0_idx(loop_ur_idx));
                    vmm_idxs.emplace(G1_idx(loop_ur_idx));
                }
                sigmoid_injector_->compute_vector_range(vmm_idxs);

                for (size_t loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                        ++loop_ur_idx) {
                    const Vmm G0(G0_idx(loop_ur_idx));
                    const Vmm G1(G1_idx(loop_ur_idx));
                    // store G0 for use in postgemm_part2
                    store(sg_addr(0, loop_ur_idx), G0, scratch_data_t,
                            current_vlen);

                    // if training we write back the gates
                    if (is_training) {
                        to_src(wg_addr(1, loop_ur_idx), G1, src_data_t,
                                current_vlen);
                        to_src(wg_addr(0, loop_ur_idx), G0, src_data_t,
                                current_vlen);
                    }

                    // states_t_l = states_tm1_l * G1
                    to_float(tmp1_vmm,
                            ptr[addr_states_tm1_l_reg + loop_ur_idx * vlen_dst],
                            src_data_t, current_vlen);
                    compute_vmulps(G1, G1, tmp1_vmm, current_vlen);
                    to_src(ptr[addr_states_t_l_reg + loop_ur_idx * vlen_dst],
                            G1, src_data_t, current_vlen);
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
                            G1, src_data_t, current_vlen, true);
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
                                    ? vlen_bias_ * current_loop_unroll
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

        // Again, only one table is needed and shared between sigmoid and tanh
        sigmoid_injector_->prepare_table(true);
        init_table(vlen);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
