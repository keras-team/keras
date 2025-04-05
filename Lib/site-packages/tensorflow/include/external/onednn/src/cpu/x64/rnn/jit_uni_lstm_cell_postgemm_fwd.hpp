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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP

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
struct jit_uni_lstm_cell_postgemm_fwd
    : public jit_uni_rnn_postgemm,
      public jit_uni_lstm_cell_postgemm_t<isa> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_fwd)

    jit_uni_lstm_cell_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name())
        , jit_uni_lstm_cell_postgemm_t<isa>(this,
                  get_last_preserved_vmm_idx(1) + 1,
                  // usage of jit_uni_rnn_postgemm::bf16_emu_ to identify bf16
                  // emulation case is illegal here, it's created in
                  // jit_uni_rnn_postgemm::init(), not in constructor, so
                  // jit_uni_rnn_postgemm::bf16_emu_ = nullptr always on this
                  // stage
                  src_data_t == data_type::bf16 && !mayiuse(avx512_core_bf16)) {
    }

    ~jit_uni_lstm_cell_postgemm_fwd() = default;

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
    using injector_t = typename jit_uni_lstm_cell_postgemm_t<isa>::injector_t;
    using Vmm = typename jit_uni_lstm_cell_postgemm_t<isa>::Vmm;

    std::unique_ptr<injector_t> sigmoid_injector_;
    std::unique_ptr<injector_t> tanh_injector_;

    // register size in bytes
    static constexpr size_t vlen_ = cpu_isa_traits<isa>::vlen;
    static constexpr size_t qscale_dt_size = sizeof(float);
    static constexpr size_t weights_peephole_dt_size_ = sizeof(float);
    const size_t vlen_dst_
            = vlen_ / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t vlen_bias_ = vlen_ / (sizeof(float) / bias_dt_size_);
    const size_t vlen_c_states_ = vlen_ / (sizeof(float) / cstate_dt_size_);
    const size_t hstate_dt_size_ = types::data_type_size(src_data_t);
    const size_t gate_dt_size_ = types::data_type_size(src_data_t);
    const size_t scratch_dt_size_ = types::data_type_size(scratch_data_t);
    int get_vmm_idx(int unroll_idx, int type_shift) const {
        const int preserved_vmm_start_idx = 1;
        // G0, G1, G2, G3, c_states;
        const int num_preserved_regs_for_loop_iter = 5;
        assert(type_shift < num_preserved_regs_for_loop_iter);
        const int unroll_idx_start = preserved_vmm_start_idx
                + num_preserved_regs_for_loop_iter * unroll_idx;
        return unroll_idx_start + type_shift;
    }

    int G0_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 0); }
    int G1_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 1); }
    int G2_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 2); }
    int G3_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 3); }
    int c_states_idx(int unroll_idx) const {
        return get_vmm_idx(unroll_idx, 4);
    }
    int get_last_preserved_vmm_idx(int current_loop_unroll) const {
        return c_states_idx(current_loop_unroll - 1);
    }

    dim_t scale_off(int gate_idx, int unroll_idx) const {
        const size_t vlen_qscale_elem = vlen_ / qscale_dt_size;
        return gate_idx * rnn_.dhc + unroll_idx * vlen_qscale_elem;
    }

    void generate() override {
        using namespace Xbyak;

        const auto is_training
                = (pd_->desc()->prop_kind == prop_kind::forward_training);

        const int mask = pd_->attr()->rnn_weights_qparams_.mask_;
        float *const weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;

        // Register map
        const Reg64 loop_cnt(rbx); // loop counter

        // We start code generations here
        preamble();

        const Reg64 n_step_reg(rbp);

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_weights_peephole_reg = r11;
        const auto addr_bias_reg = abi_param3;
        const auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
        const auto addr_states_t_l_copy_reg = r10;
        const auto addr_c_states_tm1_l_reg = rdi;
        const auto addr_c_states_t_l_reg = rsi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_c_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 16]);
        mov(addr_weights_peephole_reg, ptr[base_args + 24]);
        mov(n_step_reg, ptr[base_args + 40]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        const auto addr_c_states_tm1_l_reg = abi_param6;
        const auto addr_c_states_t_l_reg = r10;
        const auto base_args = get_stack_params_address();
        mov(addr_c_states_t_l_reg, ptr[base_args]);
        mov(addr_weights_peephole_reg, ptr[base_args + 8]);
        mov(n_step_reg, ptr[base_args + 24]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i, int j = 0) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size_
                    + j * vlen_];
        };

        const auto wg_addr = [&](int i, int j = 0) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size_
                    + j * vlen_dst_];
        };

        const auto B_addr = [&](int i, int j = 0) {
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size_
                    + j * vlen_bias_];
        };

        const auto weights_peephole_addr = [&](int i, int j = 0) {
            return ptr[addr_weights_peephole_reg
                    + i * rnn_.dhc * weights_peephole_dt_size_ + j * vlen_];
        };

        const auto loop_len = rnn_.dhc * scratch_dt_size_;
        const auto loop_tail = loop_len % vlen_;

        // initialize registers with addresses and constants
        init_regs(weights_scales, vlen_, loop_tail / scratch_dt_size_);
        sigmoid_injector_->load_table_addr();
        tanh_injector_->load_table_addr();
        if (rnn_.is_brgemm && !rnn_.unfused_post_gemm)
            mov(loop_cnt, n_step_reg);
        else
            mov(loop_cnt, loop_len);

        int loop_unroll = 1;
        int loop_unroll_tail = 0;

        const int loop_unroll_max = is_avx512 ? 4 : 1;
        if (rnn_.is_brgemm && !rnn_.unfused_post_gemm) {
            const auto block_loop_len = rnn_.n_block * scratch_dt_size_;
            for (loop_unroll = loop_unroll_max; loop_unroll > 1;
                    loop_unroll--) {
                if (block_loop_len % (loop_unroll * vlen_) == 0) break;
            }
            if (loop_unroll > 1 && rnn_.n_tail > 0
                    && rnn_.n_tail * scratch_dt_size_ - loop_tail > 0)
                loop_unroll_tail = 1;
        } else {
            for (loop_unroll = loop_unroll_max; loop_unroll > 1;
                    loop_unroll--) {
                if (loop_len >= (loop_unroll * vlen_)) break;
            }
            if (loop_unroll > 1
                    && (loop_len - loop_tail) % (loop_unroll * vlen_) > 0)
                loop_unroll_tail = 1;
        }

        auto compute_loop = [&](size_t current_vlen, int current_unroll_len) {
            this->reset_tmp_vmm_idx_range(
                    get_last_preserved_vmm_idx(current_unroll_len) + 1,
                    this->get_max_allowed_tmp_vmm_allowed_idx());

            injector_utils::vmm_index_set_t vmm_idxs;

            const bool single_tail_loop_iter
                    = current_vlen < vlen_ && current_vlen == loop_tail;
            const bool need_increment_regs = !single_tail_loop_iter;
            const auto iter_size = current_unroll_len * current_vlen;

            Label loop_start_label, loop_skip_label;
            cmp(loop_cnt, iter_size);
            jl(loop_skip_label, T_NEAR);

            L_aligned(loop_start_label, 64);
            {
                for (int ur_idx = 0; ur_idx < current_unroll_len; ur_idx++) {
                    const Vmm G0(G0_idx(ur_idx)), G1(G1_idx(ur_idx)),
                            G2(G2_idx(ur_idx)), G3(G3_idx(ur_idx)),
                            tmp_c_states(c_states_idx(ur_idx));
                    // load G0 G1 G2 G3
                    load(G0, sg_addr(0, ur_idx), scratch_data_t, current_vlen);
                    load(G1, sg_addr(1, ur_idx), scratch_data_t, current_vlen);
                    load(G2, sg_addr(2, ur_idx), scratch_data_t, current_vlen);
                    load(G3, sg_addr(3, ur_idx), scratch_data_t, current_vlen);

                    // dequantize the gates from s32 to f32 if needed, add bias
                    deq_w(src_data_t, G0, this->get_next_tmp_vmm(),
                            this->get_next_tmp_vmm(), scale_off(0, ur_idx),
                            mask, current_vlen);
                    const auto bias_g0_vmm = this->get_next_tmp_vmm();
                    to_float(bias_g0_vmm, B_addr(0, ur_idx), rnn_.bias_dt,
                            current_vlen);
                    compute_vaddps(G0, G0, bias_g0_vmm, current_vlen);

                    deq_w(src_data_t, G1, this->get_next_tmp_vmm(),
                            this->get_next_tmp_vmm(), scale_off(1, ur_idx),
                            mask, current_vlen);
                    const auto bias_g1_vmm = this->get_next_tmp_vmm();
                    to_float(bias_g1_vmm, B_addr(1, ur_idx), rnn_.bias_dt,
                            current_vlen);
                    compute_vaddps(G1, G1, bias_g1_vmm, current_vlen);

                    deq_w(src_data_t, G2, this->get_next_tmp_vmm(),
                            this->get_next_tmp_vmm(), scale_off(2, ur_idx),
                            mask, current_vlen);
                    const auto bias_g2_vmm = this->get_next_tmp_vmm();
                    to_float(bias_g2_vmm, B_addr(2, ur_idx), rnn_.bias_dt,
                            current_vlen);
                    compute_vaddps(G2, G2, bias_g2_vmm, current_vlen);

                    deq_w(src_data_t, G3, this->get_next_tmp_vmm(),
                            this->get_next_tmp_vmm(), scale_off(3, ur_idx),
                            mask, current_vlen);
                    const auto bias_g3_vmm = this->get_next_tmp_vmm();
                    to_float(bias_g3_vmm, B_addr(3, ur_idx), rnn_.bias_dt,
                            current_vlen);
                    compute_vaddps(G3, G3, bias_g3_vmm, current_vlen);

                    to_float(tmp_c_states,
                            ptr[addr_c_states_tm1_l_reg
                                    + ur_idx * vlen_c_states_],
                            rnn_.src_iter_c_dt, current_vlen);

                    // add peephole
                    if (rnn_.is_lstm_peephole) {
                        compute_vfmadd231ps(G0, tmp_c_states,
                                weights_peephole_addr(0, ur_idx), current_vlen,
                                this->maybe_get_next_tmp_vmm_for_below_avx2_isa());
                        compute_vfmadd231ps(G1, tmp_c_states,
                                weights_peephole_addr(1, ur_idx), current_vlen,
                                this->maybe_get_next_tmp_vmm_for_below_avx2_isa());
                    }

                    vmm_idxs.emplace(G0.getIdx());
                    vmm_idxs.emplace(G1.getIdx());
                    if (!rnn_.is_lstm_peephole) vmm_idxs.emplace(G3.getIdx());
                }

                // inject eltwise code
                sigmoid_injector_->load_table_addr();
                sigmoid_injector_->compute_vector_range(vmm_idxs);
                vmm_idxs.clear();

                if (is_training) {
                    for (int ur_idx = 0; ur_idx < current_unroll_len;
                            ur_idx++) {
                        to_src(wg_addr(0, ur_idx), Vmm(G0_idx(ur_idx)),
                                src_data_t, current_vlen);
                        to_src(wg_addr(1, ur_idx), Vmm(G1_idx(ur_idx)),
                                src_data_t, current_vlen);
                        if (!rnn_.is_lstm_peephole)
                            to_src(wg_addr(3, ur_idx), Vmm(G3_idx(ur_idx)),
                                    src_data_t, current_vlen);
                    }
                }
                for (int ur_idx = 0; ur_idx < current_unroll_len; ur_idx++) {
                    vmm_idxs.emplace(G2_idx(ur_idx));
                }
                tanh_injector_->load_table_addr();
                tanh_injector_->compute_vector_range(vmm_idxs);
                vmm_idxs.clear();

                for (int ur_idx = 0; ur_idx < current_unroll_len; ur_idx++) {
                    const Vmm G0(G0_idx(ur_idx)), G1(G1_idx(ur_idx)),
                            G2(G2_idx(ur_idx)),
                            tmp_c_states(c_states_idx(ur_idx));
                    if (is_training) {
                        to_src(wg_addr(2, ur_idx), G2, src_data_t,
                                current_vlen);
                    }

                    // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
                    compute_vmulps(
                            tmp_c_states, tmp_c_states, G1, current_vlen);
                    compute_vfmadd231ps(tmp_c_states, this->vmm_backup(G0), G2,
                            current_vlen);
                    to_src(ptr[addr_c_states_t_l_reg + ur_idx * vlen_c_states_],
                            tmp_c_states, rnn_.dst_iter_c_dt, current_vlen);
                }

                // add peephole
                if (rnn_.is_lstm_peephole) {
                    for (int ur_idx = 0; ur_idx < current_unroll_len;
                            ur_idx++) {
                        const int cur_g3_idx = G3_idx(ur_idx);
                        compute_vfmadd231ps(Vmm(cur_g3_idx),
                                Vmm(c_states_idx(ur_idx)),
                                weights_peephole_addr(2, ur_idx), current_vlen,
                                this->maybe_get_next_tmp_vmm_for_below_avx2_isa());
                        vmm_idxs.emplace(cur_g3_idx);
                    }
                    sigmoid_injector_->load_table_addr();
                    sigmoid_injector_->compute_vector_range(vmm_idxs);
                    vmm_idxs.clear();

                    // if training we write back the gates
                    if (is_training) {
                        for (int ur_idx = 0; ur_idx < current_unroll_len;
                                ur_idx++) {
                            to_src(wg_addr(3, ur_idx), Vmm(G3_idx(ur_idx)),
                                    src_data_t, current_vlen);
                        }
                    }
                }

                for (int ur_idx = 0; ur_idx < current_unroll_len; ur_idx++) {
                    vmm_idxs.emplace(c_states_idx(ur_idx));
                }
                // states_t_l = G3 * tanh(c_states_t_l)
                tanh_injector_->load_table_addr();
                tanh_injector_->compute_vector_range(vmm_idxs);
                vmm_idxs.clear();

                for (int ur_idx = 0; ur_idx < current_unroll_len; ur_idx++) {
                    const Vmm G3(G3_idx(ur_idx)),
                            tmp_c_states(c_states_idx(ur_idx));
                    compute_vmulps(
                            tmp_c_states, tmp_c_states, G3, current_vlen);
                }

                // downconvert/quantize and write back the state
                Label loop_inc_regs_label,
                        update_single_states_tensor_only_label;
                cmp(addr_states_t_l_copy_reg, 0);
                je(update_single_states_tensor_only_label, T_NEAR);
                // if states_t_l_copy is a non null ptr, we write the output to
                // both tensors
                for (int ur_idx = 0; ur_idx < current_unroll_len; ur_idx++) {
                    const Vmm tmp_c_states(c_states_idx(ur_idx));
                    to_src(ptr[addr_states_t_l_reg + ur_idx * vlen_dst_],
                            tmp_c_states, src_data_t, current_vlen);
                    // As to_src is called with write_only=true it's important
                    // for xf16 src_dt to execute just after to_src method with
                    // write_only=false for the same Vmm
                    to_src(ptr[addr_states_t_l_copy_reg + ur_idx * vlen_dst_],
                            tmp_c_states, src_data_t, current_vlen, true);
                }
                const size_t hstate_shift = current_vlen < vlen_
                        ? hstate_dt_size_
                        : current_unroll_len * vlen_dst_;
                if (need_increment_regs)
                    add(addr_states_t_l_copy_reg, hstate_shift);
                jmp(loop_inc_regs_label, T_NEAR);

                L_aligned(update_single_states_tensor_only_label);
                for (int ur_idx = 0; ur_idx < current_unroll_len; ur_idx++) {
                    to_src(ptr[addr_states_t_l_reg + ur_idx * vlen_dst_],
                            Vmm(c_states_idx(ur_idx)), src_data_t,
                            current_vlen);
                }

                // increment address pointers
                L_aligned(loop_inc_regs_label);
                if (need_increment_regs) {
                    const size_t scratch_shift = current_vlen < vlen_
                            ? scratch_dt_size_
                            : current_unroll_len * vlen_;
                    add(addr_scratch_gates_reg, scratch_shift);
                    if (rnn_.is_lstm_peephole) {
                        const size_t wpeephole_shift = current_vlen < vlen_
                                ? weights_peephole_dt_size_
                                : current_unroll_len * vlen_;
                        add(addr_weights_peephole_reg, wpeephole_shift);
                    }
                    const size_t bias_shift = current_vlen < vlen_
                            ? bias_dt_size_
                            : current_unroll_len * vlen_bias_;
                    add(addr_bias_reg, bias_shift);
                    add(addr_states_t_l_reg, hstate_shift);
                    const size_t cstate_shift = current_vlen < vlen_
                            ? cstate_dt_size_
                            : current_unroll_len * vlen_c_states_;
                    add(addr_c_states_tm1_l_reg, cstate_shift);
                    add(addr_c_states_t_l_reg, cstate_shift);
                    if (is_training) {
                        const size_t gate_shift = current_vlen < vlen_
                                ? gate_dt_size_
                                : current_unroll_len * vlen_dst_;
                        add(addr_ws_gates_reg, gate_shift);
                    }
                    const size_t qscale_shift = current_vlen < vlen_
                            ? qscale_dt_size
                            : current_unroll_len * vlen_;
                    inc_regs(mask, qscale_shift);
                }

                // increment loop counter
                sub(loop_cnt, iter_size);
                cmp(loop_cnt, iter_size);
                jge(loop_start_label, T_NEAR);
            }
            L_aligned(loop_skip_label, 64);
        };

        if (loop_unroll > 0) {
            // unrolled vector loop
            compute_loop(vlen_, loop_unroll);
        }

        if (loop_unroll_tail > 0) {
            // not unrolled vector loop if required
            compute_loop(vlen_, loop_unroll_tail);
        }

        if (loop_tail > 0) {
            // tail processing
            compute_loop(is_avx512 ? loop_tail : scratch_dt_size_, 1);
        }

        postamble();

        sigmoid_injector_->prepare_table(true);
        tanh_injector_->prepare_table(true);

        init_table(vlen_);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
