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

#ifndef CPU_X64_RNN_JIT_UNI_RNN_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_RNN_CELL_POSTGEMM_FWD_HPP

#include <memory>
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_rnn_cell_postgemm_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_rnn_cell_postgemm_fwd)

    using injector_t = typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_core>,
            jit_uni_eltwise_injector_f32<isa>>::type;

    jit_uni_rnn_cell_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    status_t init(data_type_t sdt) override {
        CHECK(jit_uni_rnn_postgemm::init(src_data_t));
        // we use rax for constant tables
        injector_ = utils::make_unique<injector_t>(this, pd_->activation_kind(),
                pd_->desc()->alpha, pd_->desc()->beta, 1.0f, true, rax);
        return create_kernel();
    }

protected:
    std::unique_ptr<injector_t> injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;
    static constexpr size_t cstate_dt_size = sizeof(float);
    static constexpr size_t qscale_dt_size = sizeof(float);

    const size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t vlen_bias = vlen / (sizeof(float) / bias_dt_size_);
    const size_t hstate_dt_size = types::data_type_size(src_data_t);
    const size_t gate_dt_size = types::data_type_size(src_data_t);
    const size_t scratch_dt_size = types::data_type_size(scratch_data_t);

    void generate() override {
        using namespace Xbyak;

        const int mask = pd_->attr()->rnn_weights_qparams_.mask_;
        float *weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;
        Label table_label;

        // Register map
        const Reg64 loop_cnt(r11); // loop counter
        const Reg64 n_step_reg(r12);

        // Here we do no unrolling, loop overhead should not be that dramatic
        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        const Vmm G(1), tmp1_vmm(5), tmp2_vmm(6);

        const auto is_training
                = pd_->desc()->prop_kind == prop_kind::forward_training;

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_bias_reg = abi_param3;
        const auto addr_states_t_l_reg = abi_param4;
        const auto base_args = get_stack_params_address();
#ifdef _WIN32
        const auto addr_states_t_l_copy_reg = r10;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        if (rnn_.is_brgemm && !rnn_.unfused_post_gemm)
            mov(n_step_reg, ptr[base_args + 40]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        if (rnn_.is_brgemm && !rnn_.unfused_post_gemm)
            mov(n_step_reg, ptr[base_args + 24]);
#endif

        const auto sg_addr
                = ptr[addr_scratch_gates_reg + 0 * rnn_.dhc * scratch_dt_size];
        const auto wg_addr
                = ptr[addr_ws_gates_reg + 0 * rnn_.dhc * gate_dt_size];
        const auto B_addr = ptr[addr_bias_reg + 0 * rnn_.dhc * bias_dt_size_];

        // initialize registers with addresses and constants
        init_regs(weights_scales, vlen);
        injector_->load_table_addr();

        if (rnn_.is_brgemm && !rnn_.unfused_post_gemm)
            mov(loop_cnt, n_step_reg);
        else
            mov(loop_cnt, rnn_.dhc * scratch_dt_size);

        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L_aligned(vector_loop_start_label, 64);
        {
            // load G
            uni_vmovups(G, sg_addr);

            // dequantize the gates from s32 to f32 if needed
            deq_w(src_data_t, G, tmp1_vmm, tmp2_vmm, 0, mask, vlen);

            // add biases
            to_float(tmp1_vmm, B_addr, rnn_.bias_dt, vlen);
            uni_vaddps(G, G, tmp1_vmm);

            // inject eltwise code
            injector_->compute_vector(G.getIdx());

            // if training we write back the gates
            if (is_training) to_src(wg_addr, G, src_data_t, vlen);

            to_src(ptr[addr_states_t_l_reg], G, src_data_t, vlen);
            // if states_t_l_copy is a non null ptr, we write the output to both
            // tensors
            cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
            jle(vector_loop_inc_regs);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(ptr[addr_states_t_l_copy_reg], G, src_data_t, vlen, true);

            // increment address pointers
            L(vector_loop_inc_regs);
            add(addr_scratch_gates_reg, vlen);
            add(addr_bias_reg, vlen_bias);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_states_t_l_copy_reg, vlen_dst);
            if (is_training) add(addr_ws_gates_reg, vlen_dst);
            inc_regs(mask, vlen);

            // increment loop counter
            sub(loop_cnt, vlen);
            cmp(loop_cnt, vlen);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use movuss for accessing inputs
        // TODO: smarter handling of tails with Zmm -> Ymm -> Xmm -> scalar
        L(rem_loop_start_label);
        {
            // remaping registers to Xmms
            const Xmm Gs(G.getIdx());
            const Xmm tmp1s_vmm(tmp1_vmm.getIdx());

            // load G
            uni_vmovss(Gs, sg_addr);

            // dequantize the gates from s32 to f32 if needed
            deq_w(src_data_t, G, tmp1_vmm, tmp2_vmm, 0, mask, scratch_dt_size);

            // add biases
            to_float(tmp1_vmm, B_addr, rnn_.bias_dt, sizeof(float));
            uni_vaddps(Gs, Gs, tmp1s_vmm);

            // inject eltwise code
            injector_->compute_vector(Gs.getIdx());

            // if training we write back the gates
            if (is_training) to_src(wg_addr, G, src_data_t, scratch_dt_size);

            to_src(ptr[addr_states_t_l_reg], G, src_data_t, scratch_dt_size);
            // if states_t_l_copy is a non null ptr, we write the output to both
            // tensors
            cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
            jle(rem_loop_inc_regs);
            // As to_src is called with write_only=true it's important for xf16
            // src_dt to execute just after to_src method with write_only=false
            // for the same Vmm
            to_src(ptr[addr_states_t_l_copy_reg], G, src_data_t,
                    scratch_dt_size, true);

            // increment address pointers
            L(rem_loop_inc_regs);
            add(addr_scratch_gates_reg, scratch_dt_size);
            add(addr_bias_reg, bias_dt_size_);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_states_t_l_copy_reg, hstate_dt_size);
            if (is_training) add(addr_ws_gates_reg, gate_dt_size);
            inc_regs(mask, qscale_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();

        // inject the constant table for the activation
        injector_->prepare_table();
        init_table(vlen);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
