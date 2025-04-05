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

#ifndef CPU_X64_RNN_JIT_UNI_RNN_CELL_POSTGEMM_BWD_HPP
#define CPU_X64_RNN_JIT_UNI_RNN_CELL_POSTGEMM_BWD_HPP

#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_rnn_cell_postgemm_bwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_rnn_cell_postgemm_bwd)

    jit_uni_rnn_cell_postgemm_bwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd, jit_name()) {}

    ~jit_uni_rnn_cell_postgemm_bwd() {}

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
        Label table_one_label, table_alpha_label;

        // Register map
        // aliasing with table_reg and loop_cnt since they are not used at the same time
        const Reg64 table_reg(r11);
        const Reg64 loop_cnt(r11);

        // Here we do no unrolling, loop overhead should not be that dramatic
        // Note: G has to be indexed at 0 as it is used as a mask in blend for bwd relu
        enum {
            G_idx = 0,
            dG_idx,
            dHt_idx,
            tmp1_idx,
            one_idx,
            zero_idx,
            alpha_idx
        };
        const Xbyak::Opmask kmask(1);

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_diff_states_t_lp1_reg = abi_param3;
        const auto addr_diff_states_tp1_l_reg = abi_param4;

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size];
        };
        const auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size];
        };
        // auto sc_addr = [&](int i) {
        //     return ptr[addr_scratch_cell_reg + i * rnn_.dhc * scratch_dt_size];
        // };

        // initialize registers with addresses and constants
        init_regs(vlen);

        mov(table_reg, table_one_label);
        uni_vmovups(Vmm(one_idx), ptr[table_reg]);

        if (pd_->activation_kind() == alg_kind::eltwise_relu) {
            mov(table_reg, table_alpha_label);
            uni_vmovups(Vmm(alpha_idx), ptr[table_reg]);
        }

        uni_vxorps(Vmm(zero_idx), Vmm(zero_idx), Vmm(zero_idx));

        mov(loop_cnt, rnn_.dhc * scratch_dt_size);
        cmp(loop_cnt, vlen_scratch);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            const Vmm G(G_idx), dG(dG_idx), dHt(dHt_idx), tmp1(tmp1_idx),
                    one(one_idx), zero(zero_idx), alpha(alpha_idx);

            to_float(G, wg_addr(0), src_data_t, vlen);

            // compute dHt
            uni_vmovups(dHt, ptr[addr_diff_states_tp1_l_reg]);
            uni_vmovups(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddps(dHt, dHt, tmp1);

            // compute dG
            switch (pd_->activation_kind()) {
                case alg_kind::eltwise_relu:
                    // G > 0 ? alpha : 1
                    if (G.isZMM()) {
                        vcmpps(kmask, G, zero, _cmp_nle_us);
                        vblendmps(dG | kmask, alpha, one);
                    } else {
                        // NOTE: here G is assumed to be xmm0 for sse4.1 blendvps to work
                        uni_vcmpps(G, G, zero, _cmp_nle_us);
                        uni_vmovups(dG, alpha);
                        uni_vblendvps(dG, dG, one, G);
                    }
                    break;
                case alg_kind::eltwise_tanh:
                    // 1 - G^2
                    uni_vmovups(dG, one);
                    uni_vfnmadd231ps(dG, G, G); // (1 - G2^2)
                    break;
                case alg_kind::eltwise_logistic:
                    uni_vmovups(dG, G);
                    uni_vfnmadd231ps(dG, G, G); // (G - G^2)
                    break;
                default: assert(!"unsupported");
            }

            // dG = dG * dHt
            uni_vmulps(dG, dG, dHt);

            // downconvert and write data
            to_src(sg_addr(0), dG, scratch_data_t, vlen);

            // increment address pointers
            add(addr_ws_gates_reg, vlen_scratch);
            add(addr_scratch_gates_reg, vlen_scratch);
            add(addr_diff_states_t_lp1_reg, vlen);
            add(addr_diff_states_tp1_l_reg, vlen);
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
            const Xmm G(G_idx), dG(dG_idx), dHt(dHt_idx), tmp1(tmp1_idx),
                    one(one_idx), zero(zero_idx), alpha(alpha_idx);

            to_float(G, wg_addr(0), src_data_t, hstate_dt_size);

            // compute dHt
            uni_vmovss(dHt, ptr[addr_diff_states_tp1_l_reg]);
            uni_vmovss(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddss(dHt, dHt, tmp1);

            // compute dG
            switch (pd_->activation_kind()) {
                case alg_kind::eltwise_relu:
                    // G > 0 ? alpha : 1
                    // NOTE: here G is assumed to be xmm0 for sse4.1 blendvps to work
                    uni_vcmpps(G, G, zero, _cmp_nle_us);
                    uni_vmovups(dG, alpha);
                    uni_vblendvps(dG, dG, one, G);
                    break;
                case alg_kind::eltwise_tanh:
                    // 1 - G^2
                    uni_vmovss(dG, one);
                    uni_vfnmadd231ps(dG, G, G); // (1 - G2^2)
                    break;
                case alg_kind::eltwise_logistic:
                    uni_vmovss(dG, G);
                    uni_vfnmadd231ps(dG, G, G); // (G - G^2)
                    break;
                default: assert(!"unsupported");
            }

            // dG = dG * dHt
            uni_vmulps(dG, dG, dHt);

            // downconvert and write data
            to_src(sg_addr(0), dG, scratch_data_t, hstate_dt_size);

            // increment address pointers
            add(addr_ws_gates_reg, scratch_dt_size);
            add(addr_scratch_gates_reg, scratch_dt_size);
            add(addr_diff_states_t_lp1_reg, hstate_dt_size);
            add(addr_diff_states_tp1_l_reg, hstate_dt_size);
            inc_regs(hstate_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();

        // inject the constant table for the activation
        init_table(vlen);
        L(table_one_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(1.0f));
        }
        L(table_alpha_label);
        {
            if (pd_->activation_kind() == alg_kind::eltwise_relu) {
                for (size_t i = 0; i < vlen / sizeof(float); i++)
                    dd(float2int(pd_->desc()->alpha));
            }
        }
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
