/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef CPU_X64_PRELU_JIT_PRELU_FORWARD_KERNEL_HPP
#define CPU_X64_PRELU_JIT_PRELU_FORWARD_KERNEL_HPP

#include <map>

#include "cpu/cpu_prelu_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/prelu/jit_prelu_base_kernel.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_prelu_forward_kernel_t : public jit_prelu_base_kernel_t {
public:
    static jit_prelu_forward_kernel_t *create(const cpu_prelu_fwd_pd_t *pd);

    struct call_params_t {
        const void *src = nullptr, *weights = nullptr, *dst = nullptr;
        size_t compute_data_size = 0u;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_prelu_forward_kernel_t)

    void operator()(jit_prelu_forward_kernel_t::call_params_t *params) {
        jit_generator::operator()(params);
    }

protected:
    const data_type_t src_dt_;
    const data_type_t wei_dt_;
    const data_type_t dst_dt_;
    const size_t dst_tail_block_;

    jit_prelu_forward_kernel_t(const cpu_prelu_fwd_pd_t *pd,
            const cpu_isa_t &isa, const int vlen,
            const size_t number_vmm_single_compute);
    Xbyak::Address data_ptr(int arg_num, size_t offt = 0);

private:
    bool any_tensor_bf16() const override;
    void load_kernel_call_params() override;
    void finalize() override {}

protected:
    const Xbyak::Reg64 &reg_src_ = r10;
    const Xbyak::Reg64 &reg_dst_ = r11;
    const Xbyak::Reg64 &reg_weights_ = r12;
    const cpu_prelu_fwd_pd_t *pd_;
};

template <typename Vmm>
class jit_uni_prelu_forward_kernel_t : public jit_prelu_forward_kernel_t {
public:
    jit_uni_prelu_forward_kernel_t(
            const cpu_prelu_fwd_pd_t *pd, const cpu_isa_t &isa);
    ~jit_uni_prelu_forward_kernel_t() override;

private:
    using jit_generator::uni_vfmadd132ps;

    void prepare_kernel_const_vars() override;
    void compute_dst(size_t unrolling_factor, bool tail) override;
    bool can_load_wei_from_addr_directly(bool tail) const noexcept;

    void compute_dst_body(size_t unrolling_factor, bool tail);
    void compute_ne_convert_xf16_dst_body(size_t unrolling_factor, bool tail);
    Vmm get_or_load_weights(
            const Xbyak::Address &src_addr, const Vmm &dst_vmm, bool tail);
    void uni_vfmadd132ps(
            const Vmm &x1, const Vmm &x2, const Xbyak::Operand &op, bool tail);
    std::map<data_type_t, io::io_saturation_conf_t>
    create_saturation_vmm_map() const;

    const bool saturation_needed_ = false;
    const Vmm tail_vmm_mask_; // Keep it higher to preserve idx=0 tail register
    const Vmm vmm_zeros_;
    const Vmm dst_saturate_ubound_;
    const Vmm weights_const_vmm_;
    const size_t number_vmm_single_compute_ = 0;
    const Xbyak::Opmask &tail_opmask_ = k1;
    const Xbyak::Reg64 &reg_tmp_ = r15;

    io::jit_io_multi_dt_helper_t<Vmm> io_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
