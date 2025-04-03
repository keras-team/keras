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

#ifndef CPU_X64_JIT_UNI_DW_CONV_KERNEL_UTILS_HPP
#define CPU_X64_JIT_UNI_DW_CONV_KERNEL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

#include "cpu/x64/jit_avx512_core_bf16_dw_conv_kernel.hpp"
#include "cpu/x64/jit_uni_dw_conv_kernel_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_fwd_kernel {

    jit_uni_dw_conv_fwd_kernel(
            const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md) {
        ker_ = new jit_kernel_t(ajcp, dst_md);
    }

    status_t create_kernel() {
        if (ker_) return ker_->create_kernel();
        return status::out_of_memory;
    }
    ~jit_uni_dw_conv_fwd_kernel() { delete ker_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &bias_md,
            memory_desc_t &dst_md, primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_generator *ker() const { return ker_; }
    void operator()(const jit_conv_call_s *p) const { (*ker_)(p); }

private:
    constexpr static bool ker_condition_
            = isa == avx512_core && kernel_dt == data_type::bf16;
    using jit_kernel_t = typename utils::conditional<ker_condition_,
            jit_avx512_dw_conv_fwd_kernel_bf16,
            jit_uni_dw_conv_fwd_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;
};

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_bwd_data_kernel {

    jit_uni_dw_conv_bwd_data_kernel(const jit_conv_conf_t &ajcp)
        : ker_(nullptr) {
        ker_ = new jit_kernel_t(ajcp);
    }

    status_t create_kernel() {
        if (ker_) return ker_->create_kernel();
        return status::out_of_memory;
    }
    ~jit_uni_dw_conv_bwd_data_kernel() { delete ker_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_md,
            memory_desc_t &weights_md, memory_desc_t &diff_dst_md);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void operator()(const jit_conv_call_s *p) const { (*ker_)(p); }

private:
    using jit_kernel_t = typename utils::conditional<isa == avx512_core
                    && kernel_dt == data_type::bf16,
            jit_avx512_dw_conv_bwd_data_kernel_bf16,
            jit_uni_dw_conv_bwd_data_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_dw_conv_bwd_data_kernel);
};

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_bwd_weights_kernel {

    jit_uni_dw_conv_bwd_weights_kernel(const jit_conv_conf_t &ajcp)
        : ker_(nullptr) {
        ker_ = new jit_kernel_t(ajcp);
    }

    status_t create_kernel() {
        if (ker_) return ker_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_uni_dw_conv_bwd_weights_kernel() { delete ker_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
            memory_desc_t &diff_dst_md, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    static void partition_nthr_nxc(
            jit_conv_conf_t &jcp, int nthreads, bool prioritize_threading);
    static void balance(jit_conv_conf_t &jcp, int nthreads);

    void operator()(const jit_dw_conv_call_s *p) const { (*ker_)(p); }

private:
    using jit_kernel_t = typename utils::conditional<isa == avx512_core
                    && kernel_dt == data_type::bf16,
            jit_avx512_dw_conv_bwd_weights_kernel_bf16,
            jit_uni_dw_conv_bwd_weights_kernel_f32<isa>>::type;
    jit_kernel_t *ker_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif /* CPU_X64_JIT_UNI_DW_CONV_KERNEL_UTILS_HPP */
