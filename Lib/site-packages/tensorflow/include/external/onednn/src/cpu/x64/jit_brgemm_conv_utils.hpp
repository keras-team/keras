/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_UTILS_HPP
#define CPU_X64_JIT_BRGEMM_CONV_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace brgemm_convolution_utils {

constexpr size_t P4K = 4096;

bool is_amx(cpu_isa_t isa);
bool uses_batch_elements(
        brgemm_batch_kind_t brg_type, conv_brgemm_exec_type_t exec_type);

status_t init_conf(jit_brgemm_conv_conf_t &jcp, bool use_inversion,
        cpu_isa_t isa, const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads);

status_t init_1x1_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads);

void set_amx_wsp_per_thread(jit_brgemm_conv_conf_t &jcp);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp);

status_t init_conf_bwd_w(jit_brgemm_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
        memory_desc_t &diff_dst_md, primitive_attr_t &attr, int nthreads);

status_t init_scratchpad_bwd_w(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp, memory_desc_t &src_md,
        memory_desc_t &diff_weights_md, memory_desc_t &diff_dst_md);

} // namespace brgemm_convolution_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
