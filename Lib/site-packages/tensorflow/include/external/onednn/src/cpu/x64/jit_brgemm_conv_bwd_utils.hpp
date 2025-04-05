/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_BWD_UTILS_HPP
#define CPU_X64_JIT_BRGEMM_CONV_BWD_UTILS_HPP

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

namespace brgemm_convolution_bwd_utils {

constexpr size_t P4K = 4096;

bool is_amx(cpu_isa_t isa);

void set_k_range(int P, int D, int S, dim_t i, dim_t O, int K, int &k_s,
        int &k_f, bool is_w = false);

void get_iw_range(const jit_brgemm_conv_conf_t &jcp, int iw, int iw_raw, int kw,
        int &iw_s, int &M_without_overflow);

void get_kw_range(const jit_brgemm_conv_conf_t &jcp, int iw, int iw_raw,
        int &kw_s, int &kw_full_s, int &kw_full_f, int &kw_f);

dim_t precalculate_comp_pad_kernels(const jit_brgemm_conv_conf_t &jcp,
        std::vector<dim_t> *kd_bs = nullptr,
        std::vector<dim_t> *kd_es = nullptr,
        std::vector<dim_t> *kh_bs = nullptr,
        std::vector<dim_t> *kh_es = nullptr,
        std::vector<dim_t> *kw_bs = nullptr,
        std::vector<dim_t> *kw_es = nullptr);

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads,
        bool enable_postops);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp);

} // namespace brgemm_convolution_bwd_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
