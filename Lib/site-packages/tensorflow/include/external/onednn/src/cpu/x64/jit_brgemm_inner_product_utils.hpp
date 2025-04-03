/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_X64_BRGEMM_INNER_PRODUCT_UTILS_HPP

#include "dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace brgemm_inner_product_utils {

// Common for fwd/bwd_d/bwd_w.
struct jit_brgemm_ip_conf_t : jit_brgemm_primitive_conf_t {
    // Use kernels and blocking for small os that consume less bandwidth.
    bool use_small_os_kernels = false;
    bool has_spatial_dims() const;

protected:
    status_t init_conf_base(cpu_isa_t isa, const inner_product_desc_t &ipd,
            memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md, memory_desc_t &bias_md,
            primitive_attr_t &attr, int nthreads);

    void init_scratchpad_base(memory_tracking::registrar_t &scratchpad) const;

    int get_os_block(bool try_to_adjust, bool is_adjustment) const;
    int get_oc_block(bool try_to_adjust = false) const;
    std::unordered_map<int, format_tag_t> get_desired_weights_tag() const;

    int get_adjusted_oc_block() const;
    int get_nb_oc_blocking(bool is_adjustment = false) const;
    bool adjust_thread_balance() const;

    format_tag_t get_brgemm_ip_weights_tag(
            const memory_desc_t &weights_md) const;
};

// Specific for forward.
struct jit_brgemm_ip_fwd_conf_t : jit_brgemm_ip_conf_t {
    enum loop_order_t {
        osc_occ_icc_osb_ocb,
        osc_occ_osb_ocb_icc,
        icc_osc_occ_osb_ocb,
        icc_occ_osc_ocb_osb,
    } loop_order
            = osc_occ_osb_ocb_icc;

    status_t init_conf(cpu_isa_t isa, const inner_product_desc_t &ipd,
            memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md, memory_desc_t &bias_md,
            primitive_attr_t &attr, int nthreads);

    void init_scratchpad(memory_tracking::registrar_t &scratchpad) const;

private:
    void choose_loop_order();
};

// Specific for backward by data.
struct jit_brgemm_ip_bwd_d_conf_t : jit_brgemm_ip_conf_t {
    bool global_b_transpose;

    status_t init_conf(cpu_isa_t isa, const inner_product_desc_t &ipd,
            memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md, memory_desc_t &bias_md,
            primitive_attr_t &attr, int nthreads);

    void init_scratchpad(memory_tracking::registrar_t &scratchpad) const;
};

// Specific for backward by weights.
struct jit_brgemm_ip_bwd_w_conf_t : jit_brgemm_ip_conf_t {
    enum loop_order_t {
        osc_icc_occ,
        osc_occ_icc,
        occ_icc_osc,
    } loop_order
            = occ_icc_osc;

    bool local_buffers_for_input_tensors;

    status_t init_conf(cpu_isa_t isa, const inner_product_desc_t &ipd,
            memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md, memory_desc_t &bias_md,
            primitive_attr_t &attr, int nthreads);

    void init_scratchpad(memory_tracking::registrar_t &scratchpad) const;

private:
    void thread_balance(int &nb_os_blocking_, int &nb_oc_blocking_,
            int &nb_ic_blocking_, int &nthr_, int &nthr_mb_, int &nthr_oc_b_,
            int &nthr_ic_b_) const;

    void choose_loop_order();
};

static const int max_num_brg_kernels_ip = 2 * 2 * 2 * 2 * 2;

int get_brg_kernel_index(bool is_bs_tail, bool do_initialization,
        bool is_M_tail, bool is_N_tail, bool is_K_tail);

size_t buf_dt_size(data_type_t dt, cpu_isa_t isa);

} // namespace brgemm_inner_product_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
