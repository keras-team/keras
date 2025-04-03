/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_MATMUL_BRGEMM_MATMUL_COPY_UTILS_HPP
#define CPU_AARCH64_MATMUL_BRGEMM_MATMUL_COPY_UTILS_HPP

#include "cpu/aarch64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct jit_brgemm_matmul_copy_b_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *compensation_ptr;
        const void *zp_a_compensation_ptr;
        const void *zp_a_neg_value_ptr;

        dim_t current_K_start;
        dim_t current_K_iters;
        dim_t current_N_blk;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_matmul_copy_b_t(const brgemm_matmul_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_matmul_copy_b_t() {}

    const brgemm_matmul_conf_t *conf_;
};

struct jit_brgemm_matmul_copy_a_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *zp_b_compensation_buffer_ptr;
        const void *zp_a_compensation_result_ptr;
        const void *zp_b_neg_value_ptr;
        const void *zp_ab_comp_ptr;

        dim_t current_K_start;
        dim_t current_K_blk;
        dim_t current_M_blk;
        dim_t dynamic_src_ld;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_matmul_copy_a_t(const brgemm_matmul_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_matmul_copy_a_t() {}

    const brgemm_matmul_conf_t *conf_;
};

status_t create_brgemm_matmul_copy_b(
        std::unique_ptr<jit_brgemm_matmul_copy_b_t> &copy_ker,
        const brgemm_matmul_conf_t *conf);

status_t create_brgemm_matmul_copy_a(
        std::unique_ptr<jit_brgemm_matmul_copy_a_t> &copy_ker,
        const brgemm_matmul_conf_t *conf);

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
