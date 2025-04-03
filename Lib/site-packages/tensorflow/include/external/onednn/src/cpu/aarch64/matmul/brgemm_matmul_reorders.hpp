/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef CPU_AARCH64_MATMUL_BRGEMM_MATMUL_REORDERS_HPP
#define CPU_AARCH64_MATMUL_BRGEMM_MATMUL_REORDERS_HPP

#include "cpu/aarch64/matmul/brgemm_matmul_copy_utils.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct brgemm_matmul_matrix_B_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("brgemm_matmul_matrix_B_reorder_t",
                brgemm_matmul_matrix_B_reorder_t);

        // required to re-use brgemm matmul copy_b jit kernels
        matmul::brgemm_matmul_conf_t matmul_conf_for_reorder_;
        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        void init_scratchpad() {}
        friend dnnl::impl::impl_list_item_t;
    };

    brgemm_matmul_matrix_B_reorder_t(const pd_t *apd) : primitive_t(apd) {}
    status_t init(engine_t *engine) override {
        CHECK(matmul::create_brgemm_matmul_copy_b(
                kernel_, &pd()->matmul_conf_for_reorder_));

        return status::success;
    }

private:
    status_t execute_body(const exec_ctx_t &ctx) const;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_body(ctx);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
