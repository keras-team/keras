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

#ifndef CPU_MATMUL_REF_MATMUL_INT8_HPP
#define CPU_MATMUL_REF_MATMUL_INT8_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct ref_matmul_int8_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_matmul_int8_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            bool ok = is_dense_format_kind() && utils::one_of(src_type, s8, u8)
                    && wei_type == s8
                    && IMPLICATION(with_bias(),
                            utils::one_of(bia_type, f32, bf16, s32, s8, u8))
                    && utils::one_of(dst_type, f32, bf16, s32, s8, u8)
                    && attr()->has_default_values(smask_t::scales_runtime
                                    | smask_t::zero_points_runtime
                                    | smask_t::post_ops | smask_t::sum_dt,
                            dst_type)
                    && attr_.post_ops_.check_sum_consistency(dst_type,
                            /* is_int8 */ true)
                    && ref_post_ops_t::primitive_kind_ok(attr()->post_ops_)
                    && attr_scales_ok() && attr_zero_points_ok()
                    && set_default_formats()
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            return ok ? status::success : status::unimplemented;
        }

    private:
        bool attr_zero_points_ok() const {
            int mask_src = 0, mask_wei = 0, mask_dst = 0;
            attr()->zero_points_.get(DNNL_ARG_SRC, &mask_src);
            attr()->zero_points_.get(DNNL_ARG_WEIGHTS, &mask_wei);
            attr()->zero_points_.get(DNNL_ARG_DST, &mask_dst);

            return (mask_src == 0 || (ndims() == 2 && mask_src == 1 << 1))
                    && (mask_wei == 0)
                    && (mask_dst == 0 || (ndims() == 2 && mask_dst == 1 << 1));
        }
    };

    ref_matmul_int8_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
