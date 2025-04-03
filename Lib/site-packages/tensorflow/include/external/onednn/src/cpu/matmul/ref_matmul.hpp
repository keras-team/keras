/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef CPU_MATMUL_REF_MATMUL_HPP
#define CPU_MATMUL_REF_MATMUL_HPP

#include <assert.h>

#include "common/bfloat16.hpp"
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

struct ref_matmul_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            bool ok = is_dense_format_kind()
                    && utils::one_of(src_type, f32, bf16, f16, f8_e5m2, f8_e4m3)
                    && utils::one_of(wei_type, f32, bf16, f16, f8_e5m2, f8_e4m3,
                            u8, s8, u4, s4)
                    && utils::one_of(dst_type, f32, bf16, f16, f8_e5m2, f8_e4m3)
                    && (src_type == wei_type
                            || utils::one_of(wei_type, u8, s8, u4, s4))
                    /* int8 weights decompression support */
                    && IMPLICATION(utils::one_of(wei_type, u8, s8),
                            attr_.mayiconvert(wei_type, src_type))
                    && IMPLICATION(src_type == f32, dst_type == f32)
                    && IMPLICATION(src_type == bf16,
                            utils::one_of(dst_type, f32, bf16))
                    && IMPLICATION(
                            src_type == f16, utils::one_of(dst_type, f32, f16))
                    // TODO: any implication on allowed dst data type for fp8?
                    && IMPLICATION(with_bias(),
                            utils::one_of(
                                    bia_type, f32, bf16, f16, f8_e5m2, f8_e4m3)
                                    && IMPLICATION(
                                            src_type == f32, bia_type == f32)
                                    && IMPLICATION(src_type == f16,
                                            utils::one_of(bia_type, f32, f16))
                                    && IMPLICATION(src_type == bf16,
                                            utils::one_of(bia_type, f32, bf16))
                            // TODO: any implication on allowed bias
                            // data type for fp8?
                            )
                    && platform::has_data_type_support(src_type)
                    && attr()->has_default_values(
                            smask_t::scales_runtime_data_type
                                    | smask_t::scales_runtime_groups
                                    | smask_t::zero_points_runtime_data_type
                                    | smask_t::zero_points_runtime_groups
                                    | smask_t::post_ops | smask_t::sum_dt
                                    | smask_t::fpmath_mode,
                            dst_type)
                    && attr_.post_ops_.check_sum_consistency(dst_type,
                            /* is_int8 */ false)
                    && ref_post_ops_t::primitive_kind_ok(attr()->post_ops_)
                    && attr_scales_ok() && set_default_formats()
                    && zero_points_ok()
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            return ok ? status::success : status::unimplemented;
        }

        virtual bool attr_scales_ok(
                const std::vector<int> &supported_args = {DNNL_ARG_SRC,
                        DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) const override {
            if (attr()->scales_.has_default_values()) return true;

            bool ok = attr()->scales_.has_default_values(supported_args);
            for (int arg : supported_args) {
                const auto &sc = attr()->scales_.get(arg);
                const auto &mask = sc.mask_;
                if (!sc.has_default_values()) {
                    if (arg == DNNL_ARG_WEIGHTS) {
                        ok = ok
                                && utils::one_of(mask, 0, wei_qmask_N(),
                                        wei_qmask_N() + wei_qmask_K());
                        ok = ok && utils::one_of(sc.ndims_, 0, 2)
                                && IMPLICATION(sc.ndims_ == 2,
                                        sc.group_dims_[1] == 1
                                                && K() % sc.group_dims_[0]
                                                        == 0);
                    } else
                        ok = ok && (mask == 0);
                }
            }
            return ok;
        }

        bool zero_points_ok() const {
            /* weights decompression requires zero points support */
            int mask_wei = 0;
            attr()->zero_points_.get(DNNL_ARG_WEIGHTS, &mask_wei);
            const auto wei_group_ndims
                    = attr()->zero_points_.get_groups_ndims(DNNL_ARG_WEIGHTS);
            const auto wei_group_dims
                    = attr()->zero_points_.get_groups(DNNL_ARG_WEIGHTS);

            return attr()->zero_points_.has_default_values(DNNL_ARG_SRC)
                    && attr()->zero_points_.has_default_values(DNNL_ARG_DST)
                    && utils::one_of(mask_wei, 0, wei_qmask_N(),
                            wei_qmask_N() + wei_qmask_K())
                    && utils::one_of(wei_group_ndims, 0, 2)
                    && IMPLICATION(wei_group_ndims == 2,
                            wei_group_dims[1] == 1
                                    && K() % wei_group_dims[0] == 0);
        }
    };

    ref_matmul_t(const pd_t *apd) : primitive_t(apd) {}

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
