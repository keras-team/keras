/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#ifndef CPU_REF_ELTWISE_HPP
#define CPU_REF_ELTWISE_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_eltwise_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_eltwise_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(utils::everyone_is(data_type, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(platform::has_data_type_support(data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(attr()->has_default_values(sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    ref_post_ops_t::primitive_kind_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            use_dense_ = src_d.is_dense(true) && dst_d.is_dense(true)
                    && IMPLICATION(!src_d.is_dense() || !dst_d.is_dense(),
                            is_zero_preserved());

            use_nCspBc_padded_ = !use_dense_
                    && src_d.blocking_desc().inner_nblks == 1
                    && one_of(src_d.blocking_desc().inner_blks[0], 8, 16)
                    && src_d.blocking_desc().inner_idxs[0] == 1
                    && src_d.only_padded_dim(1) && src_d.is_dense(true);

            const auto &po = attr()->post_ops_;
            if (has_zero_dim_memory() || !po.has_default_values())
                use_dense_ = use_nCspBc_padded_ = false;

            return status::success;
        }

        bool use_dense_, use_nCspBc_padded_;
    };

    ref_eltwise_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    using data_t = typename prec_traits<data_type>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->use_dense_)
            return execute_forward_dense(ctx);
        else if (pd()->use_nCspBc_padded_)
            return execute_forward_nCspBc_padded(ctx);
        else
            return execute_forward_generic(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward_nCspBc_padded(const exec_ctx_t &ctx) const;
    status_t execute_forward_dense(const exec_ctx_t &ctx) const;
    status_t execute_forward_generic(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

template <impl::data_type_t data_type>
struct ref_eltwise_bwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_bwd_pd_t {
        using cpu_eltwise_bwd_pd_t::cpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_eltwise_bwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace data_type;

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(
                    utils::everyone_is(data_type, data_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(platform::has_data_type_support(data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(diff_dst_d == diff_src_d,
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");

            use_dense_ = diff_dst_d.is_dense()
                    || (diff_dst_d.is_dense(true) && is_zero_preserved());

            if (has_zero_dim_memory()) use_dense_ = false;
            if (diff_dst_d != memory_desc_wrapper(data_md()))
                use_dense_ = false;

            if (utils::one_of(data_type, bf16, f16)) init_scratchpad();

            return status::success;
        }

        bool use_dense_;

    private:
        void init_scratchpad() {
            const memory_desc_wrapper data_d(data_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            const auto diff_dst_size = diff_dst_d.nelems(true);
            scratchpad.template book<float>(
                    key_eltwise_src, data_d.nelems(true));
            scratchpad.template book<float>(
                    key_eltwise_diff_dst, diff_dst_size);
        }
    };

    ref_eltwise_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->use_dense_)
            return execute_backward_dense(ctx);
        else
            return execute_backward_generic(ctx);
    }

private:
    status_t execute_backward_dense(const exec_ctx_t &ctx) const;
    status_t execute_backward_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
