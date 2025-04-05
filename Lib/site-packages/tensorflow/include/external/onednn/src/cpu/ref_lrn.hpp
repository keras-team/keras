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

#ifndef CPU_REF_LRN_HPP
#define CPU_REF_LRN_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_lrn_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t d_type>
struct ref_lrn_fwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_fwd_pd_t {
        using cpu_lrn_fwd_pd_t::cpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_fwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_LRN(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LRN(utils::everyone_is(d_type, src_md()->data_type,
                                  dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LRN(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LRN(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");

            dat_tag_ = memory_desc_matches_one_of_tag(
                    *src_md(), nChw16c, nChw8c, nchw, nhwc);

            return status::success;
        }

        format_tag_t dat_tag_;
    };

    ref_lrn_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<d_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace format_tag;
        switch (pd()->dat_tag_) {
            case nChw16c: return execute_forward<nChw16c>(ctx); break;
            case nChw8c: return execute_forward<nChw8c>(ctx); break;
            case nchw: return execute_forward<nchw>(ctx); break;
            case nhwc: return execute_forward<nhwc>(ctx); break;
            default: return execute_forward<any>(ctx);
        }
        return status::success;
    }

private:
    template <format_tag_t tag>
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <impl::data_type_t d_type>
struct ref_lrn_bwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_bwd_pd_t {
        using cpu_lrn_bwd_pd_t::cpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_bwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            VDISPATCH_LRN(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LRN(
                    utils::everyone_is(d_type, src_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LRN(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LRN(diff_dst_d == diff_src_d, VERBOSE_INCONSISTENT_MDS,
                    "diff_src", "diff_dst");

            dat_tag_ = memory_desc_matches_one_of_tag(
                    *src_md(), nChw16c, nChw8c, nchw, nhwc);

            return status::success;
        }

        format_tag_t dat_tag_;
    };

    ref_lrn_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<d_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace format_tag;
        switch (pd()->dat_tag_) {
            case nChw16c: return execute_backward<nChw16c>(ctx); break;
            case nChw8c: return execute_backward<nChw8c>(ctx); break;
            case nchw: return execute_backward<nchw>(ctx); break;
            case nhwc: return execute_backward<nhwc>(ctx); break;
            default: return execute_backward<any>(ctx);
        }
        return status::success;
    }

private:
    template <format_tag_t tag>
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
