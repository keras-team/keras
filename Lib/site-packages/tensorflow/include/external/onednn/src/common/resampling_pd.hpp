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

#ifndef COMMON_RESAMPLING_PD_HPP
#define COMMON_RESAMPLING_PD_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_RESAMPLING(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, resampling, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_RESAMPLING_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, resampling, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct resampling_fwd_pd_t;

struct resampling_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::resampling;

    resampling_pd_t(const resampling_desc_t *adesc,
            const primitive_attr_t *attr,
            const resampling_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}

    const resampling_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::alg_kind:
                *(alg_kind_t *)result = desc()->alg_kind;
                break;
            case query::factors:
                *(const float **)result = desc()->factors;
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common resampling aux functions */

    dim_t MB() const { return src_desc().dims[0]; }
    dim_t C() const { return src_desc().dims[1]; }
    dim_t ID() const { return ndims() >= 5 ? src_desc().dims[ndims() - 3] : 1; }
    dim_t IH() const { return ndims() >= 4 ? src_desc().dims[ndims() - 2] : 1; }
    dim_t IW() const { return ndims() >= 3 ? src_desc().dims[ndims() - 1] : 1; }
    dim_t OD() const { return ndims() >= 5 ? dst_desc().dims[ndims() - 3] : 1; }
    dim_t OH() const { return ndims() >= 4 ? dst_desc().dims[ndims() - 2] : 1; }
    dim_t OW() const { return ndims() >= 3 ? dst_desc().dims[ndims() - 1] : 1; }

    float FD() const { return ndims() >= 5 ? desc()->factors[ndims() - 5] : 1; }
    float FH() const { return ndims() >= 4 ? desc()->factors[ndims() - 4] : 1; }
    float FW() const { return ndims() >= 3 ? desc()->factors[ndims() - 3] : 1; }

    int ndims() const { return src_desc().ndims; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(src_desc()).has_zero_dim();
    }

    int n_inputs() const override { return 1 + n_binary_po_inputs(); }
    int n_outputs() const override { return 1; }

protected:
    resampling_desc_t desc_;
    const resampling_fwd_pd_t *hint_fwd_pd_;

private:
    const memory_desc_t &src_desc() const {
        return is_fwd() ? desc_.src_desc : desc_.diff_src_desc;
    }
    const memory_desc_t &dst_desc() const {
        return is_fwd() ? desc_.dst_desc : desc_.diff_dst_desc;
    }
};

struct resampling_fwd_pd_t : public resampling_pd_t {
    typedef resampling_fwd_pd_t base_class;
    typedef resampling_fwd_pd_t hint_class;

    resampling_fwd_pd_t(const resampling_desc_t *adesc,
            const primitive_attr_t *attr,
            const resampling_fwd_pd_t *hint_fwd_pd)
        : resampling_pd_t(adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc) {}

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return resampling_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

protected:
    memory_desc_t src_md_;
    memory_desc_t dst_md_;

    virtual status_t set_default_params(
            format_tag_t src_tag_hint = format_tag::undef) {
        if (dst_md()->format_kind != format_kind::any) return status::success;

        if (src_md()->format_kind != format_kind::blocked)
            return status::unimplemented;

        if (src_tag_hint != format_tag::undef) {
            return memory_desc_init_by_tag(dst_md_, src_tag_hint);
        } else {
            return memory_desc_init_by_blocking_desc(
                    dst_md_, src_md_.format_desc.blocking);
        }
    }
};

struct resampling_bwd_pd_t : public resampling_pd_t {
    typedef resampling_bwd_pd_t base_class;
    typedef resampling_fwd_pd_t hint_class;

    resampling_bwd_pd_t(const resampling_desc_t *adesc,
            const primitive_attr_t *attr,
            const resampling_fwd_pd_t *hint_fwd_pd)
        : resampling_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_DIFF_DST) return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return resampling_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *diff_src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_src_desc : &diff_src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_dst_desc : &diff_dst_md_;
        return &glob_zero_md;
    }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t diff_dst_md_;

    virtual status_t set_default_params() {
        if (diff_dst_md()->format_kind == format_kind::any && hint_fwd_pd_) {
            status_t status = memory_desc_init_by_md_and_dt(diff_dst_md_,
                    *hint_fwd_pd_->dst_md(0), diff_dst_md_.data_type);
            if (status != status::success) return status;
        }

        if (diff_src_md()->format_kind != format_kind::any)
            return status::success;

        if (diff_dst_md()->format_kind != format_kind::blocked)
            return status::unimplemented;
        return memory_desc_init_by_blocking_desc(
                diff_src_md_, diff_dst_md_.format_desc.blocking);
    }
};

} // namespace impl
} // namespace dnnl
#endif
