/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#ifndef COMMON_SHUFFLE_PD_HPP
#define COMMON_SHUFFLE_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

#define VDISPATCH_SHUFFLE(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, shuffle, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_SHUFFLE_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, shuffle, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct shuffle_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::shuffle;

    typedef shuffle_pd_t base_class;
    typedef shuffle_pd_t hint_class;

    const shuffle_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::axis_s32: *(int *)result = desc()->axis; break;
            case query::group_size_s64:
                *(dim_t *)result = desc()->group_size;
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    arg_usage_t arg_usage(int arg) const override {
        if (is_fwd()) {
            if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

            if (arg == DNNL_ARG_DST) return arg_usage_t::output;
        } else {
            if (arg == DNNL_ARG_DIFF_DST) return arg_usage_t::input;

            if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;
        }

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0 && is_fwd())
            return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0 && is_fwd())
            return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *diff_src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0 && !is_fwd())
            return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0 && !is_fwd())
            return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override { return 1; }

    /* shuffle aux functions */

    dim_t MB() const { return data_md()->dims[0]; }
    dim_t C() const { return ndims() >= 2 ? data_md()->dims[1] : 1; }
    dim_t D() const { return ndims() >= 5 ? data_md()->dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? data_md()->dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? data_md()->dims[ndims() - 1] : 1; }

    int ndims() const { return data_md()->ndims; }

    int axis() const { return desc_.axis; }
    dim_t group_size() const { return desc_.group_size; }
    dim_t axis_size() const { return data_md()->dims[axis()]; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    std::vector<memory_desc_t> hint_mds(bool is_hint) const override {
        assert(IMPLICATION(is_hint, is_fwd()));
        if (!is_hint && is_fwd()) return {};
        if (is_hint && is_fwd()) return {*dst_md(0)};
        return hint_mds_;
    }

protected:
    shuffle_desc_t desc_;
    const shuffle_pd_t *hint_fwd_pd_;
    memory_desc_t src_md_;
    memory_desc_t dst_md_;

    shuffle_pd_t(const shuffle_desc_t *adesc, const primitive_attr_t *attr,
            const shuffle_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc) {
        if (hint_fwd_pd_) hint_mds_.push_back(*hint_fwd_pd_->dst_md(0));
    }

    bool set_default_formats_common() {
        // `src_md_.format_kind == format_kind::any` may be true only for
        // backward prop kind.
        return IMPLICATION(src_md_.format_kind == format_kind::any,
                       hint_fwd_pd_
                               ? memory_desc_init_by_md_and_dt(src_md_,
                                         hint_mds(/* is_hint = */ false)[0],
                                         src_md_.data_type)
                                       == status::success
                               : memory_desc_init_by_strides(src_md_, nullptr)
                                       == status::success)
                && IMPLICATION(dst_md_.format_kind == format_kind::any,
                        memory_desc_init_by_md_and_dt(
                                dst_md_, src_md_, dst_md_.data_type)
                                == status::success);
    }

private:
    std::vector<memory_desc_t> hint_mds_;

    const memory_desc_t *data_md() const {
        return is_fwd() ? src_md() : diff_src_md();
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
