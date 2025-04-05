/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef COMMON_GROUP_NORMALIZATION_PD_HPP
#define COMMON_GROUP_NORMALIZATION_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

#define VDISPATCH_GNORM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, group_normalization, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct group_normalization_fwd_pd_t;

struct group_normalization_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::group_normalization;

    const group_normalization_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::group_size_s64:
                *(dim_t *)result = desc()->groups;
                break;
            case query::epsilon_f32:
                *(float *)result = desc()->group_norm_epsilon;
                break;
            case query::flags: *(uint32_t *)result = desc()->flags; break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    dim_t MB() const { return src_md()->dims[0]; }
    dim_t C() const { return src_md()->dims[1]; }
    dim_t G() const { return stat_md()->dims[1]; }
    dim_t D() const { return ndims() >= 5 ? src_md()->dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? src_md()->dims[ndims() - 2] : 1; }
    dim_t W() const { return src_md()->dims[ndims() - 1]; }

    int ndims() const { return src_md()->ndims; }

    const memory_desc_t *stat_md() const { return &stat_md_; }

    bool stats_is_src() const {
        return desc_.flags & normalization_flags::use_global_stats;
    }
    bool use_scale() const {
        return desc_.flags & normalization_flags::use_scale;
    }
    bool use_shift() const {
        return desc_.flags & normalization_flags::use_shift;
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training);
    }

    bool is_training() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::backward);
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(src_md()).has_zero_dim();
    }

protected:
    group_normalization_desc_t desc_;
    const group_normalization_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t src_md_;
    memory_desc_t stat_md_;
    memory_desc_t scaleshift_md_;

    group_normalization_pd_t(const group_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const group_normalization_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , stat_md_(desc_.stat_desc)
        , scaleshift_md_(desc_.scaleshift_desc) {}
};

struct group_normalization_fwd_pd_t : public group_normalization_pd_t {
    typedef group_normalization_fwd_pd_t base_class;
    typedef group_normalization_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;
        if (utils::one_of(arg, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE)) {
            if (stats_is_src()) return arg_usage_t::input;
            if (!stats_is_src() && is_training()) return arg_usage_t::output;
            return arg_usage_t::unused;
        }

        if (arg == DNNL_ARG_SCALE && use_scale()) return arg_usage_t::input;
        if (arg == DNNL_ARG_SHIFT && use_shift()) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;
        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            case DNNL_ARG_MEAN: return stats_is_src() ? src_md(1) : dst_md(1);
            case DNNL_ARG_VARIANCE:
                return stats_is_src() ? src_md(2) : dst_md(2);
            case DNNL_ARG_SCALE:
            case DNNL_ARG_SHIFT: return weights_md(0);
            default: return group_normalization_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        if (stats_is_src() && (index == 1 || index == 2)) return &stat_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        if (!stats_is_src() && is_training() && (index == 1 || index == 2))
            return &stat_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &scaleshift_md_ : &glob_zero_md;
    }

    int n_inputs() const override {
        return 1 + 2 * stats_is_src() + use_scale() + use_shift()
                + n_binary_po_inputs();
    }
    int n_outputs() const override {
        return 1 + (2 * (!stats_is_src())) * is_training();
    }

protected:
    memory_desc_t dst_md_;

    group_normalization_fwd_pd_t(const group_normalization_desc_t *adesc,
            const primitive_attr_t *attr, const hint_class *hint_fwd_pd)
        : group_normalization_pd_t(adesc, attr, hint_fwd_pd)
        , dst_md_(desc_.dst_desc) {}

    bool set_default_formats_common() {
        return IMPLICATION(dst_md_.format_kind == format_kind::any,
                memory_desc_init_by_md_and_dt(
                        dst_md_, src_md_, dst_md_.data_type)
                        == status::success);
    }
    bool check_scale_shift_data_type() const {
        return IMPLICATION(use_scale() || use_shift(),
                weights_md()->data_type == data_type::f32);
    }
    bool attr_scales_ok() const {
        using namespace data_type;
        const auto &scales = attr()->scales_;
        const std::vector<int> supported_args({DNNL_ARG_SRC, DNNL_ARG_DST});
        bool ok = scales.has_default_values(supported_args);

        for (const auto &arg : supported_args) {
            const auto &sc = scales.get(arg);
            if (!sc.has_default_values()) {
                const data_type_t dt = arg_md(arg)->data_type;
                ok = ok && utils::one_of(dt, s8, u8) && sc.mask_ == 0;
            }
        }
        return ok;
    }
};

struct group_normalization_bwd_pd_t : public group_normalization_pd_t {
    typedef group_normalization_bwd_pd_t base_class;
    typedef group_normalization_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE,
                    DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_SCALE && use_scale()) return arg_usage_t::input;

        if (arg == DNNL_ARG_WORKSPACE && !types::is_zero_md(workspace_md()))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        if (arg == DNNL_ARG_DIFF_SCALE && use_scale())
            return arg_usage_t::output;
        if (arg == DNNL_ARG_DIFF_SHIFT && use_shift())
            return arg_usage_t::output;
        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_MEAN: return src_md(1);
            case DNNL_ARG_VARIANCE: return src_md(2);
            case DNNL_ARG_SCALE: return weights_md(0);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            case DNNL_ARG_DIFF_SCALE:
            case DNNL_ARG_DIFF_SHIFT: return diff_weights_md(0);
            default: return group_normalization_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        if (index == 1 || index == 2) return &stat_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_dst_desc : &diff_dst_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_src_desc : &diff_src_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &scaleshift_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_weights_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &diff_scaleshift_md_ : &glob_zero_md;
    }

    int n_inputs() const override { return 4 + use_scale(); }
    int n_outputs() const override {
        return 1
                + (!types::is_zero_md(diff_weights_md()))
                * (use_scale() + use_shift());
    }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t diff_dst_md_;
    memory_desc_t diff_scaleshift_md_;

    group_normalization_bwd_pd_t(const group_normalization_desc_t *adesc,
            const primitive_attr_t *attr, const hint_class *hint_fwd_pd)
        : group_normalization_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , diff_dst_md_(desc_.diff_dst_desc)
        , diff_scaleshift_md_(desc_.diff_scaleshift_desc) {}

    bool set_default_formats_common() {
        return IMPLICATION(diff_dst_md_.format_kind == format_kind::any,
                       memory_desc_init_by_md_and_dt(
                               diff_dst_md_, src_md_, diff_dst_md_.data_type)
                               == status::success)
                && IMPLICATION(diff_src_md_.format_kind == format_kind::any,
                        memory_desc_init_by_md_and_dt(
                                diff_src_md_, src_md_, diff_src_md_.data_type)
                                == status::success);
    }

    bool check_scale_shift_data_type() const {
        return IMPLICATION(use_scale() || use_shift(),
                utils::everyone_is(data_type::f32, weights_md()->data_type,
                        diff_weights_md()->data_type));
    }
};

} // namespace impl
} // namespace dnnl

#endif
