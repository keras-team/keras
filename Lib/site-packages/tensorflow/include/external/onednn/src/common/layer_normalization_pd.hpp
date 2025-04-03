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

#ifndef COMMON_LAYER_NORMALIZATION_PD_HPP
#define COMMON_LAYER_NORMALIZATION_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_LNORM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, layer_normalization, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_LNORM_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, layer_normalization, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct layer_normalization_fwd_pd_t;

struct layer_normalization_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::layer_normalization;

    const layer_normalization_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::primitive_kind:
                *(primitive_kind_t *)result = desc_.primitive_kind;
                break;
            case query::epsilon_f32:
                *(float *)result = desc()->layer_norm_epsilon;
                break;
            case query::flags: *(uint32_t *)result = desc()->flags; break;

            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common layer_normalization aux functions */
    int ndims() const { return desc_.src_desc.ndims; }
    dim_t across_axis() const {
        return utils::array_product(desc_.src_desc.dims, ndims() - 1);
    }
    dim_t norm_axis() const { return desc_.src_desc.dims[ndims() - 1]; }

    bool stats_are_src() const {
        return desc_.flags & normalization_flags::use_global_stats;
    }
    bool stats_are_tmp() const { return !(stats_are_src() || is_training()); }

    bool use_scale() const {
        return desc_.flags & normalization_flags::use_scale;
    }
    bool use_shift() const {
        return desc_.flags & normalization_flags::use_shift;
    }
    bool use_global_stats() const {
        return desc_.flags & normalization_flags::use_global_stats;
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }
    bool is_training() const {
        return desc_.prop_kind == prop_kind::forward_training;
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(desc_.src_desc).has_zero_dim();
    }

    const memory_desc_t *stat_md() const { return &stat_md_; }

protected:
    layer_normalization_desc_t desc_;
    const layer_normalization_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t src_md_;
    memory_desc_t stat_md_;
    memory_desc_t scaleshift_md_;

    layer_normalization_pd_t(const layer_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const layer_normalization_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , stat_md_(desc_.stat_desc)
        , scaleshift_md_(desc_.data_scaleshift_desc) {}

    bool set_default_stat_md_format(const memory_desc_t &src_md) {
        if (stat_md_.format_kind != format_kind::any) return true;

        // src memory desc in non-blocked memory format is unsupported
        if (src_md.format_kind != format_kind::blocked) return false;

        // if the normalization axis is blocked, fallback to plain format
        bool is_norm_dim_blocked = false;
        for (int d = 0; d < src_md.format_desc.blocking.inner_nblks; ++d)
            is_norm_dim_blocked
                    |= src_md.format_desc.blocking.inner_idxs[d] == ndims() - 1;
        if (is_norm_dim_blocked)
            return memory_desc_init_by_strides(stat_md_, nullptr)
                    == status::success;

        // the default memory format for stat is derived from src_md by
        // dropping the normalization dimension and keeping the physical order
        // of other dimensions (preserving the blocked structure if any)
        return memory_desc_init_by_blocking_desc(
                       stat_md_, src_md.format_desc.blocking)
                == status::success;
    }

    // Stats and src here are compatible if:
    // `stat_strides[:] == data_strides[:] / last_data_dimension`
    // i.e. abcd & abc, bacd & bac - compatible
    status_t fill_compatible_stats_md(
            const memory_desc_t &src_md, memory_desc_t &stat_md) {
        stat_md = src_md;
        stat_md.data_type = dnnl_f32;
        stat_md.ndims -= 1;
        return memory_desc_init_by_blocking_desc(
                stat_md, src_md.format_desc.blocking);
    }

private:
    const memory_desc_t &src_desc() const { return desc_.src_desc; }
};

struct layer_normalization_fwd_pd_t : public layer_normalization_pd_t {
    typedef layer_normalization_fwd_pd_t base_class;
    typedef layer_normalization_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;
        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        if (utils::one_of(arg, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE)) {
            if (stats_are_src()) return arg_usage_t::input;
            if (!stats_are_src() && is_training()) return arg_usage_t::output;
            return arg_usage_t::unused;
        }

        if (arg == DNNL_ARG_SCALE && use_scale()) return arg_usage_t::input;
        if (arg == DNNL_ARG_SHIFT && use_shift()) return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            case DNNL_ARG_MEAN: return stats_are_src() ? src_md(1) : dst_md(1);
            case DNNL_ARG_VARIANCE:
                return stats_are_src() ? src_md(2) : dst_md(2);
            case DNNL_ARG_SCALE:
            case DNNL_ARG_SHIFT: return weights_md(0);
            default: return layer_normalization_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        if (stats_are_src() && (index == 1 || index == 2)) return &stat_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        if (!stats_are_src() && is_training() && (index == 1 || index == 2))
            return &stat_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &scaleshift_md_ : &glob_zero_md;
    }

    int n_inputs() const override {
        return 1 + 2 * stats_are_src() + use_scale() + use_shift()
                + n_binary_po_inputs();
    }
    int n_outputs() const override {
        return 1 + 2 * (!stats_are_src()) * is_training();
    }

protected:
    memory_desc_t dst_md_;

    layer_normalization_fwd_pd_t(const layer_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const layer_normalization_fwd_pd_t *hint_fwd_pd)
        : layer_normalization_pd_t(adesc, attr, hint_fwd_pd)
        , dst_md_(desc_.dst_desc) {}

    bool set_default_formats_common() {
        return IMPLICATION(dst_md_.format_kind == format_kind::any,
                       memory_desc_init_by_md_and_dt(
                               dst_md_, src_md_, dst_md_.data_type)
                               == status::success)
                && set_default_stat_md_format(src_md_);
    }

    bool check_scale_shift_data_type(
            std::initializer_list<data_type_t> supported_dts
            = {data_type::f32}) const {
        if (!use_scale() && !use_shift()) return true;

        for (auto dt : supported_dts)
            if (weights_md()->data_type == dt) return true;
        return false;
    }

    bool attr_scales_ok() const {
        const auto &scales = attr()->scales_;
        bool ok = true;
        for (const auto &e : scales.scales_) {
            ok = ok && e.second.mask_ == 0;
        }
        return ok;
    }
};

struct layer_normalization_bwd_pd_t : public layer_normalization_pd_t {
    typedef layer_normalization_bwd_pd_t base_class;
    typedef layer_normalization_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE,
                    DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_SCALE && use_scale()) return arg_usage_t::input;
        if (arg == DNNL_ARG_SHIFT && use_shift()) return arg_usage_t::input;

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
            case DNNL_ARG_SCALE:
            case DNNL_ARG_SHIFT: return weights_md(0);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            case DNNL_ARG_DIFF_SCALE:
            case DNNL_ARG_DIFF_SHIFT: return diff_weights_md(0);
            default: return layer_normalization_pd_t::arg_md(arg);
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

    int n_inputs() const override { return 4 + use_scale() + use_shift(); }
    int n_outputs() const override {
        return 1
                + (desc_.prop_kind == prop_kind::backward)
                * (use_scale() + use_shift());
    }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t diff_dst_md_;
    memory_desc_t diff_scaleshift_md_;

    layer_normalization_bwd_pd_t(const layer_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const layer_normalization_fwd_pd_t *hint_fwd_pd)
        : layer_normalization_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , diff_dst_md_(desc_.diff_dst_desc)
        , diff_scaleshift_md_(desc_.diff_data_scaleshift_desc) {}

    bool set_default_formats_common() {
        return IMPLICATION(diff_dst_md_.format_kind == format_kind::any,
                       memory_desc_init_by_md_and_dt(
                               diff_dst_md_, src_md_, diff_dst_md_.data_type)
                               == status::success)
                && IMPLICATION(diff_src_md_.format_kind == format_kind::any,
                        memory_desc_init_by_md_and_dt(
                                diff_src_md_, src_md_, diff_src_md_.data_type)
                                == status::success)
                && set_default_stat_md_format(diff_src_md_);
    }

    bool check_scale_shift_data_type(
            std::initializer_list<data_type_t> supported_dts
            = {data_type::f32}) const {
        if (!use_scale() && !use_shift()) return true;
        if (weights_md()->data_type != diff_weights_md()->data_type)
            return false;

        for (auto dt : supported_dts)
            if (weights_md()->data_type == dt) return true;
        return false;
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
