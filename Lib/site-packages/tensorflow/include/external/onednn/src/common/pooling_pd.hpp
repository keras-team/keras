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

#ifndef COMMON_POOLING_PD_HPP
#define COMMON_POOLING_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define VDISPATCH_POOLING(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, pooling, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_POOLING_IC(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, pooling, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

#define VDISPATCH_POOLING_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, pooling, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct pooling_fwd_pd_t;

struct pooling_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::pooling;

    const pooling_desc_t *desc() const { return &desc_; }
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
            case query::alg_kind:
                *(alg_kind_t *)result = desc()->alg_kind;
                break;
            case query::kernel:
                *(const dims_t **)result = &desc()->kernel;
                break;
            case query::strides:
                *(const dims_t **)result = &desc()->strides;
                break;
            case query::dilations:
                *(const dims_t **)result = &desc()->dilation;
                break;
            case query::padding_l:
                *(const dims_t **)result = &desc()->padding[0];
                break;
            case query::padding_r:
                *(const dims_t **)result = &desc()->padding[1];
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common pooling aux functions */

    dim_t MB() const { return src_desc().dims[0]; }
    dim_t IC() const { return src_desc().dims[1]; }
    dim_t OC() const { return IC(); }

    dim_t ID() const { return ndims() >= 5 ? src_desc().dims[ndims() - 3] : 1; }
    dim_t IH() const { return ndims() >= 4 ? src_desc().dims[ndims() - 2] : 1; }
    dim_t IW() const { return src_desc().dims[ndims() - 1]; }

    dim_t OD() const { return ndims() >= 5 ? dst_desc().dims[ndims() - 3] : 1; }
    dim_t OH() const { return ndims() >= 4 ? dst_desc().dims[ndims() - 2] : 1; }
    dim_t OW() const { return dst_desc().dims[ndims() - 1]; }

    dim_t KD() const { return ndims() >= 5 ? desc_.kernel[ndims() - 5] : 1; }
    dim_t KH() const { return ndims() >= 4 ? desc_.kernel[ndims() - 4] : 1; }
    dim_t KW() const { return desc_.kernel[ndims() - 3]; }

    dim_t KSD() const { return ndims() >= 5 ? desc_.strides[ndims() - 5] : 1; }
    dim_t KSH() const { return ndims() >= 4 ? desc_.strides[ndims() - 4] : 1; }
    dim_t KSW() const { return desc_.strides[ndims() - 3]; }

    dim_t KDD() const {
        return (ndims() >= 5 ? desc_.dilation[ndims() - 5] : 0);
    }
    dim_t KDH() const {
        return (ndims() >= 4 ? desc_.dilation[ndims() - 4] : 0);
    }
    dim_t KDW() const { return desc_.dilation[ndims() - 3]; }

    dim_t padFront() const {
        return ndims() >= 5 ? desc_.padding[0][ndims() - 5] : 0;
    }
    dim_t padBack() const {
        return ndims() >= 5 ? desc_.padding[1][ndims() - 5] : 0;
    }
    dim_t padT() const {
        return ndims() >= 4 ? desc_.padding[0][ndims() - 4] : 0;
    }
    dim_t padB() const {
        return ndims() >= 4 ? desc_.padding[1][ndims() - 4] : 0;
    }
    dim_t padL() const { return desc_.padding[0][ndims() - 3]; }
    dim_t padR() const { return desc_.padding[1][ndims() - 3]; }

    int ndims() const { return src_desc().ndims; }
    int spatial_ndims() const { return ndims() - 2; }

    bool is_dilated() const { return KDD() != 0 || KDH() != 0 || KDW() != 0; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(src_desc()).has_zero_dim();
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

protected:
    pooling_desc_t desc_;
    const pooling_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t ws_md_;

    pooling_pd_t(const pooling_desc_t *adesc, const primitive_attr_t *attr,
            const pooling_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , ws_md_() {}

    void init_default_ws(data_type_t dt = data_type::undef) {
        ws_md_ = is_fwd() ? *dst_md() : *diff_dst_md();
        ws_md_.data_type = (dt != data_type::undef) ? dt : indices_data_type();
    }

    data_type_t indices_data_type() const {
        /* the simplest way to express 256... */
        const int u8_max = nstl::numeric_limits<
                typename prec_traits<data_type::u8>::type>::max();
        return utils::array_product(desc()->kernel, spatial_ndims()) <= u8_max
                ? data_type::u8
                : data_type::s32;
    }

private:
    const memory_desc_t &src_desc() const {
        return is_fwd() ? desc_.src_desc : desc_.diff_src_desc;
    }
    const memory_desc_t &dst_desc() const {
        return is_fwd() ? desc_.dst_desc : desc_.diff_dst_desc;
    }
};

struct pooling_fwd_pd_t : public pooling_pd_t {
    typedef pooling_fwd_pd_t base_class;
    typedef pooling_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && (!types::is_zero_md(workspace_md())))
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return pooling_pd_t::arg_md(arg);
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
    const memory_desc_t *workspace_md(int index = 0) const override {
        return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_
                                                         : &glob_zero_md;
    }

    int n_inputs() const override { return 1 + n_binary_po_inputs(); }
    int n_outputs() const override {
        return 1 + (!types::is_zero_md(workspace_md()));
    }

    std::vector<memory_desc_t> hint_mds(bool is_hint) const override {
        if (!is_hint) return {};
        return {*src_md(0), *workspace_md(0)};
    }

protected:
    memory_desc_t src_md_;
    memory_desc_t dst_md_;

    pooling_fwd_pd_t(const pooling_desc_t *adesc, const primitive_attr_t *attr,
            const pooling_fwd_pd_t *hint_fwd_pd)
        : pooling_pd_t(adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc) {}

    virtual status_t set_default_params() {
        if (dst_md()->format_kind != format_kind::any) return status::success;

        if (src_md()->format_kind != format_kind::blocked)
            return status::unimplemented;

        return memory_desc_init_by_blocking_desc(
                dst_md_, src_md_.format_desc.blocking);
    }
};

struct pooling_bwd_pd_t : public pooling_pd_t {
    typedef pooling_bwd_pd_t base_class;
    typedef pooling_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_DIFF_DST) return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && (!types::is_zero_md(workspace_md())))
            return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return pooling_pd_t::arg_md(arg);
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
    const memory_desc_t *workspace_md(int index = 0) const override {
        return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_
                                                         : &glob_zero_md;
    }

    int n_inputs() const override {
        return 1 + (!types::is_zero_md(workspace_md()));
    }
    int n_outputs() const override { return 1; }

    std::vector<memory_desc_t> hint_mds(bool is_hint) const override {
        assert(!is_hint);
        MAYBE_UNUSED(is_hint);
        return hint_mds_;
    }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t diff_dst_md_;

    pooling_bwd_pd_t(const pooling_desc_t *adesc, const primitive_attr_t *attr,
            const pooling_fwd_pd_t *hint_fwd_pd)
        : pooling_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {
        if (hint_fwd_pd_)
            hint_mds_ = hint_fwd_pd_->hint_mds(true /* is_hint */);
    }

    virtual status_t set_default_params() {
        if (diff_src_md()->format_kind == format_kind::any) {
            status_t status = status::success;
            if (hint_fwd_pd_)
                status = memory_desc_init_by_md_and_dt(diff_src_md_,
                        hint_mds(false /* is_hint */)[0],
                        diff_src_md_.data_type);
            else
                status = memory_desc_init_by_strides(diff_src_md_, nullptr);
            if (status != status::success) return status;
        } else if (diff_src_md()->format_kind != format_kind::blocked) {
            return status::unimplemented;
        }

        if (diff_dst_md()->format_kind != format_kind::any)
            return status::success;

        if (diff_src_md()->format_kind != format_kind::blocked)
            return status::unimplemented;

        return memory_desc_init_by_blocking_desc(
                diff_dst_md_, diff_src_md_.format_desc.blocking);
    }

private:
    std::vector<memory_desc_t> hint_mds_;
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
