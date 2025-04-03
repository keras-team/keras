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

#ifndef COMMON_PRIMITIVE_DESC_HPP
#define COMMON_PRIMITIVE_DESC_HPP

#include <typeindex>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "cache_blob.hpp"
#include "cache_blob_id.hpp"
#include "memory_tracking.hpp"
#include "nstl.hpp"
#include "opdesc.hpp"
#include "primitive_attr.hpp"
#include "primitive_cache.hpp"
#include "type_helpers.hpp"
#include "verbose.hpp"

namespace dnnl {
namespace impl {

static int po_inputs(const post_ops_t &post_ops, const primitive_kind_t kind) {
    int n_inputs = 0;
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        if (post_ops.contain(kind, idx)) n_inputs++;
    }
    return n_inputs;
}

struct impl_list_item_t;
struct primitive_t;
// Primitive descriptor implementation
struct primitive_desc_t : public c_compatible {
    primitive_desc_t(const primitive_attr_t *attr, primitive_kind_t kind)
        : attr_(*attr), kind_(kind), pd_iterator_offset_(0), skip_idx_(-1) {
        is_initialized_ = is_initialized_ && attr_.is_initialized();
    }

    primitive_desc_t(primitive_kind_t kind)
        : kind_(kind), pd_iterator_offset_(0), skip_idx_(-1) {}

    bool is_initialized() const { return is_initialized_; }

    virtual ~primitive_desc_t() = default;
    virtual primitive_desc_t *clone() const = 0;

    const primitive_attr_t *attr() const { return &attr_; }
    primitive_kind_t kind() const { return kind_; }

    const char *info(engine_t *engine) const {
        if (!info_.is_initialized()) info_.init(engine, this);
        return info_.c_str();
    }

    // Returns `info` string with actual md tags and non-dense strides and dims
    //     if runtime dimensions were requested.
    // Note: doesn't use `pd_info_t` object since it's prohibited to change the
    //     state of primitive desc at `execute` time.
    // Note: lives in `primitive_desc_t` because it relies on
    //     `const char *info(...)`, which relies on its member `info_`, which is
    //     not available in verbose translation unit.
    // Note: requires all internals to be defined for ONEDNN_VERBOSE=OFF, but
    //     doesn't require any special handling since `get_verbose` is `false`.
    std::string info_with_runtime_dims(engine_t *engine,
            const memory_desc_t *src_md, const memory_desc_t *wei_md,
            const memory_desc_t *bia_md, const memory_desc_t *dst_md) {
        std::string info_str = info(engine);

        // Matmul and reorder are the only primitives supporting runtime dims.
        // Any extension of primitive list will require verbose extension for
        // `mds2str` and `dims2fmt_str`.
        if (!utils::one_of(
                    kind(), primitive_kind::matmul, primitive_kind::reorder))
            return info_str;

        assert(has_runtime_dims_or_strides() && "Runtime dims are expected.");

        // Relying on the order of mds and dims in `info()` dump.
        size_t pos = 0;
        static constexpr int mds_order_in_info = 4; // Starting from 0th.
        for (int i = 0; i < mds_order_in_info; i++) {
            pos = info_str.find_first_of(',', pos) + 1;
        }
        auto mds_len = info_str.find_first_of(',', pos) - pos;
        // Ask verbose to provide information about memory descriptors and dims.
        auto mds_updated = rt_mds2str(kind(), src_md, wei_md, bia_md, dst_md);
        info_str.replace(pos, mds_len, mds_updated);

        // Dims are always last in the line. Check position after mds replaced.
        auto dims_start_pos = info_str.find_last_of(',') + 1;
        auto dims_updated = rt_dims2fmt_str(kind(), src_md, wei_md, dst_md);
        info_str.replace(dims_start_pos, std::string::npos, dims_updated);

        return info_str;
    }

    memory_tracking::registry_t &scratchpad_registry() {
        return scratchpad_registry_;
    }
    const memory_tracking::registry_t &scratchpad_registry() const {
        return scratchpad_registry_;
    }

    virtual const op_desc_t *op_desc() const { return nullptr; }

    prop_kind_t get_prop_kind(status_t *status = nullptr) const {
        prop_kind_t prop_kind = dnnl_prop_kind_undef;
        auto st = query(query::prop_kind, 0, &prop_kind);
        if (status) *status = st;
        return prop_kind;
    }

    const std::vector<uint8_t> &get_cache_blob_id(engine_t *engine) const {
        return cache_blob_id_.get(engine, this);
    }

    static bool post_op_has_proper_input(const primitive_attr_t *attr,
            const primitive_kind_t prim, const int idx, const int arg,
            const int src_mnemonic) {
        return (attr->post_ops_.contain(prim, idx)
                && arg == (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | src_mnemonic));
    }

    virtual bool has_runtime_dims_or_strides() const {
        return memory_desc_wrapper(invariant_src_md())
                       .has_runtime_dims_or_strides()
                || memory_desc_wrapper(invariant_wei_md())
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(invariant_dst_md())
                           .has_runtime_dims_or_strides();
    };

    enum class arg_usage_t { unused, input, output };
    virtual arg_usage_t arg_usage(int arg) const {
        using types::is_zero_md;
        if (arg == DNNL_ARG_ATTR_OUTPUT_SCALES
                && !attr()->output_scales_.defined())
            return arg_usage_t::input;
        if (arg & DNNL_ARG_ATTR_ZERO_POINTS) {
            int zp_arg = arg & ~DNNL_ARG_ATTR_ZERO_POINTS;
            if (!attr()->zero_points_.defined(zp_arg))
                return arg_usage_t::input;
        }
        if (arg & DNNL_ARG_ATTR_SCALES) {
            int scale_arg = arg & ~DNNL_ARG_ATTR_SCALES;
            if (!attr()->scales_.get(scale_arg).defined())
                return arg_usage_t::input;
        }
        if ((arg == (DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0))
                && !attr()->scales_.get(DNNL_ARG_SRC_0).defined())
            return arg_usage_t::input;
        if ((arg == (DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1))
                && !attr()->scales_.get(DNNL_ARG_SRC_1).defined())
            return arg_usage_t::input;
        if (arg == DNNL_ARG_SCRATCHPAD && !is_zero_md(scratchpad_md()))
            return arg_usage_t::output;
        for (int idx = 0; idx < attr()->post_ops_.len(); ++idx) {
            using namespace primitive_kind;
            if (post_op_has_proper_input(
                        attr(), binary, idx, arg, DNNL_ARG_SRC_1)
                    || post_op_has_proper_input(
                            attr(), prelu, idx, arg, DNNL_ARG_WEIGHTS))
                return arg_usage_t::input;
        }

        return arg_usage_t::unused;
    }

    virtual const memory_desc_t *arg_md(
            int arg, bool user_input = false) const {
        // Separate binary post-ops sections due to inability to express inside
        // switch statement.
        if (arg >= DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)
                && arg < DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                           post_ops_t::post_ops_limit)) {
            const auto &po = attr()->post_ops_;
            for (int idx = 0; idx < po.len(); ++idx) {
                if (arg
                        != (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | DNNL_ARG_SRC_1))
                    continue;

                return &po.entry_[idx].binary.src1_desc;
            }
        }

        switch (arg) {
            case DNNL_ARG_WORKSPACE: return workspace_md(0);
            case DNNL_ARG_SCRATCHPAD: return scratchpad_md(0);
            default: return &glob_zero_md;
        }
    }

    virtual const memory_desc_t *invariant_src_md(
            int index = 0, bool user_input = false) const {
        return get_prop_kind() == prop_kind::backward_data ? diff_src_md(index)
                                                           : src_md(index);
    }
    virtual const memory_desc_t *invariant_wei_md(int index = 0) const {
        return get_prop_kind() == prop_kind::backward_weights
                ? diff_weights_md(index)
                : weights_md(index);
    }
    virtual const memory_desc_t *invariant_bia_md() const {
        return invariant_wei_md(1);
    }
    virtual const memory_desc_t *invariant_dst_md() const {
        const auto prop_kind = get_prop_kind();
        return utils::one_of(prop_kind, prop_kind::backward_data,
                       prop_kind::backward_weights, prop_kind::backward)
                ? diff_dst_md()
                : dst_md();
    }

    virtual format_kind_t invariant_src_user_format_kind(int index = 0) const {
        return get_prop_kind() == prop_kind::backward_data
                ? diff_src_md(index, /* user_input = */ true)->format_kind
                : src_md(index, /* user_input = */ true)->format_kind;
    }
    virtual format_kind_t invariant_wei_user_format_kind(int index = 0) const {
        return get_prop_kind() == prop_kind::backward_weights
                ? diff_weights_md(index, /* user_input = */ true)->format_kind
                : weights_md(index, /* user_input = */ true)->format_kind;
    }
    virtual format_kind_t invariant_bia_user_format_kind() const {
        return invariant_wei_user_format_kind(1);
    }
    virtual format_kind_t invariant_dst_user_format_kind(int arg = -1) const {
        const auto prop_kind = get_prop_kind();
        const int default_arg
                = utils::one_of(prop_kind, prop_kind::backward_data,
                          prop_kind::backward_weights, prop_kind::backward)
                ? DNNL_ARG_DIFF_DST
                : DNNL_ARG_DST;
        return arg_md(arg == -1 ? default_arg : arg, /* user_input = */ true)
                ->format_kind;
    }

#define DECLARE_MD_STUB(stub) \
    virtual const memory_desc_t *stub(int idx = 0, bool user_input = false) \
            const { \
        return &glob_zero_md; \
    }

    DECLARE_MD_STUB(src_md);
    DECLARE_MD_STUB(diff_src_md);
    DECLARE_MD_STUB(dst_md);
    DECLARE_MD_STUB(diff_dst_md);
    DECLARE_MD_STUB(weights_md);
    DECLARE_MD_STUB(diff_weights_md);
#undef DECLARE_MD_STUB

#define DECLARE_MD_STUB(stub) \
    virtual const memory_desc_t *stub(int idx = 0) const { \
        return &glob_zero_md; \
    }

    DECLARE_MD_STUB(workspace_md);
#undef DECLARE_MD_STUB

    const memory_desc_t *scratchpad_md(int idx = 0) const {
        return idx == 0 ? &scratchpad_md_ : &glob_zero_md;
    }

    status_t init_scratchpad_md() {
        auto size = scratchpad_size(scratchpad_mode::user);
        dims_t dims = {size};
        return memory_desc_init_by_tag(
                scratchpad_md_, size ? 1 : 0, dims, data_type::u8, dnnl_x);
    }

    /** returns the scratchpad size for the given scratchpad mode. */
    dim_t scratchpad_size(scratchpad_mode_t mode) const {
        if (mode != attr_.scratchpad_mode_) return 0;
        return scratchpad_registry().size();
    }

    virtual status_t query(query_t what, int idx, void *result) const {
        auto safe_ret_md = [&](const memory_desc_t *_) {
            if (_ == nullptr) return status::not_required;
            *(const memory_desc_t **)result = _;
            return status::success;
        };

        switch (what) {
            case query::primitive_kind:
                *(primitive_kind_t *)result = kind();
                break;

            case query::memory_consumption_s64:
                *(dim_t *)result = scratchpad_size(scratchpad_mode::library);
                break;

            case query::exec_arg_md: return safe_ret_md(arg_md(idx));
            case query::src_md: return safe_ret_md(src_md(idx));
            case query::diff_src_md: return safe_ret_md(diff_src_md(idx));
            case query::dst_md: return safe_ret_md(dst_md(idx));
            case query::diff_dst_md: return safe_ret_md(diff_dst_md(idx));
            case query::weights_md: return safe_ret_md(weights_md(idx));
            case query::diff_weights_md:
                return safe_ret_md(diff_weights_md(idx));
            case query::workspace_md:
                if (idx != 0) return status::invalid_arguments;
                return safe_ret_md(workspace_md(idx));
            case query::scratchpad_md:
                if (idx != 0) return status::invalid_arguments;
                return safe_ret_md(scratchpad_md(idx));

            case query::num_of_inputs_s32: *(int *)result = n_inputs(); break;
            case query::num_of_outputs_s32: *(int *)result = n_outputs(); break;

            case query::impl_info_str: *(const char **)result = name(); break;

            default: return status::unimplemented;
        }
        return status::success;
    }

    virtual int n_inputs() const { return 0; }
    virtual int n_outputs() const { return 0; }
    int n_binary_po_inputs() const {
        return po_inputs(attr()->post_ops_, primitive_kind::binary);
    }

    int n_prelu_po_inputs() const {
        return po_inputs(attr()->post_ops_, primitive_kind::prelu);
    }
    // The `hint_mds(bool is_hint)` returns a vector of memory descriptors
    // that might affect the equality of primitive descriptors for backward pass.
    //
    // This function is used for creating a key to fetch primitive or primitive
    // descriptor from cache.
    //
    // 1. When creating a primitive descriptor for backward pass there may be
    //    a forward primitive descriptor hint that can be used to obtain the
    //    memory descriptors. In this case the `is_hint` argument must be `true`.
    // 2. When creating a primitive this function is called for a primitive
    //    descriptor that can be either forward or backward. In this case
    //    the `is_hint` argument must be `false`.
    //       - For forward it will return an empty vector.
    //       - For backward it will return a vector of memory descriptors if
    //         the implementation depends on a forward primitive descriptor.
    //
    // The current cases are:
    // - pooling
    // - shuffle
    //
    // Later the list of primitives can be extended. For instance, currently
    // there is no convolution on the list because nthrs + op_desc
    // (even with format=`any`) + attributes fully define a particular
    // implementation.
    virtual std::vector<memory_desc_t> hint_mds(bool is_hint) const {
        UNUSED(is_hint);
        return {};
    }

    virtual status_t create_primitive(
            std::pair<std::shared_ptr<primitive_t>, bool> &primitive,
            engine_t *engine, const cache_blob_t &cache_blob) const = 0;

    // This is a proxy interface that is used for creating nested primitives.
    // It ignores the bool value that indicates whether the requested primitive
    // was taken from cache.
    status_t create_primitive(std::shared_ptr<primitive_t> &primitive,
            engine_t *engine,
            const cache_blob_t &cache_blob = cache_blob_t()) const {
        std::pair<std::shared_ptr<primitive_t>, bool> p;
        if (get_verbose(verbose_t::debuginfo) >= 1) {
            double start_ms = get_msec();
            CHECK(create_primitive(p, engine, cache_blob));
            double duration_ms = get_msec() - start_ms;
            const char *str = p.second ? ":cache_hit" : ":cache_miss";
            if (cache_blob) str = ":from_cache_blob";
            VPROF(start_ms, primitive, create_nested, str, info(engine),
                    duration_ms);
        } else {
            CHECK(create_primitive(p, engine, cache_blob));
        }
        primitive = p.first;
        return status::success;
    }

    virtual const char *name() const = 0;

    int pd_iterator_offset() const { return pd_iterator_offset_; }
    int skip_idx() const { return skip_idx_; }

protected:
    primitive_attr_t attr_;
    primitive_kind_t kind_;
    int pd_iterator_offset_;
    int skip_idx_;

    memory_desc_t scratchpad_md_;

    mutable pd_info_t info_;
    mutable cache_blob_id_t cache_blob_id_;

    memory_tracking::registry_t scratchpad_registry_;

protected:
    void init_pd_iterator_offset(int offset) { pd_iterator_offset_ = offset; }
    void init_skip_idx(int skip_idx) { skip_idx_ = skip_idx; }

    /** compares ws between fwd_pd and this (make sense to use for bwd_pd)
     * Expectation: this already set workspace, and this workspace should
     *              exactly match the one from fwd_pd */
    bool compare_ws(const primitive_desc_t *fwd_pd) const {
        if (!workspace_md()) return true; // the impl lives fine w/o workspace
        return fwd_pd && fwd_pd->workspace_md()
                && *fwd_pd->workspace_md() == *workspace_md();
    }

    primitive_desc_t &operator=(const primitive_desc_t &other) = delete;

    /* static magic */

    template <typename pd_t, typename... Args>
    static std::unique_ptr<pd_t> make_unique_pd(Args &&...args) {
        /** the only reason why this class is here is the inability of
         * utils::make_unique() to operate on protected parent classes
         * of the derivative pd_t's; compilers should optimize it out */
        class pd_t_compat : public pd_t {
        public:
            pd_t_compat(Args &&...args) : pd_t(std::forward<Args>(args)...) {}
        };
        return utils::make_unique<pd_t_compat>(std::forward<Args>(args)...);
    }

    template <typename pd_t>
    static status_t create(primitive_desc_t **pd, const op_desc_t *adesc,
            const primitive_attr_t *attr, engine_t *engine,
            const primitive_desc_t *hint_fwd) {
        using namespace dnnl::impl::status;
        using pd_op_desc_t = typename pkind_traits<pd_t::base_pkind>::desc_type;
        if (adesc->kind != pd_t::base_pkind) return invalid_arguments;
        assert(hint_fwd ? hint_fwd->kind() == pd_t::base_pkind : true);
        auto hint
                = reinterpret_cast<const typename pd_t::hint_class *>(hint_fwd);
        auto _pd
                = make_unique_pd<pd_t>((const pd_op_desc_t *)adesc, attr, hint);
        if (_pd == nullptr) return out_of_memory;
        if (!_pd->is_initialized()) return out_of_memory;
        CHECK(_pd->init(engine));
        CHECK(_pd->init_scratchpad_md());
        return safe_ptr_assign(*pd, _pd.release());
    }

    friend struct dnnl::impl::impl_list_item_t;
};

} // namespace impl
} // namespace dnnl

#define DECLARE_COMMON_PD_t(impl_name, impl_type, use_global_scratchpad) \
    pd_t *clone() const override { \
        auto new_pd = utils::make_unique<pd_t>(*this); \
        if (!new_pd->is_initialized()) return nullptr; \
        return new_pd.release(); \
    } \
    status_t create_primitive( \
            std::pair<std::shared_ptr<primitive_t>, bool> &primitive, \
            engine_t *engine, const cache_blob_t &cache_blob) const override { \
        return primitive_t::create_primitive_common<impl_type, pd_t>( \
                primitive, this, engine, use_global_scratchpad, cache_blob); \
    } \
    const char *name() const override { return impl_name; } \
    template <typename pd_t> \
    friend status_t primitive_desc_t::create(primitive_desc_t **pd, \
            const op_desc_t *adesc, const primitive_attr_t *attr, \
            engine_t *engine, const primitive_desc_t *hint_fwd);

#define DECLARE_COMMON_PD_T_USE_GLOBAL_SCRATCHPAD(impl_name, impl_type) \
    DECLARE_COMMON_PD_t(impl_name, impl_type, true)

#define DECLARE_COMMON_PD_T_(impl_name, impl_type) \
    DECLARE_COMMON_PD_t(impl_name, impl_type, false)

#define DECLARE_COMMON_PD_T(impl_name, impl_type, ...) \
    DECLARE_COMMON_PD_T_##__VA_ARGS__(impl_name, impl_type)

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
