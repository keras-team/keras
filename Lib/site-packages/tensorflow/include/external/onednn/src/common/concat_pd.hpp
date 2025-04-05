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

#ifndef COMMON_CONCAT_PD_HPP
#define COMMON_CONCAT_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"

#include "utils.hpp"

#define VDISPATCH_CONCAT(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, concat, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_CONCAT_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, concat, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct concat_pd_t : public primitive_desc_t {
    const concat_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    ~concat_pd_t() = default;

    arg_usage_t arg_usage(int arg) const override {
        if (arg >= DNNL_ARG_MULTIPLE_SRC
                && arg < DNNL_ARG_MULTIPLE_SRC + n_inputs())
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        int src_index = arg - DNNL_ARG_MULTIPLE_SRC;
        if (src_index >= 0 && src_index < n_inputs()) return src_md(src_index);
        if (arg == DNNL_ARG_DST) return dst_md(0, user_input);
        return primitive_desc_t::arg_md(arg);
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index < n_inputs())
            return user_input ? desc()->src_mds[index] : &src_mds_[index];
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? desc()->dst_md : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return n_; }
    int n_outputs() const override { return 1; }

    int concat_dim() const { return concat_dim_; }

    const memory_desc_t *src_image_md(int index = 0) const {
        return index < n_inputs() ? &src_image_mds_[index] : &glob_zero_md;
    }

protected:
    int n_, concat_dim_;
    memory_desc_t dst_md_;
    memory_desc_t original_dst_;
    std::vector<memory_desc_t> src_mds_;

    /* contains images of srcs in the dst memory (if possible)
     * Lives here to simplify some implementations. An implementation might
     * use this auxiliary array iff init() returned success */
    std::vector<memory_desc_t> src_image_mds_;

protected:
    concat_desc_t desc_;

    concat_pd_t(const primitive_attr_t *attr, const memory_desc_t *dst_md,
            int n, int concat_dim, const memory_desc_t *const *src_mds)
        : primitive_desc_t(attr, primitive_kind::concat)
        , n_(n)
        , concat_dim_(concat_dim)
        , dst_md_(*dst_md)
        , original_dst_(*dst_md) {
        src_mds_.reserve(n_);
        for (int i = 0; i < n_; ++i)
            src_mds_.push_back(*src_mds[i]);

        init_desc();
    }

    concat_pd_t(const concat_pd_t &other) : primitive_desc_t(other) {
        n_ = other.n_;
        concat_dim_ = other.concat_dim_;
        dst_md_ = other.dst_md_;
        original_dst_ = other.original_dst_;
        src_mds_ = other.src_mds_;
        src_image_mds_ = other.src_image_mds_;

        init_desc();
    }

    concat_pd_t &operator=(const concat_pd_t &other) {
        DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
        n_ = other.n_;
        concat_dim_ = other.concat_dim_;
        dst_md_ = other.dst_md_;
        original_dst_ = other.original_dst_;
        src_mds_ = other.src_mds_;
        src_image_mds_ = other.src_image_mds_;

        init_desc();
        return *this;
    }

    /* inits src_image_mds_ and dst_md_ in simple cases. It is possible to
     * override dst_md_ by using force_dst_md.
     * Rationale: if user forces particular dst_md, that cannot be used to
     *            create src_img_mds, the implementation might need to use
     *            intermediate (force_dst_md) memory with some plain format.
     *
     * @warning The call may fail. */
    status_t init(const memory_desc_t *force_dst_md = nullptr) {
        bool ok = true;
        if (force_dst_md == nullptr)
            ok = ok && set_default_params() == status::success;
        if (!ok) return status::unimplemented;

        /* work with force_dst_md */
        if (force_dst_md == nullptr) force_dst_md = &dst_md_;

        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper i_d(&src_mds_[i]);
            if (!i_d.is_blocking_desc() || i_d.is_additional_buffer())
                return status::unimplemented;
        }

        const int ndims = force_dst_md->ndims;
        dim_t current_concat_dim_offset = 0;
        for (int i = 0; i < n_; ++i) {
            const dim_t dim = src_mds_[i].dims[concat_dim_];
            dims_t dims, offsets = {};
            utils::array_copy(dims, force_dst_md->dims, ndims);
            dims[concat_dim_] = dim;
            offsets[concat_dim_] = current_concat_dim_offset;

            memory_desc_t src_img_d;
            status_t status = memory_desc_init_submemory(
                    src_img_d, *force_dst_md, dims, offsets);
            if (status != status::success) {
                src_image_mds_.clear();
                return status;
            }
            src_image_mds_.push_back(src_img_d);
            current_concat_dim_offset += dim;
        }

        return status::success;
    }

    status_t set_default_params() {
        if (dst_md_.format_kind != format_kind::any) return status::success;

        const int ndims = dst_md_.ndims;

        /* The stupidest ever heuristics (but not the same as we had before):
         *  - Pick the first non-plain format;
         *  - If all formats are plain or it is not possible to create a
         *    blocked format for the output, pick the format of the plain input
         *  - If this fails as well, use plain layout (abcd...)
         */
        status_t status = status::unimplemented;
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper src_d(src_mds_[i]);
            if (src_d.is_blocking_desc() && !src_d.is_plain()) {
                status = memory_desc_init_by_blocking_desc(
                        dst_md_, src_d.blocking_desc());
                if (status == status::success) break;
            }
        }

        if (status == status::success) {
            /* check if we can create a sub-memory for the dst */
            bool desired_format_ok = true;
            dims_t dims {}, offsets {};
            utils::array_copy(dims, dst_md_.dims, ndims);

            for (int i = 0; i < n_; ++i) {
                const auto dim = src_mds_[i].dims[concat_dim_];
                dims[concat_dim_] = dim;

                memory_desc_t src_img_d;
                status_t status = memory_desc_init_submemory(
                        src_img_d, dst_md_, dims, offsets);
                if (status != status::success) {
                    desired_format_ok = false;
                    break;
                }
                offsets[concat_dim_] += dim;
            }

            if (!desired_format_ok) status = status::unimplemented;
        }

        /* if no success so far, try using the format of the first plain input */
        if (status != status::success) {
            for (int i = 0; i < n_; ++i) {
                const memory_desc_wrapper src_d(src_mds_[i]);
                // Dim of `1` may tweak a destination format leading to
                // sub-optimal performance. Limit it to an axis case to allow
                // case like a:a->ab or a:ab->ab to work properly.
                // TODO: update the whole logic to getting string tags of
                // sources but discarding dims of one. If ndims of any source
                // coincides with dst ndims, use that tag (if they are same).
                // If dst has +1 ndim (due to concat dim), use slices as dense
                // layers inside a dst, which means axis should be the least
                // dense dimension.
                const bool axis_dim_has_one = src_d.dims()[concat_dim()] == 1;
                if (!axis_dim_has_one && src_d.is_blocking_desc()
                        && src_d.is_plain() && src_d.nelems() > 0) {
                    status = memory_desc_init_by_blocking_desc(dst_md_,
                            memory_desc_wrapper(src_mds_[i]).blocking_desc());
                    if (status == status::success) return status;
                }
            }
        }

        /* the last line of defense: use plain abcd... format */
        if (status != status::success)
            status = memory_desc_init_by_strides(dst_md_, nullptr);

        return status;
    }

private:
    void init_desc() {
        desc_ = concat_desc_t();
        desc_.primitive_kind = primitive_kind::concat;
        desc_.dst_md = &original_dst_;
        desc_.n = n_;
        desc_.concat_dimension = concat_dim_;
        for (const auto &md : src_mds_)
            desc_.src_mds.push_back(&md);
    }
};

#define DECLARE_CONCAT_PD_t(impl_name, ...) \
    static status_t create(concat_pd_t **concat_pd, engine_t *engine, \
            const primitive_attr_t *attr, const memory_desc_t *dst_md, int n, \
            int concat_dim, const memory_desc_t *const *src_mds) { \
        using namespace status; \
        auto _pd = make_unique_pd<pd_t>(attr, dst_md, n, concat_dim, src_mds); \
        if (_pd == nullptr) return out_of_memory; \
        CHECK(_pd->init(engine)); \
        CHECK(_pd->init_scratchpad_md()); \
        return safe_ptr_assign(*concat_pd, _pd.release()); \
    } \
    status_t create_primitive( \
            std::pair<std::shared_ptr<primitive_t>, bool> &primitive, \
            engine_t *engine, const cache_blob_t &cache_blob) const override { \
        return primitive_t::create_primitive_common<__VA_ARGS__, pd_t>( \
                primitive, this, engine, false, cache_blob); \
    } \
    pd_t *clone() const override { \
        auto new_pd = utils::make_unique<pd_t>(*this); \
        if (!new_pd->is_initialized()) return nullptr; \
        return new_pd.release(); \
    } \
    const char *name() const override { return impl_name; }

#define DECLARE_CONCAT_PD_T(impl_name, ...) \
    DECLARE_CONCAT_PD_t(impl_name, __VA_ARGS__)

} // namespace impl
} // namespace dnnl

#endif
