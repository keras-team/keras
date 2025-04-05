/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
* Copyright 2022 Arm Ltd. and affiliates
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

#ifndef CPU_REF_DECONVOLUTION_HPP
#define CPU_REF_DECONVOLUTION_HPP

#include <assert.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_deconvolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static status_t weights_axes_permutation(
        memory_desc_t *o_md, const memory_desc_t *i_md, bool with_groups) {
    int perm[DNNL_MAX_NDIMS] {}; // deconv to conv weight permutation
    for (int d = 0; d < DNNL_MAX_NDIMS; ++d)
        perm[d] = d;
    nstl::swap(perm[0 + with_groups], perm[1 + with_groups]);

    return memory_desc_permute_axes(*o_md, *i_md, perm);
}

static status_t conv_descr_create(const deconvolution_desc_t *dd,
        convolution_desc_t *cd, const memory_desc_t *bias_md = nullptr,
        data_type_t src_dt = data_type::undef) {
    using namespace prop_kind;
    alg_kind_t alg_kind = dd->alg_kind == alg_kind::deconvolution_direct
            ? alg_kind::convolution_direct
            : alg_kind::convolution_winograd;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    memory_desc_t src_md_patched;
    prop_kind_t prop_kind;

    if (utils::one_of(dd->prop_kind, forward_training, forward_inference)) {
        prop_kind = backward_data;
        assert(src_dt != data_type::undef);
        CHECK(memory_desc_init_by_md_and_dt(
                src_md_patched, dd->dst_desc, src_dt));
        src_md = &src_md_patched;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->weights_desc;
    } else if (dd->prop_kind == backward_data) {
        assert(src_dt == data_type::undef);
        prop_kind = forward_training;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->diff_src_desc;
        d_weights_d = &dd->weights_desc;
    } else {
        assert(src_dt == data_type::undef);
        prop_kind = dd->prop_kind;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->diff_weights_desc;
    }

    /* create weights desc for convolution */
    memory_desc_t c_weights_d;
    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;
    CHECK(weights_axes_permutation(&c_weights_d, d_weights_d, with_groups));

    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &c_weights_d,
            bias_md, dst_md, dd->strides, dd->dilates, dd->padding[0],
            dd->padding[1]);
}

struct ref_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_)
            , dst_tag_(other.dst_tag_)
            , name_(other.name_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ref_deconvolution_fwd_t);

        status_t init_convolution(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            // Create empty attributes for bwd_d conv to pick up the fastest
            // impl available and apply post-ops and/or bias update later in
            // this impl via simple loop.
            primitive_attr_t conv_attr;

            convolution_desc_t cd;
            // When no attributes were requested, try to find a bwd_d conv impl
            // which supports bias update in-place, if requested, in requested
            // dst_dt. If appropriate conv impl was not found, enforce f32
            // diff_src for conv for correct result. If attributes are
            // requested, enforce conv impl to return f32 output no matter what.
            if (attr()->has_default_values()) {
                CHECK(conv_descr_create(
                        desc(), &cd, weights_md(1), dst_md()->data_type));
                primitive_desc_iterator_t it(
                        engine, (op_desc_t *)&cd, &conv_attr, nullptr);
                if (!it.is_initialized()) return status::out_of_memory;

                while (++it != it.end()) {
                    conv_pd_ = *it;
                    if (with_bias()) {
                        conv_supports_bias_ = utils::downcast<
                                cpu_convolution_bwd_data_pd_t *>(conv_pd_.get())
                                                      ->support_bias();
                        if (!conv_supports_bias_) continue;
                    }
                    bool ok = conv_pd_->weights_md()->extra.flags == 0;
                    if (ok) return status::success;
                }
            }

            // Intermediate f32 buffer is supported only for given condition.
            if (!attr()->has_default_values() || with_bias()) {
                // Enforce f32 dt for diff src and work with f32 output for bias
                // update or post ops after conv execution.
                CHECK(conv_descr_create(desc(), &cd, nullptr, data_type::f32));
                primitive_desc_iterator_t it(
                        engine, (op_desc_t *)&cd, &conv_attr, nullptr);
                if (!it.is_initialized()) return status::out_of_memory;

                while (++it != it.end()) {
                    conv_pd_ = *it;
                    bool ok = conv_pd_->weights_md()->extra.flags == 0;
                    if (ok) return status::success;
                }
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            auto skip_mask = smask_t::post_ops | smask_t::sum_dt;
            if (utils::one_of(desc()->src_desc.data_type, s8, u8))
                skip_mask |= smask_t::scales_runtime
                        | smask_t::zero_points_runtime;

            VDISPATCH_DECONVOLUTION(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(utils::one_of(desc()->alg_kind,
                                            alg_kind::deconvolution_direct,
                                            alg_kind::deconvolution_winograd),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(attr()->has_default_values(skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_DECONVOLUTION(
                    attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_DECONVOLUTION(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_DECONVOLUTION(
                    zero_points_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);

            CHECK(init_convolution(engine));

            if (weights_md_.format_kind == format_kind::any)
                CHECK(weights_axes_permutation(
                        &weights_md_, conv_pd_->weights_md(), with_groups()));
            if (src_md_.format_kind == format_kind::any)
                src_md_ = *conv_pd_->diff_dst_md();
            if (dst_md_.format_kind == format_kind::any) {
                // re-apply dt manually since it could be changed due to bias
                const auto dst_dt = dst_md_.data_type;
                memory_desc_init_by_md_and_dt(
                        dst_md_, *conv_pd_->diff_src_md(), dst_dt);
            }
            if (bias_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(bias_md_, x));

            dst_tag_ = memory_desc_matches_one_of_tag(dst_md_,
                    utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                    utils::pick(ndims() - 3, nwc, nhwc, ndhwc),
                    utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                    utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));

            init_name();
            init_scratchpad();
            return attr_.set_default_formats(dst_md(0));
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;
        bool conv_supports_bias_ = false;
        format_tag_t dst_tag_;

    private:
        std::string name_ = "conv:any+"; // convolution-based deconvolution

        void init_name() { name_.append(conv_pd_->name()); }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, conv_pd_->scratchpad_registry());

            // This scratchpad is required for intermediate f32 conv output
            // since original memory can be of smaller size and will cause
            // out of boundary access.
            if ((with_bias() && !conv_supports_bias_)
                    || !attr()->has_default_values()) {
                const memory_desc_wrapper diff_src_d(conv_pd_->diff_src_md());
                assert(diff_src_d.data_type_size() == sizeof(float));
                scratchpad.book(key_deconv_bias, diff_src_d.nelems(true),
                        diff_src_d.data_type_size());
            }
            // This scratchpad is required to stash original dst memory for sum
            // post-op. It will be overwritten by conv execution and will not
            // be available to get the correct result.
            const memory_desc_wrapper dst_d(dst_md());
            if (attr()->post_ops_.find(primitive_kind::sum) != -1)
                scratchpad.book(key_deconv_sum, dst_d.nelems(true),
                        dst_d.data_type_size());

            if (!attr()->zero_points_.has_default_values(DNNL_ARG_SRC)) {
                scratchpad.book<int32_t>(key_deconv_zp, OC() * G());
            }
        }

        bool post_ops_ok() const {
            using namespace data_type;
            const bool is_int8 = utils::one_of(src_md()->data_type, s8, u8);
            return attr()->post_ops_.check_sum_consistency(
                           dst_md()->data_type, is_int8)
                    && ref_post_ops_t::primitive_kind_ok(attr()->post_ops_);
        }

        bool zero_points_ok() const {
            using namespace data_type;
            int mask_src = 0, mask_dst = 0;
            attr()->zero_points_.get(DNNL_ARG_SRC, &mask_src);
            attr()->zero_points_.get(DNNL_ARG_DST, &mask_dst);

            return IMPLICATION(!utils::one_of(src_md()->data_type, s8, u8),
                           attr()->zero_points_.has_default_values())
                    && attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                    && (mask_src == 0 || mask_src == 1 << 1)
                    && (mask_dst == 0 || mask_dst == 1 << 1);
        }
    };

    ref_deconvolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->conv_pd_->create_primitive(conv_p_, engine));

        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    void compute_fwd_bias_common(const exec_ctx_t &ctx, void *dst,
            const float *conv_output, bool non_default_attr) const;

    void compute_fwd_bias_ncdhw(const exec_ctx_t &ctx, void *dst,
            const float *conv_output, bool non_default_attr) const;

    void compute_fwd_bias_ndhwc(const exec_ctx_t &ctx, void *dst,
            const float *conv_output, bool non_default_attr) const;

    template <dim_t blk_size>
    void compute_fwd_bias_nCdhwXc(const exec_ctx_t &ctx, void *dst,
            const float *conv_output, bool non_default_attr) const;

    status_t compute_oscale(const exec_ctx_t &ctx, float *dst) const;

    void compute_fwd_bias(const exec_ctx_t &ctx, void *dst,
            const float *conv_output, bool non_default_attr) const;

    status_t compute_ref_attrs(const exec_ctx_t &ctx, const float *conv_output,
            void *original_dst) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

struct ref_deconvolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_bwd_data_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , name_(other.name_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ref_deconvolution_bwd_data_t);

        status_t init_convolution(engine_t *engine) {
            using namespace types;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;

            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            while (++it != it.end()) {
                conv_pd_ = *it;
                if (conv_pd_->weights_md()->extra.flags == 0)
                    return status::success;
            }

            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto dsrc_type = desc()->diff_src_desc.data_type;
            auto wei_type = desc()->weights_desc.data_type;
            auto ddst_type = desc()->diff_dst_desc.data_type;

            VDISPATCH_DECONVOLUTION(
                    desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(utils::one_of(wei_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(ddst_type == wei_type,
                    VERBOSE_INCONSISTENT_DT, "diff_dst", "weights");
            VDISPATCH_DECONVOLUTION(utils::one_of(dsrc_type, wei_type, f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(utils::one_of(desc()->alg_kind,
                                            alg_kind::deconvolution_direct,
                                            alg_kind::deconvolution_winograd),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            CHECK(init_convolution(engine));
            if (weights_md_.format_kind == format_kind::any)
                CHECK(weights_axes_permutation(
                        &weights_md_, conv_pd_->weights_md(), with_groups()));
            if (diff_src_md_.format_kind == format_kind::any)
                diff_src_md_ = *conv_pd_->dst_md();
            if (diff_dst_md_.format_kind == format_kind::any)
                diff_dst_md_ = *conv_pd_->src_md();

            init_name();
            init_scratchpad();
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;

    private:
        std::string name_ = "conv:any+"; // convolution-based deconvolution

        void init_name() { name_.append(conv_pd_->name()); }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

#if DNNL_AARCH64 && DNNL_AARCH64_USE_ACL
    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        CHECK(conv_p_->create_resource(engine, mapper));
        return status::success;
    }
#endif

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

struct ref_deconvolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_bwd_weights_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , dst_tag_(other.dst_tag_)
            , name_(other.name_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution(engine_t *engine) {
            using namespace types;
            using namespace format_tag;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;

            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            while (++it != it.end()) {
                conv_pd_ = *it;
                bool bf16_ref_deconv_supports_bias = IMPLICATION(with_bias()
                                && desc()->src_desc.data_type
                                        == data_type::bf16,
                        memory_desc_matches_one_of_tag(*conv_pd_->src_md(),
                                utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                                utils::pick(ndims() - 3, nwc, nhwc, ndhwc),
                                utils::pick(ndims() - 3, nCw16c, nChw16c,
                                        nCdhw16c)));
                if (conv_pd_->diff_weights_md()->extra.flags == 0
                        && bf16_ref_deconv_supports_bias) {
                    return status::success;
                }
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;
            auto src_type = desc()->src_desc.data_type;
            auto dwei_type = desc()->diff_weights_desc.data_type;
            auto ddst_type = desc()->diff_dst_desc.data_type;
            VDISPATCH_DECONVOLUTION(
                    desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(utils::one_of(src_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(ddst_type == src_type,
                    VERBOSE_INCONSISTENT_DT, "diff_dst", "src");
            VDISPATCH_DECONVOLUTION(utils::one_of(dwei_type, src_type, f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(utils::one_of(desc()->alg_kind,
                                            alg_kind::deconvolution_direct,
                                            alg_kind::deconvolution_winograd),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            CHECK(init_convolution(engine));
            if (diff_weights_md_.format_kind == format_kind::any)
                CHECK(weights_axes_permutation(&diff_weights_md_,
                        conv_pd_->diff_weights_md(), with_groups()));
            if (src_md_.format_kind == format_kind::any)
                src_md_ = *conv_pd_->diff_dst_md();
            if (diff_dst_md_.format_kind == format_kind::any)
                diff_dst_md_ = *conv_pd_->src_md();
            if (diff_bias_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(diff_bias_md_, x));

            dst_tag_ = memory_desc_matches_one_of_tag(diff_dst_md_,
                    utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                    utils::pick(ndims() - 3, nwc, nhwc, ndhwc),
                    utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                    utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));

            init_name();
            init_scratchpad();
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;
        format_tag_t dst_tag_;

    private:
        std::string name_ = "conv:any+"; // convolution-based deconvolution

        void init_name() { name_.append(conv_pd_->name()); }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    ref_deconvolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void compute_bwd_bias(float *diff_bias, const float *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bwd_bias_ncdhw(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bwd_bias_ndhwc(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type, dim_t blksize>
    void compute_bwd_bias_nCdhwXc(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bias(const exec_ctx_t &ctx) const;
    std::shared_ptr<primitive_t> conv_p_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
