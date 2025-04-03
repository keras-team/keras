/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef CPU_X64_IP_CONVOLUTION_HPP
#define CPU_X64_IP_CONVOLUTION_HPP

#include <string>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_inner_product_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

status_t reshape_dst(memory_desc_t *o_md, const memory_desc_t *i_md) {
    dims_t reduce {};
    const int ndims = 2; // dst is always nc for inner product
    // conv to ip: remove spatial
    for (int d = 0; d < ndims; ++d)
        reduce[d] = i_md->dims[d];

    return memory_desc_reshape(*o_md, *i_md, ndims, reduce);
}

status_t maybe_reshape_weights(memory_desc_t *o_md, const memory_desc_t *i_md,
        bool with_groups, bool to_ip = false) {
    dims_t reduce {};
    const int ndims = i_md->ndims + (to_ip ? -1 : +1) * with_groups;
    if (to_ip) {
        // conv to ip: maybe remove groups
        for (int d = 0; d < ndims; ++d)
            reduce[d] = i_md->dims[d + with_groups];
    } else {
        // ip to conv: maybe restore groups
        if (with_groups) reduce[0] = 1;
        for (int d = 0; d < ndims; ++d)
            reduce[d + with_groups] = i_md->dims[d];
    }

    return memory_desc_reshape(*o_md, *i_md, ndims, reduce);
}

status_t check_conv_ip(convolution_pd_t *self) {
    // Check if convolution is equivalent to inner product
    const bool is_ip_applicable = true
            // no dilations
            && utils::everyone_is(0, self->KDD(), self->KDH(), self->KDW())
            // no "left" padding
            && utils::everyone_is(
                    0, self->padFront(), self->padT(), self->padL())
            // no "right" padding
            && utils::everyone_is(
                    0, self->padBack(), self->padB(), self->padR())
            // no non-trivial groups or output spatial
            && utils::everyone_is(
                    1, self->G(), self->OD(), self->OH(), self->OW())
            // only unit stride
            && utils::everyone_is(1, self->KSD(), self->KSH(), self->KSW());
    if (!is_ip_applicable) return status::unimplemented;

    // Simple heuristic to only target arches and shapes that benefit.
    // TODO: Extend to other arches and shapes as performance allows.
    const dim_t ks = self->KD() * self->KH() * self->KW();
    const dim_t ks_threshold = 27; // empirical
    const bool is_performant
            = 1 < self->MB() && ks > ks_threshold && mayiuse(avx512_core);
    if (!is_performant) return status::unimplemented;

    return status::success;
}

status_t check_tag(memory_desc_t &md, const format_tag_t tag) {
    const memory_desc_wrapper mdw(&md);
    if (mdw.matches_one_of_tag(tag) == format_tag::undef)
        return status::unimplemented;
    return status::success;
}

status_t set_and_or_check_formats(const convolution_desc_t &desc,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr) {
    using namespace format_tag;
    auto atag = utils::pick(src_md.ndims - 3, nwc, nhwc, ndhwc);
    const bool is_fwd = utils::one_of(desc.prop_kind,
            prop_kind::forward_training, prop_kind::forward_inference);
    const bool with_bias = desc.prop_kind != prop_kind::backward_data;

    // Check that nspc is the default layout for convolutions,
    // or that expected performance gain outweights potential
    // cost of extra reorders.
    // Currently this means:
    // - int8 with any forward prop_kind on any isa
    // - fp32/bf16 with any prop_kind on avx512_core and higher
    // - f16
    const auto wei_dt = weights_md.data_type;
    const bool is_set_allowed = false
            || (utils::one_of(wei_dt, data_type::f32, data_type::bf16)
                    && mayiuse(avx512_core))
            || (is_fwd && wei_dt == data_type::s8)
            || (wei_dt == data_type::f16 && mayiuse(avx512_core_fp16));

    // NOTE: Only plain layouts should be supported since the dims of
    // dst_md_ must be reshaped from {N, C, H, W} to {N, C}. If the
    // conv layout is blocked by channel, then the ip layout will also
    // be blocked by channel (eg nChw16c -> nC16c). This can lead to
    // deployment of reference ip as well as strange weights layouts.
    if (is_set_allowed && src_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md, atag));
    else
        CHECK(check_tag(src_md, atag));
    if (is_set_allowed && dst_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, atag));
    else
        CHECK(check_tag(dst_md, atag));
    if (with_bias && bias_md.format_kind != format_kind::undef) {
        auto btag = x;
        if (bias_md.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, btag));
        else
            CHECK(check_tag(bias_md, btag));
    }
    return attr.set_default_formats(&dst_md);
}

} // namespace

struct ip_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_convolution_fwd_pd_t(other)
            , ip_pd_(other.ip_pd_->clone())
            , name_(other.name_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ip_convolution_fwd_t);

        status_t init_ip(engine_t *engine) {
            inner_product_desc_t ipd;
            CHECK(ip_desc_create(&ipd));
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&ipd, attr(), nullptr);
            if (!it.is_initialized()) return status::out_of_memory;

            while (++it != it.end()) {
                ip_pd_ = *it;
                const bool ok = ip_pd_->weights_md()->extra.flags == 0;
                if (ok) return status::success;
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using smask_t = primitive_attr_t::skip_mask_t;

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(attr()->has_default_values(
                                   smask_t::scales_runtime | smask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);

            CHECK(check_conv_ip(this));

            CHECK(set_and_or_check_formats(
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, attr_));

            CHECK(init_ip(engine));

            if (weights_md_.format_kind == format_kind::any)
                CHECK(maybe_reshape_weights(
                        &weights_md_, ip_pd_->weights_md(), with_groups()));

            init_name();
            init_scratchpad();
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> ip_pd_;

    private:
        std::string name_ = "ip:any+";

        void init_name() {
            const std::string ips(ip_pd_->name());
            const std::string prefix = "x64:";
            const size_t pos = ips.find(prefix);
            name_.append(ips, pos + prefix.length(), std::string::npos);
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, ip_pd_->scratchpad_registry());
        }

        status_t ip_desc_create(inner_product_desc_t *ipd) {
            const bool to_ip = true;

            // reinterpret dst without spatial
            memory_desc_t ip_dst_d;
            CHECK(reshape_dst(&ip_dst_d, &dst_md_));

            // reinterpret weights without groups
            memory_desc_t ip_weights_d;
            CHECK(maybe_reshape_weights(
                    &ip_weights_d, &weights_md_, with_groups(), to_ip));

            return ip_desc_init(ipd, desc()->prop_kind, &src_md_, &ip_weights_d,
                    &bias_md_, &ip_dst_d);
        }
    };

    ip_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->ip_pd_->create_primitive(ip_p_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> ip_p_;
};

struct ip_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_convolution_bwd_data_pd_t(other)
            , ip_pd_(other.ip_pd_->clone())
            , name_(other.name_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ip_convolution_bwd_data_t);

        status_t init_ip(engine_t *engine) {
            inner_product_desc_t ipd;
            CHECK(ip_desc_create(&ipd));
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&ipd, attr(), nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            while (++it != it.end()) {
                ip_pd_ = *it;

                // Avoid dispatching reference for f16 data-type.
                const bool is_f16 = weights_md_.data_type == data_type::f16;
                if (is_f16) {
                    const std::string impl_name(ip_pd_->name());
                    if (std::string::npos != impl_name.find("ref"))
                        return status::unimplemented;
                } else {
                    const bool ok = ip_pd_->weights_md()->extra.flags == 0;
                    if (ok) return status::success;
                }
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            CHECK(check_conv_ip(this));

            CHECK(set_and_or_check_formats(*desc(), diff_src_md_, weights_md_,
                    diff_dst_md_, bias_md_, attr_));

            CHECK(init_ip(engine));

            if (weights_md_.format_kind == format_kind::any)
                CHECK(maybe_reshape_weights(
                        &weights_md_, ip_pd_->weights_md(), with_groups()));

            init_name();
            init_scratchpad();
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> ip_pd_;

    private:
        std::string name_ = "ip:any";

        void init_name() {
            name_.append("+");
            name_.append(ip_pd_->name());
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, ip_pd_->scratchpad_registry());
        }

        status_t ip_desc_create(inner_product_desc_t *ipd) {
            const bool to_ip = true;

            // reinterpret dst without spatial
            memory_desc_t ip_diff_dst_d;
            CHECK(reshape_dst(&ip_diff_dst_d, &diff_dst_md_));

            // reinterpret weights without groups
            memory_desc_t ip_weights_d;
            CHECK(maybe_reshape_weights(
                    &ip_weights_d, &weights_md_, with_groups(), to_ip));

            return ip_desc_init(ipd, desc()->prop_kind, &diff_src_md_,
                    &ip_weights_d, nullptr, &ip_diff_dst_d);
        }
    };

    ip_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->ip_pd_->create_primitive(ip_p_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> ip_p_;
};

struct ip_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_convolution_bwd_weights_pd_t(other)
            , ip_pd_(other.ip_pd_->clone())
            , name_(other.name_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ip_convolution_bwd_weights_t);

        status_t init_ip(engine_t *engine) {
            inner_product_desc_t ipd;
            CHECK(ip_desc_create(&ipd));
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&ipd, attr(), nullptr);
            if (!it.is_initialized()) return status::out_of_memory;

            while (++it != it.end()) {
                ip_pd_ = *it;
                const bool ok = ip_pd_->weights_md()->extra.flags == 0;
                if (ok) return status::success;
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            CHECK(check_conv_ip(this));

            CHECK(set_and_or_check_formats(*desc(), src_md_, diff_weights_md_,
                    diff_dst_md_, diff_bias_md_, attr_));

            CHECK(init_ip(engine));

            if (diff_weights_md_.format_kind == format_kind::any)
                CHECK(maybe_reshape_weights(&diff_weights_md_,
                        ip_pd_->diff_weights_md(), with_groups()));

            init_name();
            init_scratchpad();
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> ip_pd_;

    private:
        std::string name_ = "ip:any";

        void init_name() {
            name_.append("+");
            name_.append(ip_pd_->name());
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, ip_pd_->scratchpad_registry());
        }

        status_t ip_desc_create(inner_product_desc_t *ipd) {
            const bool to_ip = true;

            // reinterpret dst without spatial
            memory_desc_t ip_diff_dst_d;
            CHECK(reshape_dst(&ip_diff_dst_d, &diff_dst_md_));

            // reinterpret weights without groups
            memory_desc_t ip_diff_weights_d;
            CHECK(maybe_reshape_weights(&ip_diff_weights_d, &diff_weights_md_,
                    with_groups(), to_ip));

            return ip_desc_init(ipd, desc()->prop_kind, &src_md_,
                    &ip_diff_weights_d, &diff_bias_md_, &ip_diff_dst_d);
        }
    };

    ip_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->ip_pd_->create_primitive(ip_p_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> ip_p_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
