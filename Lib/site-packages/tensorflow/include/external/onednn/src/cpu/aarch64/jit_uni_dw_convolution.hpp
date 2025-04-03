/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_DW_CONVOLUTION_HPP
#define CPU_AARCH64_JIT_UNI_DW_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/cpu_reducer.hpp"
#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/aarch64/jit_uni_dw_conv_kernel_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type = src_type>
struct jit_uni_dw_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_dw:", jcp_.isa, ""),
                jit_uni_dw_convolution_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, src_type, data_type::undef,
                            dst_type, data_type::f32)
                    && IMPLICATION(this->with_bias(),
                            utils::one_of(this->desc()->bias_desc.data_type,
                                    data_type::f32, data_type::bf16))
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, dst_type)
                    && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status
                    = jit_uni_dw_conv_fwd_kernel<isa, src_type>::init_conf(jcp_,
                            *desc(), src_md_, weights_md_, bias_md_, dst_md_,
                            *attr());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_fwd_kernel<isa, src_type>::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    jit_uni_dw_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type f32_data_t;
    typedef typename prec_traits<data_type::bf16>::type bf16_data_t;
    typedef typename prec_traits<src_type>::type data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_uni_dw_conv_fwd_kernel<isa, src_type>(pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_dw_conv_fwd_kernel<isa, src_type>> kernel_;
};

using jit_sve_512_dw_convolution_fwd_t
        = jit_uni_dw_convolution_fwd_t<sve_512, data_type::f32>;

template <cpu_isa_t isa, data_type_t diff_dst_type,
        data_type_t diff_src_type = diff_dst_type>
struct jit_uni_dw_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_dw:", jcp_.isa, ""),
                jit_uni_dw_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(diff_src_type, diff_dst_type,
                            data_type::undef, diff_dst_type, data_type::f32)
                    && attr()->has_default_values() && !has_zero_dim_memory()
                    && set_default_formats();

            if (!ok) return status::unimplemented;

            status_t status = jit_uni_dw_conv_bwd_data_kernel<isa,
                    diff_dst_type>::init_conf(jcp_, *desc(), *diff_src_md(),
                    *weights_md(), *diff_dst_md());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_bwd_data_kernel<isa,
                    diff_dst_type>::init_scratchpad(scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = nChw16c;
            auto wei_tag = Goihw16g;

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    jit_uni_dw_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<diff_dst_type>::type wei_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_uni_dw_conv_bwd_data_kernel<isa, diff_dst_type>(
                        pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_dw_conv_bwd_data_kernel<isa, diff_dst_type>>
            kernel_;
};

using jit_sve_512_dw_convolution_bwd_data_t
        = jit_uni_dw_convolution_bwd_data_t<sve_512, data_type::f32>;

template <cpu_isa_t isa, data_type_t src_type,
        data_type_t diff_weights_type = src_type>
struct jit_uni_dw_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}
        using jit_uni_dw_convolution_bwd_weights
                = jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
                        diff_weights_type>;
        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_dw:", jcp_.isa, ""),
                jit_uni_dw_convolution_bwd_weights);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, diff_weights_type,
                            data_type::undef, src_type, data_type::f32)
                    && IMPLICATION(this->with_bias(),
                            utils::one_of(
                                    this->desc()->diff_bias_desc.data_type,
                                    data_type::f32, data_type::bf16))
                    && attr()->has_default_values() && !has_zero_dim_memory()
                    && set_default_formats();
            if (!ok) return status::unimplemented;

            const int max_threads
                    = dnnl_in_parallel() ? 1 : dnnl_get_max_threads();

            status_t status = jit_uni_dw_conv_bwd_weights_kernel<isa,
                    src_type>::init_conf(jcp_, *desc(), *src_md(),
                    *diff_weights_md(), *diff_dst_md(), max_threads);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_bwd_weights_kernel<isa, src_type>::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = isa == sve_512 ? nChw16c : nChw8c;
            auto wei_tag = isa == sve_512 ? Goihw16g : Goihw8g;

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    jit_uni_dw_convolution_bwd_weights_t(const pd_t *apd);

    typedef typename prec_traits<data_type::bf16>::type bf16_data_t;
    typedef typename prec_traits<data_type::f32>::type f32_data_t;
    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<src_type>::type diff_dst_data_t;
    typedef typename prec_traits<diff_weights_type>::type diff_weights_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_uni_dw_conv_bwd_weights_kernel<isa, src_type>(
                        pd()->jcp_)));
        CHECK(kernel_->create_kernel());

        if (pd()->jcp_.nthr_mb > 1) {
            CHECK(safe_ptr_assign(
                    acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
            CHECK(acc_ker_->create_kernel());
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        execute_reduction(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    void execute_reduction(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;
    std::unique_ptr<jit_uni_dw_conv_bwd_weights_kernel<isa, src_type>> kernel_;
};

using jit_sve_512_dw_convolution_bwd_weights_t
        = jit_uni_dw_convolution_bwd_weights_t<sve_512, data_type::f32>;
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
