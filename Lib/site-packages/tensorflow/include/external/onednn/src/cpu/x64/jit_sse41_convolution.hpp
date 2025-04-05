/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_SSE41_CONVOLUTION_HPP
#define CPU_X64_JIT_SSE41_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_sse41_conv_kernel_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_sse41_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", sse41, ""),
                jit_sse41_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(expect_data_types(f32, f32, f32, f32, f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            CHECK(jit_sse41_conv_fwd_kernel_f32::init_conf(jcp_, *desc(),
                    *src_md(), *weights_md(), *dst_md(), *attr(),
                    dnnl_get_max_threads()));

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper dst_d(&dst_md_);

            const auto dat_tag_nxc = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            const auto dat_tag_ncx = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            const auto dat_tag_nCx8c
                    = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            const auto curr_src_tag = src_d.matches_one_of_tag(
                    dat_tag_nxc, dat_tag_ncx, dat_tag_nCx8c);
            const auto curr_dst_tag = dst_d.matches_one_of_tag(
                    dat_tag_nxc, dat_tag_ncx, dat_tag_nCx8c);
            const auto is_data_layout_nxc
                    = IMPLICATION(curr_src_tag != dat_tag_nxc,
                              src_d.format_kind() == format_kind::any)
                    && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                            dst_d.format_kind() == format_kind::any)
                    && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);
            const bool flat = IC() == 3;
            auto src_tag = is_data_layout_nxc ? dat_tag_nxc
                    : flat                    ? dat_tag_ncx
                                              : dat_tag_nCx8c;
            auto dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            auto wei_tag = with_groups()
                    ? utils::pick(2 * ndims() - 6 + flat, gOIw8i8o, gOwi8o,
                            gOIhw8i8o, gOhwi8o, gOIdhw8i8o, gOdhwi8o)
                    : utils::pick(2 * ndims() - 6 + flat, OIw8i8o, Owi8o,
                            OIhw8i8o, Ohwi8o, OIdhw8i8o, Odhwi8o);

            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }
    };

    jit_sse41_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_sse41_conv_fwd_kernel_f32(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_sse41_conv_fwd_kernel_f32> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
