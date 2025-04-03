/*******************************************************************************
* Copyright 2022-2024 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_DECONVOLUTION_HPP
#define CPU_AARCH64_ACL_DECONVOLUTION_HPP

#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/cpu_deconvolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_deconv_obj_t {
    arm_compute::NEDeconvolutionLayer deconv;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_deconv_conf_t {
    bool with_bias;
    // If this is true, the result of the convolution goes into a temporarily
    // allocated ACL tensor to be accumulated into the oneDNN dst during postops
    bool use_dst_acc;
    bool fast_math;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo wei_info;
    arm_compute::TensorInfo bia_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::PadStrideInfo deconv_info;
};

struct acl_deconv_resource_t : public resource_t {
    acl_deconv_resource_t()
        : acl_obj_(utils::make_unique<acl_deconv_obj_t>()) {}

    status_t configure(const acl_deconv_conf_t &adp) {
        if (!acl_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src_tensor.allocator()->init(adp.src_info);
        acl_obj_->wei_tensor.allocator()->init(adp.wei_info);
        acl_obj_->bia_tensor.allocator()->init(adp.bia_info);
        acl_obj_->dst_tensor.allocator()->init(adp.dst_info);

        // clang-format off
        acl_obj_->deconv.configure(
            &acl_obj_->src_tensor,
            &acl_obj_->wei_tensor,
            adp.with_bias ? &acl_obj_->bia_tensor : nullptr,
            &acl_obj_->dst_tensor,
            adp.deconv_info, adp.fast_math);
        // clang-format on

        return status::success;
    }

    acl_deconv_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_deconv_resource_t);

private:
    std::unique_ptr<acl_deconv_obj_t> acl_obj_;
}; // acl_deconv_resource_t

struct acl_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , acl_pd_conf()
            , post_ops() {}

        DECLARE_COMMON_PD_T("acl", acl_deconvolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            using smask_t = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper wei_d(&weights_md_);
            const memory_desc_wrapper dst_d(&dst_md_);
            const memory_desc_wrapper bia_d(&bias_md_);

            const auto src_data_t = src_d.data_type();
            const auto wei_data_t = wei_d.data_type();
            const auto dst_data_t = dst_d.data_type();
            const auto bia_data_t = bia_d.data_type();

            const bool ok = is_fwd() // Only forward deconvolutions
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::deconvolution_direct)
                    && (expect_data_types(f16, f16, f16, f16)
                            || expect_data_types(f32, f32, f32, f32))
                    && attr()->has_default_values(
                            smask_t::post_ops | smask_t::fpmath_mode,
                            dst_data_t);
            if (!ok) return status::unimplemented;

            // Strides
            const auto sh = KSH();
            const auto sw = KSW();

            // Padding
            const auto pt = padT();
            const auto pb = padB();
            const auto pl = padL();
            const auto pr = padR();

            acl_pd_conf.deconv_info = arm_compute::PadStrideInfo(sw, sh, pl, pr,
                    pt, pb, arm_compute::DimensionRoundingType::FLOOR);

            // Tensor info
            const auto mb = MB();

            const auto ih = IH();
            const auto iw = IW();
            const auto ic = IC();

            const auto oh = OH();
            const auto ow = OW();
            const auto oc = OC();

            const auto kw = KW();
            const auto kh = KH();

            const bool with_groups = G() != 1;
            const int ndims = src_d.ndims();
            const bool is_1d = ndims == 3;
            const bool is_3d = ndims == 5;

            // Compute Library unsupported shape scenarios
            if (utils::one_of(true, is_3d, is_1d, with_groups)) {
                return status::unimplemented;
            }

            acl_pd_conf.with_bias
                    = desc_.bias_desc.format_kind != format_kind::undef;

            // Data type
            auto acl_src_data_t = acl_utils::get_acl_data_t(src_data_t);
            auto acl_wei_data_t = acl_utils::get_acl_data_t(wei_data_t);
            auto acl_dst_data_t = acl_utils::get_acl_data_t(dst_data_t);
            auto acl_bia_data_t = acl_utils::get_acl_data_t(bia_data_t);

            if (acl_bia_data_t == arm_compute::DataType::UNKNOWN) {
                acl_bia_data_t = arm_compute::DataType::F32;
            }

            // Set memory formats
            auto src_tag = src_d.format_kind() == format_kind::any
                    ? nhwc
                    : memory_desc_matches_one_of_tag(src_md_, nhwc, nchw);
            auto dst_tag = dst_d.format_kind() == format_kind::any
                    ? nhwc
                    : memory_desc_matches_one_of_tag(dst_md_, nhwc, nchw);

            bool is_nspc = src_tag == nhwc;

            auto wei_tag = wei_d.format_kind() == format_kind::any
                    ? (is_nspc ? ohwi : oihw)
                    : memory_desc_matches_one_of_tag(weights_md_, ohwi, oihw);

            // Compute Library does not support mismatching layouts
            if ((src_tag != wei_tag) || (src_tag != dst_tag)) {
                return status::unimplemented;
            }

            CHECK(memory_desc_init_by_tag(src_md_, src_tag));
            CHECK(memory_desc_init_by_tag(dst_md_, dst_tag));
            CHECK(memory_desc_init_by_tag(weights_md_, wei_tag));
            if (acl_pd_conf.with_bias) {
                CHECK(memory_desc_init_by_tag(bias_md_, format_tag::a));
            }

            // Data layout
            const auto acl_layout = is_nspc ? arm_compute::DataLayout::NHWC
                                            : arm_compute::DataLayout::NCHW;

            acl_pd_conf.src_info = arm_compute::TensorInfo(is_nspc
                            ? arm_compute::TensorShape(ic, iw, ih, mb)
                            : arm_compute::TensorShape(iw, ih, ic, mb),
                    1, acl_src_data_t, acl_layout);

            auto wei_info_tensor_shape = is_nspc
                    ? arm_compute::TensorShape(ic, kw, kh, oc)
                    : arm_compute::TensorShape(kw, kh, ic, oc);
            // ACL removes last dimension if dim is 1.
            // Below fix ensures the tensor shape is correct when queried.
            wei_info_tensor_shape.set_num_dimensions(4);
            acl_pd_conf.wei_info = arm_compute::TensorInfo(
                    wei_info_tensor_shape, 1, acl_wei_data_t, acl_layout);

            acl_pd_conf.dst_info = arm_compute::TensorInfo(is_nspc
                            ? arm_compute::TensorShape(oc, ow, oh, mb)
                            : arm_compute::TensorShape(ow, oh, oc, mb),
                    1, acl_dst_data_t, acl_layout);

            acl_pd_conf.bia_info = arm_compute::TensorInfo(acl_pd_conf.with_bias
                            ? arm_compute::TensorShape(oc)
                            : arm_compute::TensorShape(),
                    1, acl_bia_data_t, acl_layout);

            acl_pd_conf.fast_math = utils::one_of(
                    attr()->fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);

            ACL_CHECK_VALID(arm_compute::NEDeconvolutionLayer::validate(
                    &acl_pd_conf.src_info, &acl_pd_conf.wei_info,
                    acl_pd_conf.with_bias ? &acl_pd_conf.bia_info : nullptr,
                    &acl_pd_conf.dst_info, acl_pd_conf.deconv_info,
                    acl_pd_conf.fast_math));

            // Calculate scaled output tensor info for determining the convolution method
            auto out_dims = arm_compute::deconvolution_output_dimensions(
                    iw, ih, kw, kh, acl_pd_conf.deconv_info);
            uint32_t deconv_pad_x = 0;
            uint32_t deconv_pad_y = 0;
            auto scale_out_shape = arm_compute::misc::shape_calculator::
                    compute_deconvolution_upsampled_shape(acl_pd_conf.src_info,
                            acl_pd_conf.wei_info, sw, sh, out_dims,
                            deconv_pad_x, deconv_pad_y);

            // If strides are unit in all dimensions, upsampling of input is skipped and a correct
            // padding is set for convolution. Otherwise, describe deconvolution as convolution of
            // upsampling input with stride = 1 and pad = 0.
            arm_compute::ConvolutionMethod conv_method;
            arm_compute::TensorInfo *conv_src_info;
            unsigned int pad_left = 0;
            unsigned int pad_right = 0;
            unsigned int pad_top = 0;
            unsigned int pad_bottom = 0;
            if (sh != 1 || sw != 1) {
                arm_compute::TensorInfo scale_out_info(
                        acl_pd_conf.src_info.clone()
                                ->set_is_resizable(true)
                                .reset_padding()
                                .set_tensor_shape(scale_out_shape));
                conv_src_info = &scale_out_info;
            } else {
                // compute correct padding here
                pad_left = pr > pl ? pr - pl : 0;
                pad_right = pl > pr ? pl - pr : 0;
                pad_top = pb > pt ? pb - pt : 0;
                pad_bottom = pt > pb ? pt - pb : 0;

                deconv_pad_x -= pad_left + pad_right;
                deconv_pad_y -= pad_top + pad_bottom;

                pad_left += deconv_pad_x / 2;
                pad_right += deconv_pad_x / 2;
                pad_top += deconv_pad_y / 2;
                pad_bottom += deconv_pad_y / 2;

                conv_src_info = &acl_pd_conf.src_info;
            }
            const arm_compute::PadStrideInfo conv_info(1, 1, pad_left,
                    pad_right, pad_top, pad_bottom,
                    arm_compute::DimensionRoundingType::CEIL);
            conv_method
                    = arm_compute::NEConvolutionLayer::get_convolution_method(
                            conv_src_info, &acl_pd_conf.wei_info,
                            &acl_pd_conf.dst_info, conv_info,
                            arm_compute::WeightsInfo(),
                            arm_compute::Size2D(1U, 1U),
                            arm_compute::ActivationLayerInfo(),
                            acl_pd_conf.fast_math);

            // Disable use of winograd based convolution algorithm when performing
            // direct deconvolution because it introduces accuracy loss.
            if (conv_method == arm_compute::ConvolutionMethod::WINOGRAD) {
                return status::unimplemented;
            }

            CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_));
            acl_pd_conf.use_dst_acc = post_ops.has_sum();

            return status::success;
        }

        acl_deconv_conf_t acl_pd_conf;
        acl_post_ops_t post_ops;

    private:
        bool post_ops_ok() const {
            return attr()->post_ops_.find(primitive_kind::convolution) == -1;
        }
    }; // pd_t

    acl_deconvolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_deconv_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        auto st = r->configure(pd()->acl_pd_conf);
        if (st == status::success) { mapper.add(this, std::move(r)); }

        CHECK(pd()->post_ops.create_resource(engine, mapper));

        return st;
    }

private:
    // To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_deconvolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
