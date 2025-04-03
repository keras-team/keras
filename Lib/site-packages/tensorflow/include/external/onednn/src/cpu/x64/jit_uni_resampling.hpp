/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef CPU_X64_UNI_RESAMPLING_HPP
#define CPU_X64_UNI_RESAMPLING_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_resampling_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_uni_resampling_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_uni_resampling_fwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_fwd_pd_t {
        using cpu_resampling_fwd_pd_t::cpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", conf_.isa, ""),
                jit_uni_resampling_fwd_t);

        status_t init(engine_t *engine);

        const jit_resampling_conf_t &get_conf() const { return conf_; }

    private:
        void fill_format_tag_info();

        jit_resampling_conf_t conf_;
    };

    jit_uni_resampling_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    virtual ~jit_uni_resampling_fwd_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t fill_data_for_interpolation();
    /*
     * Fills indices_ with the data that contains the corresponding
     * input point for each output point.
     * The data is arranged as follows:
     * od_0 = id_0 * stride_w
     * od_1 = id_1 * stride_w
     * od_2 = id_2 * stride_w
     * ...
     * ih_0 = ih_0 * stride_h
     * ih_1 = ih_1 * stride_h
     * ...
     * iw_0 = iw_1 * stride_w
     * ...
     */
    status_t fill_data_for_nearest();
    /*
     * Fills indices_ with the data that contains the corresponding
     * input point for each output point.
     * The data is arranged as follows:
     * od_0 = id_0
     * od_1 = id_1
     * od_2 = id_2
     * ...
     * oh_0 = ih_0
     * oh_1 = ih_1
     * ...
     * ow_0 = iw_0
     * ...
     */
    status_t fill_data_for_linear();
    /*
     * Fills indices_ with the data that contains the corresponding
     * corners from input tensor for each output point and fills
     * weights_ with with the data that contains weights for
     * corners from input tensor for each output point.
     * The data is arranged as follows:
     * NSPC and BLOCKED:
     *
     * indices_:
     * ow_0 = iw_0_left
     * ow_0 = iw_0_right
     * ow_1 = iw_1_left
     * ow_1 = iw_1_right
     * ...
     * oh_0 = ih_0_top
     * oh_1 = ih_1_top
     * ...
     * oh_0 = ih_0_bottom
     * oh_1 = ih_1_bottom
     * ...
     * od_0 = id_0_front
     * od_1 = id_1_front
     * ...
     * od_0 = id_0_back
     * od_1 = id_1_back
     * ...
     *
     * weights_:
     * ow_0 = weight_0_left
     * ow_0 = weight_0_right
     * ow_1 = weight_1_left
     * ow_1 = weight_1_right
     * ...
     * oh_0 = weight_0_top
     * oh_1 = weight_1_top
     * ...
     * oh_0 = weight_0_bottom
     * oh_1 = weight_1_bottom
     * ...
     * od_0 = weight_0_front
     * od_1 = weight_1_front
     * ...
     * od_0 = weight_0_back
     * od_1 = weight_1_back
     * ...
     *
     * NCSP:
     *
     * indices_:
     * sp_0 = id_0_front + ih_0_top + iw_0_left
     * sp_0 = id_0_front + ih_0_top + iw_0_right
     * sp_0 = id_0_front + ih_0_bottom + iw_0_left
     * sp_0 = id_0_front + ih_0_bottom + iw_0_right
     * sp_0 = id_0_back + ih_0_top + iw_0_left
     * sp_0 = id_0_back + ih_0_top + iw_0_right
     * sp_0 = id_0_back + ih_0_bottom + iw_0_left
     * sp_0 = id_0_back + ih_0_bottom + iw_0_right
     * sp_1 = id_1_front + ih_1_top + iw_1_left
     * sp_1 = id_1_front + ih_1_top + iw_1_right
     * sp_1 = id_1_front + ih_1_bottom + iw_1_left
     * ...
     *
     * weights_:
     * sp_0 = weight_0_front * weight_0_top * weight_0_left
     * sp_0 = weight_0_front * weight_0_top * weight_0_right
     * sp_0 = weight_0_front * weight_0_bottom * weight_0_left
     * sp_0 = weight_0_front * weight_0_bottom * weight_0_right
     * sp_0 = weight_0_back * weight_0_top * weight_0_left
     * sp_0 = weight_0_back * weight_0_top * weight_0_right
     * sp_0 = weight_0_back * weight_0_bottom * weight_0_left
     * sp_0 = weight_0_back * weight_0_bottom * weight_0_right
     * sp_1 = weight_1_front * weight_1_top * weight_1_left
     * sp_1 = weight_1_front * weight_1_top * weight_1_right
     * sp_1 = weight_1_front * weight_1_bottom * weight_1_left
     * ...
     */

    status_t interpolate_nearest(const uint8_t *src, uint8_t *dst,
            const std::vector<const void *> &post_ops_args) const;
    status_t interpolate_linear(const uint8_t *src, uint8_t *dst,
            const std::vector<const void *> &post_ops_args) const;

    status_t get_proper_kernel_for_avx512(
            const memory_desc_t *dst_md, const jit_resampling_conf_t &conf);
    status_t get_proper_kernel_for_avx(
            const memory_desc_t *dst_md, const jit_resampling_conf_t &conf);
    status_t get_proper_kernel_for_sse(
            const memory_desc_t *dst_md, const jit_resampling_conf_t &conf);

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_resampling_kernel_base_t> kernel_;

    std::vector<unsigned> indices_;
    std::vector<float> weights_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
