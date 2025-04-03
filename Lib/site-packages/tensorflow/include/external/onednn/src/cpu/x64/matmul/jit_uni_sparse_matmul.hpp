/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_SPARSE_MATMUL_HPP
#define CPU_X64_JIT_UNI_SPARSE_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

struct sparse_matmul_kernel_t;

struct jit_uni_sparse_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_sparse_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            const bool problem_dt_correct
                    = utils::everyone_is(f32, src_type, wei_type, dst_type)
                    && src_d.is_sparse_desc() && !wei_d.is_sparse_desc()
                    && utils::everyone_is(s32, src_d.metadata_type(0),
                            src_d.metadata_type(1));

            VDISPATCH_MATMUL(problem_dt_correct, VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(formats_ok(), VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        bool formats_ok() const {
            const bool is_dst_ab
                    = memory_desc_wrapper(dst_md()).matches_one_of_tag(
                            format_tag::ab);
            const bool is_wei_ab = memory_desc_wrapper(weights_md())
                                           .matches_one_of_tag(format_tag::ab);
            return is_dst_ab && is_wei_ab;
        }
    };

    jit_uni_sparse_matmul_t(const pd_t *apd);
    ~jit_uni_sparse_matmul_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<sparse_matmul_kernel_t> kernel_;
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
