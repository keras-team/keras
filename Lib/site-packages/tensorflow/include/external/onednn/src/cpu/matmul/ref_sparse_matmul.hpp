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

#ifndef CPU_MATMUL_REF_SPARSE_MATMUL_HPP
#define CPU_MATMUL_REF_SPARSE_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct ref_sparse_matmul_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_sparse_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            const bool ok
                    = utils::everyone_is(f32, src_type, wei_type, dst_type)
                    && utils::one_of(true, wei_d.is_sparse_desc(),
                            src_d.is_sparse_desc())
                    && IMPLICATION(wei_d.is_sparse_desc(),
                            wei_d.encoding() == sparse_encoding::csr)
                    && IMPLICATION(src_d.is_sparse_desc(),
                            src_d.encoding() == sparse_encoding::csr)
                    && IMPLICATION(
                            wei_d.is_sparse_desc(), !src_d.is_sparse_desc())
                    && IMPLICATION(src_d.is_sparse_desc(),
                            utils::everyone_is(s32, src_d.metadata_type(0),
                                    src_d.metadata_type(1)))
                    && IMPLICATION(wei_d.is_sparse_desc(),
                            utils::everyone_is(s32, wei_d.metadata_type(0),
                                    wei_d.metadata_type(1)))
                    && !with_bias() && attr()->has_default_values()
                    && set_default_formats() && formats_ok(src_d, wei_d);
            return ok ? status::success : status::unimplemented;
        }

        bool formats_ok(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &wei_d) const {
            if (!memory_desc_wrapper(dst_md()).matches_one_of_tag(
                        format_tag::ab))
                return false;
            if (src_d.is_sparse_desc())
                return wei_d.matches_one_of_tag(format_tag::ab);
            if (wei_d.is_sparse_desc())
                return src_d.matches_one_of_tag(format_tag::ab);
            return false;
        }
    };

    ref_sparse_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
