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

#ifndef CPU_AARCH64_JIT_UNI_BATCH_NORMALIZATION_S8_HPP
#define CPU_AARCH64_JIT_UNI_BATCH_NORMALIZATION_S8_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/cpu_batch_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace bnorm_s8_impl {
template <cpu_isa_t isa>
struct driver_t;
}

template <cpu_isa_t isa>
struct jit_uni_batch_normalization_s8_fwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("bnorm_s8_jit:", isa, ""),
                jit_uni_batch_normalization_s8_fwd_t);

        status_t init(engine_t *engine);
    };

    typedef int8_t data_t;

    jit_uni_batch_normalization_s8_fwd_t(const pd_t *apd);
    ~jit_uni_batch_normalization_s8_fwd_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    bnorm_s8_impl::driver_t<isa> *bnorm_driver_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
