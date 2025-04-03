/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_SOFTMAX_HPP
#define CPU_AARCH64_JIT_UNI_SOFTMAX_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/cpu_softmax_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace softmax_impl {
template <cpu_isa_t isa>
struct driver_t;
}

template <cpu_isa_t isa>
struct jit_uni_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_softmax_fwd_t);

        status_t init(engine_t *engine) {
            auto is_dense = [&]() {
                const memory_desc_wrapper src_d(src_md());
                const auto &bd = src_d.blocking_desc();

                if (!src_d.is_dense(true) || !src_d.only_padded_dim(axis()))
                    return false;

                if (src_d.is_plain()) return bd.strides[axis()] == 1;

                // It is fine to use float here as the kernel uses halfs of
                // vector registers.
                const auto blk_size = cpu_isa_traits<isa>::vlen / sizeof(float);
                // 31 is a general limit, 2 is for unroll_regs_ = 4;
                const size_t max_stride = (1LL << (31 - 2)) - 1;
                const int last_blk = bd.inner_nblks - 1;
                return bd.inner_blks[last_blk] == blk_size
                        && bd.inner_idxs[last_blk] == axis()
                        && sizeof(float) * bd.strides[axis()] < max_stride;
            };

            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            const auto src_dt = src_md()->data_type;
            const auto dst_dt = dst_md()->data_type;
            bool ok = mayiuse(isa) && is_fwd() && !has_zero_dim_memory()
                    && utils::one_of(src_dt, f32, s8, u8)
                    && utils::one_of(dst_dt, f32, s8, u8)
                    && (mayiuse(sve_512) || mayiuse(sve_256)
                            || mayiuse(sve_128))
                    && attr()->has_default_values(skip_mask_t::scales_runtime)
                    && attr_scales_ok()
                    && set_default_formats() == status::success;
            if (!ok) return status::unimplemented;

            ok = memory_desc_wrapper(src_md()).similar_to(
                         memory_desc_wrapper(dst_md()), true, false, 0)
                    && is_dense(); // not dense impl can be easily done
            if (!ok) return status::unimplemented;

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            return status::success;
        };

        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            if (utils::one_of(
                        dst_md()->data_type, data_type::u8, data_type::s8)) {
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<char>(
                        memory_tracking::names::key_softmax_interim_store,
                        axis_size(true) * sizeof(float) * nthr_);
            }
        }
    };

    jit_uni_softmax_fwd_t(const pd_t *apd);
    ~jit_uni_softmax_fwd_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    softmax_impl::driver_t<isa> *softmax_driver_;
};

template <cpu_isa_t isa>
struct jit_uni_softmax_bwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_bwd_pd_t {
        using cpu_softmax_bwd_pd_t::cpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_softmax_bwd_t);

        status_t init(engine_t *engine) {
            auto is_dense = [&]() {
                const memory_desc_wrapper dst_d(dst_md());
                const auto &bd = dst_d.blocking_desc();

                if (!dst_d.is_dense(true) || !dst_d.only_padded_dim(axis()))
                    return false;

                // It is fine to use float here as the kernel uses halfs of
                // vector registers.
                const auto blk_size = cpu_isa_traits<isa>::vlen / sizeof(float);
                if (dst_d.is_plain())
                    return bd.strides[axis()] == 1;
                else {
                    // 31 is a general limit, 2 is for unroll_regs_ = 4;
                    const size_t max_stride = (1LL << (31 - 2)) - 1;
                    const int last_blk = bd.inner_nblks - 1;
                    return bd.inner_blks[last_blk] == blk_size
                            && bd.inner_idxs[last_blk] == axis()
                            && sizeof(float) * bd.strides[axis()] < max_stride;
                }
            };

            using namespace data_type;
            bool ok = mayiuse(isa) && !is_fwd() && !has_zero_dim_memory()
                    && utils::one_of(dst_md()->data_type, f32)
                    && utils::one_of(diff_dst_md()->data_type, f32)
                    && utils::one_of(diff_src_md()->data_type, f32)
                    && (mayiuse(sve_512) || mayiuse(sve_256))
                    && attr()->has_default_values()
                    && set_default_formats() == status::success;
            if (!ok) return status::unimplemented;

            ok = memory_desc_wrapper(diff_src_md())
                            .similar_to(memory_desc_wrapper(diff_dst_md()),
                                    true, false, 0)
                    && memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(dst_md())
                    && is_dense(); // not dense impl can be easily done
            if (!ok) return status::unimplemented;

            return status::success;
        };
    };

    jit_uni_softmax_bwd_t(const pd_t *apd);
    ~jit_uni_softmax_bwd_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    softmax_impl::driver_t<isa> *softmax_driver_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
