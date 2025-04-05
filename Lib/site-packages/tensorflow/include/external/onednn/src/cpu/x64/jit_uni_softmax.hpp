/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_SOFTMAX_HPP
#define CPU_X64_JIT_UNI_SOFTMAX_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_softmax_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace softmax_impl {
// This class isolates primitive implementation from templates introduced by
// the kernel.
struct jit_softmax_kernel_base_t {
    static jit_softmax_kernel_base_t *create(const softmax_pd_t *pd,
            const cpu_isa_t isa, bool axis_is_plain_and_strided);

    virtual ~jit_softmax_kernel_base_t() = default;

    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const void *src, *dst, *diff_dst; // src dubs as diff_src
        const void *interim; // scratch memory for intermediate storage
        const void *src_scales; // src_scales defined for all data type cases
        const void *dst_scales; // dst_scales defined for all data type cases
        size_t process_n_elems;

        // post ops
        const void *dst_orig;
        const void *post_ops_binary_rhs_arg_vec;
    };

    virtual void operator()(const call_params_t *p) const = 0;
    virtual status_t create_kernel() = 0;

protected:
    jit_softmax_kernel_base_t(const softmax_pd_t *pd) : pd_(pd) {}

    const softmax_pd_t *pd_;
};

bcast_set_t get_supported_bcast_strategies();
std::vector<cpu_isa_t> get_supported_isa(bool is_fwd);
} // namespace softmax_impl

struct jit_uni_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        const char *impl_name() const {
            return JIT_IMPL_NAME_HELPER("jit:", isa_, "");
        }

        DECLARE_COMMON_PD_T(impl_name(), jit_uni_softmax_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            // try multiple vlen
            for (const auto &i : softmax_impl::get_supported_isa(true)) {
                if (mayiuse(i) && is_dense(i)) {
                    isa_ = i;
                    break;
                }
            }
            // not dense impl can be easily done
            VDISPATCH_SOFTMAX(isa_ != isa_undef,
                    "the axis blocking configuration is not supported");

            VDISPATCH_SOFTMAX(is_fwd(), VERBOSE_BAD_PROPKIND);

            VDISPATCH_SOFTMAX(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const auto src_dt = src_md()->data_type;
            const auto dst_dt = dst_md()->data_type;

            const bool src_dt_ok
                    = utils::one_of(src_dt, f32, bf16, f16, s8, u8);
            VDISPATCH_SOFTMAX(src_dt_ok, VERBOSE_UNSUPPORTED_DT);

            const bool dst_dt_ok
                    = utils::one_of(dst_dt, f32, bf16, f16, s8, u8);
            VDISPATCH_SOFTMAX(dst_dt_ok, VERBOSE_UNSUPPORTED_DT);

            // TODO: add int8 support for SSE41.
            const bool int8_isa_ok
                    = IMPLICATION((utils::one_of(s8, src_dt, dst_dt)
                                          || utils::one_of(u8, src_dt, dst_dt)),
                            is_superset(isa_, avx2));
            VDISPATCH_SOFTMAX(int8_isa_ok, VERBOSE_ISA_DT_MISMATCH);

            const bool bf16_isa_ok
                    = IMPLICATION(utils::one_of(bf16, src_dt, dst_dt),
                            is_superset(isa_, avx512_core)
                                    || is_superset(isa_, avx2_vnni_2));
            VDISPATCH_SOFTMAX(bf16_isa_ok, VERBOSE_ISA_DT_MISMATCH);

            const bool f16_isa_ok
                    = IMPLICATION(utils::one_of(f16, src_dt, dst_dt),
                            is_superset(isa_, avx512_core_fp16)
                                    || is_superset(isa_, avx2_vnni_2));
            VDISPATCH_SOFTMAX(f16_isa_ok, VERBOSE_ISA_DT_MISMATCH);

            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(skip_mask_t::scales_runtime
                            | skip_mask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);

            VDISPATCH_SOFTMAX(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_SOFTMAX(
                    memory_desc_wrapper(src_md()).similar_to(
                            memory_desc_wrapper(dst_md()), true, false, 0),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_SOFTMAX(
                    IMPLICATION(is_superset(isa_, avx2_vnni_2)
                                    && !is_superset(isa_, avx512_core)
                                    && (utils::one_of(bf16, src_dt, dst_dt)
                                            || utils::one_of(
                                                    f16, src_dt, dst_dt)),
                            memory_desc_wrapper(src_md()).is_plain()),
                    "avx2_vnni_2 only supports xf16 on plain layout");

            const memory_desc_wrapper dst_d(dst_md());
            axis_is_plain_and_strided_ = dst_d.is_plain() && axis_stride() > 1;
            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            return status::success;
        };

        int nthr_; // To not exceed the limit in execute used for set up.
        size_t scratch_size_per_thr_ = 0;
        cpu_isa_t isa_ = isa_undef;
        bool axis_is_plain_and_strided_ = false;

    private:
        void init_scratchpad() {
            if (dst_md()->data_type != data_type::f32) {
                auto scratchpad = scratchpad_registry().registrar();
                // When stride != 1, then each thread operates over simd at a
                // time, thus, increased scratchpad size.
                dim_t elem_per_thr = 1;
                if (axis_is_plain_and_strided_) {
                    // Strided case has scratchpad using a simd_w for an element
                    elem_per_thr = isa_max_vlen(isa_) / sizeof(float);
                }
                scratch_size_per_thr_
                        = axis_size(true) * elem_per_thr * sizeof(float);
                scratchpad.template book<char>(
                        memory_tracking::names::key_softmax_interim_store,
                        scratch_size_per_thr_ * nthr_);
            }
        }

        bool post_ops_ok() const {
            const auto &post_ops = attr()->post_ops_;
            const bool with_sum = post_ops.find(primitive_kind::sum) != -1;
            const std::vector<injector::post_op_type> accepted_post_ops
                    = {injector::eltwise, injector::binary};
            const memory_desc_wrapper dst_d(dst_md());
            injector::post_ops_ok_args_t post_ops_args(isa_, accepted_post_ops,
                    attr()->post_ops_, &dst_d, true, true, true, true,
                    softmax_impl::get_supported_bcast_strategies());

            const auto has_per_oc_bin_po = [&]() {
                for (size_t i = 0; i < static_cast<size_t>(post_ops.len());
                        i++) {
                    const auto &entry = post_ops.entry_[i];
                    if (!entry.is_like_binary()) continue;

                    const auto &src1_md
                            = binary_injector::get_src1_desc(entry, dst_d);
                    const auto &cur_strategy
                            = get_rhs_arg_broadcasting_strategy(src1_md, dst_d,
                                    softmax_impl::
                                            get_supported_bcast_strategies());
                    if (cur_strategy == broadcasting_strategy_t::per_oc
                            || cur_strategy
                                    == broadcasting_strategy_t::per_oc_spatial)
                        return true;
                }
                return false;
            };
            // `per_oc` policy is reserved for axis being 1 as it is for the
            // rest of the library, otherwise, `per_oc` doesn't make any sense.
            const bool bin_po_ok
                    = IMPLICATION(has_per_oc_bin_po(), axis() == 1);

            return !with_sum && injector::post_ops_ok(post_ops_args)
                    && bin_po_ok;
        }

        bool is_dense(const cpu_isa_t isa) const {
            const memory_desc_wrapper src_d(src_md());
            const auto &bd = src_d.blocking_desc();

            if (!src_d.is_dense(true) || !src_d.only_padded_dim(axis()))
                return false;

            if (src_d.is_plain()) return true;

            // It is fine to use float here as the kernel uses halfs of
            // vector registers.
            const dim_t blk_size = isa_max_vlen(isa) / sizeof(float);
            // 31 is a general limit, 2 is for unroll_regs_ = 4;
            const size_t max_stride = (1LL << (31 - 2)) - 1;
            const int last_blk = bd.inner_nblks - 1;
            return bd.inner_blks[last_blk] == blk_size
                    && bd.inner_idxs[last_blk] == axis()
                    && sizeof(float) * bd.strides[axis()] < max_stride;
        }
    };

    jit_uni_softmax_fwd_t(const pd_t *apd);

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<softmax_impl::jit_softmax_kernel_base_t> ker_;
};

struct jit_uni_softmax_bwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_bwd_pd_t {
        using cpu_softmax_bwd_pd_t::cpu_softmax_bwd_pd_t;

        const char *impl_name() const {
            return JIT_IMPL_NAME_HELPER("jit:", isa_, "");
        }

        DECLARE_COMMON_PD_T(impl_name(), jit_uni_softmax_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            // try multiple vlen
            for (const auto &i : softmax_impl::get_supported_isa(false)) {
                if (mayiuse(i) && is_dense(i)) {
                    isa_ = i;
                    break;
                }
            }
            // not dense impl can be easily done
            VDISPATCH_SOFTMAX(isa_ != isa_undef,
                    "the axis blocking configuration is not supported");

            VDISPATCH_SOFTMAX(!is_fwd(), VERBOSE_BAD_PROPKIND);

            VDISPATCH_SOFTMAX(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const auto dst_dt = dst_md()->data_type;
            const auto diff_dst_dt = diff_dst_md()->data_type;
            const auto diff_src_dt = diff_src_md()->data_type;

            const bool dst_dt_ok = utils::one_of(dst_dt, f32, bf16, f16);
            VDISPATCH_SOFTMAX(dst_dt_ok, VERBOSE_UNSUPPORTED_DT);

            const bool diff_dst_dt_ok
                    = utils::one_of(diff_dst_dt, f32, bf16, f16);
            VDISPATCH_SOFTMAX(diff_dst_dt_ok, VERBOSE_UNSUPPORTED_DT);

            const bool diff_src_dt_ok
                    = utils::one_of(diff_src_dt, f32, bf16, f16);
            VDISPATCH_SOFTMAX(diff_src_dt_ok, VERBOSE_UNSUPPORTED_DT);

            const bool bf16_isa_ok = IMPLICATION(
                    utils::one_of(bf16, dst_dt, diff_dst_dt, diff_src_dt),
                    is_superset(isa_, avx512_core));
            VDISPATCH_SOFTMAX(bf16_isa_ok, VERBOSE_ISA_DT_MISMATCH);

            const bool f16_isa_ok = IMPLICATION(
                    utils::one_of(f16, dst_dt, diff_dst_dt, diff_src_dt),
                    is_superset(isa_, avx512_core_fp16));
            VDISPATCH_SOFTMAX(f16_isa_ok, VERBOSE_ISA_DT_MISMATCH);

            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_SOFTMAX(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_SOFTMAX(
                    memory_desc_wrapper(diff_src_md())
                            .similar_to(memory_desc_wrapper(diff_dst_md()),
                                    true, false, 0),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
        cpu_isa_t isa_ = isa_undef;

    private:
        bool is_dense(const cpu_isa_t isa) const {
            const memory_desc_wrapper dst_d(dst_md());
            const auto &bd = dst_d.blocking_desc();

            if (!dst_d.is_dense(true) || !dst_d.only_padded_dim(axis()))
                return false;

            // It is fine to use float here as the kernel uses halfs of
            // vector registers.
            const dim_t blk_size = isa_max_vlen(isa) / sizeof(float);
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
        }
    };

    jit_uni_softmax_bwd_t(const pd_t *apd);

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<softmax_impl::jit_softmax_kernel_base_t> ker_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
