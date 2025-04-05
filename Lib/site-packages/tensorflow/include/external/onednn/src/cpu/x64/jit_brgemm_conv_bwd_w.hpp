/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_BWD_W_HPP
#define CPU_X64_JIT_BRGEMM_CONV_BWD_W_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_containers.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_brgemm_conv_comp_pad_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"

#include "cpu/x64/jit_avx512_core_amx_conv_kernel.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct brgemm_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , brgs_sz_(0)
            , bs_c(0) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("brgconv_bwd_w:", jcp_.isa, ""),
                brgemm_convolution_bwd_weights_t);

        status_t init(engine_t *engine);

        jit_brgemm_conv_conf_t jcp_;
        jit_conv_conf_t jit_jcp_;
        void copy2jit_jcp();

        int brgs_sz_;
        std::shared_ptr<brgemm_containers::brgemm_desc_container_t> brgs_;

        int bs_c;
        std::vector<int> batchsizes;
        bool are_empty_bs {false};

        int get_brg_idx(int bs, int m, bool do_initialization, bool is_N_tail,
                bool is_K_tail) const {
            auto my_bs = jcp_.var_bs ? 1 : bs;
            auto bs_idx = jcp_.use_uker ? batchsizes[my_bs] : 0;
            assert(bs_idx >= 0);
            return (((m * bs_c + bs_idx) * 2
                            + static_cast<int>(do_initialization))
                                   * 2
                           + static_cast<int>(is_N_tail))
                    * 2
                    + static_cast<int>(is_K_tail);
        }
        inline int filter_w_to_src(int kw) const {
            return kw * (jcp_.dilate_w + 1);
        }
        inline int filter_h_to_src(int kh) const {
            return kh * (jcp_.dilate_h + 1) - jcp_.t_pad;
        }
        inline int filter_d_to_src(int kd) const {
            return kd * (jcp_.dilate_d + 1) - jcp_.f_pad;
        }
        inline int get_start_ih(int kh, int oh_s) const {
            const auto real_ih = filter_h_to_src(kh) + oh_s * jcp_.stride_h;
            return utils::saturate(0, jcp_.ih,
                    real_ih
                            + utils::rnd_up(
                                    nstl::max(0, -real_ih), jcp_.stride_h));
        }
        inline int get_finish_ih(int kh, int oh_e) const {
            return utils::saturate(0, jcp_.ih,
                    filter_h_to_src(kh) + (oh_e - 1) * jcp_.stride_h + 1);
        }
        inline int get_start_id(int kd, int od_s) const {
            const auto real_id = filter_d_to_src(kd) + od_s * jcp_.stride_d;
            return utils::saturate(0, jcp_.id,
                    real_id
                            + utils::rnd_up(
                                    nstl::max(0, -real_id), jcp_.stride_d));
        }
        inline int get_finish_id(int kd, int od_e) const {
            return utils::saturate(0, jcp_.id,
                    filter_d_to_src(kd) + (od_e - 1) * jcp_.stride_d + 1);
        }

        inline int get_finish_oh(int oh_s, int start, int end) const {
            int work_rem = end - start;
            return (oh_s + work_rem > jcp_.oh ? jcp_.oh : oh_s + work_rem);
        }
        inline int get_finish_od(int od_s, int start, int end) const {
            int work_rem = end - start;
            return (od_s + work_rem > jcp_.od ? jcp_.od : od_s + work_rem);
        }
    };

    brgemm_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    struct thread_info_t;

    void execute_backward_weights(const exec_ctx_t &ctx) const;
    void prepare_scratchpad_data(const exec_ctx_t &ctx) const;
    void compute_diff_weights_2d(thread_info_t *) const;
    void compute_diff_weights_3d(thread_info_t *) const;
    void reduce_and_convert_diff_weights_and_bias(thread_info_t *) const;
    void store_in_vnni_format(thread_info_t *) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;

    std::unique_ptr<jit_diff_wei_trans_to_vnni_t> diff_wei_trans_kernel_;
    std::unique_ptr<jit_trans_src_t> trans_kernel_;
    std::unique_ptr<jit_trans_dst_t> trans_dst_kernel_;
    std::unique_ptr<jit_avx512_core_amx_bwd_bias_kernel_t> diff_bias_kernel_;

    brgemm_containers::brgemm_kernel_container_t brg_kernels_;
    brgemm_containers::brgemm_palette_container_t brgemm_palettes_;

    status_t add_brg_kernel(int bs, int M, int i_N, int i_K, int i_init);
    void call_brgemm_kernel(
            thread_info_t &btc, int brg_idx, int batch_size, void *ptr_C) const;
    inline dim_t wei_offset_int(
            int g, int oc_b, int ic_b, int kd, int kh, int kw) const {
        const auto &jcp = pd()->jcp_;
        const dim_t const_extra_offset = jcp.ic_block * jcp.oc_block;
        dim_t extra_offset
                = ((kd * jcp.kh + kh) * jcp.kw + kw) * const_extra_offset;
        return (dim_t)((g * jcp.nb_oc + oc_b) * jcp.nb_ic + ic_b) * jcp.kd
                * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block
                + extra_offset;
    }

    inline dim_t wei_offset_int(int g, int oc_b, int ic_b, int kX) const {
        const auto &jcp = pd()->jcp_;
        const dim_t const_extra_offset = jcp.kw * jcp.ic_block * jcp.oc_block;
        dim_t extra_offset = (jcp.ndims == 5) ? kX * jcp.kh * const_extra_offset
                                              : kX * const_extra_offset;
        return (dim_t)((g * jcp.nb_oc + oc_b) * jcp.nb_ic + ic_b) * jcp.kd
                * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block
                + extra_offset;
    }

    inline dim_t wei_offset_ext(int g, int oc_b, int ic_b, int kX) const {
        const auto &jcp = pd()->jcp_;
        const int nb_ic = utils::div_up(jcp.ic, 2 * jcp.ic_block);
        const dim_t const_extra_offset
                = jcp.kw * jcp.ic_block * jcp.oc_block * 2;
        dim_t extra_offset = (jcp.ndims == 5) ? kX * jcp.kh * const_extra_offset
                                              : kX * const_extra_offset;
        return (dim_t)((g * jcp.nb_oc + oc_b) * nb_ic + ic_b) * jcp.kd * jcp.kh
                * jcp.kw * jcp.ic_block * jcp.oc_block * 2
                + extra_offset;
    }

    inline int get_end(int start, int step, int limit) const {
        return nstl::min(start + step, limit);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
