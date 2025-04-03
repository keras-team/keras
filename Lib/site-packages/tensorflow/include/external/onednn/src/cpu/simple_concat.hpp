/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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

#ifndef CPU_SIMPLE_CONCAT_HPP
#define CPU_SIMPLE_CONCAT_HPP

#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_concat_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct simple_concat_t : public primitive_t {
    struct pd_t : public cpu_concat_pd_t {
        using cpu_concat_pd_t::cpu_concat_pd_t;

        pd_t(const pd_t &rhs) : cpu_concat_pd_t(rhs) { copy_from(rhs); }

        DECLARE_CONCAT_PD_T("simple:any", simple_concat_t);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper dst_d(dst_md());
            VDISPATCH_CONCAT(platform::has_data_type_support(data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONCAT(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONCAT(cpu_concat_pd_t::init() == status::success,
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "concat");
            VDISPATCH_CONCAT(dst_d.ndims() <= 6, VERBOSE_BAD_NDIMS, "dst",
                    dst_d.ndims());

            for (size_t i = 0; i < src_mds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_mds_[i]);
                const memory_desc_wrapper o_d(&src_image_mds_[i]);

                const bool ignore_strides = true;

                VDISPATCH_CONCAT(utils::everyone_is(data_type, i_d.data_type(),
                                         o_d.data_type()),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_CONCAT(utils::everyone_is(format_kind::blocked,
                                         i_d.format_kind(), o_d.format_kind()),
                        VERBOSE_UNSUPPORTED_TAG);
                VDISPATCH_CONCAT(types::blocking_desc_is_equal(
                                         *i_d.md_, *o_d.md_, ignore_strides),
                        VERBOSE_BLOCKING_FAIL, "blocking descriptor mismatch");
                VDISPATCH_CONCAT(types::blocking_desc_is_equal(
                                         *i_d.md_, *dst_d.md_, ignore_strides),
                        VERBOSE_BLOCKING_FAIL, "blocking descriptor mismatch");
                VDISPATCH_CONCAT(!i_d.is_additional_buffer(),
                        "memory format does not have additional buffer");
            }

            dst_d.compute_blocks(blocks_);
            format_perm();

            // start dim is the first dimension after which the concatenation
            // would happen contiguously
            const int start_dim = perm_[concat_dim()];

            // check that contiguous part is indeed contiguous (i.e. dense)
            VDISPATCH_CONCAT(!(nelems_to_concat(dst_d)
                                     != dst_d.padded_dims()[concat_dim()]
                                             / blocks_[concat_dim()]
                                             * dst_d.blocking_desc()
                                                       .strides[concat_dim()]),
                    VERBOSE_INCONSISTENT_NDIMS, "dst",
                    "(padded_dims, concat_dim)");

            // check that all inputs have the same strides for the
            // contiguous part [concat_dim .. ndims] for the *major* dims.
            // the block part is already checked above
            for (size_t i = 0; i < src_mds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_mds_[i]);
                for (int d = start_dim; d < dst_d.ndims(); ++d) {
                    VDISPATCH_CONCAT(
                            !(dst_d.blocking_desc().strides[iperm_[d]]
                                    != i_d.blocking_desc().strides[iperm_[d]]),
                            "inputs have inconsistent strides for major dims");
                }
            }

            init_scratchpad();

            return status::success;
        }

        int perm_[DNNL_MAX_NDIMS] {};
        int iperm_[DNNL_MAX_NDIMS] {};
        dims_t blocks_ {};

        dim_t nelems_to_concat(const memory_desc_wrapper &data_d) const {
            const int ndims = data_d.ndims();

            dim_t nelems = 1;
            for (int i = perm_[concat_dim()]; i < ndims; i++)
                nelems *= data_d.padded_dims()[iperm_[i]] / blocks_[iperm_[i]];
            for (int i = 0; i < ndims; i++)
                nelems *= blocks_[i];

            return nelems;
        }

    private:
        void format_perm() {
            const memory_desc_wrapper dst_d(dst_md());
            const int ndims = dst_d.ndims();

            dims_t blocks = {0};
            dst_d.compute_blocks(blocks);

            strides_t strides = {0};
            utils::array_copy(strides, dst_d.blocking_desc().strides, ndims);

            dims_t ou_blocks = {0};
            utils::array_copy(ou_blocks, dst_d.padded_dims(), ndims);

            for (int d = 0; d < ndims; d++) {
                iperm_[d] = d;
                ou_blocks[d] /= blocks[d];
            }

            utils::simultaneous_sort(strides, ou_blocks, iperm_, ndims,
                    [](stride_t a, stride_t b) { return b - a; });

            for (int i = 0; i < ndims; i++)
                perm_[iperm_[i]] = i;
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.template book<data_t *>(key_concat_iptrs, n_inputs());
            scratchpad.template book<data_t *>(key_concat_optrs, n_inputs());
            scratchpad.template book<dim_t>(key_concat_nelems, n_inputs());
            scratchpad.template book<strides_t>(
                    key_concat_istrides, n_inputs());
        }

        void copy_from(const pd_t &rhs) {
            int ndims = rhs.dst_md_.ndims;
            utils::array_copy(perm_, rhs.perm_, ndims);
            utils::array_copy(iperm_, rhs.iperm_, ndims);
            utils::array_copy(blocks_, rhs.blocks_, ndims);
        }
    };

    simple_concat_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

    typedef typename prec_traits<data_type>::type data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
