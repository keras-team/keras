/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_LRN_LRN_EXECUTOR_FACTORY_HPP
#define CPU_X64_LRN_LRN_EXECUTOR_FACTORY_HPP

#include <memory>
#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_utils.hpp"
#include "cpu/x64/lrn/lrn_avx512_blocked_executor.hpp"
#include "cpu/x64/lrn/lrn_avx512_nhwc_executor.hpp"
#include "cpu/x64/lrn/lrn_executor.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

class lrn_executor_factory_t {
public:
    template <::dnnl::impl::data_type_t d_type, typename PD_T>
    static std::unique_ptr<i_lrn_executor_t> create_executor(
            const PD_T *pd, direction dir) {
        const memory_desc_wrapper data_d(pd->src_md());

        if (data_d.matches_tag(format_tag::nChw16c))
            return create_jit_avx512_blocked_executor<d_type, PD_T>(pd, dir);

        return create_jit_avx512_nhwc_executor<d_type, PD_T>(pd, dir);
    }

private:
    template <::dnnl::impl::data_type_t d_type, typename PD_T>
    static std::unique_ptr<i_lrn_executor_t> create_jit_avx512_nhwc_executor(
            const PD_T *pd, direction dir) {

        if (dir == direction::forward)
            return utils::make_unique<
                    lrn_avx512_nhwc_executor_fwd_t<d_type, PD_T>>(pd);
        return utils::make_unique<lrn_avx512_nhwc_executor_bwd_t<d_type, PD_T>>(
                pd);
    }

    template <::dnnl::impl::data_type_t d_type, typename PD_T>
    static std::unique_ptr<i_lrn_executor_t> create_jit_avx512_blocked_executor(
            const PD_T *pd, direction dir) {

        if (dir == direction::forward)
            return utils::make_unique<
                    lrn_avx512_blocked_executor_fwd_t<d_type, PD_T>>(pd);
        return utils::make_unique<
                lrn_avx512_blocked_executor_bwd_t<d_type, PD_T>>(pd);
    }
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif