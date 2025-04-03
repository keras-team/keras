/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_CPU_SOFTMAX_PD_HPP
#define CPU_CPU_SOFTMAX_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/softmax_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_softmax_fwd_pd_t : public softmax_fwd_pd_t {
    using softmax_fwd_pd_t::softmax_fwd_pd_t;
};

struct cpu_softmax_bwd_pd_t : public softmax_bwd_pd_t {
    using softmax_bwd_pd_t::softmax_bwd_pd_t;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
