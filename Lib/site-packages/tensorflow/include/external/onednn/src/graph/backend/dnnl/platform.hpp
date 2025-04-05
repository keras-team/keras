/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_PLATFORM_HPP
#define GRAPH_BACKEND_DNNL_PLATFORM_HPP

#include <unordered_map>

#include "graph/interface/c_types_map.hpp"

#define IMPLICATION(cause, effect) (!(cause) || !!(effect))

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace platform {

enum dir_t {
    DIR_UNDEF = 0,
    FLAG_DAT = 1,
    FLAG_WEI = 2,
    FLAG_BIA = 4,
    FLAG_FWD = 32,
    FLAG_BWD = 64,
    FLAG_INF = 128,
};

bool has_cpu_data_type_support(data_type_t data_type);

bool has_cpu_training_support(data_type_t data_type);

bool get_dtype_support_status(engine_kind_t eng, data_type_t dtype, dir_t dir);

inline bool is_gpu(engine_kind_t eng) {
    return eng == engine_kind_t::dnnl_gpu;
}

inline bool is_cpu(engine_kind_t eng) {
    return eng == engine_kind_t::dnnl_cpu;
}

} // namespace platform
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
