/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GRAPH_UTILS_DEBUG_HPP
#define GRAPH_UTILS_DEBUG_HPP

#include <string>

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

const char *data_type2str(data_type_t v);
const char *engine_kind2str(engine_kind_t v);
const char *fpmath_mode2str(fpmath_mode_t v);
const char *layout_type2str(layout_type_t v);
const char *property_type2str(property_type_t v);

std::string partition_kind2str(partition_kind_t v);
partition_kind_t str2partition_kind(const std::string &str);

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
