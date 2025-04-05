/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef COMMON_ITTNOTIFY_HPP
#define COMMON_ITTNOTIFY_HPP

#include "c_types_map.hpp"
#include "dnnl.h"

namespace dnnl {
namespace impl {
namespace itt {

typedef enum {
    __itt_task_level_none = 0,
    __itt_task_level_low,
    __itt_task_level_high
} __itt_task_level;

struct itt_task_level_t {
    int level;
};
// Returns `true` if requested @p level is less or equal to default or specified
// one by env variable.
bool get_itt(__itt_task_level level);
void primitive_task_start(primitive_kind_t kind);
primitive_kind_t primitive_task_get_current_kind();
void primitive_task_end();
} // namespace itt
} // namespace impl
} // namespace dnnl
#endif
