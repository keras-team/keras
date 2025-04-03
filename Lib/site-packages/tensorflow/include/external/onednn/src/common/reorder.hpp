/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef COMMON_REORDER_HPP
#define COMMON_REORDER_HPP

#include "c_types_map.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;
status_t reorder_primitive_desc_create(std::shared_ptr<primitive_desc_t> &pd,
        engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *dst_md, const primitive_attr_t *attr = nullptr);

} // namespace impl
} // namespace dnnl

#endif
