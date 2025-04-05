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

#ifndef CPU_X64_LRN_LRN_EXECUTOR_HPP
#define CPU_X64_LRN_LRN_EXECUTOR_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

class i_lrn_executor_t {
public:
    virtual status_t execute(const exec_ctx_t &ctx) const = 0;
    virtual ~i_lrn_executor_t() = default;
    virtual status_t create_kernel() = 0;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif