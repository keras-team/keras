/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef COMMON_RNN_HPP
#define COMMON_RNN_HPP

#include "oneapi/dnnl/dnnl.h"

namespace dnnl {
namespace impl {
namespace rnn {

int get_gates_count(dnnl_alg_kind_t cell_kind);
} // namespace rnn
} // namespace impl
} // namespace dnnl

#endif
