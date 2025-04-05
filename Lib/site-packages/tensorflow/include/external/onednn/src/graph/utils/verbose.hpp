/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_UTILS_VERBOSE_HPP
#define GRAPH_UTILS_VERBOSE_HPP

#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <string>

#include "common/verbose.hpp"

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

void print_verbose_header();

struct partition_info_t {
    partition_info_t() = default;
    partition_info_t(const partition_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {};
    partition_info_t &operator=(const partition_info_t &rhs) {
        str_ = rhs.str_;
        is_initialized_ = rhs.is_initialized_;
        return *this;
    }

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(const engine_t *engine, const compiled_partition_t *partition);

private:
    std::string str_;

#if defined(DISABLE_VERBOSE)
    bool is_initialized_ = true;
#else
    bool is_initialized_ = false;
#endif

    std::once_flag initialization_flag_;
};

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
