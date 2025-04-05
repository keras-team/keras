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

#ifndef GRAPH_UTILS_ID_HPP
#define GRAPH_UTILS_ID_HPP

#include <atomic>
#include <cstddef>

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

struct id_t {
public:
    using value_type = size_t;
    value_type id() const { return id_; }

    id_t() : id_(++counter) {};
    id_t(const id_t &other) : id_(other.id()) {};
    id_t &operator=(const id_t &other) = delete;

protected:
    static std::atomic<value_type> counter;
    ~id_t() = default;

private:
    const value_type id_;
};
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
