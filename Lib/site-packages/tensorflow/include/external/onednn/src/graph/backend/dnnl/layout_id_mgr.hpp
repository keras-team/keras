/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_LAYOUT_ID_MGR_HPP
#define GRAPH_BACKEND_DNNL_LAYOUT_ID_MGR_HPP

#include <mutex>

#include "graph/utils/any.hpp"
#include "graph/utils/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

class dnnl_layout_id_manager_t {
    friend class dnnl_backend;

public:
    // Get a backend memory descriptor from the layout id manager according to a
    // given layout id. The layout id should be generated and manged by the
    // backend. If the layout id is not valid, graph::utils::nullopt will be
    // returned.
    graph::utils::optional_t<memory::desc> get_mem_desc(size_t layout_id) const;

    // Set a backend memory descriptor to the layout id manager and get a
    // corresponding layout id. The param `mem_desc` can be either plain or
    // opaque. The returned cache index will be used as layout id. Note that
    // this function should be called at every place where we want to convert a
    // memory descriptor to a layout id.
    graph::utils::optional_t<size_t> set_mem_desc(const memory::desc &md);

private:
    // private, only can be created in dnnl_backend
    dnnl_layout_id_manager_t() = default;

    mutable struct {
        std::vector<memory::desc> data_;
        mutable std::mutex m_;
    } mem_descs_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
