/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_PASSES_LOWER_HPP
#define GRAPH_BACKEND_DNNL_PASSES_LOWER_HPP

#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// handler is used to lower a spec op to a dnnl backend internal op. Each spec
// op has its own handler. Multiple spec ops may share same handler as long as
// their lowering rules are same, like element wise ops or binary ops.
using handler_func = std::function<status_t(
        const std::shared_ptr<op_t> &, subgraph_rewriter_t &)>;

// A template handler function that can be specialized by the dnnl backend
// internal op kind. Most spec ops can use the specialized common_handler to
// lower theirselves to the specific internal ops.
template <op_kind::kind_t op_kind>
status_t common_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(static_cast<op_kind_t>(op_kind));
    new_op->merge_attributes(op->get_attributes());

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

// This pass is used to convert a subgraph composed of spec ops to an equivalent
// subgraph composed of dnnl backend internal ops. The semantics of dnnl backend
// internal ops are closer to oneDNN primitives and has less constraints. Then
// the subsequent optimization passes (like canonicalization or fusion passes)
// can work on the lowered subgraph easier.
status_t lower_down(std::shared_ptr<subgraph_t> &sg);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
