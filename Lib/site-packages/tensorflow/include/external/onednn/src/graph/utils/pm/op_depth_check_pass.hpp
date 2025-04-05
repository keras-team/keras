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

#ifndef GRAPH_UTILS_PM_OP_DEPTH_CHECK_PASS_HPP
#define GRAPH_UTILS_PM_OP_DEPTH_CHECK_PASS_HPP

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <unordered_set>

#include "graph/utils/pm/pass_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
namespace pm {

// Figure out depth of every op in order to find longest path to the root
class graph_op_depth_check_pass_t : public graph::pass::pass_base {
public:
    explicit graph_op_depth_check_pass_t(
            std::string pbackend, std::string pname)
        : graph::pass::pass_base(std::move(pbackend), std::move(pname)) {}

    static graph::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<graph_op_depth_check_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    // the criteria of pass execution
    impl::status_t run(graph_t &agraph) override {
        std::queue<op_t *> cur_layer_ops;
        std::queue<op_t *> next_layer_ops;
        std::unordered_set<op_t *> visited;
        int64_t cur_op_depth = 0;

        for (auto &root_op : agraph.get_output_ops()) {
            cur_layer_ops.push(root_op);
            root_op->set_attr<int64_t>(op_attr::op_depth, cur_op_depth);
        }

        while (!cur_layer_ops.empty()) {
            cur_op_depth++;
            while (!cur_layer_ops.empty()) {
                op_t *cur_op = cur_layer_ops.front();
                cur_layer_ops.pop();
                // should double visit cur_op in case got shallow one.
                // so no need to check if visited.
                auto &input_values = cur_op->get_input_values();
                for (auto &input_value : input_values) {
                    if (input_value->has_producer()) {
                        auto next_layer_op = &(input_value->get_producer());
                        next_layer_op->set_attr<int64_t>(
                                op_attr::op_depth, cur_op_depth);
                        // for next round of traverse, repeated op should be checked
                        if (visited.find(next_layer_op) != visited.end())
                            continue;
                        next_layer_ops.push(next_layer_op);
                        visited.insert(next_layer_op);
                    }
                }
            }
            swap(cur_layer_ops, next_layer_ops);
            visited.clear();
        }

        return impl::status::success;
    }
};

} // namespace pm
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
