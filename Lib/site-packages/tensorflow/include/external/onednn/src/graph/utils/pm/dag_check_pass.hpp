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

#ifndef GRAPH_UTILS_PM_DAG_GRAPH_CHECK_PASS_HPP
#define GRAPH_UTILS_PM_DAG_GRAPH_CHECK_PASS_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/utils/pm/pass_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace pass {

/*!
 * \brief dag_check_pass_t analyzes the graph
 *        to see if it's DAG or not.
 */
class dag_check_pass_t : public graph::pass::pass_base {
public:
    explicit dag_check_pass_t(std::string pbackend, std::string pname)
        : graph::pass::pass_base(std::move(pbackend), std::move(pname)) {}

    static graph::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<dag_check_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    bool is_cycle_detected(const op_t *cur_op,
            std::unordered_map<size_t, bool> &visited,
            std::unordered_map<size_t, bool> &recursion_stack) {
        if (!visited.at(cur_op->get_id())) {
            visited[cur_op->get_id()] = true;
            recursion_stack[cur_op->get_id()] = true;
            // continue to visit current op's producers
            // if cur_op's producer reaches to its consumer in the
            // recursion_stack, cycle detects.
            const auto &inputs = cur_op->get_input_values();
            for (auto it = inputs.begin(); it != inputs.end(); ++it) {
                if (!((*it)->has_producer())) continue;
                op_t &next_op = (*it)->get_producer();
                if (is_cycle_detected(&next_op, visited, recursion_stack))
                    return true;
            }
            // finished visiting cur_op's producers,
            // reset recursion_stack
            recursion_stack[cur_op->get_id()] = false;
        } else if (recursion_stack.at(cur_op->get_id())) {
            return true;
        }
        return false;
    }

    // the criteria of pass execution
    status_t run(graph_t &agraph) override {
        // visited tracks all the ops that have been visited
        std::unordered_map<size_t, bool> visited;
        // recursion_stask tracks an op's visiting path (meaning the
        // op's consumers and consumers' consumers, etc.).
        std::unordered_map<size_t, bool> recursion_stack;

        // init visited and recursion_stack
        for (const std::shared_ptr<op_t> &aop : agraph.get_ops()) {
            visited[aop->get_id()] = false;
            recursion_stack[aop->get_id()] = false;
        }

        for (const std::shared_ptr<op_t> &aop : agraph.get_ops()) {
            // Check if a cycle exists in the directed graph by
            // visiting the ops one by one.
            // The algorithm is based on the idea that a cycle exists in the
            // graph when there is a back edge (i.e., an op's consumer points
            // to one of its producers).
            if (is_cycle_detected(aop.get(), visited, recursion_stack))
                return status::invalid_graph;
        }
        // if all ops are visited, and no cycles detected
        return status::success;
    }
};

} // namespace pass
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
