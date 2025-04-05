/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#ifndef GRAPH_BACKEND_FAKE_PATTERN_UTILS_HPP
#define GRAPH_BACKEND_FAKE_PATTERN_UTILS_HPP

#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "graph/interface/graph.hpp"
#include "graph/interface/partition.hpp"

#include "graph/backend/fake/fake_backend.hpp"
#include "graph/backend/fake/fake_partition_impl.hpp"

#include "graph/utils/pm/nested_matcher.hpp"
#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace fake_impl {
class pattern_utils_t {
public:
    inline void match(graph_t &backend_graph,
            std::shared_ptr<utils::pm::pb_graph_t> pgraph,
            std::vector<op_t *> &matched_op_list);
    inline void fuse(
            graph_t &backend_graph, std::vector<op_t *> &matched_op_list);
    pattern_utils_t() = default;
    pattern_utils_t(const pattern_utils_t &) = delete;
    pattern_utils_t(pattern_utils_t &&) = delete;
    pattern_utils_t &operator=(const pattern_utils_t &) = delete;
};

// function to do pattern matching
inline void pattern_utils_t::match(graph_t &backend_graph,
        std::shared_ptr<utils::pm::pb_graph_t> pgraph,
        std::vector<op_t *> &matched_op_list) {
    std::unordered_set<op_t *> selected;
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!utils::pm::match_pattern(cur_op, pgraph, candidate_fusion)) {
            return status::success;
        }

        matched_op_list.emplace_back(candidate_fusion[0]);
        return status::success;
    });
}

// function to do fusion but not rewrite the graph
inline void pattern_utils_t::fuse(
        graph_t &backend_graph, std::vector<op_t *> &matched_op_list) {
    for (auto &matched_op : matched_op_list) {
        std::shared_ptr<op_t> partition_fused_op(
                new op_t(matched_op->get_kind()));
        // merge the op ids
        partition_fused_op->add_op_ids(matched_op->get_op_ids());

        // merge the input tensor
        // FIXME(qun) Here is a potential bug: We assume that the input
        // tensors which have producer will be in prior to the input
        // tensors which have no producer, but this assumption is not
        // always true. However, Above buggy pattern will not be matched
        // by pattern matcher now, because of another bug in our current
        // pattern matcher. We will fix all these bugs in new pattern
        // matcher

        for (size_t j = 0; j < matched_op->num_inputs(); ++j) {
            std::shared_ptr<value_t> in_value = matched_op->get_input_value(j);
            // if in_value has no producer or its producer isn't in pattern,
            // add this input value to fused op
            std::shared_ptr<value_t> copied_in_value
                    = std::make_shared<value_t>(in_value->get_logical_tensor(),
                            /*internal*/ true);
            partition_fused_op->add_input(copied_in_value);
        }

        // merge the output tensor
        for (size_t k = 0; k < matched_op->num_outputs(); ++k) {
            std::shared_ptr<value_t> out_value
                    = matched_op->get_output_value(k);
            std::shared_ptr<value_t> copied_out_value
                    = std::make_shared<value_t>(out_value->get_logical_tensor(),
                            /*internal*/ true);
            partition_fused_op->add_output(copied_out_value);
        }

        std::shared_ptr<fake_partition_impl_t> pimpl
                = std::make_shared<fake_partition_impl_t>(
                        backend_graph.get_engine_kind());

        // use the fused op to initialize the partition_impl, and merge the
        // informations to it.
        pimpl->init(partition_fused_op.get());

        // transfer the ownership of fusion op from graph to partition
        // note: the fusion op will not be removed from the graph
        pimpl->add_op(matched_op->shared_from_this());
        // claim the op belong to the partition
        matched_op->set_partition(pimpl.get());

        backend_graph.add_partition(pimpl);
    }
}

} // namespace fake_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
