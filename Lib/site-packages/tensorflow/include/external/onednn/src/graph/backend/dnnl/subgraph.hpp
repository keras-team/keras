/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#ifndef BACKEND_DNNL_SUBGRAPH_HPP
#define BACKEND_DNNL_SUBGRAPH_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct op_executable_t;
class subgraph_rewriter_t;

// The subgraph_t class is a subclass of graph_t, which is used as the only
// parameter of transformation passes. Each transformation pass will process the
// subgraph_t object, and after that, the content of subgraph_t object will be
// changed.
class subgraph_t : public graph_t {
    friend class subgraph_rewriter_t;

private:
    // Make this member private so that only the friend class
    // subgraph_rewriter_t can change the subgraph structure
    std::vector<op_ptr> &get_mutable_ops() {
        return const_cast<std::vector<op_ptr> &>(get_ops());
    }

public:
    subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
            impl::fpmath_mode_t fpm_mode, bool can_use_blocked_layout,
            bool reset_layout);

    subgraph_t(const std::vector<op_ptr> &ops, bool reset_layout = true);

    // The inputs and outputs logical tensors given by users at compilation
    // stage
    std::vector<logical_tensor_t> ins_;
    std::vector<logical_tensor_t> outs_;

    // The engine that the subgraph is compiled for
    const dnnl::engine *p_engine_;

    // This manager holds each op's fusion information
    fusion_info_mgr_t fusion_info_mgr_;

    // The custom cache to store the created primitive desc
    pd_cache_t pd_cache_;

    // The vector to tell which op in the subgraph is constant and will only run
    // once
    std::vector<bool> is_constant_;

    // The executable for each op in subgraph
    std::vector<std::shared_ptr<op_executable_t>> execs_;
};

class subgraph_visualizer_t {
public:
    subgraph_visualizer_t() = default;

    subgraph_visualizer_t(size_t partition_id,
            const std::function<std::string(const value_t *)> &mem_info_func
            = {})
        : enabled_(false)
        , mem_info_func_(mem_info_func)
#ifdef DNNL_ENABLE_GRAPH_DUMP
        , partition_id_(partition_id)
        , index_(0)
#endif
    {
        MAYBE_UNUSED(partition_id);
        // Set _DNNL_BACKEND_SUBGRAPH_DUMP=1 to enable dump subgraph
        enabled_ = graph::utils::getenv_int_internal("BACKEND_SUBGRAPH_DUMP", 0)
                > 0;
    }

    status_t run(const std::shared_ptr<subgraph_t> &sg,
            const std::string &name_suffix, bool is_layout_sensitive,
            bool is_memory_sensitive = false);

private:
    bool enabled_ = false;
    std::function<std::string(const value_t *)> mem_info_func_;
#ifdef DNNL_ENABLE_GRAPH_DUMP
    size_t partition_id_;
    size_t index_;
#endif
};

class subgraph_validator_t {
public:
    subgraph_validator_t() = default;
    status_t run(const std::shared_ptr<subgraph_t> &sg);
};

// This class provide some common used utils to do subgraph rewriting. Those
// utils use "lazy rewrite" policy, which only change the connections but not
// modify the op list in subgraph. To finalize the rewriting process and modify
// the op list, we must call run() method. This class is not thread safe, we
// must not rewrite the same subgraph in multiple threads.
class subgraph_rewriter_t {
public:
    subgraph_rewriter_t(std::shared_ptr<subgraph_t> &sg) : subgraph_(sg) {}

    ~subgraph_rewriter_t();

    // Finalize the rewriting, which actually insert/remove the op to/from
    // subgraph op list
    void run();

    // Puts the op into the to_be_inserted_ops_ list
    void to_insert(const std::shared_ptr<op_t> &op) {
        to_be_inserted_ops_.emplace_back(op);
    }

    // Puts the op into the to_be_removed_ops_ list
    void to_remove(const std::shared_ptr<op_t> &op) {
        to_be_removed_ops_.emplace_back(op);
    }

    // Insert the inserted_op before the base_op, and put the inserted_op to the
    // to_be_inserted_ops_ list
    //                              in_val
    //     in_val                 \   |      /
    //   \   |      /              \  |[j]  /
    //    \  |[i]  /      -->     inserted_op
    //    base_op                     |[k]
    //                                |
    //                             \  |[i]  /
    //                              base_op
    void insert_op_before(const std::shared_ptr<op_t> &inserted_op,
            const std::shared_ptr<op_t> &base_op, size_t i,
            size_t j = std::numeric_limits<size_t>::max(),
            size_t k = std::numeric_limits<size_t>::max());

    // Insert the inserted_op after the base_op, and put the inserted_op to the
    // to_be_inserted_ops_ list
    //                           base_op
    //   base_op                /  |[i]
    //   /  |[i]  \    -->         |
    //  /   |      \            \  |[j]  /
    //    out_val               inserted_op
    //                             |[k]
    //                             |
    //                          out_val
    void insert_op_after(const std::shared_ptr<op_t> &inserted_op,
            const std::shared_ptr<op_t> &base_op, size_t i,
            size_t j = std::numeric_limits<size_t>::max(),
            size_t k = std::numeric_limits<size_t>::max());

    // Fuse a op to its successor, and put the op to the to_be_removed list.
    // The op must have only one successor and one input value
    //   in_val
    //     |
    //    op             in_val
    //     |      -->      |
    //  successor       successor
    //     |               |
    //   out_val         out_val
    void fuse_op_to_successor(const std::shared_ptr<op_t> &op);

    // Fuse a op to its predecessor, and put the op to the to_be_removed list.
    // The unfused input values of op will be add to predecessor's inputs.
    //     in_val1                  in_val1     in_val2
    //       |                         \       /
    //   predecessor  in_val2         predecessor
    //        \       /       -->          |
    //         \[i]  /                     |
    //           op                     out_val
    //            |
    //         out_val
    void fuse_op_to_predecessor(const std::shared_ptr<op_t> &op, size_t i = 0);

    // Replace the org_op with the new_op
    void replace_op(const std::shared_ptr<op_t> &org_op,
            const std::shared_ptr<op_t> &new_op);

    // Swap neighboring single input ops, can be extend to support mimo per
    // requirement
    //
    //   in_val          in_val
    //     |               |
    //  producer        consumer
    //     |      -->      |
    //  consumer        producer
    //     |               |
    //   out_val         out_val
    void swap_neighboring_si_ops(const std::shared_ptr<op_t> &producer,
            const std::shared_ptr<op_t> &consumer);

    void swap_neighboring_reshape_ops(const std::shared_ptr<op_t> &producer,
            const std::shared_ptr<op_t> &consumer);

private:
    bool is_to_be_removed(const std::shared_ptr<op_t> &op) const;

    std::shared_ptr<subgraph_t> subgraph_;
    std::vector<std::shared_ptr<op_t>> to_be_inserted_ops_;
    std::vector<std::shared_ptr<op_t>> to_be_removed_ops_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
