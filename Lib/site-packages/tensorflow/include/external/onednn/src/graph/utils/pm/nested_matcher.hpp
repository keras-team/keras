/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_UTILS_PM_NESTED_MATCHER_HPP
#define GRAPH_UTILS_PM_NESTED_MATCHER_HPP

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/interface/op.hpp"
#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
namespace pm {

//
// Nested pattern matcher is greedy graph pattern matcher that is based
// on limited look ahead.
// Matching start with a seed graph op and pattern graph.
// Pattern graph serves as a map for guiding the matching process and
// while matching, individual graph ops will be paired up with a pb_op
// in the pattern graph. pb_op_t encodes which edges need to be matched
// (topological constraints) and what other properties a a paired graph
// op should possess(attribute constraints).
// The pattern graph itself is a nested graph where nesting happens
// at alternation and repetition nodes.
//
// Each pattern graph level has a match_context associated with it.
// A match_context holds the pattern graph in current context and
// the in/out port map used to map upper-level contexts.
//
// A "binding" is pre step for graph op to pb_op_t pairing. At any pattern
// level, some pattern nodes are not pb_op. binding is between a
// graph op and a pattern node (not necessarily a pb_op).
// Matching is driven by binding. If binding is not to a pb_op,
// resolve_node() will be called which invokes matching at the nesting
// pattern level. This process will invoke match_graph, match_alternation
// and match_repetition. If binding is to a pb_op, match_node() is called,
// which checks graph op attributes with the pb_op's decision_functions
// and bind neighboring ops and corresponding pattern nodes.
//
//
// Tells how a pattern node was brought in (reached)
// during matching process.
// BIND_IN: node reached by in port
// BIND_OUT: node reached by out port
// BIND_NONE: Start of matching.
//
enum node_bind_kind {
    BIND_IN,
    BIND_OUT,
    BIND_NONE,
};

//
// Bind a graph op and pattern node for matching
// pattern node may not be a pb_op. In that case, nested(recursive)
// matching happens.
//
class binding_t {
public:
    binding_t(node_bind_kind p_kind, op_t *p_op, size_t p_op_port,
            pb_node_t *p_node, size_t p_port);

    op_t *bind_op;
    pb_node_t *bind_node;
    node_bind_kind bind_kind;
    size_t bind_port;
    size_t bind_op_port;
    // hint op and hint_op_port are the previous binding info
    // used to handle the optional case
    op_t *hint_op = nullptr;
    size_t hint_op_port = 0;
};

using graph_port_map = std::unordered_map<size_t, std::pair<op_t *, size_t>>;

//
// match context tracks a pattern graph match progress
// it consists of
// - pointer to parent match context
// - pointer to pattern graph tracked by this match context
// - a map of pattern graph input port to matched graph op and
// input port
// - a map of pattern graph output port to matched graph op and
// output port
//
class match_context_t {
public:
    // create a inherited context
    match_context_t(match_context_t *p_parent_ctx, pb_node_t *p_graph);
    match_context_t *get_parent_context() { return parent_ctx; };
    pb_graph_t *get_graph() { return graph_; };

    graph_port_map in_port_map;
    graph_port_map out_port_map;

protected:
    match_context_t *parent_ctx;
    // Can be a nullptr if context is a global for holding graph(s)
    pb_graph_t *graph_;
};

//
// match a pattern node by checking graph op's attributes with
// paired pattern node, and their inputs and outputs
//
bool match_node(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);

//
// pair pattern node input nodes (producers) with the paired op's
// input ops. If node has commutative inputs, different combination
// of input nodes and input ops are checked until one pair gets
// matched or all combination are failed
//
bool match_node_inputs(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);
//
// pair pattern node output nodes (consumers) with the paired op's
// output ops. If node output has multiple consumers. different combination
// of output nodes and output ops are checked until one pair gets matched
// or all combination are failed
//
bool match_node_outputs(op_t *op, pb_node_t *node, match_context_t *context,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);
//
// check if the matched graph causes cycles. Basically if one op in the
// matched graph has an input value produced by an external op, and the
// external op (or the external op's arbitrary producers) has an input
// value produced by another op in the matched graph, it causes a cycle,
// and the match process should not continue.
//
bool check_cyclic(
        op_t *op, const std::unordered_map<op_t *, pb_op_t *> &matched_op_map);
//
// match a graph op's attributes using decision_functions of a pb_op_t node
//
bool match_node_attributes(op_t *op, pb_node_t *node);

//
// Trigger nested matching for non pb_op_t nodes
//
bool resolve_node(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);

//
// Match a graph
//
bool match_graph(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);

bool match_graph_helper(const binding_t &local_bind, match_context_t *ctx,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);
//
// match an alternation
// iterates alternatives and apply match_graph until a matching one is found.
//
bool match_alternation(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);

//
// match a repetition including optional
// matches one iteration of repeating body at a time and matches edges
// across iterations of matched bodies.
//
bool match_repetition(const binding_t &b, match_context_t *context,
        std::unordered_map<op_t *, pb_op_t *> &matched_op_map);

//
// Based on op_depth, rearrange consumers in order of depth.
// Put consumer with larger depth the first place so that
// it is possible to meet more ops and match larger or more patterns.
//
std::vector<value_t::consumer_t> sort_op_consumers(
        std::shared_ptr<value_t> &op_out_value);

//
// Entry point of pattern matching.
// Find a match given a graph op (first_op) from an input graph
// and a pre-defined pattern.
//
bool match_pattern(op_t *first_op, const std::shared_ptr<pb_graph_t> &pattern,
        std::vector<op_t *> &fusion_ops);

//
// reorder the matched ops to make sure they are in topology order
//
inline std::vector<op_t *> reorder_matched_list(
        const std::unordered_map<op_t *, pb_op_t *> &matched_op_map);

//
// fill the current match_context's in/out port map
// to pattern match_context. Useful for nested patterns
//
void fill_parent_io_map(match_context_t *local_ctx);

//
// Two utils classes for matching node outputs/inputs
//
class node_inputs_matcher_t {
public:
    node_inputs_matcher_t(const binding_t &b, match_context_t *ctx,
            std::unordered_map<op_t *, pb_op_t *> &matched_op_map);
    std::unordered_map<op_t *, pb_op_t *> get_updated_op_map() const {
        return updated_op_map_;
    };
    //
    // If we have touched the upper boundary of the pattern
    // graph, it's time to terminate the recursion.
    // When in this case, the target node doesn't have inputs,
    // so we could confirm whether the recursion should be
    // terminated or not by checking the number of node inputs
    //
    bool check_recursion_termination();
    bool match_commutative_inputs();
    bool match_non_commutative_inputs();
    bool match_variadic_inputs();
    pb_node_t *get_node() const { return node_; };
    op_t *get_op() const { return op_; };

protected:
    op_t *op_;
    pb_node_t *node_;
    binding_t bind_;
    match_context_t *ctx_;
    std::unordered_map<op_t *, pb_op_t *> updated_op_map_;
    std::vector<std::pair<iport_t, producer_t>> node_inputs_;
    bool match_input_by_offset(
            const size_t &op_input_offset, const size_t &node_input_offset);
    bool support_optional_inputs(pb_node_t *n);
};

class node_outputs_matcher_t {
public:
    node_outputs_matcher_t(op_t *op, pb_node_t *node, match_context_t *ctx,
            std::unordered_map<op_t *, pb_op_t *> &matched_op_map);
    std::unordered_map<op_t *, pb_op_t *> get_updated_op_map() const {
        return updated_op_map_;
    };
    pb_node_t *get_node() const { return node_; };
    op_t *get_op() const { return op_; };
    bool get_optional_case_status() const { return is_optional_case_; }
    //
    // If we have touched the lower boundary of the pattern
    // graph, it's time to terminate the recursion.
    // When in this case, the target node doesn't have outputs,
    // so we could confirm whether the recursion should be
    // terminated or not by checking the number of node outputs
    //
    bool check_recursion_termination();
    bool match_output();

protected:
    op_t *op_;
    pb_node_t *node_;
    match_context_t *ctx_;
    std::unordered_map<op_t *, pb_op_t *> updated_op_map_;
    std::pair<oport_t, consumers_t> current_node_output_;
    size_t current_node_oport_;
    std::vector<std::pair<oport_t, consumers_t>> node_outputs_;
    //
    // A flag variable to mark if the optional case is
    // exist or not
    //
    bool support_optional_;
    //
    // In optional case, we need to directly return the matching
    // result(success/fail) in advance. When in this time,
    // the matching process should be terminated, and here
    // the variable "is_optional_case_" is
    // used as a flag variable to help mark that.
    //
    bool is_optional_case_ = false;
    //
    // Match the op consumers one by one
    //
    bool match_op_consumers();
    //
    // When one op consumer couldn't find any corresponding
    // matched node consumer, there may exist three cases where
    // the matching process should still be successful:
    // 1. it's the allow_internal_inputs case
    // 2. it's the allow_external_outputs case
    // 3. it's the optional case
    // For all there three cases, we should check them before
    // we return a "false" matching result.
    // If the unmatching case hits any one of the three above,
    // the variable is_unmatching_exception_ will be true, which
    // means that the iteration of matching op consumers could
    // still forward.
    //
    bool op_consumer_unmatching_checking(op_t *out_op);
    bool support_optional_outputs(pb_node_t *n);
    bool check_internal_inputs(op_t *out_op);
    bool check_external_outputs();
    bool check_optional();
    //
    // If not all the consumers of node output are matched,
    // we should check the optional case
    //
    bool check_node_consumers(
            std::unordered_set<size_t> &node_oport_matched_cons);
};

//
// A utils class for repetition graph matching process
//
class repetition_matcher_t {
public:
    repetition_matcher_t(const binding_t &bind_arg, match_context_t *parent_ctx,
            std::unordered_map<op_t *, pb_op_t *> &matched_op_map);
    //
    // A while loop for matching each repetition block and
    // update the global ctx of repetition graph
    //
    size_t match_repetition_blocks();
    //
    // After matching the repetition blocks, we need to forward
    // the matching progress by changing the value of binding
    //
    bool post_repetition_matching(size_t num_rep, const binding_t &bind_arg);
    std::unordered_map<op_t *, pb_op_t *> get_updated_op_map() const {
        return updated_op_map_;
    }

protected:
    binding_t single_iter_bind_;
    match_context_t *parent_ctx_;
    std::unordered_map<op_t *, pb_op_t *> updated_op_map_;
    //
    // A temporary binding_t type instance for single matching iteration
    //
    repetition_t *rep_node_;
    port_map pmap_;
    size_t min_rep_;
    size_t max_rep_;
    bool forward_match_;
    //
    // The matching of repetition graph is a incremental process,
    // which means that the global context of repetition graph
    // could also be obtained incrementally. In our implementation,
    // the global context of repetition graph(rep_global_ctx_) is updated
    // by the local context(local_cached_ctx), whose i/o maps will be
    // filled when calling function match_graph() in each iteration
    //
    match_context_t rep_global_ctx_;
    //
    // After successfully matching repetition blocks, we need to
    // forward the matching process, if no repetition block is
    // matched, it also means that the current op is still
    // unmatched, so we should still take it into the next
    // matching process. Similarly, if any number of blocks are matched,
    // it means that the current op has been matched, and
    // what we should do is matching the next op.
    //
    bool match_current_op(const binding_t &bind_arg);
    bool match_next_op(const binding_t &bind_arg);
    //
    // update single_iter_bind_ for the next matching round
    //
    bool prepare_next_matching_round(const match_context_t &local_cached_ctx);
    //
    // After successfully matching one repetition block,
    // for BIND_OUT matching, we need to verify if the output op
    // of the block has current number of consumers.
    // Because for BIND_OUT matching, next matching round will
    // go to traverse the input op, so the output op remains
    // unchecked.
    //
    bool verify_current_matching_round(const match_context_t &local_cached_ctx,
            const std::unordered_map<op_t *, pb_op_t *> &local_op_map) const;
};

//
// fill the current match_context's in/out port map
//
void fill_local_in_map(match_context_t *local_ctx, pb_node_t *cur_node,
        op_t *cur_op, size_t cur_op_port);
void fill_local_out_map(match_context_t *local_ctx, pb_node_t *cur_node,
        op_t *cur_op, size_t cur_op_port);

} // namespace pm
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
