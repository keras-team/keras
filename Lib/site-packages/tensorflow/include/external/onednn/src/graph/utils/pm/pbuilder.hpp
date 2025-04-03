/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef GRAPH_UTILS_PM_PBUILDER_HPP
#define GRAPH_UTILS_PM_PBUILDER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
namespace pm {
class pb_op_t;
class pb_node_t;
class pb_graph_t;
// Helper types
// VARIADIC_INPUT_NUM means the num of inputs will be depend on the op
// Using a large enough number to represent this.
#define VARIADIC_INPUT_NUM 64
using iport_t = size_t;
using oport_t = size_t;
using producer_t = std::pair<pb_node_t *, oport_t>;
using consumer_t = std::pair<pb_node_t *, iport_t>;
using consumers_t = std::vector<std::shared_ptr<consumer_t>>;
using in_edge_t = std::pair<iport_t, std::shared_ptr<producer_t>>;
using in_edges_t = std::vector<std::shared_ptr<in_edge_t>>;
using port_map = std::pair<oport_t, iport_t>;
//
// Part 1:
// Structures for representing basic topological patterns
// and attribute patterns
//

// Represents any backend defined function that takes a pointer to dnnl graph op
// and check some attribute(op type, attributes, input shapes ...)
using decision_function = std::function<bool(op_t *)>;

enum class pb_node_kind {
    PB_NODE_KIND_OP,
    PB_NODE_KIND_ALTERNATION,
    PB_NODE_KIND_REPETITION,
};

// Base class for pattern graph with input and output ports (placeholders)
// Only implements traversal methods and setting commutative input pairs.
// Suitable for representing topological patterns
class pb_node_t {
public:
    virtual ~pb_node_t() = default;
    // API for traversing
    std::shared_ptr<producer_t> get_producer(iport_t p_port);
    std::shared_ptr<consumers_t> get_consumers(oport_t p_port);

    std::vector<std::pair<iport_t, producer_t>> get_inputs();
    std::vector<std::pair<oport_t, consumers_t>> get_outputs();

    size_t get_num_decision_functions();
    decision_function get_decision_function(size_t index);
    pb_node_kind get_node_kind() { return node_kind_; };
    virtual std::string get_name() { return debug_string_; };
    virtual void set_name(std::string &&name) {
        debug_string_ = std::move(name);
    };
    const std::unordered_set<pb_op_t *> &get_contained_ops() { return p_ops_; }

protected:
    friend class pb_graph_t;
    pb_node_t() = default;
    bool set_producer(iport_t p_port, std::shared_ptr<producer_t> p_producer);
    bool add_consumer(
            oport_t p_port, const std::shared_ptr<consumer_t> &p_consumer);
    std::vector<std::shared_ptr<producer_t>> ins_;
    std::vector<std::shared_ptr<consumers_t>> outs_;
    std::vector<decision_function> decision_functions_;
    std::string debug_string_;
    pb_node_kind node_kind_;
    std::unordered_set<pb_op_t *> p_ops_;
};

std::shared_ptr<consumer_t> consumer(pb_node_t *p_node, iport_t i_t);

std::shared_ptr<consumer_t> producer(pb_node_t *p_node, oport_t o_t);

std::shared_ptr<in_edge_t> in_edge(iport_t i_t, pb_node_t *p_node, oport_t o_t);

// Helper function for op kind check
decision_function kind(dnnl::impl::graph::op_kind_t okind);
decision_function one_of_kind(
        const std::vector<dnnl::impl::graph::op_kind_t> &okind);

// pb_op_t represents a single dnnl graph  op (and future sub-class) operation
// No public constructor
// Always created by a pb_graph_t
// pb_op_t has type and attributes
// Type and attribute contraint checkers are registered in pb_op_t
// Extends "pb_node_t" to enable attribute matching including op type check.
class pb_op_t : public pb_node_t {
public:
    pb_op_t() = delete;
    // like is_commutative by callback
    bool append_decision_function(const decision_function &p_fn);

    // For overriding default side output control
    void allow_external_outputs() { accept_external_outputs_ = true; }

    bool is_allowing_external_outputs() const {
        return accept_external_outputs_;
    };

    void allow_internal_inputs() { accept_internal_inputs_ = true; };

    bool is_allowing_internal_inputs() const {
        return accept_internal_inputs_;
    };

protected:
    friend class pb_graph_t;
    pb_op_t(const decision_function &p_fn);

    /*
        The outputs could link to ops outside the pattern.
        Explained by the following example.
        The pattern:
          \   /
          Matmul
            |
           Div
            |
           Add
            |
          SoftMax
            |
           Mul
            |
        When accept_external_outputs_ is true,
        the following graph could also be matched:
          \   /
          Matmul
            |
           Div
            |
           Add
            |
         SoftMax
            |  \________________
           Mul                  \  (external output)
            |            SoftMaxBackProp
    */
    bool accept_external_outputs_ = false;

    /*
        The inputs could come from ops within the pattern.
        Explained by the following example.
        The pattern:
         \  /
         Conv
           |
           |
         Sigmoid
           |
            \     /
           Multiply
               |
        When accept_internal_inputs_ is true,
        the following graph could also be matched:
         \  /
         Conv
           |_______
           |       |
         Sigmoid   | (internal input)
           |       |
            \     /
           Multiply
               |
    */
    bool accept_internal_inputs_ = false;
};

//
// Part 2:
// Structures for extended patterns
// API may change
//
class alternation_t : public pb_node_t {
public:
    alternation_t() = delete;
    std::vector<pb_graph_t *> get_alternatives();
    size_t get_min_op_num() const { return min_op_num_; }

protected:
    friend class pb_graph_t;
    alternation_t(std::vector<std::shared_ptr<pb_graph_t>> p_nodes);
    std::vector<std::shared_ptr<pb_graph_t>> alternatives_;
    size_t min_op_num_;
};

class repetition_t : public pb_node_t {
public:
    repetition_t() = delete;
    pb_graph_t *get_body();
    port_map get_port_map(); // only support single port binding
    size_t get_min_rep() const { return min_rep_; }
    size_t get_max_rep() const { return max_rep_; }
    size_t get_min_op_num() const { return min_op_num_; }

protected:
    friend class pb_graph_t;
    // Represents p_node repeated [min_rep, max_rep) times with p_map for
    // output to input binding
    // [n, n+1) means exactly n repetitions
    // [0, n+1) means at most n repetitions
    // [n, INT64_MAX) means at least n repetitions
    repetition_t(std::shared_ptr<pb_graph_t> p_node, port_map p_map,
            size_t min_rep, size_t max_rep);
    // Usage case for Optional does not need a port map
    repetition_t(std::shared_ptr<pb_graph_t> p_node);
    std::shared_ptr<pb_graph_t> body_;
    port_map port_map_;
    size_t min_rep_;
    size_t max_rep_;
    size_t min_op_num_;
};

// "pb_graph_t" represents a group of pb_op_ts and also serves as a pb_node_t
// anywhere And provides a way to limit interface by limiting ports
// (placeholders) to outside of pb_graph_t.
// Nested/Hierarchical pb_nodes are useful for expressing patterns beyond fixed
// pb_graph_t. Regular expression like extension may works on a unit larger
// than a single pb_node_t.
// So a concept that represent grouping is going to be useful.
// pb_graph_t defines a way to forward input/output of the group
// to input/output of individual pb_nodes.
// For example, pb_graph_t "G" below wraps two connected pb_nodes "MUL" and
// "ADD" Collectively, G defines three inputs and one output. The three inputs
// of "G" are mapped to (pb_graph_t inner) inputs of "MUL" and "ADD"
// The single output of "G" maps to the single output of "ADD"
// Now, this "G" can used as part of a bigger pattern by connecting through
// the three inputs and one output just defined.
// Also, "G" can declare output ports which provides a way for backends to
// declare which outputs can be produced by compiled kernels for the pattern.
// Declaring output ports is important for exposing backend's ability to handle
// side outputs.
//    ----------------------
//    |   ------   -----   |
// 0- | 0-| MUL|---|ADD|   |
//    | 1-|    |  0|   |-0 |
// 1- |   ------   |   |   |-0
//    |          1-|   |   |
// 2- |            -----   |
//    ----------------------
//          pb_graph_t "G"
//
// G:IN0->MUL:IN0, G:IN1->MUL:IN1, G:IN2->ADD:IN1
// G:OUT0->ADD:OUT0
// G:OUTPUT PORTS = {OUT0}

class pb_graph_t : public pb_node_t {
public:
    pb_graph_t();

    // Restrict "pb_op_t" create to a pb_graph_t to avoid dangling "pb_op_t"s
    pb_op_t *append_op(
            dnnl::impl::graph::op_kind_t p_kind, const in_edges_t &p_in_edges);
    pb_op_t *append_op(dnnl::impl::graph::op_kind_t p_kind);

    pb_op_t *append_alternation(
            const std::vector<dnnl::impl::graph::op_kind_t> &p_kind,
            const in_edges_t &p_in_edges);
    pb_op_t *append_alternation(
            const std::vector<dnnl::impl::graph::op_kind_t> &p_kind);

    alternation_t *append_alternation(
            const std::vector<std::shared_ptr<pb_graph_t>> &p_nodes,
            const in_edges_t &p_in_edges);
    alternation_t *append_alternation(
            const std::vector<std::shared_ptr<pb_graph_t>> &p_nodes);

    repetition_t *append_repetition(const std::shared_ptr<pb_graph_t> &p_node,
            const port_map &p_map, size_t min_rep, size_t max_rep,
            const in_edges_t &p_in_edges);
    repetition_t *append_repetition(const std::shared_ptr<pb_graph_t> &p_node,
            const port_map &p_map, size_t min_rep, size_t max_rep);

    repetition_t *append_optional(const std::shared_ptr<pb_graph_t> &p_node,
            const in_edges_t &p_in_edges);
    repetition_t *append_optional(const std::shared_ptr<pb_graph_t> &p_node);

    std::vector<std::pair<iport_t, consumers_t>> get_inner_consumers();
    std::vector<std::pair<oport_t, producer_t>> get_inner_producers();
    std::shared_ptr<consumers_t> get_inner_consumer(iport_t);
    std::shared_ptr<producer_t> get_inner_producer(oport_t);

    bool create_input_port(iport_t, pb_node_t *, iport_t);
    bool create_output_port(oport_t, pb_node_t *, oport_t);

    std::vector<pb_node_t *> get_nodes();

    size_t get_min_op_num() const { return min_op_num_; }

protected:
    pb_op_t *append_op(const decision_function &type_checker,
            const in_edges_t &p_in_edges, std::string name = "");
    pb_op_t *append_op(
            const decision_function &type_checker, std::string name = "");

    bool set_edge(const std::shared_ptr<consumer_t> &,
            const std::shared_ptr<producer_t> &);
    bool connect_edges(pb_node_t *p_node, const in_edges_t &p_in_edges);

    bool create_input_port(iport_t, const std::shared_ptr<consumer_t> &);
    bool create_output_port(oport_t, std::shared_ptr<producer_t>);

    // Reference to all internal pb_nodes
    std::vector<std::shared_ptr<pb_node_t>> nodes_;
    std::unordered_set<oport_t> output_ports_;
    std::vector<std::shared_ptr<consumers_t>> inner_consumers_ {nullptr};
    std::vector<std::shared_ptr<producer_t>> inner_producers_ {nullptr};

    // Mininum op number required to match the pattern graph
    size_t min_op_num_;
};

} // namespace pm
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
