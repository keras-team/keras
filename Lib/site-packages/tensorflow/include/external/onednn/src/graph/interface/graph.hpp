/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_INTERFACE_GRAPH_HPP
#define GRAPH_INTERFACE_GRAPH_HPP

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.h"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/op_schema.hpp"
#include "graph/interface/partition.hpp"
#include "graph/interface/partition_impl.hpp"
#include "graph/interface/value.hpp"

#include "graph/utils/debug.hpp"
#include "graph/utils/id.hpp"
#include "graph/utils/json.hpp"
#include "graph/utils/utils.hpp"

namespace graph = dnnl::impl::graph;

namespace dnnl {
namespace impl {
namespace graph {

void rewrite(
        graph_t &agraph, const std::vector<std::vector<op_t *>> &fusion_ops);
} // namespace graph
} // namespace impl
} // namespace dnnl

struct dnnl_graph_graph : public graph::utils::id_t {
    using op_t = graph::op_t;
    using value_t = graph::value_t;
    using op_ptr = std::shared_ptr<op_t>;
    using value_ptr = std::shared_ptr<value_t>;
    using logical_tensor_t = graph::logical_tensor_t;
    using logical_tensor_wrapper_t = graph::logical_tensor_wrapper_t;
    using op_schema_t = graph::op_schema_t;
    using op_schema_registry_t = graph::op_schema_registry_t;
    using id_t = graph::utils::id_t;

private:
    /*! \brief added ops*/
    std::vector<op_ptr> ops_ {};

    /*! \brief The engine kind on which the operator will be evaluated */
    graph::engine_kind_t engine_kind_ {};

    /*! \brief The floating-point math mode */
    graph::fpmath_mode_t fpmath_mode_ {};

    std::vector<std::shared_ptr<graph::partition_impl_t>> partition_impls_;

    bool finalized_ {false};

    /*! \brief num of ops that have not been partitioned */
    size_t num_unpartitioned_ops_ {0};

public:
    dnnl_graph_graph(graph::engine_kind_t kind = graph::engine_kind::cpu)
        : engine_kind_(kind), fpmath_mode_(dnnl::impl::get_fpmath_mode()) {}

    dnnl_graph_graph(
            graph::engine_kind_t kind, graph::fpmath_mode_t fpmath_mode)
        : engine_kind_(kind), fpmath_mode_(fpmath_mode) {}

    // deep copy (except that the partition_impls_ is shallow copy)
    dnnl_graph_graph(const dnnl_graph_graph &other)
        : id_t(other)
        , ops_(deep_copy(other.ops_))
        , engine_kind_(other.engine_kind_)
        , fpmath_mode_(other.fpmath_mode_)
        , partition_impls_(other.partition_impls_) {};

    dnnl_graph_graph(const std::vector<op_ptr> &ops,
            graph::engine_kind_t kind = graph::engine_kind::cpu,
            graph::fpmath_mode_t fpmath_mode = graph::fpmath_mode::strict)
        : ops_(ops), engine_kind_(kind), fpmath_mode_(fpmath_mode) {}

    dnnl_graph_graph &operator=(const dnnl_graph_graph &other) = delete;

    ~dnnl_graph_graph() = default;

    graph::engine_kind_t get_engine_kind() const { return engine_kind_; }

    graph::fpmath_mode_t get_fpmath_mode() const { return fpmath_mode_; }

    /*!
     * \brief Check whether an operator can be added
     * \param l_n An operator in frameworks' graph.
     * \return Whether the operator is supported
     */
    graph::status_t add_op(const op_t *l_n) {
        if (!l_n) return graph::status::invalid_graph_op;

        if (std::none_of(ops_.begin(), ops_.end(),
                    [&l_n](const std::vector<op_ptr>::value_type &op) {
                        return op->get_id() == l_n->get_id();
                    })) {
            const graph::op_schema_t *opm
                    = graph::op_schema_registry_t::get_op_schema(
                            l_n->get_kind());
            op_t tmp_ln = *l_n;
            if (opm != nullptr) {
                opm->set_default_attribute(&tmp_ln);
                if (!opm->verify(&tmp_ln)) {
                    return graph::status::invalid_graph_op;
                }
            }
            ops_.push_back(std::make_shared<op_t>(tmp_ln));
            auto back_op = ops_.back().get();
            for (size_t i = 0; i < back_op->num_outputs(); i++)
                back_op->get_output_value(i)->set_producer(*back_op);
        }
        return graph::status::success;
    }

    op_t *create_op(graph::op_kind_t kind, std::string name = "") {
        ops_.push_back(std::make_shared<op_t>(kind, std::move(name)));
        return ops_.back().get();
    }

    void delete_op(op_t *op) {
        if (!op) return;

        auto pos = std::find_if(ops_.begin(), ops_.end(),
                [op](const op_ptr &n) -> bool { return *n == *op; });
        if (pos != ops_.end()) ops_.erase(pos);
    }

    /*!
     * \brief Get all the ops of this graph, including original ops and fused.
     * \return vector of ops pointers
     */
    const std::vector<op_ptr> &get_ops() const { return ops_; }

    /*! \brief how many ops in the graph */
    size_t num_ops() const { return ops_.size(); }

    /*! \brief how many ops in the graph have not been partitioned */
    size_t num_unpartitioned_ops() const { return num_unpartitioned_ops_; }

    /*!
     * \brief Get the output ops of this graph.
     * \return vector of output op pointers
     */
    std::vector<op_t *> get_output_ops() const {
        std::vector<op_t *> outputs;
        for (const op_ptr &n : ops_) {
            size_t num_consumers = 0;
            for (size_t i = 0; i < n->num_outputs(); i++) {
                num_consumers += n->num_output_consumers(i);
            }

            if (num_consumers == 0) { outputs.push_back(n.get()); }
        }
        return outputs;
    }

    /*!
     * \brief Get the input values (values whose producer are not in the graph)
     * of this graph.
     * \return vector of input values pointers
     */
    std::vector<value_t *> get_input_values() const {
        std::vector<value_t *> in_vals;
        for (const op_ptr &n : ops_) {
            for (const value_ptr &in_val : n->get_input_values()) {
                if (!in_val->has_producer()) {
                    in_vals.emplace_back(in_val.get());
                    continue;
                }

                op_t &producer = in_val->get_producer();
                if (std::none_of(ops_.begin(), ops_.end(),
                            [&producer](const op_ptr &op) {
                                return op.get() == &producer;
                            })) {
                    in_vals.emplace_back(in_val.get());
                }
            }
        }

        return in_vals;
    }

    /*!
     * \brief Get the output values (values whose consumers are not all in the
     * graph) of this graph.
     * \return vector of output values pointers
     */
    std::vector<value_t *> get_output_values() const {
        std::vector<value_t *> out_vals;
        for (const op_ptr &n : ops_) {
            for (const value_ptr &out_val : n->get_output_values()) {
                std::vector<value_t::consumer_t> consumers
                        = out_val->get_consumers();

                bool has_outer_consumer = false;
                for (const value_t::consumer_t &csm : consumers) {
                    op_t &csm_op = csm.get_op();
                    if (std::none_of(ops_.begin(), ops_.end(),
                                [&csm_op](const op_ptr &op) {
                                    return op.get() == &csm_op;
                                })) {
                        has_outer_consumer = true;
                        break;
                    }
                }

                if (consumers.empty() || has_outer_consumer)
                    out_vals.emplace_back(out_val.get());
            }
        }

        return out_vals;
    }

    void add_partition(const std::shared_ptr<graph::partition_impl_t> &pimpl) {
        partition_impls_.push_back(pimpl);
        num_unpartitioned_ops_ -= pimpl->get_ops().size();
    }

    std::vector<std::shared_ptr<graph::partition_impl_t>> &get_partitions() {
        return partition_impls_;
    }

    void clean_partitions() {
        partition_impls_.clear();
        num_unpartitioned_ops_ = num_ops();
    }

    /*!
     * \brief Get partition numbers
     * \return partition numbers
     */
    size_t get_num_partitions() const { return partition_impls_.size(); }

    /*!
     * \brief get list of partitions
     * \param list of partitions
     */
    graph::status_t get_ordered_partitions(
            std::vector<graph::partition_t *> &partitions);

    // Finalize the graph after finishing adding ops.
    graph::status_t finalize();

    // Called during finalize(), run some graph analyze passes,
    // e.g. DAG check.
    graph::status_t analyze();

    // Get the finalization status of the graph.
    bool is_finalized() const { return finalized_; }

    // This function is used to infer shape for all the ops in a graph.
    // Before calling this function, the inputs value of the graph should
    // have valid shape
    graph::status_t infer_shape() {
        using value_ptr = std::shared_ptr<value_t>;

        // Check inputs shape
        for (value_t *in : get_input_values()) {
            logical_tensor_t lt = in->get_logical_tensor();
            if (logical_tensor_wrapper_t(lt).is_shape_unknown())
                return graph::status::invalid_shape;
        }

        // call each op's infer shape function in topological order
        return graph::topo_order_visit(get_output_ops(), [](op_t *op) {
            std::vector<logical_tensor_t> tmp_inputs, tmp_outputs;
            std::vector<logical_tensor_t *> tmp_inputs_ptr, tmp_outputs_ptr;

            // avoid re-allocating
            tmp_inputs.reserve(op->num_inputs());
            tmp_outputs.reserve(op->num_outputs());
            tmp_inputs_ptr.reserve(op->num_inputs());
            tmp_outputs_ptr.reserve(op->num_outputs());

            for (const value_ptr &in : op->get_input_values()) {
                tmp_inputs.emplace_back(in->get_logical_tensor());
                tmp_inputs_ptr.emplace_back(&tmp_inputs.back());
            }
            for (const value_ptr &out : op->get_output_values()) {
                tmp_outputs.emplace_back(out->get_logical_tensor());
                tmp_outputs_ptr.emplace_back(&tmp_outputs.back());
            }

            const op_schema_t *opm
                    = op_schema_registry_t::get_op_schema(op->get_kind());
            // can't infer shape for cur op: no schema
            if (!opm) return graph::status::invalid_graph_op;

            graph::status_t ret
                    = opm->shape_infer(op, tmp_inputs_ptr, tmp_outputs_ptr);

            if (ret != graph::status::success)
                return graph::status::invalid_shape;

            for (size_t i = 0; i < op->num_outputs(); i++) {
                op->get_output_value(i)->set_logical_tensor(tmp_outputs[i]);
            }

            return graph::status::success;
        });
    }

    // This function is used to set user given logical tensors for inputs and
    // outputs of a graph.
    graph::status_t set_user_inputs_outputs(
            const std::vector<graph::logical_tensor_t> &inputs,
            const std::vector<graph::logical_tensor_t> &outputs) {
        // set the inputs's layout to subgraph's inputs value
        auto graph_in_vals = get_input_values();
        auto graph_out_vals = get_output_values();

        auto set_logical_tensors =
                [](std::vector<value_t *> &edges,
                        const std::vector<graph::logical_tensor_t> &givens,
                        bool check_given, bool must_have_shape) {
                    for (auto &edge : edges) {
                        size_t edge_id = edge->get_logical_tensor().id;

                        // partition in/outs should not have default id. There
                        // must be some errors in previous graph transformation
                        // stage
                        if (edge_id == std::numeric_limits<size_t>::max())
                            return graph::status::invalid_graph;

                        bool found = false;
                        for (const auto &given : givens) {
                            if (edge_id == given.id) {
                                if (check_given) {
                                    // check given lts
                                    bool valid = given.data_type
                                            != graph::data_type::undef;
                                    if (must_have_shape) {
                                        valid = valid && given.ndims >= 0;
                                        for (size_t i = 0;
                                                i < static_cast<size_t>(
                                                        given.ndims);
                                                i++) {
                                            const bool known = given.dims[i]
                                                    != DNNL_GRAPH_UNKNOWN_DIM;
                                            valid = valid && known;
                                        }
                                    }
                                    if (!valid)
                                        return graph::status::invalid_arguments;
                                }

                                edge->set_logical_tensor(given);
                                found = true;
                                break;
                            }
                        }

                        if (!found) return graph::status::invalid_arguments;
                    }
                    return graph::status::success;
                };

        graph::status_t ret;
        ret = set_logical_tensors(graph_in_vals, inputs, true, true);
        if (ret != graph::status::success) return ret;

        ret = set_logical_tensors(graph_out_vals, outputs, true, false);
        return ret;
    }

    // This function is used to serialize graph to a JSON file
    graph::status_t serialize(const std::string &filename) const {
        printf("onednn_graph_verbose,info,serialize graph to a json file %s\n",
                filename.c_str());
        std::ofstream of(filename);
        graph::utils::json::json_writer_t writer(&of);
        writer.begin_object();
        std::string version = std::to_string(dnnl_version()->major) + "."
                + std::to_string(dnnl_version()->minor) + "."
                + std::to_string(dnnl_version()->patch);
        writer.write_keyvalue("version", version);
        writer.write_keyvalue("engine_kind",
                std::string(graph::utils::engine_kind2str(get_engine_kind())));
        writer.write_keyvalue("fpmath_mode",
                std::string(graph::utils::fpmath_mode2str(get_fpmath_mode())));
        std::vector<size_t> inputs_id;
        inputs_id.reserve(get_input_values().size());
        for (const auto &val : get_input_values()) {
            auto lt = val->get_logical_tensor();
            auto ltw = logical_tensor_wrapper_t(lt);
            inputs_id.push_back(ltw.id());
        }
        writer.write_keyvalue("input_ports", inputs_id);
        std::vector<size_t> outputs_id;
        outputs_id.reserve(get_output_values().size());
        for (const auto &val : get_output_values()) {
            auto lt = val->get_logical_tensor();
            auto ltw = logical_tensor_wrapper_t(lt);
            outputs_id.push_back(ltw.id());
        }
        writer.write_keyvalue("output_ports", outputs_id);
        writer.write_keyvalue("graph", get_ops());
        writer.end_object();

        return graph::status::success;
    }

    static std::vector<op_ptr> deep_copy(const std::vector<op_ptr> &ops);
};

#endif
