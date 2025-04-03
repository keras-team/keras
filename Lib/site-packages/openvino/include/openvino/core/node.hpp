// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/descriptor/input.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_value.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {
namespace v0 {
class Result;
}  // namespace v0
struct AutoBroadcastSpec;
}  // namespace op
namespace pass {

class ResolveNameCollisions;

namespace pattern {
class Matcher;
}  // namespace pattern
}  // namespace pass

template <typename NodeType>
class Input;

template <typename NodeType>
class Output;

class Node;

class Model;

class SharedRTInfo;

/// EvaluationContext stores and manages a context (additional parameters, values and
/// environment) for evaluating ov::Model.
using EvaluationContext = ov::RTMap;

OPENVINO_API
std::string node_validation_failure_loc_string(const Node* node);

/// \brief Used in evaluator switch statement so that the case type and evaluate call
/// are guaranteed to have the types match.
///
/// Use this in an evaluate_*() function like this
///    switch (arg0->get_element_type())
///    {
///        TYPE_CASE(i8)(arg0, arg1, out, broadcast_spec); break;
///        TYPE_CASE(i16)(arg0, arg1, out, broadcast_spec); break;
///        ...
///    }
///
/// Each TYPE_CASE statement expands like this:
///   case element::Type_t::a: rc = evaluate<element::Type_t::a>(arg0, arg1, out,
///   broadcast_spec)
///
/// \note Don't forget to put a break after each statement or it will fall through and generate
/// a runtime error.

#define TYPE_CASE(a)         \
    case element::Type_t::a: \
        rc = evaluate<element::Type_t::a>

class NodeAccessor;

/**
 * @brief Nodes are the backbone of the graph of Value dataflow. Every node has
 * zero or more nodes as arguments and one value, which is either a tensor
 * or a (possibly empty) tuple of values.
 * @ingroup ov_model_cpp_api
 */
class OPENVINO_API Node : public std::enable_shared_from_this<Node> {
    // For access to m_outputs.
    friend class descriptor::Input;

    // For access to m_inputs and m_outputs.
    template <typename NodeType>
    friend class Input;

    // For access to m_outputs.
    template <typename NodeType>
    friend class Output;

    friend class Model;
    // To fix collisions in generated friendly name
    friend class pass::ResolveNameCollisions;

protected:
    descriptor::Input& get_input_descriptor(size_t position);
    descriptor::Output& get_output_descriptor(size_t position);

    /// \brief Construct an uninitialized Node
    Node();
    /// \brief Copying a node
    Node(const Node&);
    /// \brief Assignment operator
    Node& operator=(const Node&);

    /// \brief Construct an uninitialized Node
    /// \param output_size Number of outputs for this node
    Node(size_t output_size);

    /// \brief Constructor for Node subclasses that have metaclasses.
    /// \param arguments Output i will connect to input i
    /// \param output_size Number of outputs for this node
    Node(const OutputVector& arguments, size_t output_size = 1);
    /// \brief Moves nodes that would be deleted from inputs to nodes to avoid stack overflows
    /// on deep networks.
    void safe_delete(NodeVector& nodes, bool recurse);

    /// \brief Marks an input as being relevant or irrelevant to the output shapes of this
    ///        node.
    /// \param i The index of the input to mark as relevant or irrelevant.
    /// \param relevant true if the input is relevant to output shapes, false otherwise.
    ///
    /// This is used by the shape specialization pass to know which nodes must be statically
    /// evaluated in order to complete shape specialization. (For example, the shape input of
    /// DynReshape must be evaluated statically in order for the output shape to be
    /// determined.) By default, all inputs are marked as shape-irrelevant. Overrides of
    /// validate_and_infer_types should call this function to mark shape-relevant inputs.
    void set_input_is_relevant_to_shape(size_t i, bool relevant = true);

    /// \brief Marks an input as being relevant or irrelevant to the output values of this
    ///        node.
    /// \param i The index of the input to mark as relevant or irrelevant.
    /// \param relevant true if the input is relevant to output values, false otherwise.
    ///
    /// This is used by the shape specialization pass to cut short evaluation in cases where
    /// an input value does not actually have any effect on the output value of the node. (As
    /// of this writing, the only example of this is ShapeOf.) By default, all inputs are
    /// marked as value-relevant. Overrides of validate_and_infer_types should call this
    /// function to mark value-irrelevant inputs.
    void set_input_is_relevant_to_value(size_t i, bool relevant = true);

public:
    /// \brief Verifies that attributes and inputs are consistent and computes output shapes
    /// and element types. Must be implemented by concrete child classes so that it
    /// can be run any number of times.
    ///
    /// Throws if the node is invalid.
    virtual void validate_and_infer_types();

    // Called in constructors during transition
    void constructor_validate_and_infer_types();

    using type_info_t = DiscreteTypeInfo;

    virtual ~Node();

    virtual bool visit_attributes(AttributeVisitor&);
    /// \returns the autobroadcasr spec
    virtual const ov::op::AutoBroadcastSpec& get_autob() const;

    /// \brief Allows to get information about availability of evaluate method for the current
    /// operation
    // \returns true if evaluate is available
    virtual bool has_evaluate() const;

    /// \brief Evaluates the op on input_values putting results in output_values
    /// \param output_values Tensors for the outputs to compute. One for each result
    /// \param input_values Tensors for the inputs. One for each inputs.
    /// \returns true if successful
    virtual bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const;
    /// \brief Evaluates the op on input_values putting results in output_values
    /// \param output_values Tensors for the outputs to compute. One for each result
    /// \param input_values Tensors for the inputs. One for each inputs.
    /// \param evaluation_context Storage of additional settings and attributes that can be used
    /// when evaluating the op.
    /// \returns true if successful
    virtual bool evaluate(ov::TensorVector& output_values,
                          const ov::TensorVector& input_values,
                          const ov::EvaluationContext& evaluationContext) const;
    virtual bool evaluate_lower(ov::TensorVector& output_values) const;
    virtual bool evaluate_upper(ov::TensorVector& output_values) const;
    virtual bool evaluate_symbol(TensorSymbolVector& output_symbols) const;

    virtual bool can_constant_fold(const OutputVector& inputs_values) const;
    virtual bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values);
    /// \brief Decomposes the FusedOp into a sub-graph consisting of core openvino ops
    ///
    /// \return A vector of nodes comprising the sub-graph. The order of output
    ///         tensors must match the match output tensors of the FusedOp
    virtual OutputVector decompose_op() const {
        return OutputVector();
    }
    /// Returns the NodeTypeInfo for the node's class.
    /// During transition to type_info, returns a dummy type_info for Node if the class
    /// has not been updated yet.
    virtual const type_info_t& get_type_info() const = 0;
    const char* get_type_name() const {
        return get_type_info().name;
    }
    /// Sets/replaces the arguments with new arguments.
    void set_arguments(const NodeVector& arguments);
    /// Sets/replaces the arguments with new arguments.
    void set_arguments(const OutputVector& arguments);
    /// Sets/replaces the arguments with new arguments.
    void set_argument(size_t position, const Output<Node>& argument);

    void set_output_type(size_t i, const element::Type& element_type, const PartialShape& pshape);

    /// Sets the number of outputs
    void set_output_size(size_t output_size);

    void invalidate_values();
    virtual void revalidate_and_infer_types() {
        invalidate_values();
        validate_and_infer_types();
    }
    /// \brief Get the string name for the type of the node, such as `Add` or `Multiply`.
    ///        The class name, must not contain spaces as it is used for codegen.
    /// \returns A const reference to the node's type name
    virtual std::string description() const;
    /// \brief Get the unique name of the node.
    /// \returns A const reference to the node's unique name.
    const std::string& get_name() const;

    /// \brief Sets a friendly name for a node. This does not overwrite the unique name
    ///        of the node and is retrieved via get_friendly_name(). Used mainly for debugging.
    ///        The friendly name may be set exactly once.
    /// \param name is the friendly name to set
    void set_friendly_name(const std::string& name);

    /// \brief Gets the friendly name for a node. If no friendly name has been set via
    ///        set_friendly_name then the node's unique name is returned.
    /// \returns A const reference to the node's friendly name.
    const std::string& get_friendly_name() const;

    virtual bool is_dynamic() const;
    size_t get_instance_id() const {
        return m_instance_id;
    }
    /// \brief Writes a description of a node to a stream
    /// \param os The stream; should be returned
    /// \param depth How many levels of inputs to describe
    /// \returns The stream os
    virtual std::ostream& write_description(std::ostream& os, uint32_t depth = 0) const;

    /// Get control dependencies registered on the node
    const std::vector<std::shared_ptr<Node>>& get_control_dependencies() const;

    /// Get nodes dependent on this node
    const std::vector<Node*>& get_control_dependents() const;

    /// This node cannot execute until node executes
    void add_control_dependency(std::shared_ptr<Node> node);

    /// Remove the dependency of this node on node
    void remove_control_dependency(std::shared_ptr<Node> node);

    /// Remove all dependencies from this node
    void clear_control_dependencies();

    /// Remove this node as a dependency from all dependent nodes
    void clear_control_dependents();

    /// This node absorbs the control dependencies of source_node
    void add_node_control_dependencies(const std::shared_ptr<const Node>& source_node);

    /// This node becomes a dependent of every node dependent on source_node
    void add_node_control_dependents(const std::shared_ptr<const Node>& source_node);

    /// This node's control dependencies are replaced by replacement
    void transfer_control_dependents(std::shared_ptr<Node> replacement);

    /// Returns the number of outputs from the node.
    size_t get_output_size() const;

    /// Returns the element type for output i
    const element::Type& get_output_element_type(size_t i) const;

    /// Checks that there is exactly one output and returns its element type
    // TODO: deprecate in favor of node->get_output_element_type(0) with a suitable check in
    // the calling code, or updates to the calling code if it is making an invalid assumption
    // of only one output.
    const element::Type& get_element_type() const;

    /// Returns the shape for output i
    const Shape& get_output_shape(size_t i) const;

    /// Returns the partial shape for output i
    const PartialShape& get_output_partial_shape(size_t i) const;

    /// Return the output to use when converting to an Output<Node> with no index specified.
    /// Throws when not supported.
    Output<const Node> get_default_output() const;
    Output<Node> get_default_output();

    /// Returns the output of the default output, or throws if there is none
    virtual size_t get_default_output_index() const;
    /// Throws no default
    size_t no_default_index() const;

    /// Checks that there is exactly one output and returns its shape
    // TODO: deprecate in favor of node->get_output_shape(0) with a suitable check in the
    // calling code, or updates to the calling code if it is making an invalid assumption of
    // only one output.
    const Shape& get_shape() const;

    /// Returns the tensor for output or input i
    descriptor::Tensor& get_output_tensor(size_t i) const;
    descriptor::Tensor& get_input_tensor(size_t i) const;

    std::set<Input<Node>> get_output_target_inputs(size_t i) const;

    /// Returns the number of inputs for the op
    size_t get_input_size() const;

    /// Returns the element type of input i
    // TODO: deprecate in favor of node->get_input_element_type(i)
    const element::Type& get_input_element_type(size_t i) const;

    /// Returns the shape of input i
    // TODO: deprecate in favor of node->get_input_shape(i)
    const Shape& get_input_shape(size_t i) const;

    /// Returns the partial shape of input i
    // TODO: deprecate in favor of node->get_input_partial_shape(i)
    const PartialShape& get_input_partial_shape(size_t i) const;

    Node* get_input_node_ptr(size_t index) const;
    std::shared_ptr<Node> get_input_node_shared_ptr(size_t index) const;
    Output<Node> get_input_source_output(size_t i) const;

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const = 0;

    std::shared_ptr<Node> copy_with_new_inputs(const OutputVector& new_args) const;

    std::shared_ptr<Node> copy_with_new_inputs(const OutputVector& inputs,
                                               const std::vector<std::shared_ptr<Node>>& control_dependencies) const;

    /// True if this and node have one output with same element type and shape
    bool has_same_type(std::shared_ptr<const Node> node) const;

    using RTMap = std::map<std::string, Any>;

    RTMap& get_rt_info() {
        return m_rt_info;
    }
    const RTMap& get_rt_info() const {
        return m_rt_info;
    }

    /// Get all the nodes that uses the current node
    NodeVector get_users(bool check_is_used = false) const;

    /// Use instance ids for comparison instead of memory addresses to improve determinism
    bool operator<(const Node& other) const {
        return m_instance_id < other.m_instance_id;
    }
    /// \return A vector containing a handle for each of this node's inputs, in order.
    // TODO: Rename to get_inputs()?
    std::vector<Input<Node>> inputs();

    /// \return A vector containing a handle for each of this node's inputs, in order.
    std::vector<Input<const Node>> inputs() const;

    /// \return A vector containing the values for each input
    std::vector<Output<Node>> input_values() const;

    /// \return A vector containing a handle for each of this node's outputs, in order.
    // TODO: Rename to get_outputs()?
    std::vector<Output<Node>> outputs();

    /// \return A vector containing a handle for each of this node's outputs, in order.
    std::vector<Output<const Node>> outputs() const;

    /// \return A handle to the `input_index`th input of this node.
    /// \throw std::out_of_range if the node does not have at least `input_index+1` inputs.
    Input<Node> input(size_t input_index);

    /// \return A handle to the `input_index`th input of this node.
    /// \throw std::out_of_range if the node does not have at least `input_index+1` inputs.
    Input<const Node> input(size_t input_index) const;

    Output<Node> input_value(size_t input_index) const;

    /// \return A handle to the `output_index`th output of this node.
    /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
    Output<Node> output(size_t output_index);

    /// \return A handle to the `output_index`th output of this node.
    /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
    Output<const Node> output(size_t output_index) const;

    virtual bool match_value(ov::pass::pattern::Matcher* matcher,
                             const Output<Node>& pattern_value,
                             const Output<Node>& graph_value);

    virtual bool match_node(ov::pass::pattern::Matcher* matcher, const Output<Node>& graph_value);

protected:
    /// \brief Check constant folding disabled attribute.
    ///
    /// \return true if constant folding disabled otherwise false.
    bool is_const_fold_disabled() const;

private:
    friend class ov::NodeAccessor;
    std::vector<Node*> m_control_dependents;
    std::vector<std::shared_ptr<Node>> m_control_dependencies;
    size_t m_instance_id{m_next_instance_id.fetch_add(1)};
    std::string m_friendly_name;
    mutable std::string m_unique_name;
    mutable std::atomic_bool m_name_changing{false};
    static std::atomic<size_t> m_next_instance_id;
    std::deque<descriptor::Input> m_inputs;
    std::deque<descriptor::Output> m_outputs;
    RTMap m_rt_info;

    // The vector of SharedRTInfo attributes associated to Functions
    // where this node belongs to. SharedRTInfo is private field which
    // is used for internal purposes. For example: tracking changes
    // during graph transformations.
    std::set<std::shared_ptr<SharedRTInfo>> m_shared_rt_info;

    // As node can be included into different Functions which
    // can be executed into multiple threads means that m_shared_rt_info
    // can be updated simultaneously, so we have to guaranty exclusive
    // update of this field by having specific method with mutex.
    void insert_info(std::shared_ptr<SharedRTInfo> info);
    std::mutex m_insert_mutex;
};

using NodeTypeInfo = Node::type_info_t;

OPENVINO_API std::ostream& operator<<(std::ostream&, const Node&);
OPENVINO_API std::ostream& operator<<(std::ostream&, const Node*);

// Like an Output but with a Node* instead of a shared_ptr<Node>
struct RawNodeOutput {
    RawNodeOutput(const Output<Node>& value) : node(value.get_node()), index(value.get_index()) {}
    RawNodeOutput(Node* node, size_t index) : node(node), index(index) {}
    RawNodeOutput(const RawNodeOutput&) = default;
    RawNodeOutput() = default;
    RawNodeOutput& operator=(const RawNodeOutput&) = default;

    Node* node;
    size_t index{0};

    operator Output<Node>() {
        return Output<Node>(node, index);
    }
    bool operator==(const RawNodeOutput& other) const {
        return node == other.node && index == other.index;
    }
    bool operator!=(const RawNodeOutput& other) const {
        return !(*this == other);
    }
    bool operator<(const RawNodeOutput& other) const {
        return node < other.node || (node == other.node && index < other.index);
    }
    bool operator>(const RawNodeOutput& other) const {
        return node > other.node || (node == other.node && index > other.index);
    }
    bool operator<=(const RawNodeOutput& other) const {
        return !(*this > other);
    }
    bool operator>=(const RawNodeOutput& other) const {
        return !(*this < other);
    }
};

using RawNodeOutputMap = std::map<RawNodeOutput, Output<Node>>;

class OPENVINO_API NodeValidationFailure : public ov::AssertFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const Node* node,
                                    const std::string& explanation);

    template <class TShape>
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    std::pair<const Node*, const std::vector<TShape>*>&& ctx,
                                    const std::string& explanation);

protected:
    explicit NodeValidationFailure(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

/**
 * @brief Specialization to throw the `NodeValidationFailure` for shape inference using `PartialShape`
 *
 * @param check_loc_info Exception location details to print.
 * @param ctx            NodeValidationFailure context which got pointer to node and input shapes used for shape
 * inference.
 * @param explanation    Exception explanation string.
 */
template <>
OPENVINO_API void NodeValidationFailure::create(const char* file,
                                                int line,
                                                const char* check_string,
                                                std::pair<const Node*, const std::vector<PartialShape>*>&& ctx,
                                                const std::string& explanation);
}  // namespace ov
#define NODE_VALIDATION_CHECK(node, ...) OPENVINO_ASSERT_HELPER(::ov::NodeValidationFailure, (node), __VA_ARGS__)

/** \brief Throw NodeValidationFailure with additional information about input shapes used during shape inference. */
#define NODE_SHAPE_INFER_CHECK(node, input_shapes, ...) \
    NODE_VALIDATION_CHECK(std::make_pair(static_cast<const ::ov::Node*>((node)), &(input_shapes)), __VA_ARGS__)

namespace ov {

/**
 * @brief Check new arguments size if match node inputs count.
 *
 * This check is required in cloning ov::Node.
 *
 * @param node      Pointer to node.
 * @param new_args  Vector with new outputs to check.
 */
void OPENVINO_API check_new_args_count(const Node* const node, const OutputVector& new_args);

/// \brief Visits a reference to a node that has been registered with the visitor.
template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<ov::Node>> : public VisitorAdapter {
public:
    AttributeAdapter(std::shared_ptr<ov::Node>& value);

    bool visit_attributes(AttributeVisitor& visitor) override;
    OPENVINO_RTTI("AttributeAdapter<std::shared_ptr<Node>>");

protected:
    std::shared_ptr<ov::Node>& m_ref;
};

template <>
class OPENVINO_API AttributeAdapter<ov::NodeVector> : public VisitorAdapter {
public:
    AttributeAdapter(ov::NodeVector& ref);

    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<NodeVector>");

protected:
    ov::NodeVector& m_ref;
};

}  // namespace ov
