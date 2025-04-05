// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
class Node;

namespace descriptor {
class Output;

// Describes a tensor that is an input to an op, directly or indirectly via a tuple
class OPENVINO_API Input {
    friend class ov::Node;

public:
    /// \param node The node that owns this input
    /// \param index The position of this tensor in all input tensors
    /// \param output The output that supplies a value for this input
    Input(Node* node, size_t index, Output& output);
    /// \brief Create an Input that is not connected to an output
    /// \param node The node that owns this input
    /// \param index The position of this tensor in all input tensors
    Input(Node* node, size_t index);
    ~Input();

    /// \return the node that this is an input of
    std::shared_ptr<Node> get_node() const;

    /// \return the raw pointer to the node that this is an input of
    Node* get_raw_pointer_node() const {
        return m_node;
    }
    /// \return the position within all supplied tensors of this input
    size_t get_index() const {
        return m_index;
    }
    /// \return the connected output
    const Output& get_output() const {
        return *m_output;
    }
    /// \return the connected output
    Output& get_output() {
        return *m_output;
    }
    /// \return true if an output is connected to the input.
    bool has_output() const {
        return m_output != nullptr;
    }
    /// \return the tensor of the connected output
    const Tensor& get_tensor() const;

    /// \return the tensor of the connected output
    Tensor& get_tensor();

    RTMap& get_rt_info() {
        return m_rt_info;
    }
    const RTMap& get_rt_info() const {
        return m_rt_info;
    }

    /// \brief Replace the current output that supplies a value for this input with output i
    ///        of node
    void replace_output(const std::shared_ptr<Node>& node, size_t i);
    /// \brief Replace the current output that supplies a value for this input with output
    void replace_output(Output& output);
    /// \brief Remove the output from this input. The node will not be valid until another
    ///        output is supplied.
    void remove_output();

    /// \return true if the value of this input is relevant to the output shapes of the
    ///         corresponding node. (Usually this is false.)
    ///
    /// See Node::set_input_is_relevant_to_shape for more details.
    bool get_is_relevant_to_shape() const {
        return m_is_relevant_to_shape;
    }
    /// \return true if the value of this input is relevant to the output value of the
    ///         corresponding node. (Usually this is true.)
    ///
    /// See Node::set_input_is_relevant_to_value for more details.
    bool get_is_relevant_to_value() const {
        return m_is_relevant_to_value;
    }

    /// \return the shape of the connected output
    const Shape& get_shape() const;

    /// \return the partial shape of the connected output
    const PartialShape& get_partial_shape() const;

    /// \return the element type of the connected output
    const element::Type& get_element_type() const;

    Input(const Input&) = default;
    Input(Input&&) = default;
    Input& operator=(const Input&) = default;

protected:
    // owner of an argument node (in lieu of m_arguments)
    std::shared_ptr<Node> m_src_node;
    Node* m_node;    // The node we are an input for
    size_t m_index;  // Index into all input tensors
    Output* m_output;
    RTMap m_rt_info;

private:
    bool m_is_relevant_to_shape;
    bool m_is_relevant_to_value;
};
}  // namespace descriptor
}  // namespace ov
