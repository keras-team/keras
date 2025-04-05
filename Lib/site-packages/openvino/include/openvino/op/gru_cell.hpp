// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/op.hpp"
#include "openvino/op/util/activation_functions.hpp"
#include "openvino/op/util/rnn_cell_base.hpp"

namespace ov {
namespace op {
namespace v3 {
///
/// \brief      Class for GRU cell node.
///
/// \note       Note this class represents only single *cell* and not whole GRU *layer*.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GRUCell : public util::RNNCellBase {
public:
    OPENVINO_OP("GRUCell", "opset3", op::util::RNNCellBase);
    GRUCell();
    ///
    /// \brief      Constructs GRUCell node.
    ///
    /// \param[in]  X                     The input tensor with shape: [batch_size,
    ///                                   input_size].
    /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
    ///                                   with shape: [batch_size, hidden_size].
    /// \param[in]  W                     The weight tensor with shape:
    ///                                   [gates_count * hidden_size, input_size].
    /// \param[in]  R                     The recurrence weight tensor with shape:
    ///                                   [gates_count * hidden_size, hidden_size].
    /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
    ///
    GRUCell(const Output<Node>& X,
            const Output<Node>& initial_hidden_state,
            const Output<Node>& W,
            const Output<Node>& R,
            std::size_t hidden_size);

    ///
    /// \brief      Constructs GRUCell node.
    ///
    /// \param[in]  X                     The input tensor with shape: [batch_size,
    ///                                   input_size].
    /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
    ///                                   with shape: [batch_size, hidden_size].
    /// \param[in]  W                     The weight tensor with shape:
    ///                                   [gates_count * hidden_size, input_size].
    /// \param[in]  R                     The recurrence weight tensor with shape:
    ///                                   [gates_count * hidden_size, hidden_size].
    /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
    /// \param[in]  activations           The vector of activation functions used inside
    ///                                   recurrent cell.
    /// \param[in]  activations_alpha     The vector of alpha parameters for activation
    ///                                   functions in order respective to activation
    ///                                   list.
    /// \param[in]  activations_beta      The vector of beta parameters for activation
    ///                                   functions in order respective to activation
    ///                                   list.
    /// \param[in]  clip                  The value defining clipping range [-clip,
    ///                                   clip] on input of activation functions.
    ///
    GRUCell(const Output<Node>& X,
            const Output<Node>& initial_hidden_state,
            const Output<Node>& W,
            const Output<Node>& R,
            std::size_t hidden_size,
            const std::vector<std::string>& activations,
            const std::vector<float>& activations_alpha,
            const std::vector<float>& activations_beta,
            float clip,
            bool linear_before_reset);

    ///
    /// \brief      Constructs GRUCell node.
    ///
    /// \param[in]  X                     The input tensor with shape: [batch_size,
    ///                                   input_size].
    /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
    ///                                   with shape: [batch_size, hidden_size].
    /// \param[in]  W                     The weight tensor with shape: [gates_count *
    ///                                   hidden_size, input_size].
    /// \param[in]  R                     The recurrence weight tensor with shape:
    ///                                   [gates_count * hidden_size, hidden_size].
    /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
    /// \param[in]  B                     The sum of biases (weight and recurrence) for
    ///                                   update, reset and hidden gates.
    ///                                   If linear_before_reset := true then biases for
    ///                                   hidden gates are
    ///                                   placed separately (weight and recurrence).
    ///                                   Shape: [gates_count * hidden_size] if
    ///                                   linear_before_reset := false
    ///                                   Shape: [(gates_count + 1) * hidden_size] if
    ///                                   linear_before_reset := true
    /// \param[in]  activations           The vector of activation functions used inside
    ///                                   recurrent cell.
    /// \param[in]  activations_alpha     The vector of alpha parameters for activation
    ///                                   functions in order respective to activation
    ///                                   list.
    /// \param[in]  activations_beta      The vector of beta parameters for activation
    ///                                   functions in order respective to activation
    ///                                   list.
    /// \param[in]  clip                  The value defining clipping range [-clip,
    ///                                   clip] on input of activation functions.
    /// \param[in]  linear_before_reset   Whether or not to apply the linear
    ///                                   transformation before multiplying by the
    ///                                   output of the reset gate.
    ///
    GRUCell(const Output<Node>& X,
            const Output<Node>& initial_hidden_state,
            const Output<Node>& W,
            const Output<Node>& R,
            const Output<Node>& B,
            std::size_t hidden_size,
            const std::vector<std::string>& activations = std::vector<std::string>{"sigmoid", "tanh"},
            const std::vector<float>& activations_alpha = {},
            const std::vector<float>& activations_beta = {},
            float clip = 0.f,
            bool linear_before_reset = false);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_linear_before_reset() const {
        return m_linear_before_reset;
    }

    void set_linear_before_reset(bool linear_before_reset) {
        m_linear_before_reset = linear_before_reset;
    }

private:
    /// brief Add and initialize bias input to all zeros.
    void add_default_bias_input();

    ///
    /// \brief The Activation function f.
    ///
    util::ActivationFunction m_activation_f;
    ///
    /// \brief The Activation function g.
    ///
    util::ActivationFunction m_activation_g;

    static constexpr std::size_t s_gates_count{3};
    ///
    /// \brief Control whether or not apply the linear transformation.
    ///
    /// \note The linear transformation may be applied when computing the output of
    ///       hidden gate. It's done before multiplying by the output of the reset gate.
    ///
    bool m_linear_before_reset;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
