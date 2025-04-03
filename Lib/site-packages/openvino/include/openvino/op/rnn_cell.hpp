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
namespace v0 {
///
/// \brief      Class for single RNN cell node.
///
/// \note       It follows notation and equations defined as in ONNX standard:
///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN
///
/// \note       It calculates following equations:
///
///             Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
///
///             *       - Is a dot product,
///             f       - is activation functions.
///
/// \note       This class represents only single *cell* (for current time step)
///             and not the whole RNN Sequence layer
///
/// \sa         LSTMSequence, LSTMCell, GRUCell
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RNNCell : public util::RNNCellBase {
public:
    OPENVINO_OP("RNNCell", "opset1", util::RNNCellBase);

    RNNCell();
    ///
    /// \brief      Constructs RNNCell node.
    ///
    /// \param[in]  X                     The input tensor with shape: [batch_size,
    ///                                   input_size].
    /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
    ///                                   with shape: [batch_size, hidden_size].
    /// \param[in]  W                     The weight tensor with shape: [hidden_size,
    ///                                   input_size].
    /// \param[in]  R                     The recurrence weight tensor with shape:
    ///                                   [hidden_size, hidden_size].
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
    RNNCell(const Output<Node>& X,
            const Output<Node>& initial_hidden_state,
            const Output<Node>& W,
            const Output<Node>& R,
            std::size_t hidden_size,
            const std::vector<std::string>& activations = std::vector<std::string>{"tanh"},
            const std::vector<float>& activations_alpha = {},
            const std::vector<float>& activations_beta = {},
            float clip = 0.f);

    ///
    /// \brief      Constructs RNNCell node.
    ///
    /// \param[in]  X                     The input tensor with shape: [batch_size,
    ///                                   input_size].
    /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
    ///                                   with shape: [batch_size, hidden_size].
    /// \param[in]  W                     The weight tensor with shape: [hidden_size,
    ///                                   input_size].
    /// \param[in]  R                     The recurrence weight tensor with shape:
    ///                                   [hidden_size, hidden_size].
    /// \param[in]  B                     The bias tensor for input gate with shape:
    ///                                   [hidden_size].
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
    RNNCell(const Output<Node>& X,
            const Output<Node>& initial_hidden_state,
            const Output<Node>& W,
            const Output<Node>& R,
            const Output<Node>& B,
            std::size_t hidden_size,
            const std::vector<std::string>& activations = std::vector<std::string>{"tanh"},
            const std::vector<float>& activations_alpha = {},
            const std::vector<float>& activations_beta = {},
            float clip = 0.f);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    ///
    /// \brief      Creates the default bias input initialized with zeros.
    ///
    /// \return     The object of Output class.
    ///
    Output<Node> get_default_bias_input() const;

    ///
    /// \brief The Activation function f.
    ///
    util::ActivationFunction m_activation_f;

    static constexpr std::size_t s_gates_count{1};
};
}  // namespace v0
}  // namespace op
}  // namespace ov
