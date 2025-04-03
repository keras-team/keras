// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/activation_functions.hpp"

namespace ov {
namespace op {
namespace util {
enum class LSTMWeightsFormat {
    FICO,  // IE
    ICOF,  // PyTorch
    IFCO,  // DNNL, TF, MxNet
    IFOC,  // Caffe
    IOFC,  // ONNX
};

enum class LSTMPeepholesFormat {
    FIO,  // IE
    IOF,  // ONNX, PyTorch
    IFO,  // CAFe, DNNL, TF, MxNet
};

///
/// \brief      Change data format of provided node.
///
/// \param[in]  node  The input node to be permuted.
///
///
/// \param[in]  from_format  Original node weights format.
///
///
/// \param[in]  to_format  Weights format to convert to.
///
/// \return     Node representing reshaped tensor according to `to_format` weights
/// format.
///
std::shared_ptr<Node> OPENVINO_API convert_lstm_node_format(const Output<Node>& node,
                                                            LSTMWeightsFormat from_format,
                                                            LSTMWeightsFormat to_format = LSTMWeightsFormat::FICO,
                                                            int64_t axis = 0);

std::shared_ptr<Node> OPENVINO_API
convert_lstm_peepholes_format(const Output<Node>& node,
                              LSTMPeepholesFormat from_format,
                              LSTMPeepholesFormat to_format = LSTMPeepholesFormat::FIO,
                              int64_t axis = 0);

/// \brief      Base class for all recurrent network cells.
///
/// \note       It holds all common attributes.
///
class OPENVINO_API RNNCellBase : public Op {
public:
    OPENVINO_OP("RNNCellBase", "util");

    ///
    /// \brief      Constructs a RNNCellBase class.
    ///
    /// \param[in]  hidden_size        The number of hidden units for recurrent cell.
    /// \param[in]  clip               The value defining clipping range [-clip, clip]
    ///                                on input of activation functions.
    /// \param[in]  activations        The vector of activation functions used inside
    ///                                recurrent cell.
    /// \param[in]  activations_alpha  The vector of alpha parameters for activation
    ///                                functions in order respective to activation list.
    /// \param[in]  activations_beta   The vector of beta parameters for activation
    ///                                functions in order respective to activation list.
    ///
    RNNCellBase(const OutputVector& args,
                std::size_t hidden_size,
                float clip,
                const std::vector<std::string>& activations,
                const std::vector<float>& activations_alpha,
                const std::vector<float>& activations_beta);

    RNNCellBase();
    ~RNNCellBase() override = default;

    ///
    /// \brief      Validates static rank and dimension for provided input parameters.
    ///             Additionally input_size dimension is checked for X and W inputs.
    ///
    ///
    /// \param[in]  input           Vector with RNN-Cell op inputs in following order:
    ///                             X, initial_hidden_state, W, R and B.
    ///
    void validate_input_rank_dimension(const std::vector<PartialShape>& input);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::size_t get_hidden_size() const {
        return m_hidden_size;
    }
    void set_hidden_size(size_t hidden_size) {
        m_hidden_size = hidden_size;
    }
    float get_clip() const {
        return m_clip;
    }
    const std::vector<std::string>& get_activations() const {
        return m_activations;
    }
    const std::vector<float>& get_activations_alpha() const {
        return m_activations_alpha;
    }
    const std::vector<float>& get_activations_beta() const {
        return m_activations_beta;
    }

protected:
    ///
    /// \brief      Constructs activation function object.
    ///
    /// \param[in]  idx   The index of the activation function name.
    ///
    /// \return     The object representing activation function.
    ///
    ActivationFunction get_activation_function(std::size_t idx) const;
    ///
    /// \brief      Creates node with element-wise add operation with numpy
    ///             broadcasting.
    ///
    /// \param[in]  lhs   The left hand side argument node.
    /// \param[in]  rhs   The right hand side argument node.
    ///
    /// \return     Node with element-wise add operation.
    ///
    static std::shared_ptr<Node> add(const Output<Node>& lhs, const Output<Node>& rhs);
    ///
    /// \brief      Creates node with element-wise subtract operation with numpy
    ///             broadcasting.
    ///
    /// \param[in]  lhs   The left hand side argument node.
    /// \param[in]  rhs   The right hand side argument node.
    ///
    /// \return     Node with element-wise subtract operation.
    ///
    static std::shared_ptr<Node> sub(const Output<Node>& lhs, const Output<Node>& rhs);
    ///
    /// \brief      Creates node with element-wise multiply operation with numpy
    ///             broadcasting.
    ///
    /// \param[in]  lhs   The left hand side argument node.
    /// \param[in]  rhs   The right hand side argument node.
    ///
    /// \return     Node with element-wise multiply operation.
    ///
    static std::shared_ptr<Node> mul(const Output<Node>& lhs, const Output<Node>& rhs);
    ///
    /// \brief      Creates node with element-wise clip operation with numpy
    ///             broadcasting.
    ///
    /// \param[in]  data   The input tensor for clipping.
    ///
    /// \return     Node with element-wise clip operation.
    ///
    std::shared_ptr<Node> clip(const Output<Node>& data) const;

protected:
    std::size_t m_hidden_size;
    float m_clip;
    std::vector<std::string> m_activations;
    std::vector<float> m_activations_alpha;
    std::vector<float> m_activations_beta;
};
}  // namespace util
}  // namespace op
}  // namespace ov
