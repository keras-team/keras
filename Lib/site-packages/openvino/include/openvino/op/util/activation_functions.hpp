// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4100)
#endif

namespace ov {
namespace op {
namespace util {
namespace error {
struct UnknownActivationFunction : Exception {
    OPENVINO_SUPPRESS_DEPRECATED_START
    UnknownActivationFunction(const std::string& func_name) : Exception{"Unknown activation function: " + func_name} {}
    OPENVINO_SUPPRESS_DEPRECATED_END
};
}  // namespace error

namespace detail {
std::shared_ptr<Node> sigmoid(const std::shared_ptr<Node>& arg, float alpha, float beta);
std::shared_ptr<Node> tanh(const std::shared_ptr<Node>& arg, float alpha, float beta);
std::shared_ptr<Node> relu(const std::shared_ptr<Node>& arg, float alpha, float beta);
std::shared_ptr<Node> hardsigmoid(const std::shared_ptr<Node>& arg, float alpha, float beta);
}  // namespace detail

using ActivationFunctionType = std::shared_ptr<Node> (*)(const std::shared_ptr<Node>&, float, float);

///
/// \brief      Class representing activation function used in RNN cells.
///
class OPENVINO_API ActivationFunction {
public:
    ActivationFunction(ActivationFunctionType f, float alpha, float beta);
    ActivationFunction(ActivationFunctionType f, float alpha);
    ActivationFunction(ActivationFunctionType f);
    ActivationFunction() = default;

    ///
    /// \brief  Calls stored activation function with provided node argument.
    ///
    std::shared_ptr<Node> operator()(const std::shared_ptr<Node>& arg) const;

    void set_alpha(float alpha) {
        m_alpha = alpha;
    }
    void set_beta(float beta) {
        m_beta = beta;
    }

private:
    /// \brief Activation function wrapper.
    ActivationFunctionType m_function;
    /// \brief Activation function alpha parameter (may be unused).
    float m_alpha;
    /// \brief Activation function beta parameter (may be unused).
    float m_beta;
};

/// \brief      Gets the activation function by name.
///
/// \param[in]  func_name  The function name
///
/// \throws     UnknownActivationFunction When provided func_name is unknown.
///
/// \return     The activation function object.
///
ActivationFunction get_activation_func_by_name(const std::string& func_name);
}  // namespace util

}  // namespace op

}  // namespace ov

#ifdef _MSC_VER
#    pragma warning(pop)
#endif
