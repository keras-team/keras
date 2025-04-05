// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise natural exponential (exp) operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Exp : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Exp", "opset1", UnaryElementwiseArithmetic);

    /// \brief Constructs an exponential operation.
    Exp() = default;
    /// \brief Constructs an exponential operation.
    ///
    /// \param arg Node that produces the input tensor.
    Exp(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
