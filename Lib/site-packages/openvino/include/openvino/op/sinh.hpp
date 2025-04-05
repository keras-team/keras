// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise hyperbolic sine (sinh) operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Sinh : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Sinh", "opset1", util::UnaryElementwiseArithmetic);
    /// \brief Constructs a hyperbolic sine operation.
    ///
    /// \param arg Node that produces the input tensor.
    Sinh(const Output<Node>& arg);
    Sinh() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
