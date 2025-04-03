// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise sign operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Sign : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Sign", "opset1", util::UnaryElementwiseArithmetic);

    Sign() = default;
    /// \brief Constructs an elementwise sign operation.
    ///
    /// \param arg Node that produces the input tensor.
    Sign(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
