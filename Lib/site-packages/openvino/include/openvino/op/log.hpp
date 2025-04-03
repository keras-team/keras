// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise natural log operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Log : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Log", "opset1", op::util::UnaryElementwiseArithmetic);
    /// \brief Constructs a natural log operation.
    Log() = default;
    /// \brief Constructs a natural log operation.
    ///
    /// \param arg Node that produces the input tensor.
    Log(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
