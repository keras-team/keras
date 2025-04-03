// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Sigmoid operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Sigmoid : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Sigmoid", "opset1", util::UnaryElementwiseArithmetic);
    Sigmoid(const Output<Node>& arg);
    Sigmoid() = default;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
