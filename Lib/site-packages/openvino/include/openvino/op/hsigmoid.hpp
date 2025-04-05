// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v5 {
/// \brief A HSigmoid Activation Function
/// f(x) = min(max(x + 3, 0), 6) / 6 or
/// f(x) = min(ReLU(x + 3), 6) / 6
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API HSigmoid : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("HSigmoid", "opset5", op::util::UnaryElementwiseArithmetic);
    HSigmoid() = default;

    /// \brief Constructs a HSigmoid operation.
    ///
    /// \param data Input tensor
    HSigmoid(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v5
}  // namespace op
}  // namespace ov
