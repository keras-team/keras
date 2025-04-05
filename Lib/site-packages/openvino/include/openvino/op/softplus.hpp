// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v4 {
/// \brief A Self Regularized Non-Monotonic Neural Activation Function
/// f(x) =  ln(exp(x) + 1.)
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SoftPlus : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("SoftPlus", "opset4", util::UnaryElementwiseArithmetic);

    SoftPlus() = default;
    /// \brief Constructs an SoftPlus operation.
    ///
    /// \param data Input tensor
    SoftPlus(const Output<Node>& arg);
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v4
}  // namespace op
}  // namespace ov
