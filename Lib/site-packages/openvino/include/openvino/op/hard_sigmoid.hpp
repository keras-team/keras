// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief      Parameterized, bounded sigmoid-like, piecewise linear
///             function. min(max(alpha*x + beta, 0), 1)
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API HardSigmoid : public Op {
public:
    OPENVINO_OP("HardSigmoid", "opset1");

    HardSigmoid();

    /// \brief      Constructs a HardSigmoid operation.
    ///
    /// \param      data   Input tensor.
    /// \param[in]  alpha  A scalar value representing the alpha parameter.
    /// \param[in]  beta   A scalar value representing the beta parameter.
    ///
    HardSigmoid(const Output<Node>& data, const Output<Node>& alpha, const Output<Node>& beta);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
