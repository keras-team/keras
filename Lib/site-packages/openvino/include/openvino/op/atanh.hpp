// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Elementwise inverse hyperbolic tangent operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Atanh : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Atanh", "opset4", util::UnaryElementwiseArithmetic);

    /// \brief Constructs an Atanh operation.
    Atanh() = default;
    /// \brief Constructs an Atanh operation.
    ///
    /// \param arg Output that produces the input tensor.<br>
    /// `[d1, ...]`
    ///
    /// Output `[d1, ...]`
    ///
    Atanh(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor&) override {
        return true;
    }
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
