// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
// clang-format off
/// \brief Elementwise tangent operation.
///
/// ## Inputs
///
/// |       | Type                              | Description                                     |
/// | ----- | --------------------------------- | ----------------------------------------------- |
/// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
///
/// ## Output
///
/// | Type                   | Description                                                                          |
/// | ---------------------- | ------------------------------------------------------------------------------------ |
/// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \tan(\texttt{arg}[i_1,\dots,i_n])\f$ |
/// \ingroup ov_ops_cpp_api
// clang-format on
class OPENVINO_API Tan : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Tan", "opset1", util::UnaryElementwiseArithmetic);
    /// \brief Constructs a tangent operation.
    ///
    /// \param arg Node that produces the input tensor.
    Tan(const Output<Node>& arg);
    Tan() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
