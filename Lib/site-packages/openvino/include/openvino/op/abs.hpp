// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise absolute value operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Abs : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Abs", "opset1", util::UnaryElementwiseArithmetic);
    /// \brief Constructs an absolute value operation.
    Abs() = default;
    /// \brief Constructs an absolute value operation.
    ///
    /// \param arg Output that produces the input tensor.<br>
    /// `[d1, ...]`
    ///
    /// Output `[d1, ...]`
    ///
    Abs(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_lower(ov::TensorVector& output_values) const override;
    bool evaluate_upper(ov::TensorVector& output_values) const override;
    bool evaluate_symbol(ov::TensorSymbolVector& output_symbols) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
