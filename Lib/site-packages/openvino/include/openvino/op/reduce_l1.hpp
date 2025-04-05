// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ov {
namespace op {
namespace v4 {
/// \brief Reduction operation using L1 norm: L1(x) = sum(abs(x)) if all dimensions are
/// specified for the normalisation.
///
/// Reduces the tensor, eliminating the specified reduction axes by taking the L1-norm.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ReduceL1 : public util::ArithmeticReductionKeepDims {
public:
    OPENVINO_OP("ReduceL1", "opset4", util::ArithmeticReductionKeepDims);
    /// \brief Constructs a reducet L1-norm operation.
    ReduceL1() = default;
    /// \brief Constructs a reduce L1-norm operation.
    ///
    /// \param arg The tensor to be reduced.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    /// \param keep_dims If set to true it holds axes that are used for reduction.
    ReduceL1(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v4
}  // namespace op
}  // namespace ov
