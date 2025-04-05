// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief ReduceMax operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ReduceMax : public util::ArithmeticReductionKeepDims {
public:
    OPENVINO_OP("ReduceMax", "opset1", util::ArithmeticReductionKeepDims);
    /// \brief Constructs a summation operation.
    ReduceMax() = default;
    /// \brief Constructs a summation operation.
    ///
    /// \param arg The tensor to be summed.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    /// \param keep_dims If set to 1 it holds axes that are used for reduction.
    ReduceMax(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
