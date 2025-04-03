// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ov {
namespace op {
namespace v1 {
// clang-format off
/// \brief Tensor sum operation.
///
/// Element-wise sums the input tensor, eliminating the specified reduction axes.
/// For example:
///
/// \f[
///     \mathit{sum}\left(\{0\},
///         \left[ \begin{array}{ccc}
///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
///     \left[ (1 + 3 + 5), (2 + 4 + 6) \right] =
///     \left[ 9, 12 \right]~~~\text{(dimension 0 (rows) is eliminated)}
/// \f]
///
/// \f[
///     \mathit{sum}\left(\{1\},
///         \left[ \begin{array}{ccc}
///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
///     \left[ (1 + 2), (3 + 4), (5 + 6) \right] =
///     \left[ 3, 7, 11 \right]~~~\text{(dimension 1 (columns) is eliminated)}
/// \f]
///
/// \f[
///     \mathit{sum}\left(\{0,1\},
///         \left[ \begin{array}{ccc}
///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
///      (1 + 2) + (3 + 4) + (5 + 6) =
///      21~~~\text{(both dimensions (rows and columns) are eliminated)}
/// \f]
///
/// ## Parameters
///
/// |                      | Description                                            |
/// | -------------------- | ----------------------------------------               |
/// | `reduction_axes`     | The axes to eliminate through summation.               |
/// | `keep_dims`          | If set to 1 it holds axes that are used for reduction. |
///
/// ## Inputs
///
/// |       | Type                              | Description                                            |
/// | ----- | --------------------------------- | ------------------------------------------------------ |
/// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape and numeric element type. |
///
/// ## Output
///
/// | Type                                      | Description                                                                                                      |
/// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
/// | \f$N[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by summation. |
/// \ingroup ov_ops_cpp_api
// clang-format on
class OPENVINO_API ReduceSum : public util::ArithmeticReductionKeepDims {
public:
    OPENVINO_OP("ReduceSum", "opset1", util::ArithmeticReductionKeepDims);
    /// \brief Constructs a summation operation.
    ReduceSum() = default;
    /// \brief Constructs a summation operation.
    ///
    /// \param arg The tensor to be summed.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    /// \param keep_dims If set to 1 it holds axes that are used for reduction.
    ReduceSum(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
