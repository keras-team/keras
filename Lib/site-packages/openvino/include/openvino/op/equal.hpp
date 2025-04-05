// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_comparison.hpp"

namespace ov {
namespace op {
namespace v1 {
// clang-format off
/// \brief Elementwise is-equal operation.
///
/// ## Inputs
///
/// |        | Type                              | Description                                            |
/// | ------ | --------------------------------- | ------------------------------------------------------ |
/// | `arg0` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and element type.                |
/// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
/// | `autob`| AutoBroadcastSpec                 | Auto broadcast specification.                          |
///
/// ## Output
///
/// | Type                               | Description                                                                                                                                |
/// | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
/// | \f$\texttt{bool}[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = 1\text{ if }\texttt{arg0}[i_1,\dots,i_n] = \texttt{arg1}[i_1,\dots,i_n]\text{, else } 0\f$ |
/// \ingroup ov_ops_cpp_api
// clang-format on
class OPENVINO_API Equal : public util::BinaryElementwiseComparison {
public:
    OPENVINO_OP("Equal", "opset1", op::util::BinaryElementwiseComparison);
    /// \brief Constructs an equal operation.
    Equal() : util::BinaryElementwiseComparison(AutoBroadcastType::NUMPY) {}
    /// \brief Constructs an equal operation.
    ///
    /// \param arg0 Node that produces the first input tensor.
    /// \param arg1 Node that produces the second input tensor.
    /// \param auto_broadcast Auto broadcast specification
    Equal(const Output<Node>& arg0,
          const Output<Node>& arg1,
          const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
