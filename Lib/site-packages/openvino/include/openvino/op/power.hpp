// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v1 {
// clang-format off
/// \brief Elementwise exponentiation operation.
///
/// ## Inputs
///
/// |        | Type                              | Description                                            |
/// | ------ | --------------------------------- | ------------------------------------------------------ |
/// | `arg0` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type.        |
/// | `arg1` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
///
/// ## Output
///
/// | Type                   | Description                                                                                                    |
/// | ---------------------- | -------------------------------------------------------------------------------------------------------------- |
/// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \texttt{arg0}[i_1,\dots,i_n]^{\texttt{arg1}[i_1,\dots,i_n]}\f$ |
/// \ingroup ov_ops_cpp_api
// clang-format on
class OPENVINO_API Power : public util::BinaryElementwiseArithmetic {
public:
    OPENVINO_OP("Power", "opset1", op::util::BinaryElementwiseArithmetic);

    Power() : util::BinaryElementwiseArithmetic(AutoBroadcastType::NUMPY) {}

    /// \brief Constructs an exponentiation operation.
    ///
    /// \param arg0 Node that produces the first input tensor.
    /// \param arg1 Node that produces the second input tensor.
    /// \param auto_broadcast Auto broadcast specification
    Power(const Output<Node>& arg0,
          const Output<Node>& arg1,
          const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
