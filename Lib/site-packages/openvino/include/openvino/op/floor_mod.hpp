// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise FloorMod operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API FloorMod : public util::BinaryElementwiseArithmetic {
public:
    OPENVINO_OP("FloorMod", "opset1", op::util::BinaryElementwiseArithmetic);

    /// \brief Constructs an uninitialized addition operation
    FloorMod() : util::BinaryElementwiseArithmetic(AutoBroadcastType::NUMPY) {}

    /// \brief Constructs an Floor Mod operation.
    ///
    /// \param arg0 Output that produces the first input tensor.<br>
    /// `[d0, ...]`
    /// \param arg1 Output that produces the second input tensor.<br>
    /// `[d0, ...]`
    /// \param auto_broadcast Auto broadcast specification
    ///
    /// Output `[d0, ...]`
    ///
    FloorMod(const Output<Node>& arg0,
             const Output<Node>& arg1,
             const AutoBroadcastSpec& auto_broadcast = AutoBroadcastType::NUMPY);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
