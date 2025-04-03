// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/binary_elementwise_bitwise.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Elementwise bitwise BitwiseRightShift operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BitwiseRightShift : public util::BinaryElementwiseBitwise {
public:
    OPENVINO_OP("BitwiseRightShift", "opset15", util::BinaryElementwiseBitwise);
    /// \brief Constructs a bitwise BitwiseRightShift operation.
    BitwiseRightShift() = default;
    /// \brief Constructs a bitwise BitwiseRightShift operation.
    ///
    /// \param arg0 Node with data to be shifted.
    /// `[d0, ...]`
    /// \param arg1 Node with number of shifts.
    /// `[d0, ...]`
    /// \param auto_broadcast Auto broadcast specification. Default is Numpy-style
    ///                       implicit broadcasting.
    ///
    /// Output `[d0, ...]`
    ///
    BitwiseRightShift(const Output<Node>& arg0,
                      const Output<Node>& arg1,
                      const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v15
}  // namespace op
}  // namespace ov
