// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/pad_base.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Generic padding operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Pad : public util::PadBase {
public:
    OPENVINO_OP("Pad", "opset1", op::util::PadBase);

    /// \brief Constructs a Pad-1 operation.
    Pad() = default;
    /// \brief Constructs a Pad-1 operation.
    ///
    /// \param arg The output producing input tensor to be padded.
    /// \param pads_begin The output which specifies the number of padding elements
    /// added before position 0 on each axis of arg.
    /// \param pads_end The output which specifies the number of padding elements
    /// after the last element on each axis.
    /// \param arg_pad_value The scalar output with the value used for padding
    /// if pad_mode is CONSTANT
    /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
    /// CONSTANT initializes new elements with arg_pad_value, EDGE uses the nearest
    /// value from arg. REFLECT and SYMMETRIC tile the background by flipping arg
    /// at the edge (SYMMETRIC) or on the last row/column/etc. (REFLECT).
    Pad(const Output<Node>& arg,
        const Output<Node>& pads_begin,
        const Output<Node>& pads_end,
        const Output<Node>& arg_pad_value,
        PadMode pad_mode);

    /// \brief Constructs a Pad-1 operation.
    ///
    /// \param arg The output producing input tensor to be padded.
    /// \param pads_begin The output which specifies the number of padding elements
    /// added
    /// \param pads_end The output which specifies the number of padding elements
    /// after the last element on each axis.
    /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
    Pad(const Output<Node>& arg, const Output<Node>& pads_begin, const Output<Node>& pads_end, PadMode pad_mode);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool has_evaluate() const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
};
}  // namespace v1

namespace v12 {
/// \brief Generic padding operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Pad : public util::PadBase {
public:
    OPENVINO_OP("Pad", "opset12", op::util::PadBase);

    /// \brief Constructs a Pad-12 operation.
    Pad() = default;

    /// \brief Constructs a Pad-12 operation.
    ///
    /// \param arg The output producing input tensor to be padded.
    /// \param pads_begin The output which specifies the number of padding elements
    /// to add (or remove) before position 0 on each axis of arg.
    /// \param pads_end The output which specifies the number of padding elements
    /// to add (or remove) after the last element on each axis.
    /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
    Pad(const Output<Node>& arg, const Output<Node>& pads_begin, const Output<Node>& pads_end, PadMode pad_mode);

    /// \brief Constructs a Pad-12 operation.
    ///
    /// \param arg The output producing input tensor to be padded.
    /// \param pads_begin The output which specifies the number of padding elements
    /// to add (or remove) before position 0 on each axis of arg.
    /// \param pads_end The output which specifies the number of padding elements
    /// to add (or remove) after the last element on each axis.
    /// \param arg_pad_value The scalar output with the value used for padding
    /// if pad_mode is CONSTANT
    /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
    Pad(const Output<Node>& arg,
        const Output<Node>& pads_begin,
        const Output<Node>& pads_end,
        const Output<Node>& arg_pad_value,
        PadMode pad_mode);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool has_evaluate() const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
};
}  // namespace v12
}  // namespace op
}  // namespace ov
