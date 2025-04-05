// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API PadBase : public Op {
public:
    OPENVINO_OP("PadBase", "util");

    PadBase() = default;

    /// \brief Constructs a generic padding operation.
    ///
    /// \param arg The output producing input tensor to be padded.
    /// \param pads_begin The output which specifies the number of padding elements
    /// added
    /// before position 0 on each axis of arg.
    /// \param pads_end The output which specifies the number of padding elements
    /// after the last element on each axis.
    /// \param arg_pad_value The scalar output with the value used for padding
    /// if pad_mode is CONSTANT
    /// \param pad_mode The padding mode
    PadBase(const Output<Node>& arg,
            const Output<Node>& pads_begin,
            const Output<Node>& pads_end,
            const Output<Node>& arg_pad_value,
            PadMode pad_mode);

    /// \brief Constructs a generic padding operation.
    ///
    /// \param arg The output producing input tensor to be padded.
    /// \param pads_begin The output which specifies the number of padding elements
    /// added
    /// \param pads_end The output which specifies the number of padding elements
    /// after the last element on each axis.
    /// \param pad_mode The padding mode
    PadBase(const Output<Node>& arg, const Output<Node>& pads_begin, const Output<Node>& pads_end, PadMode pad_mode);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    /// return The node which specifies the number of padding elements
    /// added at the beginning of each axis
    CoordinateDiff get_pads_begin() const;
    /// return The node which specifies the number of padding elements
    /// added at the end of each axis
    CoordinateDiff get_pads_end() const;

    /// \return The padding mode.
    PadMode get_pad_mode() const {
        return m_pad_mode;
    }
    void set_pad_mode(PadMode pad_mode) {
        m_pad_mode = pad_mode;
    }

    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;

protected:
    PadMode m_pad_mode{PadMode::CONSTANT};
    bool evaluate_pad(TensorVector& outputs, const TensorVector& inputs) const;
};
}  // namespace util
}  // namespace op
}  // namespace ov
