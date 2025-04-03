// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API BroadcastBase : public Op {
protected:
    BroadcastBase() = default;
    /// \brief Constructs a broadcast operation.
    ///
    /// \param arg            The input tensor to be broadcast.
    /// \param target_shape   The shape of the output tensor.
    /// \param axes_mapping   The axis positions (0-based) in the result that correspond
    ///                       to input axes.
    /// \param broadcast_mode Broadcast specification to use for determining broadcast
    ///                       axes. 'axes_mapping' should not be provided if mode other
    ///
    BroadcastBase(const Output<Node>& arg,
                  const Output<Node>& target_shape,
                  const Output<Node>& axes_mapping,
                  const BroadcastModeSpec& broadcast_mode = BroadcastType::EXPLICIT);

    /// \brief Constructs a broadcast operation.
    ///
    /// \param arg            The input tensor to be broadcast.
    /// \param target_shape   The shape of the output tensor.
    /// \param broadcast_mode Broadcast specification to use for determining broadcast
    ///                       axes
    BroadcastBase(const Output<Node>& arg,
                  const Output<Node>& target_shape,
                  const BroadcastModeSpec& broadcast_mode = BroadcastType::NUMPY);

public:
    OPENVINO_OP("BroadcastBase", "util");

    void validate_and_infer_types() override;
    /// \return true and the AxisSet if broadcast axes can be fully determined.
    virtual std::pair<bool, AxisSet> get_broadcast_axes() const;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    const BroadcastModeSpec& get_broadcast_spec() const {
        return m_mode;
    }

protected:
    BroadcastModeSpec m_mode;

    bool evaluate_broadcast(const ov::Tensor& arg0,
                            ov::Tensor& out,
                            const std::pair<bool, AxisSet>& pair_broadcast_axes,
                            const Shape& output_shape) const;

    bool evaluate_broadcast(const ov::Tensor& arg0, ov::Tensor& out, const AxisSet& broadcast_axes) const;

    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(ov::TensorSymbolVector& output_symbols) const override;

    PartialShape get_result_shape_pdpd(const PartialShape& arg0_shape,
                                       const PartialShape& target_shape,
                                       const op::BroadcastModeSpec& broadcast_spec) const;

    void validate_target_shape_numpy(const PartialShape& arg_shape, const PartialShape& target_shape) const;

    static std::pair<bool, AxisSet> get_broadcast_axes_numpy_pdpd(const Shape& arg_shape,
                                                                  const Shape& result_shape,
                                                                  const op::BroadcastModeSpec& broadcast_spec);

    static std::pair<bool, AxisSet> get_broadcast_axes_none(const AxisVector& axes_mapping_val,
                                                            const size_t target_shape);

    void validate_target_shape_none(const PartialShape& arg_shape,
                                    const AxisVector& axes_mapping_val,
                                    const PartialShape& target_shape) const;

    Shape get_target_shape(const ov::Tensor& input1) const;
};
}  // namespace util
}  // namespace op
}  // namespace ov
