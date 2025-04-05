// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API MaxPoolBase : public Op {
public:
    OPENVINO_OP("MaxPoolBase", "util");
    MaxPoolBase() = default;

    /// \param arg The node producing the input data batch tensor.
    /// \param strides The strides.
    /// \param pads_begin The beginning of padding shape.
    /// \param pads_end The end of padding shape.
    /// \param kernel The kernel shape.
    /// \param rounding_mode Whether to use ceiling or floor rounding type while
    /// computing output shape.
    /// \param auto_pad The pad type for automatically computing padding sizes.
    MaxPoolBase(const Output<Node>& arg,
                const Strides& strides,
                const Shape& pads_begin,
                const Shape& pads_end,
                const Shape& kernel,
                const op::RoundingType rounding_mode = op::RoundingType::FLOOR,
                const PadType auto_pad = op::PadType::EXPLICIT);

    void validate_and_infer_types() override;

    /// \return The kernel shape.
    const Shape& get_kernel() const {
        return m_kernel;
    }
    void set_kernel(const Shape& kernel) {
        m_kernel = kernel;
    }
    /// \return The strides.
    const Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const Strides& strides) {
        m_strides = strides;
    }
    /// \return The beginning of padding shape.
    const Shape& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const Shape& pads_begin) {
        m_pads_begin = pads_begin;
    }
    /// \return The end of padding shape.
    const Shape& get_pads_end() const {
        return m_pads_end;
    }
    void set_pads_end(Shape pads_end);

    /// \return The pad type for pooling.
    PadType get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const PadType auto_pad) {
        m_auto_pad = auto_pad;
    }
    /// \return The ceiling mode being used for output shape computations
    op::RoundingType get_rounding_type() const {
        return m_rounding_type;
    }
    void set_rounding_type(op::RoundingType rounding_type) {
        m_rounding_type = rounding_type;
    }

protected:
    Shape m_kernel;
    Strides m_strides;
    Shape m_pads_begin;
    Shape m_pads_end;
    PadType m_auto_pad;
    op::RoundingType m_rounding_type;
};
}  // namespace util
}  // namespace op
}  // namespace ov
