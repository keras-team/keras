// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API AvgPoolBase : public Op {
public:
    OPENVINO_OP("AvgPool", "util");
    AvgPoolBase() = default;

    /// \brief      Constructs a batched average pooling operation.
    ///
    /// \param      arg            The output producing the input data batch tensor.<br>
    ///                            `[d1, dn]`
    /// \param      strides        The strides.<br> `[n]`
    /// \param      pads_begin     The beginning of padding shape.<br> `[n]`
    /// \param      pads_end       The end of padding shape.<br> `[n]`
    /// \param      kernel         The kernel shape.<br> `[n]`
    /// \param      exclude_pad    If false then averages include padding elements, each
    ///                            treated as the number zero.  If true, padding
    ///                            elements
    ///                            are entirely ignored when computing averages.
    /// \param      rounding_type  Whether to use ceiling or floor rounding type while
    ///                            computing output shape.
    /// \param      auto_pad       Padding type to use for additional padded dimensions
    AvgPoolBase(const Output<Node>& arg,
                const Strides& strides,
                const Shape& pads_begin,
                const Shape& pads_end,
                const Shape& kernel,
                bool exclude_pad,
                RoundingType rounding_type = RoundingType::FLOOR,
                const PadType& auto_pad = PadType::EXPLICIT);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    /// \return The kernel shape.
    const Shape& get_kernel() const;
    void set_kernel(const Shape& kernel);

    /// \return The strides.
    const Strides& get_strides() const;
    void set_strides(const Strides& strides);

    /// \return The beginning of padding shape.
    const Shape& get_pads_begin() const;
    void set_pads_begin(const Shape& pads_begin);

    /// \return The end of padding shape.
    const Shape& get_pads_end() const;
    void set_pads_end(const Shape& pads_end);

    /// \return Exclude zero-values in padding area.
    bool get_exclude_pad() const;
    void set_exclude_pad(bool exclude_pad);

    /// \return The pad type for pooling.
    const PadType& get_auto_pad() const;
    void set_auto_pad(const PadType& auto_pad);

    /// \return The ceiling mode being used for output shape computations
    RoundingType get_rounding_type() const;
    void set_rounding_type(RoundingType rounding_type);

protected:
    Shape m_kernel;
    Strides m_strides;
    Shape m_pads_begin;
    Shape m_pads_end;
    bool m_exclude_pad{true};
    PadType m_auto_pad{PadType::EXPLICIT};
    RoundingType m_rounding_type{RoundingType::FLOOR};
};
}  // namespace util
}  // namespace op
}  // namespace ov
