// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/avg_pool_base.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Batched average pooling operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API AvgPool : public util::AvgPoolBase {
public:
    OPENVINO_OP("AvgPool", "opset1", util::AvgPoolBase);

    /// \brief Constructs a batched average pooling operation.
    AvgPool() = default;

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
    AvgPool(const Output<Node>& arg,
            const Strides& strides,
            const Shape& pads_begin,
            const Shape& pads_end,
            const Shape& kernel,
            bool exclude_pad,
            RoundingType rounding_type = RoundingType::FLOOR,
            const PadType& auto_pad = PadType::EXPLICIT);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v1

namespace v14 {
/// \brief Batched average pooling operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API AvgPool : public util::AvgPoolBase {
public:
    OPENVINO_OP("AvgPool", "opset14", util::AvgPoolBase);

    /// \brief Constructs a batched average pooling operation.
    AvgPool() = default;

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
    AvgPool(const Output<Node>& arg,
            const Strides& strides,
            const Shape& pads_begin,
            const Shape& pads_end,
            const Shape& kernel,
            bool exclude_pad,
            RoundingType rounding_type = RoundingType::FLOOR,
            const PadType& auto_pad = PadType::EXPLICIT);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v14
}  // namespace op
}  // namespace ov
