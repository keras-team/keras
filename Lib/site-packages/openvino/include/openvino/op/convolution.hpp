// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/convolution_backprop_base.hpp"
#include "openvino/op/util/convolution_base.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Batched convolution operation, with optional window dilation and stride.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Convolution : public util::ConvolutionFwdPropBase {
public:
    OPENVINO_OP("Convolution", "opset1", op::util::ConvolutionFwdPropBase);

    /// \brief Constructs a batched convolution operation.
    Convolution() = default;
    /// \brief Constructs a batched convolution operation.
    ///
    /// \param data_batch The node producing the input data batch tensor.<br>
    /// `[N, C_IN, D1, ... Df]`
    /// \param filters The node producing the filters tensor.<br>
    /// `[C_OUT, C_IN, F1, ... Ff]`
    /// \param strides The strides.<br>
    /// `[f]`
    /// \param dilations The dilations.<br>
    /// `[f]`
    /// \param pads_begin The beginning of padding shape.<br>
    /// `[f]`
    /// \param pads_end The end of padding shape.<br>
    /// `[f]`
    /// \param auto_pad The pad type for automatically computing padding sizes.<br>
    /// `[f]`
    ///
    /// Output `[N, C_OUT, R1, ... Rf]`
    ///
    Convolution(const Output<Node>& data_batch,
                const Output<Node>& filters,
                const Strides& strides,
                const CoordinateDiff& pads_begin,
                const CoordinateDiff& pads_end,
                const Strides& dilations,
                const PadType& auto_pad = PadType::EXPLICIT);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

/// \brief Data batch backprop for batched convolution operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ConvolutionBackpropData : public util::ConvolutionBackPropBase {
public:
    OPENVINO_OP("ConvolutionBackpropData", "opset1", op::util::ConvolutionBackPropBase);

    /// \brief Constructs a batched-convolution data batch-backprop operation.
    ConvolutionBackpropData() = default;
    // clang-format off
    //
    // \brief      Constructs a batched-convolution data batch-backprop operation.
    //
    // \param      data            The node producing data from forward-prop. Shape: [N,
    //                             C_INPUT, X1, ..., XD].
    // \param      filters         The node producing the filter from forward-prop. Shape:
    //                             [C_INPUT, C_OUTPUT, K_D, ..., K_1]
    // \param      output_shape    The shape of the data batch from forward-prop. It's size
    //                             should be equal to number of data spatial dimensions.
    // \param      strides         The strides from forward-prop.
    // \param      pads_begin      The padding-below sizes from forward-prop.
    // \param      pads_end        The padding-above sizes from forward-prop.
    // \param      dilations       The dilations from forward-prop.
    // \param      auto_pad        The pad type for automatically computing padding sizes.
    // \param      output_padding  The output padding adds additional amount of paddings per
    //                             each spatial axis in the output tensor. clang-format on
    //
    // clang-format on
    ConvolutionBackpropData(const Output<Node>& data,
                            const Output<Node>& filters,
                            const Output<Node>& output_shape,
                            const Strides& strides,
                            const CoordinateDiff& pads_begin,
                            const CoordinateDiff& pads_end,
                            const Strides& dilations,
                            const PadType& auto_pad = PadType::EXPLICIT,
                            const CoordinateDiff& output_padding = {});

    // clang-format off
    //
    // \brief      Constructs a batched-convolution data batch-backprop operation.
    //
    // \param      data            The node producing data from forward-prop. Shape: [N,
    //                             C_INPUT, X1, ..., XD].
    // \param      filters         The node producing the filter from forward-prop. Shape:
    //                             [C_INPUT, C_OUTPUT, K_D, ..., K_1]
    // \param      strides         The strides from forward-prop.
    // \param      pads_begin      The padding-below sizes from forward-prop.
    // \param      pads_end        The padding-above sizes from forward-prop.
    // \param      dilations       The dilations from forward-prop.
    // \param      auto_pad        The pad type for automatically computing padding sizes.
    // \param      output_padding  The output padding adds additional amount of paddings per
    //                             each spatial axis in the output tensor. clang-format on
    //
    // clang-format on
    ConvolutionBackpropData(const Output<Node>& data,
                            const Output<Node>& filters,
                            const Strides& strides,
                            const CoordinateDiff& pads_begin,
                            const CoordinateDiff& pads_end,
                            const Strides& dilations,
                            const PadType& auto_pad = PadType::EXPLICIT,
                            const CoordinateDiff& output_padding = {});

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    bool is_dynamic() const override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The output spatial dimensions shape.
    const PartialShape get_output_shape() const;
    void set_output_shape(const Shape& output_shape);
};
}  // namespace v1
}  // namespace op
}  // namespace ov
