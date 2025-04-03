// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/convolution.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/convolution_base.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Batched convolution operation, with optional window dilation and stride.
class OPENVINO_API GroupConvolution : public util::ConvolutionFwdPropBase {
public:
    OPENVINO_OP("GroupConvolution", "opset1", op::util::ConvolutionFwdPropBase);

    /// \brief Constructs a batched convolution operation.
    GroupConvolution() = default;
    /// \brief Constructs a batched convolution operation.
    ///
    /// \param data_batch The node producing the input data batch tensor.<br>
    /// `[N, C_IN, D1, ... Df]`
    /// \param filters The node producing the filters tensor.<br>
    /// `[GROUPS, FC_OUT, FC_IN, F1, ... Ff]`
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
    /// Output `[N, FC_OUT * GROUPS, R1, ... Rf]`
    ///
    GroupConvolution(const Output<Node>& data_batch,
                     const Output<Node>& filters,
                     const Strides& strides,
                     const CoordinateDiff& pads_begin,
                     const CoordinateDiff& pads_end,
                     const Strides& dilations,
                     const PadType& auto_pad = PadType::EXPLICIT);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

/// \brief Data batch backprop for batched convolution operation.
class OPENVINO_API GroupConvolutionBackpropData : public util::ConvolutionBackPropBase {
public:
    OPENVINO_OP("GroupConvolutionBackpropData", "opset1", op::util::ConvolutionBackPropBase);

    /// \brief Constructs a batched-convolution data batch-backprop operation.
    GroupConvolutionBackpropData();
    // clang-format off
                //
                // \brief      Constructs a batched-convolution data batch-backprop operation.
                //
                // \param      data            The node producing data from forward-prop. Shape: [N,
                //                             C_INPUT * GROUPS, X1, ..., XD].
                // \param      filter          The node producing the filter from forward-prop. Shape:
                //                             [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      output_shape    The shape of the data batch from forward-prop. It's size
                //                             should be equal to number of data spatial dimensions.
                // \param      strides         The strides from forward-prop.
                // \param      pads_begin      The padding-below sizes from forward-prop.
                // \param      pads_end        The padding-above sizes from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor.
                //
    // clang-format on
    //
    GroupConvolutionBackpropData(const Output<Node>& data,
                                 const Output<Node>& filter,
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
                //                             C_INPUT * GROUPS, X1, ..., XD].
                // \param      filter          The node producing the filter from forward-prop. Shape:
                //                             [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      output_shape    The shape of the data batch from forward-prop. It's size
                //                             should be equal to number of data spatial dimensions.
                // \param      strides         The strides from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor.
                //
    // clang-format on
    //
    GroupConvolutionBackpropData(const Output<Node>& data,
                                 const Output<Node>& filter,
                                 const Output<Node>& output_shape,
                                 const Strides& strides,
                                 const Strides& dilations,
                                 const PadType& auto_pad,
                                 const CoordinateDiff& output_padding = {});

    // clang-format off
                //
                // \brief      Constructs a batched-convolution data batch-backprop operation.
                //
                // \param      data            The node producing data from forward-prop. Shape:
                //                             [N, C_INPUT * GROUPS, X1, ..., XD].
                // \param      filter          The node producing the filter from forward-prop. Shape:
                //                             [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
                // \param      strides         The strides from forward-prop.
                // \param      pads_begin      The padding-below sizes from forward-prop.
                // \param      pads_end        The padding-above sizes from forward-prop.
                // \param      dilations       The dilations from forward-prop.
                // \param      auto_pad        The pad type for automatically computing padding sizes.
                // \param      output_padding  The output padding adds additional amount of paddings per
                //                             each spatial axis in the output tensor.
                //
    // clang-format on
    GroupConvolutionBackpropData(const Output<Node>& data,
                                 const Output<Node>& filter,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const PadType& auto_pad = PadType::EXPLICIT,
                                 const CoordinateDiff& output_padding = {});
    ///
    /// \brief      Calculates output spatial features size.
    ///
    /// \param[in]  input_data_shape      The input data partial shape
    /// \param[in]  filters_shape         The filters partial shape
    /// \param[in]  strides               The strides values.
    /// \param[in]  dilations             The dilations values.
    /// \param[in]  pads_begin            The paddings at the beginning of axis.
    /// \param[in]  pads_end              The paddings at the end of axis.
    /// \param[in]  output_padding    The output padding values.
    /// \param      output_spatial_shape  The placeholder for computed output spatial
    /// partial
    /// shape.
    ///
    void infer_conv_backprop_output_spatial_shape(const std::vector<Dimension>& input_data_shape,
                                                  const std::vector<Dimension>& filters_shape,
                                                  const Strides& strides,
                                                  const Strides& dilations,
                                                  const CoordinateDiff& pads_begin,
                                                  const CoordinateDiff& pads_end,
                                                  const CoordinateDiff& output_padding,
                                                  std::vector<Dimension>& output_spatial_shape);

    bool visit_attributes(AttributeVisitor& visitor) override;
    bool is_dynamic() const override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The spatial shape of the output.
    const PartialShape get_convolution_output_shape() const;
    void set_output_shape(const Shape& output_shape);
};
}  // namespace v1
}  // namespace op
}  // namespace ov
