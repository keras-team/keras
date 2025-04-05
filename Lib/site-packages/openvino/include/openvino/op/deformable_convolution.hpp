// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/deformable_convolution_base.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief DeformableConvolution operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API DeformableConvolution : public op::util::DeformableConvolutionBase {
public:
    OPENVINO_OP("DeformableConvolution", "opset1", op::util::DeformableConvolutionBase);

    /// \brief Constructs a conversion operation.
    DeformableConvolution() = default;
    /// \brief Constructs a conversion operation.
    ///
    /// \param arg                Node that produces the input tensor.
    /// \param offsets            Node producing the deformable values tensor.
    /// \param filters            Node producing the filters(kernels) tensor with OIZYX
    ///                           layout.
    /// \param strides            Convolution strides.
    /// \param pads_begin         Amount of padding to be added to the beginning along
    ///                           each axis. For example in case of a 2D input the value
    ///                           of (1, 2) means that 1 element will be added to the
    ///                           top and 2 elements to the left.
    /// \param pads_end           Amount of padding to be added to the end along each
    ///                           axis.
    /// \param dilations          The distance in width and height between the weights
    ///                           in the filters tensor.
    /// \param auto_pad           Specifies how the automatic calculation of padding
    ///                           should be done.
    /// \param group              The number of groups which both output and input
    ///                           should be split into.
    /// \param deformable_group   The number of groups which deformable values and
    ///                           output should be split into along the channel axis.
    DeformableConvolution(const Output<Node>& arg,
                          const Output<Node>& offsets,
                          const Output<Node>& filters,
                          const Strides& strides,
                          const CoordinateDiff& pads_begin,
                          const CoordinateDiff& pads_end,
                          const Strides& dilations,
                          const PadType& auto_pad = PadType::EXPLICIT,
                          const int64_t group = 1,
                          const int64_t deformable_group = 1);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};
}  // namespace v1

namespace v8 {
/// \brief DeformableConvolution operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API DeformableConvolution : public op::util::DeformableConvolutionBase {
public:
    OPENVINO_OP("DeformableConvolution", "opset8", op::util::DeformableConvolutionBase);

    /// \brief Constructs a conversion operation.
    DeformableConvolution() = default;
    /// \brief Constructs a conversion operation.
    ///
    /// \param arg                Node that produces the input tensor.
    /// \param offsets            Node producing the deformable values tensor.
    /// \param filters            Node producing the filters(kernels) tensor with OIZYX
    ///                           layout.
    /// \param strides            Convolution strides.
    /// \param pads_begin         Amount of padding to be added to the beginning along
    ///                           each axis. For example in case of a 2D input the value
    ///                           of (1, 2) means that 1 element will be added to the
    ///                           top and 2 elements to the left.
    /// \param pads_end           Amount of padding to be added to the end along each
    ///                           axis.
    /// \param dilations          The distance in width and height between the weights
    ///                           in the filters tensor.
    /// \param auto_pad           Specifies how the automatic calculation of padding
    ///                           should be done.
    /// \param group              The number of groups which both output and input
    ///                           should be split into.
    /// \param deformable_group   The number of groups which deformable values and
    ///                           output should be split into along the channel axis.
    /// \param bilinear_interpolation_pad
    ///                           The flag that determines the mode of bilinear
    ///                           interpolation execution.
    ///                           If the flag is `true` and the sampling location is
    ///                           within one pixel outside of the feature map boundary,
    ///                           then bilinear interpolation is performed on the zero
    ///                           padded feature map. If the flag is `false` and the
    ///                           sampling location is within one pixel outside of the
    ///                           feature map boundary, then the sampling location
    ///                           shifts to the inner boundary of the feature map.`
    DeformableConvolution(const Output<Node>& arg,
                          const Output<Node>& offsets,
                          const Output<Node>& filters,
                          const Strides& strides,
                          const CoordinateDiff& pads_begin,
                          const CoordinateDiff& pads_end,
                          const Strides& dilations,
                          const PadType& auto_pad = PadType::EXPLICIT,
                          const int64_t group = 1,
                          const int64_t deformable_group = 1,
                          const bool bilinear_interpolation_pad = false);

    /// \brief Constructs a conversion operation.
    ///
    /// \param arg                Node that produces the input tensor.
    /// \param offsets            Node producing the deformable values tensor.
    /// \param filters            Node producing the filters(kernels) tensor with OIZYX
    ///                           layout.
    /// \param mask               Node producing the mask(mask) tensor.
    /// \param strides            Convolution strides.
    /// \param pads_begin         Amount of padding to be added to the beginning along
    ///                           each axis. For example in case of a 2D input the value
    ///                           of (1, 2) means that 1 element will be added to the
    ///                           top and 2 elements to the left.
    /// \param pads_end           Amount of padding to be added to the end along each
    ///                           axis.
    /// \param dilations          The distance in width and height between the weights
    ///                           in the filters tensor.
    /// \param auto_pad           Specifies how the automatic calculation of padding
    ///                           should be done.
    /// \param group              The number of groups which both output and input
    ///                           should be split into.
    /// \param deformable_group   The number of groups which deformable values and
    ///                           output should be split into along the channel axis.
    /// \param bilinear_interpolation_pad
    ///                           The flag that determines the mode of bilinear
    ///                           interpolation execution.
    ///                           If the flag is `true` and the sampling location is
    ///                           within one pixel outside of the feature map boundary,
    ///                           then bilinear interpolation is performed on the zero
    ///                           padded feature map. If the flag is `false` and the
    ///                           sampling location is within one pixel outside of the
    ///                           feature map boundary, then the sampling location
    ///                           shifts to the inner boundary of the feature map.
    DeformableConvolution(const Output<Node>& arg,
                          const Output<Node>& offsets,
                          const Output<Node>& filters,
                          const Output<Node>& mask,
                          const Strides& strides,
                          const CoordinateDiff& pads_begin,
                          const CoordinateDiff& pads_end,
                          const Strides& dilations,
                          const PadType& auto_pad = PadType::EXPLICIT,
                          const int64_t group = 1,
                          const int64_t deformable_group = 1,
                          const bool bilinear_interpolation_pad = false);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_bilinear_interpolation_pad() const {
        return m_bilinear_interpolation_pad;
    }

    void set_bilinear_interpolation_pad(const bool bilinear_interpolation_pad) {
        m_bilinear_interpolation_pad = bilinear_interpolation_pad;
    }

private:
    bool m_bilinear_interpolation_pad{false};
};
}  // namespace v8
}  // namespace op
}  // namespace ov
