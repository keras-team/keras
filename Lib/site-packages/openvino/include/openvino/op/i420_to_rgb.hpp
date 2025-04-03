// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/convert_color_i420_base.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief Color conversion operation from I420 to RGB format.
///    Input:
///        - Input NV12 image can be represented in two ways:
///            a) Single plane (as it is in the file): I420 height dimension is 1.5x bigger than image height. 'C'
///               dimension shall be 1.
///            b) Three separate planes (used this way in many physical video sources): Y, U and V. In
///               this case
///               b1) Y plane has height same as image height. 'C' dimension equals to 1
///               b2) U plane has dimensions: 'H' = image_h / 2; 'W' = image_w / 2; 'C' = 1.
///               b3) V plane has dimensions: 'H' = image_h / 2; 'W' = image_w / 2; 'C' = 1.
///        - Supported element types: u8 or any supported floating-point type.
///    Output:
///        - Output node will have NHWC layout and shape HxW same as image spatial dimensions.
///        - Number of output channels 'C' will be 3, as per interleaved RGB format, first channel is R, last is B
///
/// \details Conversion of each pixel from I420 (YUV) to RGB space is represented by following formulas:
///        R = 1.164 * (Y - 16) + 1.596 * (V - 128)
///        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
///        B = 1.164 * (Y - 16) + 2.018 * (U - 128)
///        Then R, G, B values are clipped to range (0, 255)
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API I420toRGB : public util::ConvertColorI420Base {
public:
    OPENVINO_OP("I420toRGB", "opset8", util::ConvertColorI420Base);

    I420toRGB() = default;

    /// \brief Constructs a conversion operation from input image in I420 format
    /// As per I420 format definition, node height dimension shall be 1.5 times bigger than image height
    /// so that image (w=640, h=480) is represented by NHWC shape {N,720,640,1} (height*1.5 x width)
    ///
    /// \param arg          Node that produces the input tensor. Input tensor represents image in NV12 format (YUV).
    explicit I420toRGB(const Output<Node>& arg);

    /// \brief Constructs a conversion operation from 2-plane input image in NV12 format
    /// In general case Y channel of image can be separated from UV channel which means that operation needs two nodes
    /// for Y and UV planes respectively. Y plane has one channel, and UV has 2 channels, both expect 'NHWC' layout
    ///
    /// \param arg_y       Node that produces the input tensor for Y plane (NHWC layout). Shall have WxH dimensions
    /// equal to image dimensions. 'C' dimension equals to 1.
    ///
    /// \param arg_u       Node that produces the input tensor for U plane (NHWC layout). 'H' is half of image height,
    /// 'W' is half of image width, 'C' dimension equals to 1.
    ///
    /// \param arg_v       Node that produces the input tensor for V plane (NHWC layout). 'H' is half of image height,
    /// 'W' is half of image width, 'C' dimension equals to 1.
    ///
    explicit I420toRGB(const Output<Node>& arg_y, const Output<Node>& arg_u, const Output<Node>& arg_v);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
