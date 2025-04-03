// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for color conversion operation from NV12 to RGB/BGR format.
///    Input:
///        - Operation expects input shape in NHWC layout.
///        - Input NV12 image can be represented in a two ways:
///            a) Single plane: NV12 height dimension is 1.5x bigger than image height. 'C' dimension shall be 1
///            b) Two separate planes: Y and UV. In this case
///               b1) Y plane has height same as image height. 'C' dimension equals to 1
///               b2) UV plane has dimensions: 'H' = image_h / 2; 'W' = image_w / 2; 'C' = 2.
///        - Supported element types: u8 or any supported floating-point type.
///    Output:
///        - Output node will have NHWC layout and shape HxW same as image spatial dimensions.
///        - Number of output channels 'C' will be 3
///
/// \details Conversion of each pixel from NV12 (YUV) to RGB space is represented by following formulas:
///        R = 1.164 * (Y - 16) + 1.596 * (V - 128)
///        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
///        B = 1.164 * (Y - 16) + 2.018 * (U - 128)
///        Then R, G, B values are clipped to range (0, 255)
///
class OPENVINO_API ConvertColorNV12Base : public Op {
public:
    /// \brief Exact conversion format details
    /// Currently supports conversion from NV12 to RGB or BGR, in future can be extended with NV21_to_RGBA/BGRA, etc
    enum class ColorConversion : int { NV12_TO_RGB = 0, NV12_TO_BGR = 1 };

protected:
    ConvertColorNV12Base() = default;

    /// \brief Constructs a conversion operation from input image in NV12 format
    /// As per NV12 format definition, node height dimension shall be 1.5 times bigger than image height
    /// so that image (w=640, h=480) is represented by NHWC shape {N,720,640,1} (height*1.5 x width)
    ///
    /// \param arg          Node that produces the input tensor. Input tensor represents image in NV12 format (YUV).
    /// \param format       Conversion format.
    explicit ConvertColorNV12Base(const Output<Node>& arg, ColorConversion format);

    /// \brief Constructs a conversion operation from 2-plane input image in NV12 format
    /// In general case Y channel of image can be separated from UV channel which means that operation needs two nodes
    /// for Y and UV planes respectively. Y plane has one channel, and UV has 2 channels, both expect 'NHWC' layout
    ///
    /// \param arg_y        Node that produces the input tensor for Y plane (NHWC layout). Shall have WxH dimensions
    /// equal to image dimensions. 'C' dimension equals to 1.
    ///
    /// \param arg_uv       Node that produces the input tensor for UV plane (NHWC layout). 'H' is half of image height,
    /// 'W' is half of image width, 'C' dimension equals to 2. Channel 0 represents 'U', channel 1 represents 'V'
    /// channel
    ///
    /// \param format       Conversion format.
    ConvertColorNV12Base(const Output<Node>& arg_y, const Output<Node>& arg_uv, ColorConversion format);

public:
    OPENVINO_OP("ConvertColorNV12Base", "util");

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    bool is_type_supported(const ov::element::Type& type) const;

    ColorConversion m_format = ColorConversion::NV12_TO_RGB;
};
}  // namespace util
}  // namespace op
}  // namespace ov
