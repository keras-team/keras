// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief ROIPooling operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ROIPooling : public Op {
public:
    OPENVINO_OP("ROIPooling", "opset2");

    ROIPooling() = default;
    /// \brief Constructs a ROIPooling operation
    ///
    /// \param input          Input feature map {N, C, H, W}
    /// \param coords         Coordinates of bounding boxes
    /// \param output_size    Height/Width of ROI output features
    /// \param spatial_scale  Ratio of input feature map over input image size
    /// \param method         Method of pooling - Max or Bilinear
    ROIPooling(const Output<Node>& input,
               const Output<Node>& coords,
               const Shape& output_size,
               const float spatial_scale,
               const std::string& method = "max");

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \brief Set the output ROI feature map (pooled_h, pooled_w).
    /// \param output_size Shape with pooling attributes pooled_h and pooled_w sizes.
    void set_output_roi(Shape output_size);

    /// \brief Get the output ROI feature map shape (H x W)
    /// \return Shape with pooled_h and pooled_w attributes.
    const Shape& get_output_roi() const;

    /// \brief Set the spatial scale value.
    /// \param scale Scale value to set.
    void set_spatial_scale(float scale);
    float get_spatial_scale() const {
        return m_spatial_scale;
    }

    /// \brief Set the method of pooling
    /// \param method_name Pooling method name.
    void set_method(std::string method_name);
    const std::string& get_method() const {
        return m_method;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    Shape m_output_size{0, 0};
    float m_spatial_scale{0.0f};
    std::string m_method = "max";
};
}  // namespace v0
}  // namespace op
}  // namespace ov
