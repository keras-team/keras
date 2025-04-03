// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/roi_align_base.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief ROIAlignRotated operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ROIAlignRotated : public util::ROIAlignBase {
public:
    OPENVINO_OP("ROIAlignRotated", "opset15", util::ROIAlignBase);

    ROIAlignRotated() = default;
    /// \brief Constructs a ROIAlignRotated operation.
    /// \param input           Input feature map {N, C, H, W}
    /// \param rois            Regions of interest to pool over
    /// \param batch_indices   Indices of images in the batch matching
    ///                        the number or ROIs
    /// \param pooled_h        Height of the ROI output features
    /// \param pooled_w        Width of the ROI output features
    /// \param sampling_ratio  Number of sampling points used to compute
    ///                        an output element
    /// \param spatial_scale   Spatial scale factor used to translate ROI coordinates
    ///
    /// \param clockwise_mode  If true, rotation angle is interpreted as clockwise, otherwise as counterclockwise
    ROIAlignRotated(const Output<Node>& input,
                    const Output<Node>& rois,
                    const Output<Node>& batch_indices,
                    const int pooled_h,
                    const int pooled_w,
                    const int sampling_ratio,
                    const float spatial_scale,
                    const bool clockwise_mode);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int get_rois_input_second_dim_size() const override {
        return 5;
    }

    bool get_clockwise_mode() const {
        return m_clockwise_mode;
    }

    void set_clockwise_mode(const bool clockwise_mode) {
        m_clockwise_mode = clockwise_mode;
    }

private:
    bool m_clockwise_mode{};
};
}  // namespace v15
}  // namespace op
}  // namespace ov
