// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {

///
/// \brief Base class for ROIAlignXXX operators.
///
class OPENVINO_API ROIAlignBase : public Op {
public:
    OPENVINO_OP("ROIAlignBase", "util");

protected:
    ROIAlignBase() = default;
    /// \brief Constructs a ROIAlignBase operation.
    ///
    /// \param input           Input feature map {N, C, H, W}
    /// \param rois            Regions of interest to pool over
    /// \param batch_indices   Indices of images in the batch matching
    ///                        the number or ROIs
    /// \param pooled_h        Height of the ROI output features
    /// \param pooled_w        Width of the ROI output features
    /// \param sampling_ratio  Number of sampling points used to compute
    ///                        an output element
    /// \param spatial_scale   Spatial scale factor used to translate ROI coordinates
    ROIAlignBase(const Output<Node>& input,
                 const Output<Node>& rois,
                 const Output<Node>& batch_indices,
                 const int pooled_h,
                 const int pooled_w,
                 const int sampling_ratio,
                 const float spatial_scale);

public:
    // Return size of second dimension of the ROI input.
    // Needed for validation check.
    virtual int get_rois_input_second_dim_size() const = 0;

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    int get_pooled_h() const {
        return m_pooled_h;
    }

    void set_pooled_h(const int h) {
        m_pooled_h = h;
    }

    int get_pooled_w() const {
        return m_pooled_w;
    }

    void set_pooled_w(const int w) {
        m_pooled_w = w;
    }

    int get_sampling_ratio() const {
        return m_sampling_ratio;
    }

    void set_sampling_ratio(const int ratio) {
        m_sampling_ratio = ratio;
    }

    float get_spatial_scale() const {
        return m_spatial_scale;
    }

    void set_spatial_scale(const float scale) {
        m_spatial_scale = scale;
    }

private:
    int m_pooled_h;
    int m_pooled_w;
    int m_sampling_ratio;
    float m_spatial_scale;
};

}  // namespace util
}  // namespace op
}  // namespace ov
