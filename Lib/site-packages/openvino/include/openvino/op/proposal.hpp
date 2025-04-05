// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

namespace v0 {
/// \brief Proposal operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Proposal : public Op {
public:
    OPENVINO_OP("Proposal", "opset1");
    // base_size       Anchor sizes
    // pre_nms_topn    Number of boxes before nms
    // post_nms_topn   Number of boxes after nms
    // nms_thresh      Threshold for nms
    // feat_stride     Feature stride
    // min_size        Minimum box size
    // ratio   Ratios for anchor generation
    // scale   Scales for anchor generation
    // clip_before_nms Clip before NMs
    // clip_after_nms  Clip after NMs
    // normalize       Normalize boxes to [0,1]
    // box_size_scale  Scale factor for scaling box size
    // box_coordinate_scale Scale factor for scaling box coordiate
    // framework            Calculation frameworkrithm to use
    struct Attributes {
        size_t base_size;
        size_t pre_nms_topn;
        size_t post_nms_topn;
        float nms_thresh = 0.0f;
        size_t feat_stride = 1;
        size_t min_size = 1;
        std::vector<float> ratio;
        std::vector<float> scale;
        bool clip_before_nms = true;
        bool clip_after_nms = false;
        bool normalize = false;
        float box_size_scale = 1.0f;
        float box_coordinate_scale = 1.0f;
        std::string framework;
        bool infer_probs = false;
    };
    Proposal() = default;
    /// \brief Constructs a Proposal operation
    ///
    /// \param class_probs     Class probability scores
    /// \param bbox_deltas     Prediction of bounding box deltas
    /// \param image_shape     Shape of image
    /// \param attrs           Proposal op attributes
    Proposal(const Output<Node>& class_probs,
             const Output<Node>& bbox_deltas,
             const Output<Node>& image_shape,
             const Attributes& attrs);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    const Attributes& get_attrs() const {
        return m_attrs;
    }

    /**
     * @brief Set the Proposal operator attributes.
     * @param attrs  Attributes to be set.
     */
    void set_attrs(Attributes attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    Attributes m_attrs;
    void validate_element_types();
};
}  // namespace v0

namespace v4 {
/// \brief Proposal operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Proposal : public op::v0::Proposal {
public:
    OPENVINO_OP("Proposal", "opset4", op::v0::Proposal);
    Proposal() = default;
    /// \brief Constructs a Proposal operation
    ///
    /// \param class_probs     Class probability scores
    /// \param bbox_deltas     Prediction of bounding box deltas
    /// \param image_shape     Shape of image
    /// \param attrs           Proposal op attributes
    Proposal(const Output<Node>& class_probs,
             const Output<Node>& bbox_deltas,
             const Output<Node>& image_shape,
             const Attributes& attrs);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v4
}  // namespace op
}  // namespace ov
