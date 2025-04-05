// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v9 {
/// \brief An operation GenerateProposals
/// computes ROIs and their scores based on input data.
class OPENVINO_API GenerateProposals : public Op {
public:
    OPENVINO_OP("GenerateProposals", "opset9");

    /// \brief Structure that specifies attributes of the operation
    struct Attributes {
        // minimum box width & height
        float min_size;
        // specifies NMS threshold
        float nms_threshold;
        // number of top-n proposals before NMS
        int64_t pre_nms_count;
        // number of top-n proposals after NMS
        int64_t post_nms_count;
        // specify whether the bbox is normalized or not.
        // For example if *normalized* is true, width = x_right - x_left
        // If *normalized* is false, width = x_right - x_left + 1.
        bool normalized = true;
        // specify eta parameter for adaptive NMS in generate proposals
        float nms_eta = 1.0;
    };

    GenerateProposals() = default;
    /// \brief Constructs a GenerateProposals operation.
    ///
    /// \param im_info Input image info
    /// \param anchors Input anchors
    /// \param deltas Input deltas
    /// \param scores Input scores
    /// \param attrs Operation attributes
    /// \param roi_num_type roi_num type
    GenerateProposals(const Output<Node>& im_info,
                      const Output<Node>& anchors,
                      const Output<Node>& deltas,
                      const Output<Node>& scores,
                      const Attributes& attrs,
                      const element::Type& roi_num_type = element::i64);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }

    void set_attrs(Attributes attrs);

    const element::Type& get_roi_num_type() const {
        return m_roi_num_type;
    }
    void set_roi_num_type(const element::Type& output_type) {
        NODE_VALIDATION_CHECK(this,
                              (output_type == ov::element::i64) || (output_type == ov::element::i32),
                              "The third output type must be int64 or int32.");
        m_roi_num_type = output_type;
        set_output_type(2, output_type, get_output_partial_shape(2));
    }

private:
    Attributes m_attrs;
    ov::element::Type m_roi_num_type = ov::element::i64;
};
}  // namespace v9
}  // namespace op
}  // namespace ov
