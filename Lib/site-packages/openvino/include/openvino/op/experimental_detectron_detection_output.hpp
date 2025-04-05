// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v6 {
/// \brief An operation ExperimentalDetectronDetectionOutput performs
/// non-maximum suppression to generate the detection output using
/// information on location and score predictions.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ExperimentalDetectronDetectionOutput : public Op {
public:
    OPENVINO_OP("ExperimentalDetectronDetectionOutput", "opset6", op::Op);

    /// \brief Structure that specifies attributes of the operation
    struct Attributes {
        // specifies score threshold
        float score_threshold;
        // specifies NMS threshold
        float nms_threshold;
        // specifies maximal delta of logarithms for width and height
        float max_delta_log_wh;
        // specifies number of detected classes
        int64_t num_classes;
        // specifies maximal number of detections per class
        int64_t post_nms_count;
        // specifies maximual number of detections per image
        size_t max_detections_per_image;
        // a flag specifies whether to delete background classes or not
        // `true`  means background classes should be deleted,
        // `false` means background classes shouldn't be deleted.
        bool class_agnostic_box_regression;
        // specifies deltas of weights
        std::vector<float> deltas_weights;
    };

    ExperimentalDetectronDetectionOutput() = default;
    /// \brief Constructs a ExperimentalDetectronDetectionOutput operation.
    ///
    /// \param input_rois  Input rois
    /// \param input_deltas Input deltas
    /// \param input_scores Input scores
    /// \param input_im_info Input image info
    /// \param attrs  Attributes attributes
    ExperimentalDetectronDetectionOutput(const Output<Node>& input_rois,
                                         const Output<Node>& input_deltas,
                                         const Output<Node>& input_scores,
                                         const Output<Node>& input_im_info,
                                         const Attributes& attrs);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    /// \brief Returns attributes of the operation ExperimentalDetectronDetectionOutput
    const Attributes& get_attrs() const {
        return m_attrs;
    }

    /// \brief Set the attributes of the operation ExperimentalDetectronDetectionOutput.
    /// \param attrs  Attributes to set.
    void set_attrs(Attributes attrs);

private:
    Attributes m_attrs;
};
}  // namespace v6
}  // namespace op
}  // namespace ov
