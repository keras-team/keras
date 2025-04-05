// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief DetectionOutputBase basic class for DetectionOutput v0 and v8
class OPENVINO_API DetectionOutputBase : public Op {
public:
    struct AttributesBase {
        int background_label_id = 0;
        int top_k = -1;
        bool variance_encoded_in_target = false;
        std::vector<int> keep_top_k;
        std::string code_type = std::string{"caffe.PriorBoxParameter.CORNER"};
        bool share_location = true;
        float nms_threshold;
        float confidence_threshold = 0;
        bool clip_after_nms = false;
        bool clip_before_nms = false;
        bool decrease_label_id = false;
        bool normalized = false;
        size_t input_height = 1;
        size_t input_width = 1;
        float objectness_score = 0;
    };

    OPENVINO_OP("DetectionOutputBase", "util");
    DetectionOutputBase() = default;
    DetectionOutputBase(const OutputVector& args);

    void validate_base(const AttributesBase& attrs);

    bool visit_attributes_base(AttributeVisitor& visitor, AttributesBase& attrs);
    Dimension compute_num_classes(const AttributesBase& attrs);
};
}  // namespace util
}  // namespace op
}  // namespace ov
