// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v6 {
/// \brief An operation ExperimentalDetectronGenerateProposalsSingleImage
/// computes ROIs and their scores based on input data.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ExperimentalDetectronGenerateProposalsSingleImage : public Op {
public:
    OPENVINO_OP("ExperimentalDetectronGenerateProposalsSingleImage", "opset6", op::Op);

    /// \brief Structure that specifies attributes of the operation
    struct Attributes {
        // minimum box width & height
        float min_size;
        // specifies NMS threshold
        float nms_threshold;
        // number of top-n proposals after NMS
        int64_t post_nms_count;
        // number of top-n proposals before NMS
        int64_t pre_nms_count;
    };

    ExperimentalDetectronGenerateProposalsSingleImage() = default;
    /// \brief Constructs a ExperimentalDetectronGenerateProposalsSingleImage operation.
    ///
    /// \param im_info Input image info
    /// \param anchors Input anchors
    /// \param deltas Input deltas
    /// \param scores Input scores
    /// \param attrs Operation attributes
    ExperimentalDetectronGenerateProposalsSingleImage(const Output<Node>& im_info,
                                                      const Output<Node>& anchors,
                                                      const Output<Node>& deltas,
                                                      const Output<Node>& scores,
                                                      const Attributes& attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }

    void set_attrs(Attributes attrs);

private:
    Attributes m_attrs;
};
}  // namespace v6
}  // namespace op
}  // namespace ov
