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
/// \brief An operation ExperimentalDetectronTopKROIs, according to the repository
/// is TopK operation applied to probabilities of input ROIs.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ExperimentalDetectronTopKROIs : public Op {
public:
    OPENVINO_OP("ExperimentalDetectronTopKROIs", "opset6", op::Op);

    ExperimentalDetectronTopKROIs() = default;
    /// \brief Constructs a ExperimentalDetectronTopKROIs operation.
    ///
    /// \param input_rois  Input rois
    /// \param rois_probs Probabilities for input rois
    /// \param max_rois Maximal numbers of output rois
    ExperimentalDetectronTopKROIs(const Output<Node>& input_rois, const Output<Node>& rois_probs, size_t max_rois = 0);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void set_max_rois(size_t max_rois);

    size_t get_max_rois() const {
        return m_max_rois;
    }

private:
    size_t m_max_rois{0};
};
}  // namespace v6
}  // namespace op
}  // namespace ov
