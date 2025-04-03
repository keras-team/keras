// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

namespace v13 {
/// \brief NMSRotated operation
///
class OPENVINO_API NMSRotated : public Op {
public:
    OPENVINO_OP("NMSRotated", "opset13", op::Op);

    NMSRotated() = default;

    /// \brief Constructs a NMSRotated operation.
    ///
    /// \param boxes Node containing the coordinates of the bounding boxes
    /// \param scores Node containing the scores of the bounding boxes
    /// \param max_output_boxes_per_class Node containing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node containing intersection over union threshold
    /// \param score_threshold Node containing minimum score threshold
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output type of the first and third output
    /// \param clockwise Specifies the direction of the rotation
    NMSRotated(const Output<Node>& boxes,
               const Output<Node>& scores,
               const Output<Node>& max_output_boxes_per_class,
               const Output<Node>& iou_threshold,
               const Output<Node>& score_threshold,
               const bool sort_result_descending = true,
               const ov::element::Type& output_type = ov::element::i64,
               const bool clockwise = true);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_sort_result_descending() const;
    void set_sort_result_descending(const bool sort_result_descending);

    element::Type get_output_type_attr() const;
    void set_output_type_attr(const element::Type& output_type);

    bool get_clockwise() const;
    void set_clockwise(const bool clockwise);

protected:
    bool m_sort_result_descending = true;
    ov::element::Type m_output_type = ov::element::i64;
    bool m_clockwise = true;
};
}  // namespace v13
}  // namespace op

}  // namespace ov
