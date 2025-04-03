// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/detection_output_base.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Layer which performs non-max suppression to
/// generate detection output using location and confidence predictions
class OPENVINO_API DetectionOutput : public op::util::DetectionOutputBase {
public:
    struct Attributes : public op::util::DetectionOutputBase::AttributesBase {
        int num_classes;
    };

    OPENVINO_OP("DetectionOutput", "opset1", op::util::DetectionOutputBase);

    DetectionOutput() = default;
    /// \brief Constructs a DetectionOutput operation
    ///
    /// \param box_logits			Box logits
    /// \param class_preds			Class predictions
    /// \param proposals			Proposals
    /// \param aux_class_preds		Auxilary class predictions
    /// \param aux_box_preds		Auxilary box predictions
    /// \param attrs				Detection Output attributes
    DetectionOutput(const Output<Node>& box_logits,
                    const Output<Node>& class_preds,
                    const Output<Node>& proposals,
                    const Output<Node>& aux_class_preds,
                    const Output<Node>& aux_box_preds,
                    const Attributes& attrs);

    /// \brief Constructs a DetectionOutput operation
    ///
    /// \param box_logits			Box logits
    /// \param class_preds			Class predictions
    /// \param proposals			Proposals
    /// \param attrs				Detection Output attributes
    DetectionOutput(const Output<Node>& box_logits,
                    const Output<Node>& class_preds,
                    const Output<Node>& proposals,
                    const Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }
    void set_attrs(const Attributes& attrs) {
        m_attrs = attrs;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    Attributes m_attrs;
};
}  // namespace v0

namespace v8 {
/// \brief Layer which performs non-max suppression to
/// generate detection output using location and confidence predictions
class OPENVINO_API DetectionOutput : public op::util::DetectionOutputBase {
public:
    using Attributes = op::util::DetectionOutputBase::AttributesBase;

    OPENVINO_OP("DetectionOutput", "opset8", op::util::DetectionOutputBase);

    DetectionOutput() = default;
    /// \brief Constructs a DetectionOutput operation
    ///
    /// \param box_logits			Box logits
    /// \param class_preds			Class predictions
    /// \param proposals			Proposals
    /// \param aux_class_preds		Auxilary class predictions
    /// \param aux_box_preds		Auxilary box predictions
    /// \param attrs				Detection Output attributes
    DetectionOutput(const Output<Node>& box_logits,
                    const Output<Node>& class_preds,
                    const Output<Node>& proposals,
                    const Output<Node>& aux_class_preds,
                    const Output<Node>& aux_box_preds,
                    const Attributes& attrs);

    /// \brief Constructs a DetectionOutput operation
    ///
    /// \param box_logits			Box logits
    /// \param class_preds			Class predictions
    /// \param proposals			Proposals
    /// \param attrs				Detection Output attributes
    DetectionOutput(const Output<Node>& box_logits,
                    const Output<Node>& class_preds,
                    const Output<Node>& proposals,
                    const Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }
    void set_attrs(const Attributes& attrs) {
        m_attrs = attrs;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    Attributes m_attrs;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
