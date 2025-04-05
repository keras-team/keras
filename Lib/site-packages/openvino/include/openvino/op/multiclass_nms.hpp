// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/multiclass_nms_base.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief MulticlassNms operation
///
class OPENVINO_API MulticlassNms : public util::MulticlassNmsBase {
public:
    OPENVINO_OP("MulticlassNms", "opset8", op::util::MulticlassNmsBase);

    /// \brief Constructs a conversion operation.
    MulticlassNms() = default;

    /// \brief Constructs a MulticlassNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v8

namespace v9 {
/// \brief MulticlassNms operation
///
class OPENVINO_API MulticlassNms : public util::MulticlassNmsBase {
public:
    OPENVINO_OP("MulticlassNms", "opset9", op::util::MulticlassNmsBase);

    /// \brief Constructs a conversion operation.
    MulticlassNms() = default;

    /// \brief Constructs a MulticlassNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs);

    /// \brief Constructs a MulticlassNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param roisnum Node producing the number of rois
    /// \param attrs Attributes of the operation
    MulticlassNms(const Output<Node>& boxes,
                  const Output<Node>& scores,
                  const Output<Node>& roisnum,
                  const Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v9
}  // namespace op
}  // namespace ov
