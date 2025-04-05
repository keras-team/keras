// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v16 {
/// \brief Identity operation is used as a placeholder op.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Identity : public Op {
public:
    OPENVINO_OP("Identity", "opset16");
    Identity() = default;
    /**
     * @brief Identity operation is used as a placeholder. It copies the tensor data to the output.
     */
    Identity(const Output<Node>& data);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v16
}  // namespace op
}  // namespace ov
