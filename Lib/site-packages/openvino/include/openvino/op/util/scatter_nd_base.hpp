// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
///
/// \brief      Base class for ScatterNDXXX operators.
///
class OPENVINO_API ScatterNDBase : public Op {
public:
    OPENVINO_OP("ScatterNDBase", "util");
    // Respective input ordinal number.
    static constexpr int INPUTS = 0;
    static constexpr int INDICES = 1;
    static constexpr int UPDATES = 2;
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    ScatterNDBase() = default;

    ///
    /// \brief      Constructs ScatterNDBase object.
    ///
    /// \param      inputs   The input tensor to be updated.
    /// \param      indices  The tensor with indexes which will be updated.
    /// \param      updates  The tensor with update values.
    ///
    ScatterNDBase(const Output<Node>& inputs, const Output<Node>& indices, const Output<Node>& updates);
};
}  // namespace util
}  // namespace op
}  // namespace ov
