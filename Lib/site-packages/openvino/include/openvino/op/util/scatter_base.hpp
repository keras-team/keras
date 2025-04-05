// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
///
/// \brief      Base class for ScatterXXX operators.
///
class OPENVINO_API ScatterBase : public Op {
public:
    OPENVINO_OP("ScatterBase", "util");
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    ScatterBase() = default;

    ///
    /// \brief      Constructs ScatterBase object.
    ///
    /// \param      inputs   The input tensor to be updated.
    /// \param      indices  The tensor with indexes which will be updated.
    /// \param      updates  The tensor with update values.
    /// \param[in]  axis     The axis at which elements will be updated.
    ///
    ScatterBase(const Output<Node>& inputs,
                const Output<Node>& indices,
                const Output<Node>& updates,
                const Output<Node>& axis);

private:
    // Respective input ordinal number.
    static constexpr int DATA = 0;
    static constexpr int INDICES = 1;
    static constexpr int UPDATES = 2;
    static constexpr int AXIS = 3;
};
}  // namespace util
}  // namespace op
}  // namespace ov
