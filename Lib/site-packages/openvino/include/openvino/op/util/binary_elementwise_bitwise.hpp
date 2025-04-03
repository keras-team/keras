// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API BinaryElementwiseBitwise : public Op {
protected:
    BinaryElementwiseBitwise();

    /// \brief Constructs a binary elementwise bitwise operation.
    ///
    /// \param arg0 Output that produces the first input tensor.
    /// \param arg1 Output that produces the second input tensor.
    /// \param auto_broadcast Auto broadcast specification. Default is Numpy-style
    ///                       implicit broadcasting.
    BinaryElementwiseBitwise(const Output<Node>& arg0,
                             const Output<Node>& arg1,
                             const AutoBroadcastSpec& autob = AutoBroadcastSpec());

public:
    OPENVINO_OP("BinaryElementwiseBitwise", "util");

    void validate_and_infer_types() override;

    virtual const AutoBroadcastSpec& get_autob() const override;

    void set_autob(const AutoBroadcastSpec& autob);
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    AutoBroadcastSpec m_autob = AutoBroadcastType::NUMPY;
};
}  // namespace util
}  // namespace op
}  // namespace ov
