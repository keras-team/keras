// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/logical_reduction.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API LogicalReductionKeepDims : public util::LogicalReduction {
protected:
    LogicalReductionKeepDims() = default;

    /// \param arg The tensor to be reduced.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    /// \param keep_dims If set to 1 it holds axes that are used for reduction.
    LogicalReductionKeepDims(const Output<Node>& arg, const Output<Node>& reduction_axes, const bool keep_dims = false);

    bool visit_attributes(AttributeVisitor& visitor) override;

public:
    OPENVINO_OP("LogicalReductionKeepDims", "util", util::LogicalReduction);
    void validate_and_infer_types() override;

    /// \return If set to 1 it holds axes that are used for reduction.
    /// For each such axis, output dimension is equal to 1.
    bool get_keep_dims() const override {
        return m_keep_dims;
    }
    void set_keep_dims(bool keep_dims) {
        m_keep_dims = keep_dims;
    }

private:
    bool m_keep_dims = false;
};
}  // namespace util
}  // namespace op
}  // namespace ov
