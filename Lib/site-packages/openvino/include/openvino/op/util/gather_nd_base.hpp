// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief GatherNDBase basic class for GatherND v5 and v8
class OPENVINO_API GatherNDBase : public Op {
public:
    OPENVINO_OP("GatherNDBase", "util");
    GatherNDBase() = default;

    /// \brief Constructs a GatherND operation.
    ///
    /// \param data Node producing data that are gathered
    /// \param indices Node producing indices by which the operation gathers elements
    /// or slices from data
    /// \param batch_dims Specifies a leading number of dimensions representing the batches
    GatherNDBase(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims = 0);

    size_t get_batch_dims() const {
        return m_batch_dims;
    }

    void set_batch_dims(size_t batch_dims) {
        m_batch_dims = batch_dims;
    }

    void validate_inputs_and_infer_shape();

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    size_t m_batch_dims = 0;
};
}  // namespace util
}  // namespace op
}  // namespace ov
