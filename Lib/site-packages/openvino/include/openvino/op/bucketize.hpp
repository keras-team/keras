// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Operation that bucketizes the input based on boundaries
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Bucketize : public Op {
public:
    OPENVINO_OP("Bucketize", "opset3", op::Op);

    Bucketize() = default;
    /// \brief Constructs a Bucketize node

    /// \param data              Input data to bucketize
    /// \param buckets           1-D of sorted unique boundaries for buckets
    /// \param output_type       Output tensor type, "i64" or "i32", defaults to i64
    /// \param with_right_bound  indicates whether bucket includes the right or left
    ///                          edge of interval. default true = includes right edge
    Bucketize(const Output<Node>& data,
              const Output<Node>& buckets,
              const element::Type output_type = element::i64,
              const bool with_right_bound = true);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(element::Type output_type) {
        m_output_type = output_type;
    }
    // Overload collision with method on Node
    using Node::set_output_type;

    bool get_with_right_bound() const {
        return m_with_right_bound;
    }
    void set_with_right_bound(bool with_right_bound) {
        m_with_right_bound = with_right_bound;
    }

private:
    element::Type m_output_type;
    bool m_with_right_bound{true};
};
}  // namespace v3
}  // namespace op
}  // namespace ov
