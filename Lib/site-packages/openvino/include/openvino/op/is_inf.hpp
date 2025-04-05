// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v10 {
/// \brief Boolean mask that maps infinite values to true.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API IsInf : public Op {
public:
    OPENVINO_OP("IsInf", "opset10");
    /// \brief A Structure which contains all IsInf attributes
    struct Attributes {
        // A flag which specifies whether to map negative infinities to true.
        // If set to false, negative infinite will be mapped to false.
        bool detect_negative = true;
        // A flag which specifies whether to map positive infinities to true.
        // If set to false, positive infinite will be mapped to false.
        bool detect_positive = true;

        Attributes() = default;
        Attributes(bool detect_negative, bool detect_positive)
            : detect_negative{detect_negative},
              detect_positive{detect_positive} {}
    };

    IsInf() = default;
    /// \brief Constructs a IsInf operation
    ///
    /// \param data   Input data tensor
    IsInf(const Output<Node>& data);
    /// \brief Constructs a IsInf operation
    ///
    /// \param data   Input data tensor
    /// \param attrs  IsInf attributes
    IsInf(const Output<Node>& data, const Attributes& attributes);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attributes() const {
        return m_attributes;
    }

    void set_attributes(const Attributes& attributes) {
        m_attributes = attributes;
    }

private:
    Attributes m_attributes = {};
};
}  // namespace v10
}  // namespace op
}  // namespace ov
