// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief Adaptive max pooling operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API AdaptiveMaxPool : public Op {
public:
    OPENVINO_OP("AdaptiveMaxPool", "opset8");

    AdaptiveMaxPool() = default;

    ///
    /// \brief    Constructs adaptive max pooling operation.
    ///
    /// \param    data                  Input data
    ///
    /// \param    output_shape          1D tensor describing output shape for spatial
    ///                                 dimensions.
    ///
    /// \param    index_element_type    Specifies the output tensor type for indices
    /// output
    ///
    AdaptiveMaxPool(const Output<Node>& data,
                    const Output<Node>& output_shape,
                    const ov::element::Type& index_element_type = ov::element::i64);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    element::Type get_index_element_type() const {
        return m_index_element_type;
    }
    void set_index_element_type(const element::Type& type);

protected:
    ov::element::Type m_index_element_type = ov::element::i64;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
