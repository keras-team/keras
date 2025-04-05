// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v10 {

/// \brief Operator which selects and returns unique elements or unique slices of the input tensor
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Unique : public Op {
public:
    OPENVINO_OP("Unique", "opset10");

    Unique() = default;

    /// \brief Constructs a Unique operation
    ///
    /// \param data     Input data tensor
    /// \param sorted   Controls the order of the returned unique values (sorts ascendingly when true)
    /// \param index_element_type    The data type for outputs containing indices
    /// \param count_element_type    The data type for output containing repetition count
    Unique(const Output<Node>& data,
           const bool sorted = true,
           const element::Type& index_element_type = element::i64,
           const element::Type& count_element_type = element::i64);

    /// \brief Constructs a Unique operation
    ///
    /// \param data     Input data tensor
    /// \param axis     An input tensor containing the axis value
    /// \param sorted   Controls the order of the returned unique values (sorts ascendingly when true)
    /// \param index_element_type    The data type for outputs containing indices
    /// \param count_element_type    The data type for output containing repetition count
    Unique(const Output<Node>& data,
           const Output<Node>& axis,
           const bool sorted = true,
           const element::Type& index_element_type = element::i64,
           const element::Type& count_element_type = element::i64);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_sorted() const {
        return m_sorted;
    }

    void set_sorted(const bool sorted) {
        m_sorted = sorted;
    }

    element::Type get_index_element_type() const {
        return m_index_element_type;
    }

    void set_index_element_type(const element::Type& index_element_type) {
        m_index_element_type = index_element_type;
    }

    element::Type get_count_element_type() const {
        return m_count_element_type;
    }

    void set_count_element_type(const element::Type& count_element_type) {
        m_count_element_type = count_element_type;
    }

private:
    bool m_sorted = true;
    element::Type m_index_element_type = element::i64;
    element::Type m_count_element_type = element::i64;
};
}  // namespace v10
}  // namespace op
}  // namespace ov
