// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief SearchSorted operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SearchSorted : public Op {
public:
    OPENVINO_OP("SearchSorted", "opset15", Op);

    SearchSorted() = default;
    /// \brief Constructs a SearchSorted operation.
    /// \param sorted_sequence Sorted sequence to search in.
    /// \param values          Values to search indexs for.
    /// \param right_mode      If False, return the first suitable index that is found for given value. If True, return
    /// the last such index.
    /// \param output_type The element type of the output tensor. This is purely an implementation flag, which
    /// is used to convert the output type for CPU plugin in ConvertPrecision transformation (and potentially other
    /// plugins as well). Setting this flag to element::i32 will result in the output tensor of i32 element type.
    /// Setting this flag to element::i64 will generally not give any effect, since it will be converted to i32 anyway,
    /// at least for CPU plugin.
    SearchSorted(const Output<Node>& sorted_sequence,
                 const Output<Node>& values,
                 bool right_mode = false,
                 const element::Type& output_type = element::i64);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_right_mode() const {
        return m_right_mode;
    }

    void set_right_mode(bool right_mode) {
        m_right_mode = right_mode;
    }

    void set_output_type_attr(const element::Type& output_type) {
        m_output_type = output_type;
    }

    element::Type get_output_type_attr() const {
        return m_output_type;
    }

private:
    bool m_right_mode{};
    element::Type m_output_type = element::i64;
};
}  // namespace v15
}  // namespace op
}  // namespace ov
