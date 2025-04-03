// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API IndexReduction : public Op {
protected:
    IndexReduction();

    IndexReduction(const Output<Node>& arg, uint64_t axis, const element::Type& index_element_type);

public:
    uint64_t get_reduction_axis() const;
    void set_reduction_axis(uint64_t value);
    element::Type get_index_element_type() const;
    void set_index_element_type(const element::Type& index_element_type);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    uint64_t m_axis{0};
    element::Type m_index_element_type;
};
}  // namespace util
}  // namespace op
}  // namespace ov
