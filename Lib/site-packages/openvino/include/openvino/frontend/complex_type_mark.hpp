// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

// ComplexTypeMark serves to mark places that require complex type propagation
// that means to represent native complex type with simulating floating-point tensor
// that has one extra dimension to concatenate real and imaginary parts of complex tensor.
// For example, a tensor of complex type with shape [N1, N2, ..., Nk] will be transformed
// into a floating-point tensor [N1, N2, ..., Nk, 2]
// where a slice with index [..., 0] represents a real part and
// a slice with index [..., 1] represents a imaginary part.
class FRONTEND_API ComplexTypeMark : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("ComplexTypeMark", "util", ov::op::util::FrameworkNode);

    ComplexTypeMark(const ov::Output<ov::Node>& input, const ov::element::Type& complex_part_type)
        : ov::op::util::FrameworkNode(ov::OutputVector{input}, 1),
          m_complex_part_type(complex_part_type) {
        validate_and_infer_types();
    }

    ~ComplexTypeMark() override;

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto complex_type_mark = std::make_shared<ComplexTypeMark>(inputs[0], m_complex_part_type);
        complex_type_mark->set_attrs(get_attrs());
        return complex_type_mark;
    }

    ov::element::Type get_complex_part_type() const {
        return m_complex_part_type;
    }

private:
    ov::element::Type m_complex_part_type;
};

}  // namespace frontend
}  // namespace ov
