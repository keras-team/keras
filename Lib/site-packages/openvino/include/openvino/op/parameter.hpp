// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/layout.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief A model parameter.
///
/// Parameters are nodes that represent the arguments that will be passed to
/// user-defined models. Model creation requires a sequence of parameters.
/// Basic graph operations do not need parameters attached to a model.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Parameter : public op::Op {
public:
    OPENVINO_OP("Parameter", "opset1");
    /// \brief Constructions a tensor-typed parameter node.
    Parameter() = default;
    /// \brief Constructions a tensor-typed parameter node.
    ///
    /// \param element_type The element type of the parameter.
    /// \param pshape The partial shape of the parameter.
    Parameter(const ov::element::Type& element_type, const PartialShape& pshape);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool is_relevant_to_shapes() const;
    void set_is_relevant_to_shapes(bool is_relevant);

    const PartialShape& get_partial_shape() const {
        return m_partial_shape;
    }
    PartialShape& get_partial_shape() {
        return m_partial_shape;
    }

    void set_partial_shape(const PartialShape& partial_shape);

    const element::Type& get_element_type() const {
        return m_element_type;
    }
    void set_element_type(const element::Type& element_type) {
        m_element_type = element_type;
    }

    /// \brief Returns current layout, or empty Layout if it is not set
    Layout get_layout() const;

    /// \brief Sets layout runtime information to tensor.
    ///
    /// \param layout Layout to set. If empty (default constructed), layout runtime information is erased.
    void set_layout(const Layout& layout);

protected:
    PartialShape m_partial_shape;
    element::Type m_element_type;
    bool m_is_relevant_to_shapes{false};
};
}  // namespace v0
}  // namespace op
using ParameterVector = std::vector<std::shared_ptr<op::v0::Parameter>>;

template <>
class OPENVINO_API AttributeAdapter<ParameterVector> : public VisitorAdapter {
public:
    AttributeAdapter(ParameterVector& ref);

    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<ParameterVector>");

protected:
    ParameterVector& m_ref;
};

}  // namespace ov
