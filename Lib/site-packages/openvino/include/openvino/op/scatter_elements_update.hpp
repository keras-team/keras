// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/scatter_elements_update_base.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief ScatterElementsUpdate operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScatterElementsUpdate : public util::ScatterElementsUpdateBase {
public:
    OPENVINO_OP("ScatterElementsUpdate", "opset3", util::ScatterElementsUpdateBase);

    ScatterElementsUpdate() = default;
    /// \brief Constructs a ScatterElementsUpdate node

    /// \param data            Input data
    /// \param indices         Data entry index that will be updated
    /// \param updates         Update values
    /// \param axis            Axis to scatter on
    ScatterElementsUpdate(const Output<Node>& data,
                          const Output<Node>& indices,
                          const Output<Node>& updates,
                          const Output<Node>& axis);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
};
}  // namespace v3
namespace v12 {
class OPENVINO_API ScatterElementsUpdate : public op::util::ScatterElementsUpdateBase {
public:
    OPENVINO_OP("ScatterElementsUpdate", "opset12", op::util::ScatterElementsUpdateBase);

    /// \brief Lists the supported reduction types for this version of the operator.
    ///        See the specification for the description of how reduction works with ScatterElementsUpdate.
    enum class Reduction { NONE, SUM, PROD, MIN, MAX, MEAN };

    ScatterElementsUpdate() = default;
    /// \brief Constructs a ScatterElementsUpdate node

    /// \param data            Input data
    /// \param indices         Data entry index that will be updated
    /// \param updates         Update values
    /// \param axis            Axis to scatter on
    ScatterElementsUpdate(const Output<Node>& data,
                          const Output<Node>& indices,
                          const Output<Node>& updates,
                          const Output<Node>& axis,
                          const Reduction reduction = Reduction::NONE,
                          const bool use_init_val = true);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    Reduction get_reduction() const {
        return m_reduction;
    }

    void set_reduction(const Reduction reduction) {
        m_reduction = reduction;
    }

    bool get_use_init_val() const {
        return m_use_init_val;
    }

    void set_use_init_val(const bool use_init_val) {
        m_use_init_val = use_init_val;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;

private:
    Reduction m_reduction = Reduction::NONE;
    bool m_use_init_val = true;
};
}  // namespace v12
OPENVINO_API
std::ostream& operator<<(std::ostream& s, const v12::ScatterElementsUpdate::Reduction& reduction);

}  // namespace op
template <>
class OPENVINO_API AttributeAdapter<op::v12::ScatterElementsUpdate::Reduction>
    : public EnumAttributeAdapterBase<op::v12::ScatterElementsUpdate::Reduction> {
public:
    AttributeAdapter(op::v12::ScatterElementsUpdate::Reduction& value)
        : EnumAttributeAdapterBase<op::v12::ScatterElementsUpdate::Reduction>(value) {}

    OPENVINO_RTTI("AttributeAdapter<v12::ScatterElementsUpdate::Reduction>");
};
}  // namespace ov
