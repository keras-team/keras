// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/scatter_nd_base.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Add updates to slices from inputs addressed by indices
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScatterNDUpdate : public util::ScatterNDBase {
public:
    OPENVINO_OP("ScatterNDUpdate", "opset4", util::ScatterNDBase);
    ScatterNDUpdate() = default;
    /// \param inputs Tensor
    /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
    /// \param updates Tensor: Must have same type as inputs
    ScatterNDUpdate(const Output<Node>& inputs, const Output<Node>& indices, const Output<Node>& updates)
        : util::ScatterNDBase(inputs, indices, updates) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool has_evaluate() const override;
};
}  // namespace v3
namespace v15 {
/// \brief Add updates to slices from inputs addressed by indices
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScatterNDUpdate : public util::ScatterNDBase {
public:
    OPENVINO_OP("ScatterNDUpdate", "opset15", util::ScatterNDBase);

    /// \brief Lists the supported reduction types for this version of the operator.
    ///        See the specification for the description of how reduction works with ScatterNDUpdate.
    enum class Reduction { NONE, SUM, SUB, PROD, MIN, MAX };

    ScatterNDUpdate() = default;
    /// \param inputs Tensor
    /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
    /// \param updates Tensor: Must have same type as inputs
    /// \param reduction Reduction: Type of operation to perform on inputs
    ScatterNDUpdate(const Output<Node>& inputs,
                    const Output<Node>& indices,
                    const Output<Node>& updates,
                    const Reduction reduction = Reduction::NONE);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool evaluate_lower(TensorVector& output_values) const override;
    bool evaluate_upper(TensorVector& output_values) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool has_evaluate() const override;

    Reduction get_reduction() const;

    void set_reduction(const Reduction reduction);

private:
    Reduction m_reduction = Reduction::NONE;
};
}  // namespace v15
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v15::ScatterNDUpdate::Reduction& reduction);

template <>
class OPENVINO_API AttributeAdapter<op::v15::ScatterNDUpdate::Reduction>
    : public EnumAttributeAdapterBase<op::v15::ScatterNDUpdate::Reduction> {
public:
    AttributeAdapter(op::v15::ScatterNDUpdate::Reduction& value)
        : EnumAttributeAdapterBase<op::v15::ScatterNDUpdate::Reduction>(value) {}

    OPENVINO_RTTI("AttributeAdapter<v15::ScatterNDUpdate::Reduction>");
};
}  // namespace ov
