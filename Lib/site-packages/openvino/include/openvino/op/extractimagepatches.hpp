// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief ExtractImagePatches operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ExtractImagePatches : public Op {
public:
    OPENVINO_OP("ExtractImagePatches", "opset3", op::Op);

    ExtractImagePatches() = default;
    /// \brief Constructs a ExtractImagePatches operation
    ///
    /// \param data 4-D Input data to extract image patches
    /// \param sizes Patch size in the format of [size_rows, size_cols]
    /// \param strides Patch movement stride in the format of [stride_rows, stride_cols]
    /// \param rates Element seleciton rate for creating a patch. in the format of
    /// [rate_rows, rate_cols]
    /// \param auto_pad Padding type. it can be any value from
    /// valid, same_lower, same_upper
    ExtractImagePatches(const Output<Node>& image,
                        const Shape& sizes,
                        const Strides& strides,
                        const Shape& rates,
                        const PadType& auto_pad);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Shape& get_sizes() const {
        return m_patch_sizes;
    }
    void set_sizes(const Shape& sizes) {
        m_patch_sizes = sizes;
    }
    const Strides& get_strides() const {
        return m_patch_movement_strides;
    }
    void set_strides(const Strides& strides) {
        m_patch_movement_strides = strides;
    }
    const Shape& get_rates() const {
        return m_patch_selection_rates;
    }
    void set_rates(const Shape& rates) {
        m_patch_selection_rates = rates;
    }
    const PadType& get_auto_pad() const {
        return m_padding;
    }
    void set_auto_pad(const PadType& padding) {
        m_padding = padding;
    }

private:
    Shape m_patch_sizes;
    Strides m_patch_movement_strides;
    Shape m_patch_selection_rates;
    PadType m_padding{PadType::EXPLICIT};
};
}  // namespace v3
}  // namespace op
}  // namespace ov
