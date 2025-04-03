// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v9 {

/// \brief Operator performing interpolated sampling of the input tensor
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GridSample : public Op {
public:
    OPENVINO_OP("GridSample", "opset9");

    enum class InterpolationMode { BILINEAR, BICUBIC, NEAREST };
    enum class PaddingMode { ZEROS, BORDER, REFLECTION };

    /// \brief A Structure which contains all GridSample attributes
    struct Attributes {
        // A flag which specifies whether to align the grid extrema values with the borders or center points
        // of the input tensor's border pixels.
        bool align_corners = false;
        // Specifies the type of interpolation: `bilinear`, `bicubic`, `nearest`
        InterpolationMode mode = InterpolationMode::BILINEAR;
        // Specifies how the out-of-bounds coordinates should be handled: `zeros`, `border`, `reflection`
        PaddingMode padding_mode = PaddingMode::ZEROS;

        Attributes() = default;
        Attributes(bool align_corners, InterpolationMode mode, PaddingMode padding_mode)
            : align_corners{align_corners},
              mode{mode},
              padding_mode{padding_mode} {}
    };

    GridSample() = default;
    /// \brief Constructs a GridSample operation
    ///
    /// \param data   Input data tensor (input image)
    /// \param grid   Normalized interpolation coordinates
    /// \param attrs  GridSample attributes
    GridSample(const Output<Node>& data, const Output<Node>& grid, const Attributes& attributes);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attributes() const {
        return m_attributes;
    }

    void set_attributes(const Attributes& attributes) {
        m_attributes = attributes;
    }

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    Attributes m_attributes = {};
};
}  // namespace v9
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v9::GridSample::InterpolationMode& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v9::GridSample::PaddingMode& type);

template <>
class OPENVINO_API AttributeAdapter<op::v9::GridSample::InterpolationMode>
    : public EnumAttributeAdapterBase<op::v9::GridSample::InterpolationMode> {
public:
    AttributeAdapter(op::v9::GridSample::InterpolationMode& value)
        : EnumAttributeAdapterBase<op::v9::GridSample::InterpolationMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v9::GridSample::InterpolationMode>");
};
template <>
class OPENVINO_API AttributeAdapter<op::v9::GridSample::PaddingMode>
    : public EnumAttributeAdapterBase<op::v9::GridSample::PaddingMode> {
public:
    AttributeAdapter(op::v9::GridSample::PaddingMode& value)
        : EnumAttributeAdapterBase<op::v9::GridSample::PaddingMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v9::GridSample::PaddingMode>");
};
}  // namespace ov
