// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/interpolate_base.hpp"

namespace ov {
namespace op {
namespace v0 {

/// \brief Layer which performs bilinear interpolation
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Interpolate : public Op {
public:
    OPENVINO_OP("Interpolate", "opset1");
    /// \brief Structure that specifies attributes for interpolation
    struct Attributes {
        // specify dimension indices where interpolation is applied, and `axes` is any
        // unordered list of indices of different dimensions of input tensor. Required.
        AxisSet axes;
        // specifies type of interpolation
        // one of `nearest`, `linear`, `cubic`, `area`. Required.
        std::string mode;
        // a flag that specifies whether to align corners or not.
        // `true` (default) means the alignment is applied,
        // `false` means the alignment isn't applied.
        bool align_corners = true;
        // a flag that specifies whether to perform anti-aliasing. default is `false`
        bool antialias = false;
        // specify the number of pixels to add to the beginning of the image being
        // interpolated. This addition of pixels is done before interpolation calculation.
        std::vector<size_t> pads_begin;
        // specify the number of pixels to add to the end of the image being interpolated.
        // This addition of pixels is done before interpolation calculation.
        std::vector<size_t> pads_end;
    };

    enum class InterpolateMode { NEAREST, LINEAR, CUBIC, AREA };

    Interpolate() = default;
    /// \brief Constructs a Interpolate operation
    ///
    /// \param image        Input image
    /// \param output_shape Output shape of spatial axes
    /// \param attrs        Interpolation attributes
    Interpolate(const Output<Node>& image, const Output<Node>& output_shape, const Attributes& attrs);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }

    void set_attrs(Attributes attrs);

private:
    Attributes m_attrs;
};
}  // namespace v0

namespace v4 {
/// \brief Interpolate operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Interpolate : public util::InterpolateBase {
public:
    OPENVINO_OP("Interpolate", "opset4", util::InterpolateBase);

    Interpolate() = default;
    /// \brief Constructs a Interpolate operation without 'axes' input.
    ///
    /// \param image  Input image
    /// \param output_shape Output shape of spatial axes
    /// \param scales Scales of spatial axes, i.e. output_shape / input_shape
    /// \param attrs  Interpolation attributes
    Interpolate(const Output<Node>& image,
                const Output<Node>& output_shape,
                const Output<Node>& scales,
                const InterpolateAttrs& attrs);

    /// \brief Constructs a Interpolate operation with 'axes' input.
    ///
    /// \param image  Input image
    /// \param output_shape Output shape of spatial axes
    /// \param scales Scales of spatial axes, i.e. output_shape / input_shape
    /// \param axes   Interpolation axes
    /// \param attrs  Interpolation attributes
    Interpolate(const Output<Node>& image,
                const Output<Node>& output_shape,
                const Output<Node>& scales,
                const Output<Node>& axes,
                const InterpolateAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    bool evaluate_interpolate(TensorVector& outputs, const TensorVector& inputs) const;
};
}  // namespace v4

namespace v11 {
/// \brief Interpolate operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Interpolate : public util::InterpolateBase {
public:
    OPENVINO_OP("Interpolate", "opset11", util::InterpolateBase);
    Interpolate() = default;
    /// \brief Constructs a Interpolate operation without 'axes' input.
    ///
    /// \param image  Input image
    /// \param scales_or_sizes Scales of spatial axes, i.e. output_shape / input_shape
    /// \param attrs  Interpolation attributes
    Interpolate(const Output<Node>& image, const Output<Node>& scales_or_sizes, const InterpolateAttrs& attrs);

    /// \brief Constructs a Interpolate operation with 'axes' input.
    ///
    /// \param image  Input image
    /// \param scales_or_sizes Scales of spatial axes, i.e. output_shape / input_shape
    /// \param axes   Interpolation axes
    /// \param attrs  Interpolation attributes
    Interpolate(const Output<Node>& image,
                const Output<Node>& scales_or_sizes,
                const Output<Node>& axes,
                const InterpolateAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool has_evaluate() const override {
        return false;
    }
};
}  // namespace v11
}  // namespace op

//---------------------------------------- v0 --------------------------------------------------
OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v0::Interpolate::InterpolateMode& type);

template <>
class OPENVINO_API AttributeAdapter<op::v0::Interpolate::InterpolateMode>
    : public EnumAttributeAdapterBase<op::v0::Interpolate::InterpolateMode> {
public:
    AttributeAdapter(op::v0::Interpolate::InterpolateMode& value)
        : EnumAttributeAdapterBase<op::v0::Interpolate::InterpolateMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v0::Interpolate::InterpolateMode>");
};
}  // namespace ov
