// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API InterpolateBase : public Op {
public:
    OPENVINO_OP("InterpolateBase", "util");
    /// \brief PartialShape calculation mode
    ///
    /// SIZES  - output shape for interpolated axes is calculated using input `sizes`
    /// SCALES - output shape for interpolated axes is calculated using input `scales`
    enum class ShapeCalcMode { SIZES, SCALES };

    /// \brief Interpolation mode
    ///
    /// NEAREST         - nearest interpolation
    /// LINEAR          - linear interpolation as in TensorFlow
    /// LINEAR_ONNX     - linear interpolation as in ONNX
    /// CUBIC           - cubic interpolation
    /// BILINEAR_PILLOW - bilinear interpolation as in Pillow
    /// BICUBIC_PILLOW  - bicubic interpolation  as in Pillow
    enum class InterpolateMode { NEAREST, LINEAR, LINEAR_ONNX, CUBIC, BILINEAR_PILLOW, BICUBIC_PILLOW };

    /// \brief Mode of the calculation of the source coordinate from resized one
    ///
    /// These modes are modes from ONNXRuntime.
    enum class CoordinateTransformMode {
        HALF_PIXEL,
        PYTORCH_HALF_PIXEL,
        ASYMMETRIC,
        TF_HALF_PIXEL_FOR_NN,
        ALIGN_CORNERS
    };

    /// \brief Rounding modes for the NEAREST interpolation.
    enum class NearestMode { ROUND_PREFER_FLOOR, ROUND_PREFER_CEIL, FLOOR, CEIL, SIMPLE };

    struct InterpolateAttrs {
        // Specifies the type of interpolation. Required.
        InterpolateMode mode = InterpolateMode::NEAREST;
        // Specifies the shape calculation mode. Required.
        ShapeCalcMode shape_calculation_mode = ShapeCalcMode::SIZES;
        // Specifies the number of pixels to add to the beginning of the image being
        // interpolated. This addition of pixels is done before the interpolation calculation.
        std::vector<size_t> pads_begin;
        // Specifies the number of pixels to add to the end of the image being
        // interpolated. This addition of pixels is done before the interpolation calculation.
        std::vector<size_t> pads_end;
        // Specifies how to transform the coordinate in the resized tensor to the
        // coordinate in the original tensor.
        CoordinateTransformMode coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
        // Specifies rounding mode when `mode == NEAREST` and takes effect only when `mode == NEAREST`.
        NearestMode nearest_mode = NearestMode::ROUND_PREFER_FLOOR;
        // A flag that specifies whether to perform anti-aliasing.
        bool antialias = false;
        // Specifies the parameter *a* for the cubic interpolation .
        // (see, e.g. [article](https://ieeexplore.ieee.org/document/1163711/)).
        // *cube_coeff* takes effect only when `mode == CUBIC` or `mode == BICUBIC_PILLOW`
        double cube_coeff = -0.75f;

        InterpolateAttrs() = default;

        InterpolateAttrs(InterpolateMode mode,
                         ShapeCalcMode shape_calculation_mode,
                         const std::vector<size_t>& pads_begin,
                         const std::vector<size_t>& pads_end,
                         CoordinateTransformMode coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL,
                         NearestMode nearest_mode = NearestMode::ROUND_PREFER_FLOOR,
                         bool antialias = false,
                         double cube_coeff = -0.75)
            : mode(mode),
              shape_calculation_mode(shape_calculation_mode),
              pads_begin(pads_begin),
              pads_end(pads_end),
              coordinate_transformation_mode(coordinate_transformation_mode),
              nearest_mode(nearest_mode),
              antialias(antialias),
              cube_coeff(cube_coeff) {}

        bool operator==(const InterpolateAttrs& other) const {
            return std::tie(mode,
                            shape_calculation_mode,
                            pads_begin,
                            pads_end,
                            coordinate_transformation_mode,
                            nearest_mode,
                            antialias,
                            cube_coeff) == std::tie(other.mode,
                                                    other.shape_calculation_mode,
                                                    other.pads_begin,
                                                    other.pads_end,
                                                    other.coordinate_transformation_mode,
                                                    other.nearest_mode,
                                                    other.antialias,
                                                    other.cube_coeff);
        }

        bool operator!=(const InterpolateAttrs& other) const {
            return !operator==(other);
        }
    };

    InterpolateBase() = default;

    InterpolateBase(const Output<Node>& image, const Output<Node>& scales_or_sizes, const InterpolateAttrs& attrs);

    // Since v11::Interpolate the second input serves different purpose depending on
    // the value of attrs.shape_calculation_mode.
    //
    // If this constructor is used by v4::Interpolate the 3 inputs serve as: image, output_shape and scales
    InterpolateBase(const Output<Node>& image,
                    const Output<Node>& scales_or_sizes,  // v4::Interpolate -> output_shape
                    const Output<Node>& axes,             // v4::Interpolate -> scales
                    const InterpolateAttrs& attrs);

    // This constructor should only be used by v4::Interpolate
    InterpolateBase(const Output<Node>& image,
                    const Output<Node>& output_shape,
                    const Output<Node>& scales,
                    const Output<Node>& axes,
                    const InterpolateAttrs& attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    const InterpolateAttrs& get_attrs() const {
        return m_attrs;
    }
    void set_attrs(const InterpolateAttrs& attrs) {
        this->m_attrs = attrs;
    }

protected:
    InterpolateAttrs m_attrs;

    void validate_scales_element_type(const element::Type& et) const;
    void validate_sizes_element_type(const element::Type& et) const;
    void validate_axes_element_type(const element::Type& et) const;
};
}  // namespace util
}  // namespace op

template <>
class OPENVINO_API AttributeAdapter<op::util::InterpolateBase::InterpolateMode>
    : public EnumAttributeAdapterBase<op::util::InterpolateBase::InterpolateMode> {
public:
    AttributeAdapter(op::util::InterpolateBase::InterpolateMode& value)
        : EnumAttributeAdapterBase<op::util::InterpolateBase::InterpolateMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::util::InterpolateBase::InterpolateMode>");
};

template <>
class OPENVINO_API AttributeAdapter<op::util::InterpolateBase::CoordinateTransformMode>
    : public EnumAttributeAdapterBase<op::util::InterpolateBase::CoordinateTransformMode> {
public:
    AttributeAdapter(op::util::InterpolateBase::CoordinateTransformMode& value)
        : EnumAttributeAdapterBase<op::util::InterpolateBase::CoordinateTransformMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::util::InterpolateBase::CoordinateTransformMode>");
};

template <>
class OPENVINO_API AttributeAdapter<op::util::InterpolateBase::NearestMode>
    : public EnumAttributeAdapterBase<op::util::InterpolateBase::NearestMode> {
public:
    AttributeAdapter(op::util::InterpolateBase::NearestMode& value)
        : EnumAttributeAdapterBase<op::util::InterpolateBase::NearestMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::util::InterpolateBase::NearestMode>");
};

template <>
class OPENVINO_API AttributeAdapter<op::util::InterpolateBase::ShapeCalcMode>
    : public EnumAttributeAdapterBase<op::util::InterpolateBase::ShapeCalcMode> {
public:
    AttributeAdapter(op::util::InterpolateBase::ShapeCalcMode& value)
        : EnumAttributeAdapterBase<op::util::InterpolateBase::ShapeCalcMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::util::InterpolateBase::ShapeCalcMode>");
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::util::InterpolateBase::InterpolateMode& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::util::InterpolateBase::CoordinateTransformMode& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::util::InterpolateBase::NearestMode& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::util::InterpolateBase::ShapeCalcMode& type);
}  // namespace ov
