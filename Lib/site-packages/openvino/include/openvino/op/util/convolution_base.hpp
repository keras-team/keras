// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for operations like convolutions
class OPENVINO_API ConvolutionBase : public Op {
public:
    OPENVINO_OP("ConvolutionBase", "util");

    /// \brief Constructs a conversion operation.
    ConvolutionBase() = default;

    /// \brief Constructs a conversion operation.
    /// \param strides            Convolution strides.
    /// \param pads_begin         Amount of padding to be added to the beginning along
    ///                           each axis. For example in case of a 2D input the value
    ///                           of (1, 2) means that 1 element will be added to the
    ///                           top and 2 elements to the left.
    /// \param pads_end           Amount of padding to be added to the end along each
    ///                           axis.
    /// \param dilations          The distance in width and height between the weights
    ///                           in the filters tensor.
    /// \param auto_pad           Specifies how the automatic calculation of padding
    ///                           should be done.
    ConvolutionBase(const OutputVector& arguments,
                    const Strides& strides,
                    const CoordinateDiff& pads_begin,
                    const CoordinateDiff& pads_end,
                    const Strides& dilations,
                    const PadType& auto_pad = PadType::EXPLICIT)
        : Op(arguments),
          m_strides(strides),
          m_dilations(dilations),
          m_pads_begin(pads_begin),
          m_pads_end(pads_end),
          m_auto_pad(auto_pad) {}

    const Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const Strides& strides) {
        m_strides = strides;
    }
    const Strides& get_dilations() const {
        return m_dilations;
    }
    void set_dilations(const Strides& dilations) {
        m_dilations = dilations;
    }
    const CoordinateDiff& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const CoordinateDiff& pads_begin) {
        m_pads_begin = pads_begin;
    }
    const CoordinateDiff& get_pads_end() const {
        return m_pads_end;
    }
    void set_pads_end(const CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }
    const PadType& get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }

protected:
    Strides m_strides;
    Strides m_dilations;
    CoordinateDiff m_pads_begin;
    CoordinateDiff m_pads_end;
    PadType m_auto_pad;
    size_t m_num_spatial = std::numeric_limits<size_t>::max();

    void resize_attributes(size_t num_spatial) {
        if (m_strides.empty()) {
            m_strides.resize(num_spatial, 1);
        }
        if (m_dilations.empty()) {
            m_dilations.resize(num_spatial, 1);
        }
    }

    void set_num_spatial(size_t num_spatial, const std::vector<PartialShape>& input_shapes) {
        if (input_shapes[0].rank().is_static() && input_shapes[1].rank().is_static()) {
            m_num_spatial = num_spatial;
        }
    }

private:
    friend bool is_attr_validation_required(const ConvolutionBase* op);
    friend size_t get_num_spatial(const ConvolutionBase* op);
};

/// \brief Base class for operations like back propagation convolution
class OPENVINO_API ConvolutionFwdPropBase : public ConvolutionBase {
public:
    OPENVINO_OP("ConvolutionFwdPropBase", "util", ConvolutionBase);

    /// \brief Constructs a conversion operation.
    ConvolutionFwdPropBase() = default;

    /// \brief Constructs a conversion operation.
    /// \param strides            Convolution strides.
    /// \param pads_begin         Amount of padding to be added to the beginning along
    ///                           each axis. For example in case of a 2D input the value
    ///                           of (1, 2) means that 1 element will be added to the
    ///                           top and 2 elements to the left.
    /// \param pads_end           Amount of padding to be added to the end along each
    ///                           axis.
    /// \param dilations          The distance in width and height between the weights
    ///                           in the filters tensor.
    /// \param auto_pad           Specifies how the automatic calculation of padding
    ///                           should be done.
    ConvolutionFwdPropBase(const OutputVector& arguments,
                           const Strides& strides,
                           const CoordinateDiff& pads_begin,
                           const CoordinateDiff& pads_end,
                           const Strides& dilations,
                           const PadType& auto_pad = PadType::EXPLICIT)
        : ConvolutionBase(arguments, strides, pads_begin, pads_end, dilations, auto_pad) {}

private:
    friend bool is_attr_validation_required(const ConvolutionBase* op);
};

}  // namespace util
}  // namespace op
}  // namespace ov
