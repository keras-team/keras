// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/convolution_base.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for operations DeformableConvolution v1 and DeformableConvolution
/// v8.
class OPENVINO_API DeformableConvolutionBase : public util::ConvolutionBase {
public:
    OPENVINO_OP("DeformableConvolutionBase", "util", util::ConvolutionBase);

    /// \brief Constructs a conversion operation.
    DeformableConvolutionBase() = default;

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
    /// \param group              The number of groups which both output and input
    ///                           should be split into.
    /// \param deformable_group   The number of groups which deformable values and
    ///                           output should be split into along the channel axis.
    DeformableConvolutionBase(const OutputVector& arguments,
                              const Strides& strides,
                              const CoordinateDiff& pads_begin,
                              const CoordinateDiff& pads_end,
                              const Strides& dilations,
                              const PadType& auto_pad = PadType::EXPLICIT,
                              int64_t group = 1,
                              int64_t deformable_group = 1);

    bool visit_attributes(AttributeVisitor& visitor) override;

    int64_t get_group() const {
        return m_group;
    }
    void set_group(const int64_t group) {
        m_group = group;
    }
    int64_t get_deformable_group() const {
        return m_deformable_group;
    }
    void set_deformable_group(const int64_t deformable_group) {
        m_deformable_group = deformable_group;
    }

protected:
    int64_t m_group;
    int64_t m_deformable_group;
};
}  // namespace util
}  // namespace op
}  // namespace ov
