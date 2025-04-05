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
namespace v1 {
/// \brief BinaryConvolution operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BinaryConvolution : public util::ConvolutionFwdPropBase {
public:
    OPENVINO_OP("BinaryConvolution", "opset1", op::util::ConvolutionFwdPropBase);

    enum class BinaryConvolutionMode {
        // Interpret input data and kernel values: 0 as -1, 1 as 1
        XNOR_POPCOUNT
    };

    /// \brief Constructs a binary convolution operation.
    BinaryConvolution() = default;
    /// \brief Constructs a binary convolution operation.
    /// \param data The node producing the input data batch tensor.
    /// \param kernel The node producing the filters tensor.
    /// \param strides The strides.
    /// \param pads_begin The beginning of padding shape.
    /// \param pads_end The end of padding shape.
    /// \param dilations The dilations.
    /// \param mode Defines how input tensor 0/1 values and weights 0/1 are interpreted.
    /// \param pad_value Floating-point value used to fill pad area.
    /// \param auto_pad The pad type for automatically computing padding sizes.
    ///
    /// Output `[N, C_OUT, R1, ... Rf]`
    BinaryConvolution(const Output<Node>& data,
                      const Output<Node>& kernel,
                      const Strides& strides,
                      const CoordinateDiff& pads_begin,
                      const CoordinateDiff& pads_end,
                      const Strides& dilations,
                      BinaryConvolutionMode mode,
                      float pad_value,
                      const PadType& auto_pad = PadType::EXPLICIT);

    BinaryConvolution(const Output<Node>& data,
                      const Output<Node>& kernel,
                      const Strides& strides,
                      const CoordinateDiff& pads_begin,
                      const CoordinateDiff& pads_end,
                      const Strides& dilations,
                      const std::string& mode,
                      float pad_value,
                      const PadType& auto_pad = PadType::EXPLICIT);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The mode of convolution.
    const BinaryConvolutionMode& get_mode() const {
        return m_mode;
    }
    void set_mode(const BinaryConvolutionMode& mode) {
        m_mode = mode;
    }
    /// \return The pad value.
    float get_pad_value() const {
        return m_pad_value;
    }
    void set_pad_value(float pad_value) {
        m_pad_value = pad_value;
    }

protected:
    BinaryConvolutionMode mode_from_string(const std::string& mode) const;

    BinaryConvolutionMode m_mode;
    float m_pad_value;
};
}  // namespace v1
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v1::BinaryConvolution::BinaryConvolutionMode& type);

template <>
class OPENVINO_API AttributeAdapter<op::v1::BinaryConvolution::BinaryConvolutionMode>
    : public EnumAttributeAdapterBase<op::v1::BinaryConvolution::BinaryConvolutionMode> {
public:
    AttributeAdapter(op::v1::BinaryConvolution::BinaryConvolutionMode& value)
        : EnumAttributeAdapterBase<op::v1::BinaryConvolution::BinaryConvolutionMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v1::BinaryConvolution::BinaryConvolutionMode>");
};

}  // namespace ov
