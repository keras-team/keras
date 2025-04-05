// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \note       FakeConvert is an experimental operation and subject to change.
///
/// \brief      FakeConvert performs element-wise quantization of input values
///             into a set of values corresponding to a target low-precision type.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API FakeConvert : public Op {
public:
    OPENVINO_OP("FakeConvert", "opset13");

    FakeConvert() = default;

    /// \brief    Constructs FakeConvert operation (default shift).
    ///
    /// \param data                 The input data tensor.
    /// \param scale                Tensor with a scale factor for the data input.
    /// \param destination_type     The low precision type to be emulated.
    FakeConvert(const ov::Output<ov::Node>& data,
                const ov::Output<ov::Node>& scale,
                std::string destination_type = "f8e4m3");

    /// \brief    Constructs FakeConvert operation.
    ///
    /// \param data                 The input data tensor.
    /// \param scale                Tensor with a scale factor for the data input.
    /// \param shift                Tensor with a shift factor for the data input.
    /// \param destination_type     The low precision type to be emulated.
    FakeConvert(const ov::Output<ov::Node>& data,
                const ov::Output<ov::Node>& scale,
                const ov::Output<ov::Node>& shift,
                std::string destination_type = "f8e4m3");

    /// \brief    Constructs FakeConvert operation (default shift).
    ///
    /// \param data                 The input data tensor.
    /// \param scale                Tensor with a scale factor for the data input.
    /// \param destination_type     The low precision type to be emulated.
    FakeConvert(const ov::Output<ov::Node>& data,
                const ov::Output<ov::Node>& scale,
                const ov::element::Type& destination_type);

    /// \brief    Constructs FakeConvert operation.
    ///
    /// \param data                 The input data tensor.
    /// \param scale                Tensor with a scale factor for the data input.
    /// \param shift                Tensor with a shift factor for the data input.
    /// \param destination_type     The low precision type to be emulated.
    FakeConvert(const ov::Output<ov::Node>& data,
                const ov::Output<ov::Node>& scale,
                const ov::Output<ov::Node>& shift,
                const ov::element::Type& destination_type);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    std::string get_destination_type() const;
    void set_destination_type(ov::element::Type destination_type);
    const ov::element::Type& get_destination_element_type() const;

private:
    void validate_destination_type() const;

    ov::element::Type m_destination_type = ov::element::f8e4m3;
};
}  // namespace v13
}  // namespace op
}  // namespace ov
