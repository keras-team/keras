// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/fft_base.hpp"

namespace ov {
namespace op {
namespace v9 {
/// \brief An operation RDFT that computes the discrete real-to-complex Fourier transformation.
class OPENVINO_API RDFT : public util::FFTBase {
public:
    OPENVINO_OP("RDFT", "opset9", util::FFTBase);
    RDFT() = default;

    /// \brief Constructs a RDFT operation. RDFT is performed for full size axes.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform RDFT
    RDFT(const Output<Node>& data, const Output<Node>& axes);

    /// \brief Constructs a RDFT operation.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform RDFT
    /// \param signal_size Signal sizes for 'axes'
    RDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v9
}  // namespace op
}  // namespace ov
