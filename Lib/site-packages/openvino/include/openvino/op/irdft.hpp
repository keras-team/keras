// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/fft_base.hpp"

namespace ov {
namespace op {
namespace v9 {
/// \brief An operation IRDFT that computes the discrete inverse complex-to-real Fourier transformation.
class OPENVINO_API IRDFT : public util::FFTBase {
public:
    OPENVINO_OP("IRDFT", "opset9", util::FFTBase);
    IRDFT() = default;

    /// \brief Constructs a IRDFT operation. IRDFT is performed for full size axes.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform IRDFT
    IRDFT(const Output<Node>& data, const Output<Node>& axes);

    /// \brief Constructs a IRDFT operation.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform IRDFT
    /// \param signal_size Signal sizes for 'axes'
    IRDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v9
}  // namespace op
}  // namespace ov
