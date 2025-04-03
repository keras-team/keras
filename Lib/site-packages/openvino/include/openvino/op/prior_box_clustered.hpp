// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

namespace v0 {
/// \brief Layer which generates prior boxes of specified sizes
/// normalized to input image size
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PriorBoxClustered : public Op {
public:
    OPENVINO_OP("PriorBoxClustered", "opset1");
    struct Attributes {
        // widths         Desired widths of prior boxes
        // heights        Desired heights of prior boxes
        // clip           Clip output to [0,1]
        // step_widths    Distance between prior box centers
        // step_heights   Distance between prior box centers
        // step           Distance between prior box centers (when step_w = step_h)
        // offset         Box offset relative to top center of image
        // variances      Values to adjust prior boxes with
        std::vector<float> widths;
        std::vector<float> heights;
        bool clip = true;
        float step_widths = 0.0f;
        float step_heights = 0.0f;
        float step = 0.0f;
        float offset = 0.0f;
        std::vector<float> variances;
    };

    PriorBoxClustered() = default;
    /// \brief Constructs a PriorBoxClustered operation
    ///
    /// \param layer_shape    Shape of layer for which prior boxes are computed
    /// \param image_shape    Shape of image to which prior boxes are scaled
    /// \param attrs          PriorBoxClustered attributes
    PriorBoxClustered(const Output<Node>& layer_shape, const Output<Node>& image_shape, const Attributes& attrs);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    const Attributes& get_attrs() const {
        return m_attrs;
    }
    void set_attrs(Attributes attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    Attributes m_attrs;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
