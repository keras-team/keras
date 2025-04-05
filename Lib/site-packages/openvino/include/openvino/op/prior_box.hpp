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
class OPENVINO_API PriorBox : public Op {
public:
    OPENVINO_OP("PriorBox", "opset1");
    struct Attributes {
        // min_size         Desired min_size of prior boxes
        // max_size         Desired max_size of prior boxes
        // aspect_ratio     Aspect ratios of prior boxes
        // clip             Clip output to [0,1]
        // flip             Flip aspect ratios
        // step             Distance between prior box centers
        // offset           Box offset relative to top center of image
        // variance         Values to adjust prior boxes with
        // scale_all_sizes  Scale all sizes
        std::vector<float> min_size;
        std::vector<float> max_size;
        std::vector<float> aspect_ratio;
        std::vector<float> density;
        std::vector<float> fixed_ratio;
        std::vector<float> fixed_size;
        bool clip = false;
        bool flip = false;
        float step = 0.0f;
        float offset = 0.0f;
        std::vector<float> variance;
        bool scale_all_sizes = true;
    };

    PriorBox() = default;
    /// \brief Constructs a PriorBox operation
    ///
    /// \param layer_shape    Shape of layer for which prior boxes are computed
    /// \param image_shape    Shape of image to which prior boxes are scaled
    /// \param attrs          PriorBox attributes
    PriorBox(const Output<Node>& layer_shape, const Output<Node>& image_shape, const Attributes& attrs);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    static int64_t number_of_priors(const Attributes& attrs);

    static std::vector<float> normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip);
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

namespace v8 {
/// \brief Layer which generates prior boxes of specified sizes
/// normalized to input image size
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PriorBox : public Op {
public:
    OPENVINO_OP("PriorBox", "opset8");
    struct Attributes {
        // min_size                    Desired min_size of prior boxes
        // max_size                    Desired max_size of prior boxes
        // aspect_ratio                Aspect ratios of prior boxes
        // clip                        Clip output to [0,1]
        // flip                        Flip aspect ratios
        // step                        Distance between prior box centers
        // offset                      Box offset relative to top center of image
        // variance                    Values to adjust prior boxes with
        // scale_all_sizes             Scale all sizes
        // min_max_aspect_ratios_order Order of output prior box
        std::vector<float> min_size;
        std::vector<float> max_size;
        std::vector<float> aspect_ratio;
        std::vector<float> density;
        std::vector<float> fixed_ratio;
        std::vector<float> fixed_size;
        bool clip = false;
        bool flip = false;
        float step = 0.0f;
        float offset = 0.0f;
        std::vector<float> variance;
        bool scale_all_sizes = true;
        bool min_max_aspect_ratios_order = true;
    };

    PriorBox() = default;
    /// \brief Constructs a PriorBox operation
    ///
    /// \param layer_shape    Shape of layer for which prior boxes are computed
    /// \param image_shape    Shape of image to which prior boxes are scaled
    /// \param attrs          PriorBox attributes
    PriorBox(const Output<Node>& layer_shape, const Output<Node>& image_shape, const Attributes& attrs);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    static int64_t number_of_priors(const Attributes& attrs);

    static std::vector<float> normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip);
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
}  // namespace v8
}  // namespace op
}  // namespace ov
