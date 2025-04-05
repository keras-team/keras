// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief MatrixNms operation
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API MatrixNms : public Op {
public:
    OPENVINO_OP("MatrixNms", "opset8");

    enum class DecayFunction { GAUSSIAN, LINEAR };

    enum class SortResultType {
        CLASSID,  // sort selected boxes by class id (ascending) in each batch element
        SCORE,    // sort selected boxes by score (descending) in each batch element
        NONE      // do not guarantee the order in each batch element
    };

    /// \brief Structure that specifies attributes of the operation
    struct Attributes {
        // specifies order of output elements
        SortResultType sort_result_type = SortResultType::NONE;
        // specifies whenever it is necessary to sort selected boxes across batches or
        // not
        bool sort_result_across_batch = false;
        // specifies the output tensor type
        ov::element::Type output_type = ov::element::i64;
        // specifies minimum score to consider box for the processing
        float score_threshold = 0.0f;
        // specifies maximum number of boxes to be selected per class, -1 meaning to
        // keep all boxes
        int nms_top_k = -1;
        // specifies maximum number of boxes to be selected per batch element, -1
        // meaning to keep all boxes
        int keep_top_k = -1;
        // specifies the background class id, -1 meaning to keep all classes
        int background_class = -1;
        // specifies decay function used to decay scores
        DecayFunction decay_function = DecayFunction::LINEAR;
        // specifies gaussian_sigma parameter for gaussian decay_function
        float gaussian_sigma = 2.0f;
        // specifies threshold to filter out boxes with low confidence score after
        // decaying
        float post_threshold = 0.0f;
        // specifies whether boxes are normalized or not
        bool normalized = true;

        Attributes() {}

        Attributes(op::v8::MatrixNms::SortResultType sort_result_type,
                   bool sort_result_across_batch,
                   ov::element::Type output_type,
                   float score_threshold,
                   int nms_top_k,
                   int keep_top_k,
                   int background_class,
                   op::v8::MatrixNms::DecayFunction decay_function,
                   float gaussian_sigma,
                   float post_threshold,
                   bool normalized)
            : sort_result_type(sort_result_type),
              sort_result_across_batch(sort_result_across_batch),
              output_type(output_type),
              score_threshold(score_threshold),
              nms_top_k(nms_top_k),
              keep_top_k(keep_top_k),
              background_class(background_class),
              decay_function(decay_function),
              gaussian_sigma(gaussian_sigma),
              post_threshold(post_threshold),
              normalized(normalized) {}
    };

    /// \brief Constructs a conversion operation.
    MatrixNms() = default;

    /// \brief Constructs a MatrixNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    MatrixNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    /// \brief Returns attributes of the operation MatrixNms
    const Attributes& get_attrs() const {
        return m_attrs;
    }

    void set_attrs(Attributes attrs);

    void set_output_type(const element::Type& output_type) {
        m_attrs.output_type = output_type;
    }
    using Node::set_output_type;

protected:
    Attributes m_attrs;

    void validate();  // helper
};
}  // namespace v8
}  // namespace op
OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v8::MatrixNms::DecayFunction& type);

template <>
class OPENVINO_API AttributeAdapter<op::v8::MatrixNms::DecayFunction>
    : public EnumAttributeAdapterBase<op::v8::MatrixNms::DecayFunction> {
public:
    AttributeAdapter(op::v8::MatrixNms::DecayFunction& value)
        : EnumAttributeAdapterBase<op::v8::MatrixNms::DecayFunction>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v8::MatrixNms::DecayFunction>");
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v8::MatrixNms::SortResultType& type);

template <>
class OPENVINO_API AttributeAdapter<op::v8::MatrixNms::SortResultType>
    : public EnumAttributeAdapterBase<op::v8::MatrixNms::SortResultType> {
public:
    AttributeAdapter(op::v8::MatrixNms::SortResultType& value)
        : EnumAttributeAdapterBase<op::v8::MatrixNms::SortResultType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v8::MatrixNms::SortResultType>");
};

}  // namespace ov
