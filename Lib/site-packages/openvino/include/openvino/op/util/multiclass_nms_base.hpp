// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for operations MulticlassNMS v8 and MulticlassNMS
/// v9.
class OPENVINO_API MulticlassNmsBase : public Op {
public:
    OPENVINO_OP("MulticlassNmsBase", "util");

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
        // specifies intersection over union threshold
        float iou_threshold = 0.0f;
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
        // specifies eta parameter for adpative NMS, in close range [0, 1.0]
        float nms_eta = 1.0f;
        // specifies whether boxes are normalized or not
        bool normalized = true;
    };

    /// \brief Constructs a conversion operation.
    MulticlassNmsBase() = default;

    /// \brief Constructs a MulticlassNmsBase operation
    ///
    /// \param arguments Node list producing the box coordinates, scores, etc.
    /// \param attrs Attributes of the operation
    MulticlassNmsBase(const OutputVector& arguments, const Attributes& attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

    /// \brief Returns attributes of the operation MulticlassNmsBase
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
}  // namespace util
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::util::MulticlassNmsBase::SortResultType& type);

template <>
class OPENVINO_API AttributeAdapter<op::util::MulticlassNmsBase::SortResultType>
    : public EnumAttributeAdapterBase<op::util::MulticlassNmsBase::SortResultType> {
public:
    AttributeAdapter(op::util::MulticlassNmsBase::SortResultType& value)
        : EnumAttributeAdapterBase<op::util::MulticlassNmsBase::SortResultType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::util::MulticlassNmsBase::SortResultType>");
};
}  // namespace ov
