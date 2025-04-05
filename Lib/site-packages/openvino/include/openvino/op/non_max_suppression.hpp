// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise addition operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NonMaxSuppression : public Op {
public:
    enum class BoxEncodingType { CORNER, CENTER };

    OPENVINO_OP("NonMaxSuppression", "opset1", op::Op);

    NonMaxSuppression() = default;

    /// \brief Constructs a NonMaxSuppression operation.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true);

    /// \brief Constructs a NonMaxSuppression operation with default values for the last
    ///        3 inputs
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box coordinates
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    BoxEncodingType get_box_encoding() const {
        return m_box_encoding;
    }
    void set_box_encoding(const BoxEncodingType box_encoding) {
        m_box_encoding = box_encoding;
    }
    bool get_sort_result_descending() const {
        return m_sort_result_descending;
    }
    void set_sort_result_descending(const bool sort_result_descending) {
        m_sort_result_descending = sort_result_descending;
    }

protected:
    BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
    bool m_sort_result_descending = true;

private:
    int64_t max_boxes_output_from_input() const;
};
}  // namespace v1

namespace v3 {
/// \brief NonMaxSuppression operation
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NonMaxSuppression : public Op {
public:
    enum class BoxEncodingType { CORNER, CENTER };

    OPENVINO_OP("NonMaxSuppression", "opset3", op::Op);
    NonMaxSuppression() = default;

    /// \brief Constructs a NonMaxSuppression operation.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default values for the last
    ///        3 inputs
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box coordinates
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    BoxEncodingType get_box_encoding() const {
        return m_box_encoding;
    }
    void set_box_encoding(const BoxEncodingType box_encoding) {
        m_box_encoding = box_encoding;
    }
    bool get_sort_result_descending() const {
        return m_sort_result_descending;
    }
    void set_sort_result_descending(const bool sort_result_descending) {
        m_sort_result_descending = sort_result_descending;
    }

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(const element::Type& output_type) {
        m_output_type = output_type;
    }
    using Node::set_output_type;

protected:
    BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
    bool m_sort_result_descending = true;
    ov::element::Type m_output_type = ov::element::i64;
    int64_t max_boxes_output_from_input() const;
};
}  // namespace v3

namespace v4 {
/// \brief NonMaxSuppression operation
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NonMaxSuppression : public op::v3::NonMaxSuppression {
public:
    OPENVINO_OP("NonMaxSuppression", "opset4", op::v3::NonMaxSuppression);
    NonMaxSuppression() = default;

    /// \brief Constructs a NonMaxSuppression operation.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default values for the last
    ///        3 inputs
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box coordinates
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v4

namespace v5 {
/// \brief NonMaxSuppression operation
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NonMaxSuppression : public Op {
public:
    OPENVINO_OP("NonMaxSuppression", "opset5", op::Op);
    enum class BoxEncodingType { CORNER, CENTER };

    NonMaxSuppression() = default;

    /// \brief Constructs a NonMaxSuppression operation with default values in the last
    ///        4 inputs.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default values in the last.
    ///        3 inputs.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default values in the last.
    ///        2 inputs.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default value in the last.
    ///        input.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param soft_nms_sigma Node specifying the sigma parameter for Soft-NMS
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const Output<Node>& soft_nms_sigma,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    BoxEncodingType get_box_encoding() const {
        return m_box_encoding;
    }
    void set_box_encoding(const BoxEncodingType box_encoding) {
        m_box_encoding = box_encoding;
    }
    bool get_sort_result_descending() const {
        return m_sort_result_descending;
    }
    void set_sort_result_descending(const bool sort_result_descending) {
        m_sort_result_descending = sort_result_descending;
    }

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(const element::Type& output_type) {
        m_output_type = output_type;
    }
    using Node::set_output_type;

    int64_t max_boxes_output_from_input() const;
    float iou_threshold_from_input() const;
    float score_threshold_from_input() const;
    float soft_nms_sigma_from_input() const;
    bool is_soft_nms_sigma_constant_and_default() const;

protected:
    BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
    bool m_sort_result_descending = true;
    ov::element::Type m_output_type = ov::element::i64;
};
}  // namespace v5

namespace v9 {
/// \brief NonMaxSuppression operation
///
class OPENVINO_API NonMaxSuppression : public Op {
public:
    OPENVINO_OP("NonMaxSuppression", "opset9", op::Op);
    enum class BoxEncodingType { CORNER, CENTER };

    NonMaxSuppression() = default;

    /// \brief Constructs a NonMaxSuppression operation with default values in the last
    ///        4 inputs.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default values in the last
    ///        3 inputs.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default values in the last.
    ///        2 inputs.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation with default value in the last.
    ///        input.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    /// \brief Constructs a NonMaxSuppression operation.
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
    /// selected per class
    /// \param iou_threshold Node producing intersection over union threshold
    /// \param score_threshold Node producing minimum score threshold
    /// \param soft_nms_sigma Node specifying the sigma parameter for Soft-NMS
    /// \param box_encoding Specifies the format of boxes data encoding
    /// \param sort_result_descending Specifies whether it is necessary to sort selected
    /// boxes across batches
    /// \param output_type Specifies the output tensor type
    NonMaxSuppression(const Output<Node>& boxes,
                      const Output<Node>& scores,
                      const Output<Node>& max_output_boxes_per_class,
                      const Output<Node>& iou_threshold,
                      const Output<Node>& score_threshold,
                      const Output<Node>& soft_nms_sigma,
                      const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                      const bool sort_result_descending = true,
                      const ov::element::Type& output_type = ov::element::i64);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    BoxEncodingType get_box_encoding() const {
        return m_box_encoding;
    }
    void set_box_encoding(const BoxEncodingType box_encoding) {
        m_box_encoding = box_encoding;
    }

    bool get_sort_result_descending() const {
        return m_sort_result_descending;
    }
    void set_sort_result_descending(const bool sort_result_descending) {
        m_sort_result_descending = sort_result_descending;
    }

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(const element::Type& output_type) {
        m_output_type = output_type;
    }

    using Node::set_output_type;

    int64_t max_boxes_output_from_input() const;
    float iou_threshold_from_input() const;
    float score_threshold_from_input() const;
    float soft_nms_sigma_from_input() const;
    bool is_soft_nms_sigma_constant_and_default() const;

protected:
    BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
    bool m_sort_result_descending = true;
    ov::element::Type m_output_type = ov::element::i64;
};
}  // namespace v9
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v1::NonMaxSuppression::BoxEncodingType& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v3::NonMaxSuppression::BoxEncodingType& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v5::NonMaxSuppression::BoxEncodingType& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v9::NonMaxSuppression::BoxEncodingType& type);

template <>
class OPENVINO_API AttributeAdapter<op::v1::NonMaxSuppression::BoxEncodingType>
    : public EnumAttributeAdapterBase<op::v1::NonMaxSuppression::BoxEncodingType> {
public:
    AttributeAdapter(op::v1::NonMaxSuppression::BoxEncodingType& value)
        : EnumAttributeAdapterBase<op::v1::NonMaxSuppression::BoxEncodingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::v1::NonMaxSuppression::BoxEncodingType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::v3::NonMaxSuppression::BoxEncodingType>
    : public EnumAttributeAdapterBase<op::v3::NonMaxSuppression::BoxEncodingType> {
public:
    AttributeAdapter(op::v3::NonMaxSuppression::BoxEncodingType& value)
        : EnumAttributeAdapterBase<op::v3::NonMaxSuppression::BoxEncodingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::v3::NonMaxSuppression::BoxEncodingType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::v5::NonMaxSuppression::BoxEncodingType>
    : public EnumAttributeAdapterBase<op::v5::NonMaxSuppression::BoxEncodingType> {
public:
    AttributeAdapter(op::v5::NonMaxSuppression::BoxEncodingType& value)
        : EnumAttributeAdapterBase<op::v5::NonMaxSuppression::BoxEncodingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::v5::NonMaxSuppression::BoxEncodingType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::v9::NonMaxSuppression::BoxEncodingType>
    : public EnumAttributeAdapterBase<op::v9::NonMaxSuppression::BoxEncodingType> {
public:
    AttributeAdapter(op::v9::NonMaxSuppression::BoxEncodingType& value)
        : EnumAttributeAdapterBase<op::v9::NonMaxSuppression::BoxEncodingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::v9::NonMaxSuppression::BoxEncodingType>");
};

}  // namespace ov
