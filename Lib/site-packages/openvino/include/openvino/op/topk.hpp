// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/topk_base.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Computes indices and values of the k maximum/minimum values
///        for each slice along specified axis.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API TopK : public util::TopKBase {
public:
    OPENVINO_OP("TopK", "opset1", op::util::TopKBase);

    using SortType = TopKSortType;
    using Mode = TopKMode;

    /// \brief Constructs a TopK operation
    TopK() = default;
    /// \brief Constructs a TopK operation with two outputs: values and indices.
    ///        By default the indices output is described by i32 data type.
    ///
    /// \param data The input tensor
    /// \param k Specifies how many maximum/minimum elements should be computed
    ///          (note: scalar input tensor)
    /// \param axis The axis along which to compute top k indices
    /// \param mode Specifies which operation (min or max) is used to select
    ///             the biggest element of two.
    /// \param sort Specifies order of output elements and/or indices
    ///             Accepted values: none, index, value
    /// \param index_element_type Specifies type of produced indices
    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const std::string& mode,
         const std::string& sort,
         const element::Type& index_element_type = element::i32);

    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const Mode mode,
         const SortType sort,
         const element::Type& index_element_type = element::i32);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

protected:
    virtual void k_type_check(const element::Type& k_element_type) const override;
};
}  // namespace v1

namespace v3 {
/// \brief Computes indices and values of the k maximum/minimum values
///        for each slice along specified axis.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API TopK : public util::TopKBase {
public:
    OPENVINO_OP("TopK", "opset3", op::util::TopKBase);
    /// \brief Constructs a TopK operation
    TopK() = default;
    /// \brief Constructs a TopK operation with two outputs: values and indices.
    ///        By default the indices output is described by i32 data type.
    ///
    /// \param data The input tensor
    /// \param k Specifies how many maximum/minimum elements should be computed
    ///          (note: scalar input tensor)
    /// \param axis The axis along which to compute top k indices
    /// \param mode Specifies which operation (min or max) is used to select
    ///             the biggest element of two.
    /// \param sort Specifies order of output elements and/or indices
    ///             Accepted values: none, index, value
    /// \param index_element_type Specifies type of produced indices
    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const std::string& mode,
         const std::string& sort,
         const element::Type& index_element_type = element::i32);

    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const TopKMode mode,
         const TopKSortType sort,
         const element::Type& index_element_type = element::i32);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v3

namespace v11 {
/// \brief Computes the top K elements of a given tensor along the specified axis.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API TopK : public util::TopKBase {
public:
    OPENVINO_OP("TopK", "opset11", op::util::TopKBase);
    /// \brief Constructs a TopK operation
    TopK() = default;
    /// \brief Constructs a TopK operation with two outputs: values and indices.
    ///
    /// \param data The input tensor
    /// \param k Specifies how many maximum/minimum elements should be computed
    /// \param axis The axis along which the TopK operation should be executed
    /// \param mode Specifies whether TopK selects the largest or the smallest elements from each slice
    /// \param sort Specifies the order of corresponding elements of the output tensor
    /// \param index_element_type Specifies the data type of the elements in the 'indices' output tensor.
    /// \param stable Specifies whether the equivalent elements should maintain their relative order
    ///               from the input tensor during sorting.
    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const std::string& mode,
         const std::string& sort,
         const element::Type& index_element_type = element::i32,
         const bool stable = false);

    /// \brief Constructs a TopK operation with two outputs: values and indices.
    ///
    /// \param data The input tensor
    /// \param k Specifies how many maximum/minimum elements should be computed
    /// \param axis The axis along which the TopK operation should be executed
    /// \param mode Specifies whether TopK selects the largest or the smallest elements from each slice
    /// \param sort Specifies the order of corresponding elements of the output tensor
    /// \param index_element_type Specifies the data type of the elements in the 'indices' output tensor.
    /// \param stable Specifies whether the equivalent elements should maintain their relative order
    ///               from the input tensor during sorting.
    TopK(const Output<Node>& data,
         const Output<Node>& k,
         const int64_t axis,
         const TopKMode mode,
         const TopKSortType sort,
         const element::Type& index_element_type = element::i32,
         const bool stable = false);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    bool get_stable() const {
        return m_stable;
    }

    void set_stable(const bool stable) {
        m_stable = stable;
    }

private:
    bool m_stable = false;
};
}  // namespace v11
}  // namespace op
}  // namespace ov
