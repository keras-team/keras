// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v0 {
class Constant;
}

namespace util {
class OPENVINO_API TopKBase : public Op {
public:
    using Mode = TopKMode;
    using SortType = TopKSortType;

    OPENVINO_OP("TopKBase", "util");
    TopKBase() = default;

    /// \brief The common base class for all TopK operator versions
    ///
    /// \param data The input tensor
    /// \param k Specifies how many maximum/minimum elements should be computed
    /// \param axis The axis along which TopK should be computed
    /// \param mode Specifies whether the maximum or minimum elements are selected
    /// \param sort Specifies the order of output elements and/or indices
    ///             Accepted values: none, index, value
    /// \param index_element_type Specifies the type of produced indices
    TopKBase(const Output<Node>& data,
             const Output<Node>& k,
             const int64_t axis,
             const std::string& mode,
             const std::string& sort,
             const element::Type& index_element_type = element::i32);

    TopKBase(const Output<Node>& data,
             const Output<Node>& k,
             const int64_t axis,
             const TopKMode mode,
             const TopKSortType sort,
             const element::Type& index_element_type = element::i32);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    /// \brief Returns axis value after normalization
    /// \note If input rank required to normalization is dynamic, the exception is
    /// thrown
    uint64_t get_axis() const;
    /// \brief Returns axis value before normalization
    int64_t get_provided_axis() const {
        return m_axis;
    }
    void set_axis(const int64_t axis);
    void set_axis(const Rank& input_rank, const int64_t axis);
    TopKMode get_mode() const {
        return m_mode;
    }
    void set_mode(const TopKMode mode) {
        m_mode = mode;
    }
    TopKSortType get_sort_type() const {
        return m_sort;
    }
    void set_sort_type(const TopKSortType sort) {
        m_sort = sort;
    }
    element::Type get_index_element_type() const {
        return m_index_element_type;
    }
    void set_index_element_type(const element::Type& index_element_type) {
        m_index_element_type = index_element_type;
    }
    /// \brief Returns the value of K, if available
    ///
    /// \note If the second input to this op is a constant, the value is retrieved
    ///       and returned. If the input is not constant(dynamic) this method returns 0
    size_t get_k() const;
    void set_k(size_t k);
    size_t get_default_output_index() const override {
        return no_default_index();
    }

protected:
    int64_t m_axis;
    uint64_t m_normalized_axis;
    TopKMode m_mode;
    TopKSortType m_sort;
    element::Type m_index_element_type{element::i32};

    virtual void k_type_check(const element::Type& k_element_type) const;
    size_t read_k_from_constant_node(const std::shared_ptr<Node>& node, const element::Type& k_element_type) const;
    template <typename T>
    size_t validate_and_get_k(const std::shared_ptr<op::v0::Constant>& k_constant) const;
};
}  // namespace util
}  // namespace op
}  // namespace ov
