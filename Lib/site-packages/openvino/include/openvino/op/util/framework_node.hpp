// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace ov {
namespace op {
namespace util {

class OPENVINO_API FrameworkNodeAttrs {
public:
    using attrs_t = std::unordered_map<std::string, std::string>;

    void set_opset_name(const std::string& opset_name) {
        m_opset_name = opset_name;
    }

    void set_type_name(const std::string& type_name) {
        m_type_name = type_name;
    }

    const std::string& get_opset_name() const {
        return m_opset_name;
    }

    const std::string& get_type_name() const {
        return m_type_name;
    }

    attrs_t::iterator begin() {
        return m_attrs.begin();
    }

    attrs_t::iterator end() {
        return m_attrs.end();
    }

    attrs_t::const_iterator begin() const {
        return m_attrs.begin();
    }

    attrs_t::const_iterator end() const {
        return m_attrs.end();
    }

    std::string& operator[](const std::string& key) {
        return m_attrs[key];
    }

    std::string at(const std::string& key) const {
        return m_attrs.at(key);
    }

    attrs_t::const_iterator find(const std::string& key) const {
        return m_attrs.find(key);
    }

    bool operator==(const FrameworkNodeAttrs& other) const {
        return m_type_name == other.m_type_name && m_opset_name == other.m_opset_name && m_attrs == other.m_attrs;
    }

private:
    std::string m_type_name;
    std::string m_opset_name;

    std::unordered_map<std::string, std::string> m_attrs;
};

class OPENVINO_API FrameworkNode : public MultiSubGraphOp {
public:
    OPENVINO_OP("FrameworkNode", "util", MultiSubGraphOp);

    FrameworkNode() = default;

    explicit FrameworkNode(const OutputVector& inputs, size_t output_size = 1, size_t num_subgraphs = 0);

    virtual void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    const FrameworkNodeAttrs& get_attrs() const {
        return m_attrs;
    }

    void set_attrs(const FrameworkNodeAttrs& attrs) {
        m_attrs = attrs;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void cache_output_descriptor();

protected:
    FrameworkNode(const FrameworkNode&);

private:
    void clone_to(FrameworkNode& dst) const;

    std::vector<std::tuple<ov::PartialShape, ov::element::Type>> m_inputs_desc;
    std::vector<std::tuple<ov::PartialShape, ov::element::Type>> m_output_desc;

    FrameworkNodeAttrs m_attrs;
    size_t m_num_bodies;
};
}  // namespace util
}  // namespace op
}  // namespace ov

namespace ov {

template <>
class OPENVINO_API AttributeAdapter<ov::op::util::FrameworkNodeAttrs>
    : public DirectValueAccessor<ov::op::util::FrameworkNodeAttrs> {
public:
    AttributeAdapter(ov::op::util::FrameworkNodeAttrs& value);

    OPENVINO_RTTI("AttributeAdapter<FrameworkNodeAttr>");
};

}  // namespace ov
