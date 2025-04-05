// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {
class OPENVINO_API WrapType : public Pattern {
public:
    OPENVINO_RTTI("patternAnyType");

    explicit WrapType(
        NodeTypeInfo wrapped_type,
        const ValuePredicate& pred =
            [](const Output<Node>& output) {
                return true;
            },
        const OutputVector& input_values = {})
        : Pattern(input_values, pred),
          m_wrapped_types({wrapped_type}) {
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    explicit WrapType(
        std::vector<NodeTypeInfo> wrapped_types,
        const ValuePredicate& pred =
            [](const Output<Node>& output) {
                return true;
            },
        const OutputVector& input_values = {})
        : Pattern(input_values, pred),
          m_wrapped_types(std::move(wrapped_types)) {
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;

    NodeTypeInfo get_wrapped_type() const;

    const std::vector<NodeTypeInfo>& get_wrapped_types() const;
    std::ostream& write_type_description(std::ostream& out) const override;

private:
    std::vector<NodeTypeInfo> m_wrapped_types;
};
}  // namespace op

template <class T>
void collect_wrap_info(std::vector<DiscreteTypeInfo>& info) {
    info.emplace_back(T::get_type_info_static());
}

template <class T, class... Targs, typename std::enable_if<sizeof...(Targs) != 0, bool>::type = true>
void collect_wrap_info(std::vector<DiscreteTypeInfo>& info) {
    collect_wrap_info<T>(info);
    collect_wrap_info<Targs...>(info);
}

template <class... Args>
std::shared_ptr<Node> wrap_type(const OutputVector& inputs, const pattern::op::ValuePredicate& pred) {
    std::vector<DiscreteTypeInfo> info;
    collect_wrap_info<Args...>(info);
    return std::make_shared<op::WrapType>(info, pred, inputs);
}

template <class... Args>
std::shared_ptr<Node> wrap_type(const OutputVector& inputs = {}) {
    return wrap_type<Args...>(inputs, [](const Output<Node>& output) {
        return true;
    });
}

template <class... Args>
std::shared_ptr<Node> wrap_type(const pattern::op::ValuePredicate& pred) {
    return wrap_type<Args...>({}, pred);
}
}  // namespace pattern
}  // namespace pass
}  // namespace ov
