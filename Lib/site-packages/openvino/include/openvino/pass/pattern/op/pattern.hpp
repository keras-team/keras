// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "openvino/core/node.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {
class Label;
}

class Matcher;
class MatcherState;

using RPatternValueMap = std::map<std::shared_ptr<Node>, OutputVector>;
using PatternValueMap = std::map<std::shared_ptr<Node>, Output<Node>>;
using PatternValueMaps = std::vector<PatternValueMap>;

using PatternMap = std::map<std::shared_ptr<Node>, std::shared_ptr<Node>>;

PatternMap as_pattern_map(const PatternValueMap& pattern_value_map);
PatternValueMap as_pattern_value_map(const PatternMap& pattern_map);

template <typename T>
std::function<bool(std::shared_ptr<Node>)> has_class() {
    auto pred = [](std::shared_ptr<Node> node) -> bool {
        return ov::is_type<T>(std::move(node));
    };

    return pred;
}
template <typename T>
std::function<bool(std::shared_ptr<Node>)> class_other_than() {
    auto pred = [](std::shared_ptr<Node> node) -> bool {
        return !ov::is_type<T>(std::move(node));
    };

    return pred;
}

OPENVINO_API
std::function<bool(Output<Node>)> consumers_count(size_t n);

OPENVINO_API
std::function<bool(Output<Node>)> consumers_more_than(size_t n);

OPENVINO_API
std::function<bool(Output<Node>)> has_static_dim(size_t pos);

OPENVINO_API
std::function<bool(Output<Node>)> has_static_dims(const std::vector<size_t>& dims);

OPENVINO_API
std::function<bool(Output<Node>)> has_static_shape();

OPENVINO_API
std::function<bool(Output<Node>)> has_static_rank();

OPENVINO_API
std::function<bool(Output<Node>)> rank_equals(const Dimension& expected_rank);

OPENVINO_API
std::function<bool(Output<Node>)> type_matches(const element::Type& type);

OPENVINO_API
std::function<bool(Output<Node>)> type_matches_any(const std::vector<element::Type>& types);

OPENVINO_API
std::function<bool(Output<Node>)> all_of(const std::vector<std::function<bool(Output<Node>)>>& predicates);

namespace op {
using NodePredicate = std::function<bool(std::shared_ptr<Node>)>;
using ValuePredicate = std::function<bool(const Output<Node>& value)>;

OPENVINO_API
ValuePredicate as_value_predicate(NodePredicate pred);

class OPENVINO_API Pattern : public Node {
public:
    /// \brief \p a base class for \sa Skip and \sa Label
    ///
    Pattern(const OutputVector& patterns, ValuePredicate pred);

    Pattern(const OutputVector& patterns) : Pattern(patterns, nullptr) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& /* new_args */) const override {
        OPENVINO_THROW("Uncopyable");
    }

    ValuePredicate get_predicate() const;

    std::ostream& write_description(std::ostream& out, uint32_t depth) const override;
    virtual std::ostream& write_type_description(std::ostream& out) const;

protected:
    ValuePredicate m_predicate;
};
}  // namespace op
}  // namespace pattern
}  // namespace pass
OPENVINO_API pass::pattern::op::ValuePredicate operator||(const std::function<bool(Output<Node>)>& a,
                                                          const std::function<bool(Output<Node>)>& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator||(const pass::pattern::op::ValuePredicate& a,
                                                          const pass::pattern::op::ValuePredicate& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator||(const pass::pattern::op::NodePredicate& a,
                                                          const pass::pattern::op::NodePredicate& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator||(const pass::pattern::op::ValuePredicate& a,
                                                          const pass::pattern::op::NodePredicate& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator||(const pass::pattern::op::NodePredicate& a,
                                                          const pass::pattern::op::ValuePredicate& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator&&(const std::function<bool(Output<Node>)>& a,
                                                          const std::function<bool(Output<Node>)>& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator&&(const pass::pattern::op::ValuePredicate& a,
                                                          const pass::pattern::op::ValuePredicate& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator&&(const pass::pattern::op::NodePredicate& a,
                                                          const pass::pattern::op::NodePredicate& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator&&(const pass::pattern::op::ValuePredicate& a,
                                                          const pass::pattern::op::NodePredicate& b);
OPENVINO_API pass::pattern::op::ValuePredicate operator&&(const pass::pattern::op::NodePredicate& a,
                                                          const pass::pattern::op::ValuePredicate& b);
}  // namespace ov
