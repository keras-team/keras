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
/// The graph value is to the matched value list. If the predicate is true for the node
/// and the arguments match, the match succeeds.
class OPENVINO_API Any : public Pattern {
public:
    OPENVINO_RTTI("patternAny");
    /// \brief creates a Any node containing a sub-pattern described by \sa type and \sa
    ///        shape.
    Any(const element::Type& type, const PartialShape& s, ValuePredicate pred, const OutputVector& wrapped_values)
        : Pattern(wrapped_values, pred) {
        set_output_type(0, type, s);
    }
    Any(const element::Type& type, const PartialShape& s, NodePredicate pred, const NodeVector& wrapped_values)
        : Any(type, s, as_value_predicate(pred), as_output_vector(wrapped_values)) {}
    /// \brief creates a Any node containing a sub-pattern described by the type and
    ///        shape of \sa node.
    Any(const Output<Node>& node, ValuePredicate pred, const OutputVector& wrapped_values)
        : Any(node.get_element_type(), node.get_partial_shape(), pred, wrapped_values) {}
    Any(const Output<Node>& node, NodePredicate pred, const NodeVector& wrapped_values)
        : Any(node.get_element_type(),
              node.get_partial_shape(),
              as_value_predicate(pred),
              as_output_vector(wrapped_values)) {}

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;
};
}  // namespace op
}  // namespace pattern
}  // namespace pass
}  // namespace ov
