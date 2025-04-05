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
/// The graph value is added to the matched values list. If the predicate is true for
/// the
/// graph node, a submatch is performed on the input of AnyOf and each input of the
/// graph node. The first match that succeeds results in a successful match. Otherwise
/// the match fails.
///
/// AnyOf may be given a type and shape for use in strict mode.
class OPENVINO_API AnyOf : public Pattern {
public:
    OPENVINO_RTTI("patternAnyOf");
    /// \brief creates a AnyOf node containing a sub-pattern described by \sa type and
    ///        \sa shape.
    AnyOf(const element::Type& type, const PartialShape& s, ValuePredicate pred, const OutputVector& wrapped_values)
        : Pattern(wrapped_values, pred) {
        if (wrapped_values.size() != 1) {
            OPENVINO_THROW("AnyOf expects exactly one argument");
        }
        set_output_type(0, type, s);
    }
    AnyOf(const element::Type& type, const PartialShape& s, NodePredicate pred, const NodeVector& wrapped_values)
        : AnyOf(
              type,
              s,
              [pred](const Output<Node>& value) {
                  return pred(value.get_node_shared_ptr());
              },
              as_output_vector(wrapped_values)) {}

    /// \brief creates a AnyOf node containing a sub-pattern described by the type and
    ///        shape of \sa node.
    AnyOf(const Output<Node>& node, ValuePredicate pred, const OutputVector& wrapped_values)
        : AnyOf(node.get_element_type(), node.get_partial_shape(), pred, wrapped_values) {}
    AnyOf(const std::shared_ptr<Node>& node, NodePredicate pred, const NodeVector& wrapped_values)
        : AnyOf(node, as_value_predicate(pred), as_output_vector(wrapped_values)) {}
    bool match_value(Matcher* matcher, const Output<Node>& pattern_value, const Output<Node>& graph_value) override;
};
}  // namespace op
}  // namespace pattern
}  // namespace pass
}  // namespace ov
