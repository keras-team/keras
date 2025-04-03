// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {

/// A submatch on the graph value which contains optional op types defined in constructor.
/// `Optional` pattern supports multi input operations. In this case the pattern checks
/// inputs with optional node type or 1st input.
/// The match is succeed in case of full graphs matching or extended by one of optional type graph or pattern.
/// Otherwise fails.
//
//  +------+   +------+                       +------+  +------+      +------+
//  | op_0 |   | op_1 |                       | op_0 |  | op_1 |      | op_0 |
//  +------+   +------+                       +------+  +------+      +------+
//     |           |                              |        |              |
//     V           V                              V        V              |
//  +-------------------+                    +---------------------+      |
//  | optional<op_type> |      =======>>>    | wrap_type<op_types> |      |
//  +-------------------+                    +---------------------+      |
//          |                                        |                    |
//          V                                        +------------------+ |
//       +------+                                                       | |
//       | op_3 |                                                       V V
//       +------+                                                    +--------+
//                                                                   |   Or   |
//                                                                   +--------+
//                                                                        |
//                                                                        V
//                                                                   +--------+
//                                                                   |  op_3  |
//                                                                   +--------+

// Known limitations:
// 1. The pattern matching does not support operations with optional inputs.
//    For example, ov::op::v5::NonMaxSupression can be created without some optional input nodes (like
//    `max_output_boxes_per_class`) (In case we would not specify input in constructor, the node input won't be created
//    by default as a constant). Arguments matching will be failed due to different number of pattern and graph input
//    args. Issue: 139835
// 2. The optional nodes with cumulative inputs will be matched by 1st input.
//    Issue: 139839
class OPENVINO_API Optional : public Pattern {
public:
    OPENVINO_RTTI("patternOptional");
    /// \brief creates an optional node matching one pattern. Add nodes to match list.
    /// \param type_infos Optional operation types to exclude them from the matching
    /// in case the following op types do not exist in a pattern to match.
    /// \param patterns The pattern to match a graph.
    Optional(
        const std::vector<DiscreteTypeInfo>& type_infos,
        const OutputVector& inputs = {},
        const pattern::op::ValuePredicate& pred =
            [](const Output<Node>& output) {
                return true;
            })
        : Pattern(inputs, pred),
          optional_types(type_infos){};

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;

    std::vector<DiscreteTypeInfo> get_optional_types() const;

protected:
    std::vector<DiscreteTypeInfo> optional_types;
};
}  // namespace op

template <class NodeType>
void collect_type_info(std::vector<DiscreteTypeInfo>& type_info_vec) {
    type_info_vec.push_back(NodeType::get_type_info_static());
}

template <class NodeType,
          class... NodeTypeArgs,
          typename std::enable_if<sizeof...(NodeTypeArgs) != 0, bool>::type = true>
void collect_type_info(std::vector<DiscreteTypeInfo>& type_info_vec) {
    collect_type_info<NodeType>(type_info_vec);
    collect_type_info<NodeTypeArgs...>(type_info_vec);
}

template <class... NodeTypes>
std::shared_ptr<Node> optional(const OutputVector& inputs, const pattern::op::ValuePredicate& pred = nullptr) {
    std::vector<DiscreteTypeInfo> optional_type_info_vec;
    collect_type_info<NodeTypes...>(optional_type_info_vec);
    return std::make_shared<op::Optional>(optional_type_info_vec, inputs, pred);
}

template <class... NodeTypes>
std::shared_ptr<Node> optional(const Output<Node>& input, const pattern::op::ValuePredicate& pred = nullptr) {
    return optional<NodeTypes...>(OutputVector{input}, pred);
}

template <class... NodeTypes>
std::shared_ptr<Node> optional(const pattern::op::ValuePredicate& pred = nullptr) {
    return optional<NodeTypes...>(OutputVector{}, pred);
}

}  // namespace pattern
}  // namespace pass
}  // namespace ov
