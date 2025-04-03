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
/// Matches any output of a node
class OPENVINO_API AnyOutput : public Pattern {
public:
    OPENVINO_RTTI("patternAnyOutput");
    /// \brief creates an AnyOutput node matching any output of a node
    /// \param node The node to match
    AnyOutput(const std::shared_ptr<Node>& pattern) : Pattern({pattern->output(0)}) {}

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;
};
}  // namespace op
}  // namespace pattern
}  // namespace pass
}  // namespace ov
