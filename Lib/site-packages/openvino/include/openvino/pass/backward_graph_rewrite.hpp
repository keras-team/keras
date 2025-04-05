#pragma once

#include <functional>
#include <memory>
#include <set>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
class OPENVINO_API BackwardGraphRewrite : public GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::pass::BackwardGraphRewrite");

    BackwardGraphRewrite() = default;

    explicit BackwardGraphRewrite(const std::shared_ptr<MatcherPass>& pass) : GraphRewrite(pass) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
}  // namespace pass
}  // namespace ov
