// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/rtti.hpp"
#include "openvino/pass/node_registry.hpp"

#define _OPENVINO_MATCHER_PASS_RTTI_WITH_TYPE(TYPE_NAME) _OPENVINO_MATCHER_PASS_RTTI_WITH_TYPE_VERSION(TYPE_NAME, "0")

#define _OPENVINO_MATCHER_PASS_RTTI_WITH_TYPE_VERSION(TYPE_NAME, VERSION_NAME) \
    _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT(TYPE_NAME, VERSION_NAME, ::ov::pass::MatcherPass)

#define OPENVINO_MATCHER_PASS_RTTI(...)                                                                       \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR_2(__VA_ARGS__,                                   \
                                                               _OPENVINO_MATCHER_PASS_RTTI_WITH_TYPE_VERSION, \
                                                               _OPENVINO_MATCHER_PASS_RTTI_WITH_TYPE)(__VA_ARGS__))

namespace ov {
using matcher_pass_callback = std::function<bool(pass::pattern::Matcher& m)>;
using graph_rewrite_callback = std::function<bool(pass::pattern::Matcher& m)>;
using handler_callback = std::function<bool(const std::shared_ptr<Node>& node)>;
namespace pass {
/// \brief MatcherPass is a basic block for pattern based transformations. It describes
/// pattern and
/// action that is applied if pattern is matched.
///
/// MatcherPass consists of Matcher and matcher_pass_callback that needs to be implemented
/// and
/// finally registered by using \sa register_matcher. MatcherPass can be executed on node
/// within
/// \sa apply method. To run matcher pass on Function use GraphRewrite.
/// In addition MatcherPass provides a way for adding new operations into GraphRewrite
/// execution
/// queue. That means that operations that were created inside transformation callback can
/// be added
/// for matching. To register node use \sa register_new_node method. GraphRewrite
/// automatically
/// takes registered nodes and put them to execution queue. If multiple nodes were register
/// make
/// sure that they were registered in topological order.
/// Note: when implementing pattern for Matcher make sure that root node is an operation
/// from opset
/// or has ov::pass::pattern::op::WrapType. That will help GraphRewrite to execute matcher
/// passes more
/// efficient.
/// \ingroup ov_pass_cpp_api
class OPENVINO_API MatcherPass : public PassBase {
public:
    OPENVINO_RTTI("ov::pass::MatcherPass");

    MatcherPass() = default;

    MatcherPass(const MatcherPass&) = delete;
    MatcherPass& operator=(const MatcherPass&) = delete;

    explicit MatcherPass(const std::string& name,
                         const std::shared_ptr<pattern::Matcher>& m,
                         const handler_callback& handler,
                         const PassPropertyMask& property = PassProperty::CHANGE_DYNAMIC_STATE)
        : PassBase(),
          m_handler(handler),
          m_matcher(m) {
        set_name(name);
        set_property(property, true);
    }

    MatcherPass(const std::shared_ptr<pattern::Matcher>& m, const matcher_pass_callback& callback) : PassBase() {
        register_matcher(m, callback);
    }

    bool apply(std::shared_ptr<ov::Node> node);

    template <typename T, class... Args>
    std::shared_ptr<T> register_new_node(Args&&... args) {
        return m_new_nodes.make<T>(std::forward<Args>(args)...);
    }

    template <typename T>
    std::shared_ptr<T> register_new_node(const std::shared_ptr<T>& node) {
        return m_new_nodes.add(node);
    }

    std::shared_ptr<ov::Node> register_new_node_(const std::shared_ptr<ov::Node>& node) {
        return register_new_node(node);
    }

    const std::vector<std::shared_ptr<ov::Node>>& get_new_nodes() {
        return m_new_nodes.get();
    }

    void clear_new_nodes() {
        m_new_nodes.clear();
    }

    std::shared_ptr<pattern::Matcher> get_matcher() {
        return m_matcher;
    }

protected:
    void register_matcher(const std::shared_ptr<pattern::Matcher>& m,
                          const matcher_pass_callback& callback,
                          const PassPropertyMask& property);

    void register_matcher(const std::shared_ptr<pattern::Matcher>& m, const matcher_pass_callback& callback);

private:
    handler_callback m_handler;
    std::shared_ptr<pattern::Matcher> m_matcher;
    NodeRegistry m_new_nodes;
};
}  // namespace pass
}  // namespace ov
