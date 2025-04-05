// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <set>

#include "openvino/core/rtti.hpp"
#include "openvino/pass/matcher_pass.hpp"

#define _OPENVINO_GRAPH_REWRITE_RTTI_WITH_TYPE(TYPE_NAME) _OPENVINO_GRAPH_REWRITE_RTTI_WITH_TYPE_VERSION(TYPE_NAME, "0")

#define _OPENVINO_GRAPH_REWRITE_RTTI_WITH_TYPE_VERSION(TYPE_NAME, VERSION_NAME) \
    _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT(TYPE_NAME, VERSION_NAME, ::ov::pass::GraphRewrite)

#define OPENVINO_GRAPH_REWRITE_RTTI(...)                                                                       \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR_2(__VA_ARGS__,                                    \
                                                               _OPENVINO_GRAPH_REWRITE_RTTI_WITH_TYPE_VERSION, \
                                                               _OPENVINO_GRAPH_REWRITE_RTTI_WITH_TYPE)(__VA_ARGS__))

namespace ov {
namespace pass {
/// \brief GraphRewrite is a container for MatcherPasses that allows to run them on Function
/// in
/// efficient way
///
/// Graph rewrite pass is used for matcher passes execution on Function.
/// To register MatcherPass use \sa add_matcher<T>(args) method where T is a MatcherPass
/// class.
/// As a default algorithm graph rewrite pass traverse Function in topological order and
/// applies
/// registered matcher passes for each node. But if all registered matcher passes have type
/// based
/// root node in Matcher pattern then efficient mechanism is used to execute them.
/// Matcher pattern root is type based if it's operation from opset or
/// pattern::op::WrapType.
/// Note: when implementing pattern for Matcher make sure that root node is an operation
/// from opset
/// or has ov::pattern::op::WrapType. That will help GraphRewrite to execute matcher
/// passes more
/// efficient.
/// \ingroup ov_pass_cpp_api
class OPENVINO_API GraphRewrite : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::pass::GraphRewrite");

    GraphRewrite() = default;

    explicit GraphRewrite(const std::shared_ptr<MatcherPass>& pass) : ModelPass() {
        m_matchers.push_back(pass);
    }

    /// \brief Register given transformation class type to GraphRewrite execution list
    /// All registered transformations will be executed in a single graph traversal.
    /// Example below show the basic usage of pass::GraphRewrite
    ///
    ///     pass::Manager manager;
    ///     auto anchor = manager.register_pass<GraphRewrite>();
    ///     anchor->add_matcher<MatcherPassA>();
    ///     anchor->add_matcher<MatcherPassB>();
    ///     anchor->set_name("CommonMatchers");
    ///     manager.run_passes(f);
    ///
    /// For some purposes transformation can be registered and disabled by default.
    ///
    ///     anchor->add_matcher<MatcherPassB, false>();
    ///
    /// \return shared_ptr to the transformation instance
    template <typename T,
              bool Enabled = true,
              class... Args,
              typename std::enable_if<std::is_base_of<pass::MatcherPass, T>::value, bool>::type = true>
    std::shared_ptr<T> add_matcher(Args&&... args) {
        static_assert(std::is_base_of<pass::MatcherPass, T>::value, "pass not derived from MatcherPass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_config = get_pass_config();
        pass->set_pass_config(pass_config);
        if (!Enabled && !pass_config->is_enabled<T>()) {
            pass_config->disable<T>();
        }
        m_matchers.push_back(pass);
        return pass;
    }

    /// \brief Register passes from GraphRewrite class that contains sequence of matcher
    /// passes registered in its ctor.
    /// For example:
    ///
    ///    class ov::pass::LinFusions: public ov::pass::GraphRewrite {
    ///    public:
    ///         OPENVINO_GRAPH_REWRITE_RTTI("LinFusion");
    ///         Fusions() {
    ///             add_matcher<ov::pass::AddFusion>();
    ///             add_matcher<ov::pass::MulFusion>();
    ///         }
    ///     };
    ///
    ///     pass::Manager manager;
    ///     auto anchor = manager.register_pass<GraphRewrite>();
    ///     anchor->add_matcher<LinFusions>();
    ///     anchor->add_matcher<OtherFusions>();
    ///     anchor->set_name("CommonFusions");
    ///     manager.run_passes(f);
    ///
    /// In this case all matcher passes from LinFusions pass will be united with other
    /// registered matchers.
    template <typename T,
              class... Args,
              typename std::enable_if<std::is_base_of<pass::GraphRewrite, T>::value, bool>::type = true>
    void add_matcher(Args&&... args) {
        static_assert(std::is_base_of<pass::GraphRewrite, T>::value, "pass not derived from GraphRewrite");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_config = get_pass_config();

        for (auto& matcher : pass->m_matchers) {
            pass->set_pass_config(pass_config);
            m_matchers.push_back(matcher);
        }
    }

    std::shared_ptr<MatcherPass> add_matcher(const std::shared_ptr<MatcherPass>& pass);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    void set_pass_config(const std::shared_ptr<PassConfig>& pass_config) override;

protected:
    bool apply_matcher_passes(std::shared_ptr<Model> f, std::deque<std::weak_ptr<Node>> nodes_to_run);

    bool m_enable_shape_inference = false;

    std::vector<std::shared_ptr<ov::pass::MatcherPass>> m_matchers;
};
}  // namespace pass
}  // namespace ov
