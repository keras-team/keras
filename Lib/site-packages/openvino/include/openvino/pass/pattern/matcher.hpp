// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>

#include <algorithm>
#include <functional>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/any_of.hpp"
#include "openvino/pass/pattern/op/any_output.hpp"
#include "openvino/pass/pattern/op/label.hpp"

namespace ov {
namespace pass {
class GraphRewrite;

namespace pattern {
class Matcher;

class OPENVINO_API MatcherState {
public:
    MatcherState(Matcher*);
    bool finish(bool is_successful);
    ~MatcherState();

protected:
    Matcher* m_matcher;
    PatternValueMap m_pattern_value_map;
    PatternValueMaps m_pattern_value_maps;
    size_t m_watermark;
    size_t m_capture_size;
    bool m_restore{true};
};

/// Matcher looks for node patterns in a computation graph. The patterns are described by an
/// automaton that is described by an extended computation graph. The matcher executes
/// by attempting to match the start node of the pattern to a computation graph value
/// (output of a Node). In addition to determing if a match occurs, a pattern node may add
/// graph nodes to a list of matched nodes, associate nodes with graph values, and start
/// submatches. Submatches add match state changes to the enclosing match if the submatch
/// succeeds; otherwise the state is reverted.
///
/// The default match behavior of a pattern node with a graph nodes is that the computation
/// graph value is added to the end of the matched value list and the match succeeds if the
/// node/pattern types match and the input values match. In the case of a commutative node,
/// the inputs can match in any order. If the matcher is in strict mode, the graph value
/// element type and shape must also match.
///
/// Pattern nodes that have different match behavior are in ov::pass::pattern::op and have
/// descriptions of their match behavior.
class OPENVINO_API Matcher {
public:
    using PatternMap = ov::pass::pattern::PatternMap;

    // Avoid implicit string construction from nullptr.
    Matcher(const std::shared_ptr<Node> pattern_node, std::nullptr_t name) = delete;

    Matcher()
        : m_match_root{},
          m_pattern_node{},
          m_pattern_map{},
          m_pattern_value_maps{},
          m_matched_list{},
          m_name{""},
          m_strict_mode{false} {}
    Matcher(Output<Node>& pattern_node)
        : m_match_root{},
          m_pattern_node{pattern_node},
          m_pattern_map{},
          m_pattern_value_maps{},
          m_matched_list{},
          m_name{""},
          m_strict_mode{false} {}

    Matcher(Output<Node>& pattern_node, const std::string& name)
        : m_match_root{},
          m_pattern_node{pattern_node},
          m_pattern_map{},
          m_pattern_value_maps{},
          m_matched_list{},
          m_name{name},
          m_strict_mode{false} {}

    /// \brief Constructs a Matcher object
    ///
    /// \param pattern_node is a pattern sub graph that will be matched against input graphs
    /// \param name is a string which is used for logging and disabling a matcher
    /// \param strict_mode forces a matcher to consider shapes and ET of nodes
    Matcher(const Output<Node>& pattern_node, const std::string& name, bool strict_mode)
        : m_match_root{},
          m_pattern_node{pattern_node},
          m_pattern_map{},
          m_pattern_value_maps{},
          m_matched_list{},
          m_name{name},
          m_strict_mode{strict_mode} {}

    // Some matches should start on a node rather than an output. These three constructors
    // are transition until we work out the right way to do that.
    Matcher(std::shared_ptr<Node> pattern_node);
    Matcher(std::shared_ptr<Node> pattern_node, const std::string& name);
    Matcher(std::shared_ptr<Node> pattern_node, const std::string& name, bool strict_mode);

    virtual ~Matcher();

    /// \brief Matches a pattern to \p graph_node
    ///
    /// \param graph_value is an input graph to be matched against
    bool match(const Output<Node>& graph_value);

    bool match(std::shared_ptr<Node> graph_node);

    /// \brief Matches a pattern to \p graph_node
    ///
    /// \param graph_value is an input graph to be matched against
    /// \param previous_matches contains previous mappings from labels to nodes to use
    bool match(const Output<Node>& graph_value, const PatternMap& previous_matches);
    bool match(const Output<Node>& graph_value, const PatternValueMap& previous_matches);

    template <typename T>
    static std::shared_ptr<T> unique_match(const std::shared_ptr<Node>& node) {
        std::shared_ptr<T> matched;
        for (const auto& arg : node->input_values()) {
            if (auto t_casted = ov::as_type_ptr<T>(arg.get_node_shared_ptr())) {
                if (matched) {
                    OPENVINO_THROW("There's more than two arguments of the same type");
                } else {
                    matched = t_casted;
                }
            }
        }
        return matched;
    }

    bool is_contained_match(const NodeVector& exclusions = {}, bool ignore_unused = true);
    const NodeVector get_matched_nodes() {
        return as_node_vector(m_matched_list);
    }
    const OutputVector& get_matched_values() const {
        return m_matched_list;
    }
    OutputVector& get_matched_values() {
        return m_matched_list;
    }
    void reset() {}
    const std::string& get_name() {
        return m_name;
    }
    std::shared_ptr<Node> get_pattern() {
        return m_pattern_node.get_node_shared_ptr();
    }
    Output<Node> get_pattern_value() {
        return m_pattern_node;
    }
    std::shared_ptr<Node> get_match_root();
    Output<Node> get_match_value();
    PatternMap get_pattern_map() const;
    PatternValueMap& get_pattern_value_map() {
        return m_pattern_map;
    }
    PatternValueMaps& get_pattern_value_maps() {
        return m_pattern_value_maps;
    }
    /// \brief Low-level helper to match recurring patterns
    ///
    /// \param graph is a graph to be matched against
    /// \param pattern is a recurring pattern
    /// \param rpattern specifies a node to recur from next
    /// \param patterns a map from labels to matches

    size_t add_node(Output<Node> node);

    virtual bool match_value(const ov::Output<Node>& pattern_value, const ov::Output<Node>& graph_value);

    bool is_strict_mode() {
        return m_strict_mode;
    }
    virtual bool match_arguments(Node* pattern_node, const std::shared_ptr<Node>& graph_node);

    void capture(const std::set<Node*>& static_nodes);

    void clear_state();

    size_t get_number_of_recurrent_matches() const {
        return m_pattern_value_maps.size();
    }
    NodeVector get_bound_nodes_for_pattern(const Output<Node>& pattern) const;
    size_t get_number_of_bound_labels() const;
    /// \brief Try a match
    MatcherState start_match();

    Output<Node> m_match_root;
    Output<Node> m_pattern_node;
    PatternValueMap m_pattern_map;
    PatternValueMaps m_pattern_value_maps;
    OutputVector m_matched_list;

protected:
    bool match_permutation(const OutputVector& pattern_args, const OutputVector& args);

    std::string m_name{"unnamed"};
    bool m_strict_mode{false};
};

}  // namespace pattern
}  // namespace pass
}  // namespace ov
