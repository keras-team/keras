// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/serialize.hpp"

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
#    include <filesystem>
#endif

namespace ov {

OPENVINO_API
void traverse_nodes(const std::shared_ptr<const Model>& p, const std::function<void(const std::shared_ptr<Node>&)>& f);

OPENVINO_API
void traverse_nodes(const Model* p, const std::function<void(const std::shared_ptr<Node>&)>& f);

/// \brief Visit each node in a sub-graph of the entire graph
/// \param subgraph_results The output nodes of the sub-graph
/// \param f Model to execute at each node in the traversal
/// \param subgraph_params Input nodes of the sub-graph (optional)
///
/// Traverses a sub-graph starting from subgraph_results moving up
/// towards parameter nodes. Traversal stops if it hits a node in
/// subgraph_params.
///
/// Most useful for finding parameters of a graph directly from the
/// result nodes and not from function parameters or extracting a
/// subgraph relevant to the computation of certain outputs
OPENVINO_API
void traverse_nodes(const NodeVector& subgraph_results,
                    const std::function<void(const std::shared_ptr<Node>&)>& f,
                    const NodeVector& subgraph_params = {});

/// \brief Replace the node `target` with the node `replacement`, i.e.,
///        redirect all users and control dependencies of `target` to
///        `replacement`.
///
/// \param target Node to be replaced.
/// \param replacement Node to replace `target` with.
/// \param output_order Vector determines order of replacement node's outputs.
///
/// This is primarily used in graph-rewriting passes. For example, we
/// might "fuse" two Concat operations as follows:
///
/// (Step 0: Original graph)
///
///   A                       B
///   |                       |
///   v                       v
/// N0[Concat, concatenation_axis=3]     C
///          |                           |
///          v                           v
///        N1[Concat, concatenation_axis=3]
///          |                |
///          v                v
///       some_user         another_user
///
/// (Step 1: Construct replacement)
///
///    shared_ptr<Node> new_N1 = make_shared<op::Concat>({A,B,C},3);
///
///   A----------------------------------------.
///   |                                        |
///   |                       B----------------)--.
///   |                       |                |  |
///   v                       v                |  |
/// N0[Concat, concatenation_axis=3]     C-----)--)--.
///          |                           |     |  |  |
///          v                           v     v  v  v
///        N1[Concat, concatenation_axis=3]   new_N1[Concat, concatenation_axis=3]
///          |                |
///          v                v
///       some_user         another_user
///
/// (Step 2: Replace N1 with new_N1)
///
///    replace_node(N1, new_N1);
///
///   A----------------------------------------.
///   |                                        |
///   |                       B----------------)--.
///   |                       |                |  |
///   v                       v                |  |
/// N0[Concat, concatenation_axis=3]     C-----)--)--.
///          |                           |     |  |  |
///          v                           v     v  v  v
///        N1[Concat, concatenation_axis=3]   new_N1[Concat, concatenation_axis=3]
///                                                  |                |
///                                                  v                v
///                                               some_user         another_user
///
/// (Step 3: N0 and N1 are now dead, nodes will be freed)
///
///    [happens automatically, once all shared_ptrs to N1 are released]
///
///   A----------------------------------------.
///                                            |
///                           B----------------)--.
///                                            |  |
///                                            |  |
///                                      C-----)--)--.
///                                            |  |  |
///                                            v  v  v
///                                           new_N1[Concat, concatenation_axis=3]
///                                                  |                |
///                                                  v                v
///                                               some_user         another_user
///
/// NOTE 1: replace_node is not type-safe (the graph is not revalidated).
/// For example, the following is allowed, even if node `some_user`
/// requires an input of shape 2x2:
///
/// (Before)
///      A(shape=2x2)  B(shape=3x3)
///      |
///      v
///   some_user(requires 2x2 input)
///
/// (After -- graph is now invalid)
///
///      replace_node(A, B);
///
///      A(shape=2x2)  B(shape=3x3)
///                    |
///                    v
///                 some_user(requires 2x2 input)
///
/// NOTE 2: it is possible to insert a cycle into the graph with
/// replace_node, resulting in an invalid graph. Care must be taken to
/// avoid this. One common example is when you are attempting to insert a
/// new node `M` "after"` a node `N`. For example, you might expect this
/// to work:
///
///    shared_ptr<Node> M = make_shared<SomeUnaryOp>(N);
///    replace_node(M, N);
///
/// The problem is that at replacement time, N itself is a user of M. So
/// we end up introducing a cycle as follows:
///
///       N
///       |
///       v
///  other users...
///
///      |||
///      vvv
///
///       N------------>M
///       |
///       v
///  other users...
///
///      |||
///      vvv
///
///               .----.
///              |      |
///              |      |
///       N      `----->M
///                     |
///                     v
///                other users...
///
/// To avoid the cycle, a valid way to perform the above desired insertion would be,
///
///        auto new_N = N->clone_with_new_inputs(N->input_values());
///        shared_ptr<Node> M = make_shared<SomeUnaryOp>(new_N);
///        replace_node(N, M);
OPENVINO_API
void replace_node(const std::shared_ptr<Node>& target,
                  const std::shared_ptr<Node>& replacement,
                  const std::vector<int64_t>& output_order);

/// Replace target.outputs[i] with replacement_values[i] and transfer control dependents and
OPENVINO_API
void replace_node(const std::shared_ptr<Node>& target, const OutputVector& replacement_values);

OPENVINO_API
void replace_node(const std::shared_ptr<Node>& target, const std::shared_ptr<Node>& replacement);

/// \brief Replace multiple nodes in a function.
/// \param f Model where replacement is taking place.
/// \param parameter_replacement_map A mapping from parameter shared pointers to parameter
///                                  shared pointers. For each pair (k,v) in the map, parameter
///                                  k is replaced by parameter v, except if k==v or k is not a
///                                  parameter bound by f, in which case the pair (k,v) is
///                                  ignored.
/// \param body_replacement_map A mapping from node shared pointers to node shared pointers.
///                             For each pair (k,v) in the map, node k is replaced by node v,
///                             except if k==v, the pair (k,v) is ignored.
///                             Note that if k is a parameter, its users will be redirected to
///                             v, but k will _not_ be replaced in the function's parameter
///                             list.
///
/// Limitations:
///
///    - No check is made that the replaced nodes in `parameter_replacement_map` are actually
///      among the bound parameters of `f`. (If a parameter appears in the map that is not
///      bound by `f`, it will be silently ignored.)
///    - If a parameter node appears as a key in both `parameter_replacement_map` _and_ in
///      `body_replacement_map`, behavior is unspecified.
OPENVINO_API
void replace_nodes(const std::shared_ptr<Model>& f,
                   const std::unordered_map<std::shared_ptr<op::v0::Parameter>, std::shared_ptr<op::v0::Parameter>>&
                       parameter_replacement_map,
                   const std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>>& body_replacement_map);

/// Topological sort of nodes needed to compute root_nodes
template <typename T>
std::vector<std::shared_ptr<Node>> topological_sort(T root_nodes) {
    std::stack<Node*, std::vector<Node*>> nodes_to_do;
    std::unordered_set<Node*> nodes_done;
    std::unordered_map<Node*, uint8_t /*is_visited*/> nodes_visited;
    std::vector<std::shared_ptr<Node>> result;

    for (auto& node : root_nodes) {
        nodes_to_do.push(node.get());
    }
    while (nodes_to_do.size() > 0) {
        Node* node = nodes_to_do.top();
        if (nodes_done.count(node) == 0) {
            bool can_add = true;
            if (++nodes_visited[node] > 2)
                // Node may be at the top of `nodes_to_do` not more than twice before it's added to `nodes_done` -
                // when visited and placed in `nodes_to_do` and after the subtree traversal is finished.
                // Otherwise it's a loop.
                OPENVINO_THROW("Loop detected during topological sort starting from '",
                               node->get_friendly_name(),
                               "' node.");

            size_t arg_count = node->get_input_size();
            for (size_t i = 0; i < arg_count; ++i) {
                Node* dep = node->get_input_node_ptr(arg_count - i - 1);
                if (nodes_done.count(dep) == 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            for (auto& depptr : node->get_control_dependencies()) {
                Node* dep = depptr.get();
                if (nodes_done.count(dep) == 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            if (can_add) {
                result.push_back(node->shared_from_this());
                nodes_to_do.pop();
                nodes_done.insert(node);
            }
        } else {
            nodes_to_do.pop();
        }
    }
    return result;
}

OPENVINO_API
bool compare_constants(const std::shared_ptr<Node>& n1, const std::shared_ptr<Node>& n2);

OPENVINO_API
bool replace_output_update_name(Output<Node> node, const Output<Node>& node_input);

OPENVINO_API
bool replace_node_update_name(const std::shared_ptr<Node>& target, const std::shared_ptr<Node>& replacement);

/// \brief Serialize given model into IR. The generated .xml and .bin files will be saved into provided paths.
/// This method serializes model "as-is" that means no weights compression and other possible transformations
/// are applied. It is recommended to use ov::save_model function instead of ov::serialize, because it is aligned
/// with default model conversion flow.
/// \param m Model which will be converted to IR representation.
/// \param xml_path Path where .xml file will be saved.
/// \param bin_path Path where .bin file will be saved (optional).
///                 The same name as for xml_path will be used by default.
/// \param version Version of the generated IR (optional).
/// \{
OPENVINO_API
void serialize(const std::shared_ptr<const ov::Model>& m,
               const std::string& xml_path,
               const std::string& bin_path = "",
               ov::pass::Serialize::Version version = ov::pass::Serialize::Version::UNSPECIFIED);

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
template <class Path, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
void serialize(const std::shared_ptr<const ov::Model>& m,
               const Path& xml_path,
               const Path& bin_path = {""},
               ov::pass::Serialize::Version version = ov::pass::Serialize::Version::UNSPECIFIED) {
    serialize(m, xml_path.string(), bin_path.string(), version);
}
#endif
/// \}

/// \brief Save given model into IR. Floating point weights are compressed to FP16 by default.
/// This method saves a model to IR applying all necessary transformations that usually applied
/// in model conversion flow provided by OVC tool. Particularly, floating point weights are compressed to FP16.
/// \param model Model which will be converted to IR representation.
/// \param output_model Path to the output model file, must have extension .xml
/// \param compress_to_fp16 Whether to compress floating point weights to FP16 (true by default)
OPENVINO_API
void save_model(const std::shared_ptr<const ov::Model>& model,
                const std::string& output_model,
                bool compress_to_fp16 = true);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
OPENVINO_API
void save_model(const std::shared_ptr<const ov::Model>& model,
                const std::wstring& output_model,
                bool compress_to_fp16 = true);
#endif

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
template <class Path, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
void save_model(const std::shared_ptr<const ov::Model>& model, const Path& output_model, bool compress_to_fp16 = true) {
    save_model(model, output_model.string(), compress_to_fp16);
}
#endif
}  // namespace ov
