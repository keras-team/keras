// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/inliner/inliner.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/common/constants.h"
#include "onnx/common/interned_strings.h"
#include "onnx/common/proto_util.h"
#include "onnx/common/visitor.h"
#include "onnx/shape_inference/attribute_binder.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE {
namespace inliner {

namespace { // internal/private API

using namespace internal;

using OpsetMapBase = std::unordered_map<std::string, int64_t>;

// A representation of the opset versions required by a model or a function.
// Used to check for compatibility between a model and a function or between
// two functions.
struct OpsetMap : public OpsetMapBase {
 public:
  // Construct a map representing the opset versions required by a model.
  explicit OpsetMap(const ModelProto& model) {
    (void)Add(model.opset_import());
  }

  // Adds the opset versions required by a function to the map. Returns true
  // iff the function is compatible with the map, i.e., if the function does
  // not require a different version for any domain already in the map.
  bool Add(const FunctionProto& function) {
    return Add(function.opset_import());
  }

  // Returns the set of mismatches in the opset requirements of given
  // function and the map.
  OpsetMapBase Mismatches(const FunctionProto& function) const {
    return Mismatches(function.opset_import());
  }

 private:
  OpsetMapBase Mismatches(const google::protobuf::RepeatedPtrField<OperatorSetIdProto>& list) const {
    OpsetMapBase result;
    for (const auto& pair : list) {
      auto iter = this->find(NormalizeDomain(pair.domain()));
      if ((iter != this->end()) && (iter->second != pair.version()))
        result.insert(*iter);
    }
    return result;
  }

  bool Add(const google::protobuf::RepeatedPtrField<OperatorSetIdProto>& list) {
    for (const auto& pair : list) {
      auto domain = NormalizeDomain(pair.domain());
      auto version = pair.version();
      auto iter = this->find(domain);
      if (iter != this->end()) {
        if (iter->second != version)
          return false;
      } else {
        (*this)[domain] = version;
      }
    }
    return true;
  }
};

using RepeatedNodeProto = google::protobuf::RepeatedPtrField<NodeProto>;

class NameGenerator : private Visitor {
 public:
  explicit NameGenerator(const GraphProto& graph) : index_(0) {
    VisitGraph(graph);
  }

  explicit NameGenerator(const FunctionProto& function) : index_(0) {
    VisitFunction(function);
  }

  // Creates a new unique name, based on a suggested name, and adds it to the set
  // of existing names. Returns the newly created name.
  std::string CreateNew(const std::string& suggested) {
    std::string name = suggested;
    while (existing_names_.count(name) > 0) {
      name = suggested + "_" + std::to_string(index_++);
    }
    existing_names_.insert(name);
    return name;
  }

  void Add(const std::string& name) {
    // We don't bother to check for empty string names. Ok to add them.
    existing_names_.insert(name);
  }

  bool ProcessGraph(const GraphProto& graph) override {
    for (const auto& x : graph.input())
      Add(x.name());
    for (const auto& x : graph.initializer())
      Add(x.name());
    // Adding graph outputs is redundant for a valid graph, but we do it anyway,
    // to produce better results for invalid graphs.
    for (const auto& x : graph.output())
      Add(x.name());
    return true;
  }

  bool ProcessFunction(const FunctionProto& function) override {
    for (const auto& x : function.input())
      Add(x);
    for (const auto& x : function.output())
      Add(x);
    return true;
  }

  bool ProcessNode(const NodeProto& node) override {
    // We use a single name-space for node names and variable names, to keep name-generation simple.
    Add(node.name());
    for (const std::string& name : node.input()) {
      Add(name);
    }
    for (const std::string& name : node.output()) {
      Add(name);
    }
    return true;
  }

 private:
  unsigned int index_;
  std::unordered_set<std::string> existing_names_;
};

class InliningRenamer : private MutableVisitor {
 private:
  std::string suffix;
  NameGenerator& generator;
  std::vector<std::unordered_map<std::string, std::string>> rename_scopes{};

  InliningRenamer(std::string suffix_, NameGenerator& generator_) : suffix(std::move(suffix_)), generator(generator_) {
    // Create an empty mapping for the top-level scope.
    rename_scopes.emplace_back();
  }

  // We use a two-level renaming scheme to generate names for variables when inlined in the
  // main graph. First, we add a suffix (specific to the call-site being inlined).
  // Thus, "temp" in called-function becomes "temp__1" for the first inlined function-call
  // and "temp__2" for the second inlined function-call. In addition, there is a subsequent
  // iterative check that ensures that this names does not clash with any pre-existing names,
  // and tries another counter-based suffix in the case of a clash, stopping when successful.
  std::string MakeUnique(const std::string& name) {
    return generator.CreateNew(name + suffix);
  }

  // Replace given name with a unique version of the name, and cache the
  // renaming-binding in current scope.
  void Rename(std::string& name) {
    auto new_name = MakeUnique(name);
    auto& current_scope = rename_scopes.back();
    current_scope[name] = new_name;
    name = new_name;
  }

  void LookupOrRename(std::string& name, bool is_new_def) {
    if (name.empty())
      return;
    for (auto i = rename_scopes.size(); i > 0; --i) {
      const auto& map = rename_scopes[i - 1];
      auto iter = map.find(name);
      if (iter != map.end()) {
        name = iter->second;
        return;
      }
    }
    if (is_new_def) {
      Rename(name);
    }
    // Otherwise, it is a reference to an outer-scope variable that should not be renamed.
  }

  template <bool isOutput>
  void Bind(
      google::protobuf::RepeatedPtrField<std::string>& formals,
      const google::protobuf::RepeatedPtrField<std::string>& actuals) {
    // Every formal parameter name FP should be replace by the corresponding actual parameter name AP.
    // However, if AP is empty, it is a missing optional parameter. This does not make any difference
    // for inputs. However, for outputs we use a unique dummy name to handle the case that it
    // is used in an output-context where it is not optional.
    ONNX_ASSERTM(
        actuals.size() <= formals.size(), "Number of actual parameters cannot exceed number of formal parameters");
    auto& current_scope = rename_scopes.back();
    int i = 0;
    for (; i < actuals.size(); ++i) {
      std::string& formal = *formals.Mutable(i);
      std::string rename_as = actuals.Get(i);
      if (isOutput)
        if (rename_as.empty())
          rename_as = MakeUnique(formal);
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
    for (; i < formals.size(); ++i) {
      std::string& formal = *formals.Mutable(i);
      std::string rename_as = isOutput ? MakeUnique(formal) : std::string("");
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
  }

  // Process a node:
  bool ProcessNode(NodeProto* node) override {
    if (!node->name().empty())
      node->set_name(MakeUnique(node->name()));

    for (auto& x : *node->mutable_input()) {
      LookupOrRename(x, false);
    }
    for (auto& y : *node->mutable_output()) {
      LookupOrRename(y, true);
    }
    return true; // Process attribute subgraphs in traversal
  }

  // Process a sub-graph, contained as an attribute in a control-flow op node.
  // Since we need both pre-processing and post-processing in the traversal, we
  // override the VisitGraph method.
  void VisitGraph(GraphProto* graph) override {
    rename_scopes.emplace_back();
    for (auto& x : *graph->mutable_input())
      Rename(*x.mutable_name());
    for (auto& init : *graph->mutable_initializer())
      Rename(*init.mutable_name());
    for (auto& y : *graph->mutable_output())
      Rename(*y.mutable_name());
    for (auto& n : *graph->mutable_node())
      VisitNode(&n);
    rename_scopes.pop_back();
  }

 public:
  // Renames variables in a FunctionProto for inlining a particular call-site. This does the following:
  // (i)  Rename all intermediate variables in the function to ensure that they are unique (wrt the main graph).
  // (ii) Rename inputs and outputs using names of actual parameters.
  static void
  Rename(const NodeProto& callnode, FunctionProto& callee, std::string unique_suffix, NameGenerator& generator) {
    InliningRenamer renamer(std::move(unique_suffix), generator);

    renamer.Bind<false>(*callee.mutable_input(), callnode.input());
    renamer.Bind<true>(*callee.mutable_output(), callnode.output());

    renamer.VisitFunction(&callee);
    for (auto& v : *callee.mutable_value_info())
      renamer.LookupOrRename(*v.mutable_name(), false);
  }
};

// Identify the set of all "input" variables used by a given node.
// This includes the variables listed as node.input, as well as
// implicit inputs referred to in any graph-valued-attribute of the node.
// In the case of variables referenced in sub-graphs, only non-local variables
// are treated as implicit inputs.

class ComputeInputs : private Visitor {
 private:
  std::vector<std::unordered_set<std::string>> namescopes;

  bool InNestedScope() const {
    return !namescopes.empty();
  }

  std::unordered_set<std::string>& CurrentScope() {
    return namescopes.back();
  }

  bool IsLocalVar(const std::string& name) const {
    for (auto& scope : namescopes) {
      if (scope.count(name) > 0) {
        return true;
      }
    }
    return false;
  }

  void VisitGraph(const GraphProto& graph) override {
    namescopes.emplace_back();
    for (auto& x : graph.input())
      CurrentScope().insert(x.name());
    for (auto& init : graph.initializer())
      CurrentScope().insert(init.name());
    for (auto& n : graph.node())
      VisitNode(n);
    namescopes.pop_back();
  }

  bool ProcessNode(const NodeProto& node) override {
    for (auto& var : node.input()) {
      if (!var.empty() && !IsLocalVar(var)) {
        result.push_back(var);
      }
    }
    if (InNestedScope()) {
      for (auto& var : node.output()) {
        if (!var.empty()) {
          CurrentScope().insert(var);
        }
      }
    }
    return true; // process sub-graphs
  }

 public:
  std::vector<std::string> result;

  ComputeInputs(const NodeProto& node) {
    result.reserve(node.input_size());
    VisitNode(node);
  }
};

std::vector<std::string> GetUsedVars(const NodeProto& node) {
  return ComputeInputs(node).result;
}

using ConstNodeMap = std::unordered_map<std::string, const NodeProto*>;

ConstNodeMap FindConstantNodes(const GraphProto& graph) {
  ConstNodeMap result;
  for (const NodeProto& node : graph.node()) {
    if (IsOnnxDomain(node.domain()) && (node.op_type() == "Constant")) {
      result[node.output(0)] = &node;
    }
  }
  return result;
}

const TypeProto& GetType(const ModelProto& model, std::string var) {
  for (auto& vi : model.graph().value_info()) {
    if (vi.name() == var)
      return vi.type();
  }
  for (auto& vi : model.graph().input()) {
    if (vi.name() == var)
      return vi.type();
  }
  for (auto& vi : model.graph().output()) {
    if (vi.name() == var)
      return vi.type();
  }
  ONNX_ASSERTM(false, "Type unknown for %s", var.c_str());
}

void ConvertVersion(ModelProto& model, const NodeProto& call_node, FunctionProto& function, int target_version) {
  shape_inference::InferShapes(model);

  ModelProto function_as_model;
  function_as_model.set_ir_version(model.ir_version());
  *function_as_model.mutable_opset_import() = function.opset_import();

  GraphProto& graph = *function_as_model.mutable_graph();
  // The graph's inputs are all the variables used in the call_node.
  auto used_vars = GetUsedVars(call_node);
  auto constant_node_map = FindConstantNodes(model.graph());

  RepeatedNodeProto& function_nodes = *function.mutable_node();
  RepeatedNodeProto& nodes = *graph.mutable_node();
  nodes.Reserve(function_nodes.size() + used_vars.size());

  auto* inputs = graph.mutable_input();
  for (const auto& var : used_vars) {
    auto* new_input = inputs->Add();
    new_input->set_name(var);
    *new_input->mutable_type() = GetType(model, var);
    // Create a copy of constants used by the call_node.
    // We do not handle initializers-as-constants for now.
    auto it = constant_node_map.find(var);
    if (it != constant_node_map.end()) {
      *nodes.Add() = *(it->second);
    }
  }

  // outputs: from call_node node outputs
  auto* outputs = graph.mutable_output();
  for (const auto& var : call_node.output()) {
    if (!var.empty()) {
      auto* new_output = outputs->Add();
      new_output->set_name(var);
      *new_output->mutable_type() = GetType(model, var);
    }
  }

  // TODO: Use std::move when it is fully supported on all protobuf platforms used
  for (auto& function_node : function_nodes)
    *nodes.Add() = function_node;
  function_nodes.Clear();

  auto converted = ONNX_NAMESPACE::version_conversion::ConvertVersion(function_as_model, target_version);

  function_nodes.Swap(converted.mutable_graph()->mutable_node());

  // Append new initializers to main graph initializers
  for (auto& added_initializer : converted.graph().initializer())
    *model.mutable_graph()->mutable_initializer()->Add() = added_initializer;
  for (auto& added_initializer : converted.graph().sparse_initializer())
    *model.mutable_graph()->mutable_sparse_initializer()->Add() = added_initializer;
}

constexpr int64_t kNoConversion = -1;
using FunctionMap = std::unordered_map<FunctionImplId, std::pair<const FunctionProto*, int64_t>>;

using NodeList = google::protobuf::RepeatedPtrField<NodeProto>;

/** Utility function used for inlining into a GraphProto.
 * @param graph Mutable graph
 * @param map Map from function-id to function for functions to be inlined
 * @param name_generator Name generator for generating unique names for inlined variables
 * @param model If non-null, the model being inlined into. Used for version conversion.
 * @param inline_count Mutable counter for number of inlined calls. Used for name generation.
 */
void InlineFunctions(
    GraphProto& graph,
    const FunctionMap& map,
    NameGenerator& name_generator,
    ModelProto* model,
    int& inline_count);

/** Shared utility function used for inlining into either a GraphProto or a FunctionProto.
 * @param nodes Mutable list of nodes (of function or graph)
 * @param value_infos Mutable list of value_infos (of function or graph)
 * @param map Map from function-id to function for functions to be inlined
 * @param name_generator Name generator for generating unique names for inlined variables
 * @param model If non-null, the model being inlined into. Used for version conversion.
 * @param inline_count Mutable counter for number of inlined calls. Used for name generation.
 */
void InlineFunctions(
    NodeList& nodes,
    ValueInfoList& value_infos,
    const FunctionMap& map,
    NameGenerator& name_generator,
    ModelProto* model,
    int& inline_count) {
  NodeList original_nodes;
  // Move all nodes into original_nodes
  original_nodes.Swap(&nodes);

  std::function<void(NodeProto & node)> append_node = [&](NodeProto& node) {
    FunctionProto callee;
    auto iter = map.find(GetCalleeId(node));
    if (iter != map.end()) {
      callee = *iter->second.first;
      int64_t target_version = iter->second.second;
      // Bind attribute parameters
      internal::AttributeBinder::BindAttributes(node, callee);

      // Rename variable names in callee
      InliningRenamer::Rename(node, callee, "__" + std::to_string(++inline_count), name_generator);
      if (target_version != kNoConversion) {
        ONNX_ASSERTM(
            model != nullptr,
            "Internal Error: Inlining function %s::%s requires version conversion, "
            "but version conversion is supported only for models, not functions.",
            callee.domain().c_str(),
            callee.name().c_str());
        ConvertVersion(*model, node, callee, target_version);
      }
      std::unordered_set<std::string> actual_parameters;
      for (const auto& x : node.input())
        actual_parameters.insert(x);
      for (const auto& x : node.output())
        actual_parameters.insert(x);
      // Append valueinfos of called function
      for (auto& callee_vi : callee.value_info()) {
        if (actual_parameters.count(callee_vi.name()) == 0) {
          *value_infos.Add() = callee_vi;
        }
      }
      // Append nodes of called function
      for (auto& callee_node : *callee.mutable_node())
        append_node(callee_node);
    } else {
      // Append node without inlining.
      // TODO: use std::move instead of copying. Use of move doesn't seem to work with
      // protobuf in some platforms/settings. [nodes->Add(std::move(node));]

      for (auto& attr : *node.mutable_attribute()) {
        if (attr.has_g()) {
          InlineFunctions(*attr.mutable_g(), map, name_generator, model, inline_count);
        }
        for (auto& g : *attr.mutable_graphs()) {
          InlineFunctions(g, map, name_generator, model, inline_count);
        }
      }

      *nodes.Add() = node;
    }
  };
  for (auto& node : original_nodes) {
    append_node(node);
  }
}

/** Utility function used for inlining into a GraphProto.
 * @param graph Mutable graph
 * @param map Map from function-id to function for functions to be inlined
 * @param name_generator Name generator for generating unique names for inlined variables
 * @param model If non-null, the model being inlined into. Used for version conversion.
 * @param inline_count Mutable counter for number of inlined calls. Used for name generation.
 */
void InlineFunctions(
    GraphProto& graph,
    const FunctionMap& map,
    NameGenerator& name_generator,
    ModelProto* model,
    int& inline_count) {
  auto* nodes = graph.mutable_node();
  auto* value_infos = graph.mutable_value_info();
  InlineFunctions(*nodes, *value_infos, map, name_generator, model, inline_count);
}

/** Utility function used for inlining into a ModelProto.
 * @param model Mutable model
 * @param map Map from function-id to function for functions to be inlined
 */
void InlineFunctions(ModelProto& model, FunctionMap& map) {
  int inline_count = 0;
  auto* graph = model.mutable_graph();
  NameGenerator name_generator(*graph);
  auto* nodes = graph->mutable_node();
  auto* value_infos = graph->mutable_value_info();
  InlineFunctions(*nodes, *value_infos, map, name_generator, &model, inline_count);
}

/** Utility function used for inlining into a FunctionProto.
 * @param function Mutable function
 * @param map Map from function-id to function for functions to be inlined
 */
void InlineFunctions(FunctionProto& function, FunctionMap& map) {
  int inline_count = 0;
  NameGenerator name_generator(function);
  InlineFunctions(*function.mutable_node(), *function.mutable_value_info(), map, name_generator, nullptr, inline_count);
}

class VectorSet : public FunctionIdSet {
 public:
  VectorSet(FunctionIdVector&& function_ids, bool invert) : function_ids_(std::move(function_ids)), invert_(invert) {}

  bool Contains(const std::string& function_domain, const std::string& function_name) const override {
    bool found =
        std::find(function_ids_.begin(), function_ids_.end(), std::make_pair(function_domain, function_name)) !=
        function_ids_.end();
    return invert_ ? !found : found;
  }

 private:
  FunctionIdVector function_ids_;
  bool invert_;
};

} // namespace

// Public API implementation:

std::unique_ptr<FunctionIdSet> ONNX_NAMESPACE::inliner::FunctionIdSet::Create(
    FunctionIdVector&& function_ids,
    bool invert) {
  auto* p = new VectorSet(std::move(function_ids), invert);
  return std::unique_ptr<FunctionIdSet>(p);
}

void InlineLocalFunctions(ModelProto& model, bool convert_version) {
  OpsetMap model_imports(model);
  FunctionMap map;

  // For every function, we check if there is a mismatch between the opset versions
  // required for the function and the model. If there is no mismatch, we can inline
  // this function. If there is a mismatch only for the standard ONNX domain, we
  // can inline after version-conversion (if the version-conversion is successful).
  // Otherwise, we cannot inline, since currently version-conversion supports only
  // standard ONNX domain.

  for (auto& function : model.functions()) {
    auto mismatches = model_imports.Mismatches(function);
    auto iter = mismatches.find(ONNX_DOMAIN);
    int64_t target_onnx_version = kNoConversion;
    if (convert_version && (iter != mismatches.end())) {
      target_onnx_version = iter->second;
      mismatches.erase(iter);
    }
    if (mismatches.empty()) {
      map[GetFunctionImplId(function)] = std::pair<const FunctionProto*, int64_t>(&function, target_onnx_version);
    }
  }

  InlineFunctions(model, map);

  // Remove all model-local functions. We do not remove functions with a mis-matched
  // opset version. They need to be handled some other way, eg., using a version-adapter.
  auto* local_functions = model.mutable_functions();
  for (auto it = local_functions->begin(); it != local_functions->end();) {
    if (map.count(GetFunctionImplId(*it)) > 0)
      it = local_functions->erase(it);
    else
      ++it;
  }
}

void InlineSelectedFunctions(ModelProto& model, const FunctionIdSet& to_inline) {
  OpsetMap model_imports(model);
  FunctionMap map;
  std::vector<FunctionProto*> non_inlined_functions;

  // If there is any mismatch between the opset versions required for any of the
  // functions and the model, the inliner will fail.

  for (auto& function : *model.mutable_functions()) {
    // auto& function = *function_ptr;
    if (!model_imports.Add(function))
      ONNX_THROW("Model has functions with incompatible opset versions.");
    if (to_inline.Contains(function.domain(), function.name())) {
      map[GetFunctionImplId(function)] = std::pair<const FunctionProto*, int64_t>(&function, kNoConversion);
    } else {
      non_inlined_functions.push_back(&function);
    }
  }

  InlineFunctions(model, map);

  for (auto* function_ptr : non_inlined_functions) {
    InlineFunctions(*function_ptr, map);
  }

  // Remove all inlined model-local functions.
  auto* local_functions = model.mutable_functions();
  for (auto it = local_functions->begin(); it != local_functions->end();) {
    if (map.count(GetFunctionImplId(*it)) > 0)
      it = local_functions->erase(it);
    else
      ++it;
  }
}

} // namespace inliner
} // namespace ONNX_NAMESPACE
