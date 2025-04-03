//===- PredicateTree.h - Predicate tree node definitions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for nodes of a tree structure for representing
// the general control flow within a pattern match.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_PDLTOPDLINTERP_PREDICATETREE_H_
#define MLIR_LIB_CONVERSION_PDLTOPDLINTERP_PREDICATETREE_H_

#include "Predicate.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
class ModuleOp;

namespace pdl_to_pdl_interp {

class MatcherNode;

/// A PositionalPredicate is a predicate that is associated with a specific
/// positional value.
struct PositionalPredicate {
  PositionalPredicate(Position *pos,
                      const PredicateBuilder::Predicate &predicate)
      : position(pos), question(predicate.first), answer(predicate.second) {}

  /// The position the predicate is applied to.
  Position *position;

  /// The question that the predicate applies.
  Qualifier *question;

  /// The expected answer of the predicate.
  Qualifier *answer;
};

//===----------------------------------------------------------------------===//
// MatcherNode
//===----------------------------------------------------------------------===//

/// This class represents the base of a predicate matcher node.
class MatcherNode {
public:
  virtual ~MatcherNode() = default;

  /// Given a module containing PDL pattern operations, generate a matcher tree
  /// using the patterns within the given module and return the root matcher
  /// node. `valueToPosition` is a map that is populated with the original
  /// pdl values and their corresponding positions in the matcher tree.
  static std::unique_ptr<MatcherNode>
  generateMatcherTree(ModuleOp module, PredicateBuilder &builder,
                      DenseMap<Value, Position *> &valueToPosition);

  /// Returns the position on which the question predicate should be checked.
  Position *getPosition() const { return position; }

  /// Returns the predicate checked on this node.
  Qualifier *getQuestion() const { return question; }

  /// Returns the node that should be visited if this, or a subsequent node
  /// fails.
  std::unique_ptr<MatcherNode> &getFailureNode() { return failureNode; }

  /// Sets the node that should be visited if this, or a subsequent node fails.
  void setFailureNode(std::unique_ptr<MatcherNode> node) {
    failureNode = std::move(node);
  }

  /// Returns the unique type ID of this matcher instance. This should not be
  /// used directly, and is provided to support type casting.
  TypeID getMatcherTypeID() const { return matcherTypeID; }

protected:
  MatcherNode(TypeID matcherTypeID, Position *position = nullptr,
              Qualifier *question = nullptr,
              std::unique_ptr<MatcherNode> failureNode = nullptr);

private:
  /// The position on which the predicate should be checked.
  Position *position;

  /// The predicate that is checked on the given position.
  Qualifier *question;

  /// The node to visit if this node fails.
  std::unique_ptr<MatcherNode> failureNode;

  /// An owning store for the failure node if it is owned by this node.
  std::unique_ptr<MatcherNode> failureNodeStorage;

  /// A unique identifier for the derived matcher node, used for type casting.
  TypeID matcherTypeID;
};

//===----------------------------------------------------------------------===//
// BoolNode

/// A BoolNode denotes a question with a boolean-like result. These nodes branch
/// to a single node on a successful result, otherwise defaulting to the failure
/// node.
struct BoolNode : public MatcherNode {
  BoolNode(Position *position, Qualifier *question, Qualifier *answer,
           std::unique_ptr<MatcherNode> successNode,
           std::unique_ptr<MatcherNode> failureNode = nullptr);

  /// Returns if the given matcher node is an instance of this class, used to
  /// support type casting.
  static bool classof(const MatcherNode *node) {
    return node->getMatcherTypeID() == TypeID::get<BoolNode>();
  }

  /// Returns the expected answer of this boolean node.
  Qualifier *getAnswer() const { return answer; }

  /// Returns the node that should be visited on success.
  std::unique_ptr<MatcherNode> &getSuccessNode() { return successNode; }

private:
  /// The expected answer of this boolean node.
  Qualifier *answer;

  /// The next node if this node succeeds. Otherwise, go to the failure node.
  std::unique_ptr<MatcherNode> successNode;
};

//===----------------------------------------------------------------------===//
// ExitNode

/// An ExitNode is a special sentinel node that denotes the end of matcher.
struct ExitNode : public MatcherNode {
  ExitNode() : MatcherNode(TypeID::get<ExitNode>()) {}

  /// Returns if the given matcher node is an instance of this class, used to
  /// support type casting.
  static bool classof(const MatcherNode *node) {
    return node->getMatcherTypeID() == TypeID::get<ExitNode>();
  }
};

//===----------------------------------------------------------------------===//
// SuccessNode

/// A SuccessNode denotes that a given high level pattern has successfully been
/// matched. This does not terminate the matcher, as there may be multiple
/// successful matches.
struct SuccessNode : public MatcherNode {
  explicit SuccessNode(pdl::PatternOp pattern, Value root,
                       std::unique_ptr<MatcherNode> failureNode);

  /// Returns if the given matcher node is an instance of this class, used to
  /// support type casting.
  static bool classof(const MatcherNode *node) {
    return node->getMatcherTypeID() == TypeID::get<SuccessNode>();
  }

  /// Return the high level pattern operation that is matched with this node.
  pdl::PatternOp getPattern() const { return pattern; }

  /// Return the chosen root of the pattern.
  Value getRoot() const { return root; }

private:
  /// The high level pattern operation that was successfully matched with this
  /// node.
  pdl::PatternOp pattern;

  /// The chosen root of the pattern.
  Value root;
};

//===----------------------------------------------------------------------===//
// SwitchNode

/// A SwitchNode denotes a question with multiple potential results. These nodes
/// branch to a specific node based on the result of the question.
struct SwitchNode : public MatcherNode {
  SwitchNode(Position *position, Qualifier *question);

  /// Returns if the given matcher node is an instance of this class, used to
  /// support type casting.
  static bool classof(const MatcherNode *node) {
    return node->getMatcherTypeID() == TypeID::get<SwitchNode>();
  }

  /// Returns the children of this switch node. The children are contained
  /// within a mapping between the various case answers to destination matcher
  /// nodes.
  using ChildMapT = llvm::MapVector<Qualifier *, std::unique_ptr<MatcherNode>>;
  ChildMapT &getChildren() { return children; }

  /// Returns the child at the given index.
  std::pair<Qualifier *, std::unique_ptr<MatcherNode>> &getChild(unsigned i) {
    assert(i < children.size() && "invalid child index");
    return *std::next(children.begin(), i);
  }

private:
  /// Switch predicate "answers" select the child. Answers that are not found
  /// default to the failure node.
  ChildMapT children;
};

} // namespace pdl_to_pdl_interp
} // namespace mlir

#endif // MLIR_CONVERSION_PDLTOPDLINTERP_PREDICATETREE_H_
