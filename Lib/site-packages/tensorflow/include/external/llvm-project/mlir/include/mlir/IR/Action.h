//===- Action.h -  Action Support ---------------------*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the action framework. This framework
// allows for external entities to control certain actions taken by the compiler
// by registering handler functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ACTION_H
#define MLIR_IR_ACTION_H

#include "mlir/IR/Unit.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <type_traits>

namespace mlir {
namespace tracing {

/// An action is a specific action that is to be taken by the compiler,
/// that can be toggled and controlled by an external user. There are no
/// constraints on the granularity of an action, it could be as simple as
/// "perform this fold" and as complex as "run this pass pipeline".
///
/// This class represents the base class of the ActionImpl class (see below).
/// This holds the template-invariant elements of the Action class.
class Action {
public:
  virtual ~Action() = default;

  /// Return the unique action id of this action, use for casting
  /// functionality.
  TypeID getActionID() const { return actionID; }

  /// Return a string "tag" which intends to uniquely identify this type of
  /// action. For example "pass-application" or "pattern-rewrite".
  virtual StringRef getTag() const = 0;

  virtual void print(raw_ostream &os) const {
    os << "Action \"" << getTag() << "\"";
  }

  /// Return the set of IR units that are associated with this action.
  virtual ArrayRef<IRUnit> getContextIRUnits() const { return irUnits; }

protected:
  Action(TypeID actionID, ArrayRef<IRUnit> irUnits)
      : actionID(actionID), irUnits(irUnits) {}

  /// The type of the derived action class, used for `isa`/`dyn_cast`.
  TypeID actionID;

  /// Set of IR units (operations, regions, blocks, values) that are associated
  /// with this action.
  ArrayRef<IRUnit> irUnits;
};

/// CRTP Implementation of an action. This class provides a base class for
/// implementing specific actions.
///  Derived classes are expected to provide the following:
///   * static constexpr StringLiteral tag = "...";
///     - This method returns a unique string identifier, similar to a command
///       line flag or DEBUG_TYPE.
template <typename Derived>
class ActionImpl : public Action {
public:
  ActionImpl(ArrayRef<IRUnit> irUnits = {})
      : Action(TypeID::get<Derived>(), irUnits) {}

  /// Provide classof to allow casting between action types.
  static bool classof(const Action *action) {
    return action->getActionID() == TypeID::get<Derived>();
  }

  /// Forward tag access to the derived class.
  StringRef getTag() const final { return Derived::tag; }
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_IR_ACTION_H
