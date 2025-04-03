//===- BufferViewFlowAnalysis.h - Buffer dependency analysis ---*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERVIEWFLOWANALYSIS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERVIEWFLOWANALYSIS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {

/// A straight-forward alias analysis which ensures that all dependencies of all
/// values will be determined. This is a requirement for the BufferPlacement
/// class since you need to determine safe positions to place alloc and
/// deallocs. This alias analysis only finds aliases that might have been
/// created on top of the specified view. To find all aliases, resolve the
/// intial alloc/argument value.
class BufferViewFlowAnalysis {
public:
  using ValueSetT = SmallPtrSet<Value, 16>;
  using ValueMapT = llvm::DenseMap<Value, ValueSetT>;

  /// Constructs a new alias analysis using the op provided.
  BufferViewFlowAnalysis(Operation *op);

  /// Find all immediate dependencies this value could potentially have.
  ValueMapT::const_iterator find(Value value) const {
    return dependencies.find(value);
  }

  /// Returns the begin iterator to iterate over all dependencies.
  ValueMapT::const_iterator begin() const { return dependencies.begin(); }

  /// Returns the end iterator that can be used in combination with find.
  ValueMapT::const_iterator end() const { return dependencies.end(); }

  /// Find all immediate and indirect views upon this value. This will find all
  /// dependencies on this value that can potentially be later in the execution
  /// of the program, but will not return values that this alias might have been
  /// created from (such as if the value is created by a subview, this will not
  /// return the parent view if there is no cyclic behavior). Note that the
  /// resulting set will also contain the value provided as it is an alias of
  /// itself.
  ///
  /// A = *
  /// B = subview(A)
  /// C = B
  ///
  /// Results in resolve(B) returning {B, C}
  ValueSetT resolve(Value value) const;
  ValueSetT resolveReverse(Value value) const;

  /// Removes the given values from all alias sets.
  void remove(const SetVector<Value> &aliasValues);

  /// Replaces all occurrences of 'from' in the internal datastructures with
  /// 'to'. This is useful when the defining operation of a value has to be
  /// re-built because additional results have to be added or the types of
  /// results have to be changed.
  void rename(Value from, Value to);

  /// Returns "true" if the given value may be a terminal.
  bool mayBeTerminalBuffer(Value value) const;

private:
  /// This function constructs a mapping from values to its immediate
  /// dependencies.
  void build(Operation *op);

  /// Maps values to all immediate dependencies this value can have.
  ValueMapT dependencies;
  ValueMapT reverseDependencies;

  /// A set of all SSA values that may be terminal buffers.
  DenseSet<Value> terminals;
};

/// An is-same-buffer analysis that checks if two SSA values belong to the same
/// buffer allocation or not.
class BufferOriginAnalysis {
public:
  BufferOriginAnalysis(Operation *op);

  /// Return "true" if `v1` and `v2` originate from the same buffer allocation.
  /// Return "false" if `v1` and `v2` originate from different allocations.
  /// Return "nullopt" if we do not know for sure.
  ///
  /// Example 1: isSameAllocation(%0, %1) == true
  /// ```
  /// %0 = memref.alloc()
  /// %1 = memref.subview %0
  /// ```
  ///
  /// Example 2: isSameAllocation(%0, %1) == false
  /// ```
  /// %0 = memref.alloc()
  /// %1 = memref.alloc()
  /// ```
  ///
  /// Example 3: isSameAllocation(%0, %2) == nullopt
  /// ```
  /// %0 = memref.alloc()
  /// %1 = memref.alloc()
  /// %2 = arith.select %c, %0, %1
  /// ```
  std::optional<bool> isSameAllocation(Value v1, Value v2);

private:
  BufferViewFlowAnalysis analysis;
};

} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERVIEWFLOWANALYSIS_H
