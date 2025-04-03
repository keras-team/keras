//===- IRMapping.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility class for maintaining a mapping of SSA values,
// blocks, and operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_IRMAPPING_H
#define MLIR_IR_IRMAPPING_H

#include "mlir/IR/Block.h"

namespace mlir {
/// This is a utility class for mapping one set of IR entities to another. New
/// mappings can be inserted via 'map'. Existing mappings can be
/// found via the 'lookup*' functions. There are three variants that differ only
/// in return value when an existing is not found for the provided key: SSA
/// values, blocks, and operations. 'lookupOrNull' returns nullptr where as
/// 'lookupOrDefault' will return the lookup key.
class IRMapping {
public:
  /// Inserts a new mapping for 'from' to 'to'. If there is an existing mapping,
  /// it is overwritten.
  void map(Value from, Value to) { valueMap[from] = to; }
  void map(Block *from, Block *to) { blockMap[from] = to; }
  void map(Operation *from, Operation *to) { operationMap[from] = to; }

  template <typename S, typename T,
            std::enable_if_t<!std::is_assignable_v<Value, S> &&
                             !std::is_assignable_v<Block *, S> &&
                             !std::is_assignable_v<Operation *, S>> * = nullptr>
  void map(S &&from, T &&to) {
    for (auto [fromValue, toValue] : llvm::zip(from, to))
      map(fromValue, toValue);
  }

  /// Erases a mapping for 'from'.
  template <typename T>
  void erase(T from) {
    getMap<T>().erase(from);
  }

  /// Checks to see if a mapping for 'from' exists.
  template <typename T>
  bool contains(T from) const {
    return getMap<T>().count(from);
  }

  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return nullptr.
  template <typename T>
  auto lookupOrNull(T from) const {
    return lookupOrValue(from, T(nullptr));
  }

  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return the provided value.
  template <typename T>
  auto lookupOrDefault(T from) const {
    return lookupOrValue(from, from);
  }

  /// Lookup a mapped value within the map. This asserts the provided value
  /// exists within the map.
  template <typename T>
  auto lookup(T from) const {
    auto result = lookupOrNull(from);
    assert(result && "expected 'from' to be contained within the map");
    return result;
  }

  /// Clears all mappings held by the mapper.
  void clear() { valueMap.clear(); }

  /// Return the held value mapping.
  const DenseMap<Value, Value> &getValueMap() const { return valueMap; }

  /// Return the held block mapping.
  const DenseMap<Block *, Block *> &getBlockMap() const { return blockMap; }

  /// Return the held operation mapping.
  const DenseMap<Operation *, Operation *> &getOperationMap() const {
    return operationMap;
  }

private:
  /// Return the map for the given value type.
  template <typename T>
  auto &getMap() const {
    if constexpr (std::is_convertible_v<T, Value>)
      return const_cast<DenseMap<Value, Value> &>(valueMap);
    else if constexpr (std::is_convertible_v<T, Block *>)
      return const_cast<DenseMap<Block *, Block *> &>(blockMap);
    else
      return const_cast<DenseMap<Operation *, Operation *> &>(operationMap);
  }

  /// Utility lookupOrValue that looks up an existing key or returns the
  /// provided value.
  template <typename T>
  auto lookupOrValue(T from, T value) const {
    auto &map = getMap<T>();
    auto it = map.find(from);
    return it != map.end() ? it->second : value;
  }

  DenseMap<Value, Value> valueMap;
  DenseMap<Block *, Block *> blockMap;
  DenseMap<Operation *, Operation *> operationMap;
};

} // namespace mlir

#endif // MLIR_IR_IRMAPPING_H
