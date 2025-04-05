//===- AsmParserState.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ASMPARSER_ASMPARSERSTATE_H
#define MLIR_ASMPARSER_ASMPARSERSTATE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/SMLoc.h"
#include <cstddef>

namespace mlir {
class Block;
class BlockArgument;
class FileLineColLoc;
class Operation;
class OperationName;
class SymbolRefAttr;
class Value;

/// This class represents state from a parsed MLIR textual format string. It is
/// useful for building additional analysis and language utilities on top of
/// textual MLIR. This should generally not be used for traditional compilation.
class AsmParserState {
public:
  /// This class represents a definition within the source manager, containing
  /// it's defining location and locations of any uses. SMDefinitions are only
  /// provided for entities that have uses within an input file, e.g. SSA
  /// values, Blocks, and Symbols.
  struct SMDefinition {
    SMDefinition() = default;
    SMDefinition(SMRange loc) : loc(loc) {}

    /// The source location of the definition.
    SMRange loc;
    /// The source location of all uses of the definition.
    SmallVector<SMRange> uses;
  };

  /// This class represents the information for an operation definition within
  /// an input file.
  struct OperationDefinition {
    struct ResultGroupDefinition {
      ResultGroupDefinition(unsigned index, SMRange loc)
          : startIndex(index), definition(loc) {}

      /// The result number that starts this group.
      unsigned startIndex;
      /// The source definition of the result group.
      SMDefinition definition;
    };

    OperationDefinition(Operation *op, SMRange loc, SMLoc endLoc)
        : op(op), loc(loc), scopeLoc(loc.Start, endLoc) {}

    /// The operation representing this definition.
    Operation *op;

    /// The source location for the operation, i.e. the location of its name.
    SMRange loc;

    /// The full source range of the operation definition, i.e. a range
    /// encompassing the start and end of the full operation definition.
    SMRange scopeLoc;

    /// Source definitions for any result groups of this operation.
    SmallVector<ResultGroupDefinition> resultGroups;

    /// If this operation is a symbol operation, this vector contains symbol
    /// uses of this operation.
    SmallVector<SMRange> symbolUses;
  };

  /// This class represents the information for a block definition within the
  /// input file.
  struct BlockDefinition {
    BlockDefinition(Block *block, SMRange loc = {})
        : block(block), definition(loc) {}

    /// The block representing this definition.
    Block *block;

    /// The source location for the block, i.e. the location of its name, and
    /// any uses it has.
    SMDefinition definition;

    /// Source definitions for any arguments of this block.
    SmallVector<SMDefinition> arguments;
  };

  /// This class represents the information for an attribute alias definition
  /// within the input file.
  struct AttributeAliasDefinition {
    AttributeAliasDefinition(StringRef name, SMRange loc = {},
                             Attribute value = {})
        : name(name), definition(loc), value(value) {}

    /// The name of the attribute alias.
    StringRef name;

    /// The source location for the alias.
    SMDefinition definition;

    /// The value of the alias.
    Attribute value;
  };

  /// This class represents the information for type definition within the input
  /// file.
  struct TypeAliasDefinition {
    TypeAliasDefinition(StringRef name, SMRange loc, Type value)
        : name(name), definition(loc), value(value) {}

    /// The name of the attribute alias.
    StringRef name;

    /// The source location for the alias.
    SMDefinition definition;

    /// The value of the alias.
    Type value;
  };

  AsmParserState();
  ~AsmParserState();
  AsmParserState &operator=(AsmParserState &&other);

  //===--------------------------------------------------------------------===//
  // Access State
  //===--------------------------------------------------------------------===//

  using BlockDefIterator = llvm::pointee_iterator<
      ArrayRef<std::unique_ptr<BlockDefinition>>::iterator>;
  using OperationDefIterator = llvm::pointee_iterator<
      ArrayRef<std::unique_ptr<OperationDefinition>>::iterator>;
  using AttributeDefIterator = llvm::pointee_iterator<
      ArrayRef<std::unique_ptr<AttributeAliasDefinition>>::iterator>;
  using TypeDefIterator = llvm::pointee_iterator<
      ArrayRef<std::unique_ptr<TypeAliasDefinition>>::iterator>;

  /// Return a range of the BlockDefinitions held by the current parser state.
  iterator_range<BlockDefIterator> getBlockDefs() const;

  /// Return the definition for the given block, or nullptr if the given
  /// block does not have a definition.
  const BlockDefinition *getBlockDef(Block *block) const;

  /// Return a range of the OperationDefinitions held by the current parser
  /// state.
  iterator_range<OperationDefIterator> getOpDefs() const;

  /// Return the definition for the given operation, or nullptr if the given
  /// operation does not have a definition.
  const OperationDefinition *getOpDef(Operation *op) const;

  /// Return a range of the AttributeAliasDefinitions held by the current parser
  /// state.
  iterator_range<AttributeDefIterator> getAttributeAliasDefs() const;

  /// Return the definition for the given attribute alias, or nullptr if the
  /// given alias does not have a definition.
  const AttributeAliasDefinition *getAttributeAliasDef(StringRef name) const;

  /// Return a range of the TypeAliasDefinitions held by the current parser
  /// state.
  iterator_range<TypeDefIterator> getTypeAliasDefs() const;

  /// Return the definition for the given type alias, or nullptr if the given
  /// alias does not have a definition.
  const TypeAliasDefinition *getTypeAliasDef(StringRef name) const;

  /// Returns (heuristically) the range of an identifier given a SMLoc
  /// corresponding to the start of an identifier location.
  static SMRange convertIdLocToRange(SMLoc loc);

  //===--------------------------------------------------------------------===//
  // Populate State
  //===--------------------------------------------------------------------===//

  /// Initialize the state in preparation for populating more parser state under
  /// the given top-level operation.
  void initialize(Operation *topLevelOp);

  /// Finalize any in-progress parser state under the given top-level operation.
  void finalize(Operation *topLevelOp);

  /// Start a definition for an operation with the given name.
  void startOperationDefinition(const OperationName &opName);

  /// Finalize the most recently started operation definition.
  void finalizeOperationDefinition(
      Operation *op, SMRange nameLoc, SMLoc endLoc,
      ArrayRef<std::pair<unsigned, SMLoc>> resultGroups = std::nullopt);

  /// Start a definition for a region nested under the current operation.
  void startRegionDefinition();

  /// Finalize the most recently started region definition.
  void finalizeRegionDefinition();

  /// Add a definition of the given entity.
  void addDefinition(Block *block, SMLoc location);
  void addDefinition(BlockArgument blockArg, SMLoc location);
  void addAttrAliasDefinition(StringRef name, SMRange location,
                              Attribute value);
  void addTypeAliasDefinition(StringRef name, SMRange location, Type value);

  /// Add a source uses of the given value.
  void addUses(Value value, ArrayRef<SMLoc> locations);
  void addUses(Block *block, ArrayRef<SMLoc> locations);
  void addAttrAliasUses(StringRef name, SMRange locations);
  void addTypeAliasUses(StringRef name, SMRange locations);

  /// Add source uses for all the references nested under `refAttr`. The
  /// provided `locations` should match 1-1 with the number of references in
  /// `refAttr`, i.e.:
  ///   nestedReferences.size() + /*leafReference=*/1 == refLocations.size()
  void addUses(SymbolRefAttr refAttr, ArrayRef<SMRange> refLocations);

  /// Refine the `oldValue` to the `newValue`. This is used to indicate that
  /// `oldValue` was a placeholder, and the uses of it should really refer to
  /// `newValue`.
  void refineDefinition(Value oldValue, Value newValue);

private:
  struct Impl;

  /// A pointer to the internal implementation of this class.
  std::unique_ptr<Impl> impl;
};

} // namespace mlir

#endif // MLIR_ASMPARSER_ASMPARSERSTATE_H
