//===- Builders.h - Helpers for constructing MLIR Classes -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILDERS_H
#define MLIR_IR_BUILDERS_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace mlir {

class AffineExpr;
class IRMapping;
class UnknownLoc;
class FileLineColLoc;
class Type;
class PrimitiveType;
class IntegerType;
class FloatType;
class FunctionType;
class IndexType;
class MemRefType;
class VectorType;
class RankedTensorType;
class UnrankedTensorType;
class TupleType;
class NoneType;
class BoolAttr;
class IntegerAttr;
class FloatAttr;
class StringAttr;
class TypeAttr;
class ArrayAttr;
class SymbolRefAttr;
class ElementsAttr;
class DenseElementsAttr;
class DenseIntElementsAttr;
class AffineMapAttr;
class AffineMap;
class UnitAttr;

/// This class is a general helper class for creating context-global objects
/// like types, attributes, and affine expressions.
class Builder {
public:
  explicit Builder(MLIRContext *context) : context(context) {}
  explicit Builder(Operation *op) : Builder(op->getContext()) {}

  MLIRContext *getContext() const { return context; }

  // Locations.
  Location getUnknownLoc();
  Location getFusedLoc(ArrayRef<Location> locs,
                       Attribute metadata = Attribute());

  // Types.
  FloatType getFloat6E2M3FNType();
  FloatType getFloat6E3M2FNType();
  FloatType getFloat8E5M2Type();
  FloatType getFloat8E4M3Type();
  FloatType getFloat8E4M3FNType();
  FloatType getFloat8E5M2FNUZType();
  FloatType getFloat8E4M3FNUZType();
  FloatType getFloat8E4M3B11FNUZType();
  FloatType getFloat8E3M4Type();
  FloatType getBF16Type();
  FloatType getF16Type();
  FloatType getTF32Type();
  FloatType getF32Type();
  FloatType getF64Type();
  FloatType getF80Type();
  FloatType getF128Type();

  IndexType getIndexType();

  IntegerType getI1Type();
  IntegerType getI2Type();
  IntegerType getI4Type();
  IntegerType getI8Type();
  IntegerType getI16Type();
  IntegerType getI32Type();
  IntegerType getI64Type();
  IntegerType getIntegerType(unsigned width);
  IntegerType getIntegerType(unsigned width, bool isSigned);
  FunctionType getFunctionType(TypeRange inputs, TypeRange results);
  TupleType getTupleType(TypeRange elementTypes);
  NoneType getNoneType();

  /// Get or construct an instance of the type `Ty` with provided arguments.
  template <typename Ty, typename... Args>
  Ty getType(Args &&...args) {
    return Ty::get(context, std::forward<Args>(args)...);
  }

  /// Get or construct an instance of the attribute `Attr` with provided
  /// arguments.
  template <typename Attr, typename... Args>
  Attr getAttr(Args &&...args) {
    return Attr::get(context, std::forward<Args>(args)...);
  }

  // Attributes.
  NamedAttribute getNamedAttr(StringRef name, Attribute val);

  UnitAttr getUnitAttr();
  BoolAttr getBoolAttr(bool value);
  DictionaryAttr getDictionaryAttr(ArrayRef<NamedAttribute> value);
  IntegerAttr getIntegerAttr(Type type, int64_t value);
  IntegerAttr getIntegerAttr(Type type, const APInt &value);
  FloatAttr getFloatAttr(Type type, double value);
  FloatAttr getFloatAttr(Type type, const APFloat &value);
  StringAttr getStringAttr(const Twine &bytes);
  ArrayAttr getArrayAttr(ArrayRef<Attribute> value);

  // Returns a 0-valued attribute of the given `type`. This function only
  // supports boolean, integer, and 16-/32-/64-bit float types, and vector or
  // ranked tensor of them. Returns null attribute otherwise.
  TypedAttr getZeroAttr(Type type);
  // Returns a 1-valued attribute of the given `type`.
  // Type constraints are the same as `getZeroAttr`.
  TypedAttr getOneAttr(Type type);

  // Convenience methods for fixed types.
  FloatAttr getF16FloatAttr(float value);
  FloatAttr getF32FloatAttr(float value);
  FloatAttr getF64FloatAttr(double value);

  IntegerAttr getI8IntegerAttr(int8_t value);
  IntegerAttr getI16IntegerAttr(int16_t value);
  IntegerAttr getI32IntegerAttr(int32_t value);
  IntegerAttr getI64IntegerAttr(int64_t value);
  IntegerAttr getIndexAttr(int64_t value);

  /// Signed and unsigned integer attribute getters.
  IntegerAttr getSI32IntegerAttr(int32_t value);
  IntegerAttr getUI32IntegerAttr(uint32_t value);

  /// Vector-typed DenseIntElementsAttr getters. `values` must not be empty.
  DenseIntElementsAttr getBoolVectorAttr(ArrayRef<bool> values);
  DenseIntElementsAttr getI32VectorAttr(ArrayRef<int32_t> values);
  DenseIntElementsAttr getI64VectorAttr(ArrayRef<int64_t> values);
  DenseIntElementsAttr getIndexVectorAttr(ArrayRef<int64_t> values);

  DenseFPElementsAttr getF32VectorAttr(ArrayRef<float> values);
  DenseFPElementsAttr getF64VectorAttr(ArrayRef<double> values);

  /// Tensor-typed DenseIntElementsAttr getters. `values` can be empty.
  /// These are generally preferable for representing general lists of integers
  /// as attributes.
  DenseIntElementsAttr getI32TensorAttr(ArrayRef<int32_t> values);
  DenseIntElementsAttr getI64TensorAttr(ArrayRef<int64_t> values);
  DenseIntElementsAttr getIndexTensorAttr(ArrayRef<int64_t> values);

  /// Tensor-typed DenseArrayAttr getters.
  DenseBoolArrayAttr getDenseBoolArrayAttr(ArrayRef<bool> values);
  DenseI8ArrayAttr getDenseI8ArrayAttr(ArrayRef<int8_t> values);
  DenseI16ArrayAttr getDenseI16ArrayAttr(ArrayRef<int16_t> values);
  DenseI32ArrayAttr getDenseI32ArrayAttr(ArrayRef<int32_t> values);
  DenseI64ArrayAttr getDenseI64ArrayAttr(ArrayRef<int64_t> values);
  DenseF32ArrayAttr getDenseF32ArrayAttr(ArrayRef<float> values);
  DenseF64ArrayAttr getDenseF64ArrayAttr(ArrayRef<double> values);

  ArrayAttr getAffineMapArrayAttr(ArrayRef<AffineMap> values);
  ArrayAttr getBoolArrayAttr(ArrayRef<bool> values);
  ArrayAttr getI32ArrayAttr(ArrayRef<int32_t> values);
  ArrayAttr getI64ArrayAttr(ArrayRef<int64_t> values);
  ArrayAttr getIndexArrayAttr(ArrayRef<int64_t> values);
  ArrayAttr getF32ArrayAttr(ArrayRef<float> values);
  ArrayAttr getF64ArrayAttr(ArrayRef<double> values);
  ArrayAttr getStrArrayAttr(ArrayRef<StringRef> values);
  ArrayAttr getTypeArrayAttr(TypeRange values);

  // Affine expressions and affine maps.
  AffineExpr getAffineDimExpr(unsigned position);
  AffineExpr getAffineSymbolExpr(unsigned position);
  AffineExpr getAffineConstantExpr(int64_t constant);

  // Special cases of affine maps and integer sets
  /// Returns a zero result affine map with no dimensions or symbols: () -> ().
  AffineMap getEmptyAffineMap();
  /// Returns a single constant result affine map with 0 dimensions and 0
  /// symbols.  One constant result: () -> (val).
  AffineMap getConstantAffineMap(int64_t val);
  // One dimension id identity map: (i) -> (i).
  AffineMap getDimIdentityMap();
  // Multi-dimensional identity map: (d0, d1, d2) -> (d0, d1, d2).
  AffineMap getMultiDimIdentityMap(unsigned rank);
  // One symbol identity map: ()[s] -> (s).
  AffineMap getSymbolIdentityMap();

  /// Returns a map that shifts its (single) input dimension by 'shift'.
  /// (d0) -> (d0 + shift)
  AffineMap getSingleDimShiftAffineMap(int64_t shift);

  /// Returns an affine map that is a translation (shift) of all result
  /// expressions in 'map' by 'shift'.
  /// Eg: input: (d0, d1)[s0] -> (d0, d1 + s0), shift = 2
  ///   returns:    (d0, d1)[s0] -> (d0 + 2, d1 + s0 + 2)
  AffineMap getShiftedAffineMap(AffineMap map, int64_t shift);

protected:
  MLIRContext *context;
};

/// This class helps build Operations. Operations that are created are
/// automatically inserted at an insertion point. The builder is copyable.
class OpBuilder : public Builder {
public:
  class InsertPoint;
  struct Listener;

  /// Create a builder with the given context.
  explicit OpBuilder(MLIRContext *ctx, Listener *listener = nullptr)
      : Builder(ctx), listener(listener) {}

  /// Create a builder and set the insertion point to the start of the region.
  explicit OpBuilder(Region *region, Listener *listener = nullptr)
      : OpBuilder(region->getContext(), listener) {
    if (!region->empty())
      setInsertionPoint(&region->front(), region->front().begin());
  }
  explicit OpBuilder(Region &region, Listener *listener = nullptr)
      : OpBuilder(&region, listener) {}

  /// Create a builder and set insertion point to the given operation, which
  /// will cause subsequent insertions to go right before it.
  explicit OpBuilder(Operation *op, Listener *listener = nullptr)
      : OpBuilder(op->getContext(), listener) {
    setInsertionPoint(op);
  }

  OpBuilder(Block *block, Block::iterator insertPoint,
            Listener *listener = nullptr)
      : OpBuilder(block->getParent()->getContext(), listener) {
    setInsertionPoint(block, insertPoint);
  }

  /// Create a builder and set the insertion point to before the first operation
  /// in the block but still inside the block.
  static OpBuilder atBlockBegin(Block *block, Listener *listener = nullptr) {
    return OpBuilder(block, block->begin(), listener);
  }

  /// Create a builder and set the insertion point to after the last operation
  /// in the block but still inside the block.
  static OpBuilder atBlockEnd(Block *block, Listener *listener = nullptr) {
    return OpBuilder(block, block->end(), listener);
  }

  /// Create a builder and set the insertion point to before the block
  /// terminator.
  static OpBuilder atBlockTerminator(Block *block,
                                     Listener *listener = nullptr) {
    auto *terminator = block->getTerminator();
    assert(terminator != nullptr && "the block has no terminator");
    return OpBuilder(block, Block::iterator(terminator), listener);
  }

  //===--------------------------------------------------------------------===//
  // Listeners
  //===--------------------------------------------------------------------===//

  /// Base class for listeners.
  struct ListenerBase {
    /// The kind of listener.
    enum class Kind {
      /// OpBuilder::Listener or user-derived class.
      OpBuilderListener = 0,

      /// RewriterBase::Listener or user-derived class.
      RewriterBaseListener = 1
    };

    Kind getKind() const { return kind; }

  protected:
    ListenerBase(Kind kind) : kind(kind) {}

  private:
    const Kind kind;
  };

  /// This class represents a listener that may be used to hook into various
  /// actions within an OpBuilder.
  struct Listener : public ListenerBase {
    Listener() : ListenerBase(ListenerBase::Kind::OpBuilderListener) {}

    virtual ~Listener() = default;

    /// Notify the listener that the specified operation was inserted.
    ///
    /// * If the operation was moved, then `previous` is the previous location
    ///   of the op.
    /// * If the operation was unlinked before it was inserted, then `previous`
    ///   is empty.
    ///
    /// Note: Creating an (unlinked) op does not trigger this notification.
    virtual void notifyOperationInserted(Operation *op, InsertPoint previous) {}

    /// Notify the listener that the specified block was inserted.
    ///
    /// * If the block was moved, then `previous` and `previousIt` are the
    ///   previous location of the block.
    /// * If the block was unlinked before it was inserted, then `previous`
    ///   is "nullptr".
    ///
    /// Note: Creating an (unlinked) block does not trigger this notification.
    virtual void notifyBlockInserted(Block *block, Region *previous,
                                     Region::iterator previousIt) {}

  protected:
    Listener(Kind kind) : ListenerBase(kind) {}
  };

  /// Sets the listener of this builder to the one provided.
  void setListener(Listener *newListener) { listener = newListener; }

  /// Returns the current listener of this builder, or nullptr if this builder
  /// doesn't have a listener.
  Listener *getListener() const { return listener; }

  //===--------------------------------------------------------------------===//
  // Insertion Point Management
  //===--------------------------------------------------------------------===//

  /// This class represents a saved insertion point.
  class InsertPoint {
  public:
    /// Creates a new insertion point which doesn't point to anything.
    InsertPoint() = default;

    /// Creates a new insertion point at the given location.
    InsertPoint(Block *insertBlock, Block::iterator insertPt)
        : block(insertBlock), point(insertPt) {}

    /// Returns true if this insert point is set.
    bool isSet() const { return (block != nullptr); }

    Block *getBlock() const { return block; }
    Block::iterator getPoint() const { return point; }

  private:
    Block *block = nullptr;
    Block::iterator point;
  };

  /// RAII guard to reset the insertion point of the builder when destroyed.
  class InsertionGuard {
  public:
    InsertionGuard(OpBuilder &builder)
        : builder(&builder), ip(builder.saveInsertionPoint()) {}

    ~InsertionGuard() {
      if (builder)
        builder->restoreInsertionPoint(ip);
    }

    InsertionGuard(const InsertionGuard &) = delete;
    InsertionGuard &operator=(const InsertionGuard &) = delete;

    /// Implement the move constructor to clear the builder field of `other`.
    /// That way it does not restore the insertion point upon destruction as
    /// that should be done exclusively by the just constructed InsertionGuard.
    InsertionGuard(InsertionGuard &&other) noexcept
        : builder(other.builder), ip(other.ip) {
      other.builder = nullptr;
    }

    InsertionGuard &operator=(InsertionGuard &&other) = delete;

  private:
    OpBuilder *builder;
    OpBuilder::InsertPoint ip;
  };

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = Block::iterator();
  }

  /// Return a saved insertion point.
  InsertPoint saveInsertionPoint() const {
    return InsertPoint(getInsertionBlock(), getInsertionPoint());
  }

  /// Restore the insert point to a previously saved point.
  void restoreInsertionPoint(InsertPoint ip) {
    if (ip.isSet())
      setInsertionPoint(ip.getBlock(), ip.getPoint());
    else
      clearInsertionPoint();
  }

  /// Set the insertion point to the specified location.
  void setInsertionPoint(Block *block, Block::iterator insertPoint) {
    // TODO: check that insertPoint is in this rather than some other block.
    this->block = block;
    this->insertPoint = insertPoint;
  }

  /// Sets the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void setInsertionPoint(Operation *op) {
    setInsertionPoint(op->getBlock(), Block::iterator(op));
  }

  /// Sets the insertion point to the node after the specified operation, which
  /// will cause subsequent insertions to go right after it.
  void setInsertionPointAfter(Operation *op) {
    setInsertionPoint(op->getBlock(), ++Block::iterator(op));
  }

  /// Sets the insertion point to the node after the specified value. If value
  /// has a defining operation, sets the insertion point to the node after such
  /// defining operation. This will cause subsequent insertions to go right
  /// after it. Otherwise, value is a BlockArgument. Sets the insertion point to
  /// the start of its block.
  void setInsertionPointAfterValue(Value val) {
    if (Operation *op = val.getDefiningOp()) {
      setInsertionPointAfter(op);
    } else {
      auto blockArg = llvm::cast<BlockArgument>(val);
      setInsertionPointToStart(blockArg.getOwner());
    }
  }

  /// Sets the insertion point to the start of the specified block.
  void setInsertionPointToStart(Block *block) {
    setInsertionPoint(block, block->begin());
  }

  /// Sets the insertion point to the end of the specified block.
  void setInsertionPointToEnd(Block *block) {
    setInsertionPoint(block, block->end());
  }

  /// Return the block the current insertion point belongs to.  Note that the
  /// insertion point is not necessarily the end of the block.
  Block *getInsertionBlock() const { return block; }

  /// Returns the current insertion point of the builder.
  Block::iterator getInsertionPoint() const { return insertPoint; }

  /// Returns the current block of the builder.
  Block *getBlock() const { return block; }

  //===--------------------------------------------------------------------===//
  // Block Creation
  //===--------------------------------------------------------------------===//

  /// Add new block with 'argTypes' arguments and set the insertion point to the
  /// end of it. The block is inserted at the provided insertion point of
  /// 'parent'. `locs` contains the locations of the inserted arguments, and
  /// should match the size of `argTypes`.
  Block *createBlock(Region *parent, Region::iterator insertPt = {},
                     TypeRange argTypes = std::nullopt,
                     ArrayRef<Location> locs = std::nullopt);

  /// Add new block with 'argTypes' arguments and set the insertion point to the
  /// end of it. The block is placed before 'insertBefore'. `locs` contains the
  /// locations of the inserted arguments, and should match the size of
  /// `argTypes`.
  Block *createBlock(Block *insertBefore, TypeRange argTypes = std::nullopt,
                     ArrayRef<Location> locs = std::nullopt);

  //===--------------------------------------------------------------------===//
  // Operation Creation
  //===--------------------------------------------------------------------===//

  /// Insert the given operation at the current insertion point and return it.
  Operation *insert(Operation *op);

  /// Creates an operation given the fields represented as an OperationState.
  Operation *create(const OperationState &state);

  /// Creates an operation with the given fields.
  Operation *create(Location loc, StringAttr opName, ValueRange operands,
                    TypeRange types = {},
                    ArrayRef<NamedAttribute> attributes = {},
                    BlockRange successors = {},
                    MutableArrayRef<std::unique_ptr<Region>> regions = {});

private:
  /// Helper for sanity checking preconditions for create* methods below.
  template <typename OpT>
  RegisteredOperationName getCheckRegisteredInfo(MLIRContext *ctx) {
    std::optional<RegisteredOperationName> opName =
        RegisteredOperationName::lookup(TypeID::get<OpT>(), ctx);
    if (LLVM_UNLIKELY(!opName)) {
      llvm::report_fatal_error(
          "Building op `" + OpT::getOperationName() +
          "` but it isn't known in this MLIRContext: the dialect may not "
          "be loaded or this operation hasn't been added by the dialect. See "
          "also https://mlir.llvm.org/getting_started/Faq/"
          "#registered-loaded-dependent-whats-up-with-dialects-management");
    }
    return *opName;
  }

public:
  /// Create an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    OperationState state(location,
                         getCheckRegisteredInfo<OpTy>(location.getContext()));
    OpTy::build(*this, state, std::forward<Args>(args)...);
    auto *op = create(state);
    auto result = dyn_cast<OpTy>(op);
    assert(result && "builder didn't return the right type");
    return result;
  }

  /// Create an operation of specific op type at the current insertion point,
  /// and immediately try to fold it. This functions populates 'results' with
  /// the results of the operation.
  template <typename OpTy, typename... Args>
  void createOrFold(SmallVectorImpl<Value> &results, Location location,
                    Args &&...args) {
    // Create the operation without using 'create' as we want to control when
    // the listener is notified.
    OperationState state(location,
                         getCheckRegisteredInfo<OpTy>(location.getContext()));
    OpTy::build(*this, state, std::forward<Args>(args)...);
    Operation *op = Operation::create(state);
    if (block)
      block->getOperations().insert(insertPoint, op);

    // Attempt to fold the operation.
    if (succeeded(tryFold(op, results)) && !results.empty()) {
      // Erase the operation, if the fold removed the need for this operation.
      // Note: The fold already populated the results in this case.
      op->erase();
      return;
    }

    ResultRange opResults = op->getResults();
    results.assign(opResults.begin(), opResults.end());
    if (block && listener)
      listener->notifyOperationInserted(op, /*previous=*/{});
  }

  /// Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value>
  createOrFold(Location location, Args &&...args) {
    SmallVector<Value, 1> results;
    createOrFold<OpTy>(results, location, std::forward<Args>(args)...);
    return results.front();
  }

  /// Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy>
  createOrFold(Location location, Args &&...args) {
    auto op = create<OpTy>(location, std::forward<Args>(args)...);
    SmallVector<Value, 0> unused;
    (void)tryFold(op.getOperation(), unused);

    // Folding cannot remove a zero-result operation, so for convenience we
    // continue to return it.
    return op;
  }

  /// Attempts to fold the given operation and places new results within
  /// `results`. Returns success if the operation was folded, failure otherwise.
  /// If the fold was in-place, `results` will not be filled.
  /// Note: This function does not erase the operation on a successful fold.
  LogicalResult tryFold(Operation *op, SmallVectorImpl<Value> &results);

  /// Creates a deep copy of the specified operation, remapping any operands
  /// that use values outside of the operation using the map that is provided
  /// ( leaving them alone if no entry is present).  Replaces references to
  /// cloned sub-operations to the corresponding operation that is copied,
  /// and adds those mappings to the map.
  Operation *clone(Operation &op, IRMapping &mapper);
  Operation *clone(Operation &op);

  /// Creates a deep copy of this operation but keep the operation regions
  /// empty. Operands are remapped using `mapper` (if present), and `mapper` is
  /// updated to contain the results.
  Operation *cloneWithoutRegions(Operation &op, IRMapping &mapper) {
    return insert(op.cloneWithoutRegions(mapper));
  }
  Operation *cloneWithoutRegions(Operation &op) {
    return insert(op.cloneWithoutRegions());
  }
  template <typename OpT>
  OpT cloneWithoutRegions(OpT op) {
    return cast<OpT>(cloneWithoutRegions(*op.getOperation()));
  }

  /// Clone the blocks that belong to "region" before the given position in
  /// another region "parent". The two regions must be different. The caller is
  /// responsible for creating or updating the operation transferring flow of
  /// control to the region and passing it the correct block arguments.
  void cloneRegionBefore(Region &region, Region &parent,
                         Region::iterator before, IRMapping &mapping);
  void cloneRegionBefore(Region &region, Region &parent,
                         Region::iterator before);
  void cloneRegionBefore(Region &region, Block *before);

protected:
  /// The optional listener for events of this builder.
  Listener *listener;

private:
  /// The current block this builder is inserting into.
  Block *block = nullptr;
  /// The insertion point within the block that this builder is inserting
  /// before.
  Block::iterator insertPoint;
};

} // namespace mlir

#endif
