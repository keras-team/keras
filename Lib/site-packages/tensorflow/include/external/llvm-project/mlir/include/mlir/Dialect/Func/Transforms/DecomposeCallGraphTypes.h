//===- DecomposeCallGraphTypes.h - CG type decompositions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Conversion patterns for decomposing types along call graph edges. That is,
// decomposing types for calls, returns, and function args.
//
// TODO: Make this handle dialect-defined functions, calls, and returns.
// Currently, the generic interfaces aren't sophisticated enough for the
// types of mutations that we are doing here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_TRANSFORMS_DECOMPOSECALLGRAPHTYPES_H
#define MLIR_DIALECT_FUNC_TRANSFORMS_DECOMPOSECALLGRAPHTYPES_H

#include "mlir/Transforms/DialectConversion.h"
#include <optional>

namespace mlir {

/// This class provides a hook that expands one Value into multiple Value's,
/// with a TypeConverter-inspired callback registration mechanism.
///
/// For folks that are familiar with the dialect conversion framework /
/// TypeConverter, this is effectively the inverse of a source/argument
/// materialization. A target materialization is not what we want here because
/// it always produces a single Value, but in this case the whole point is to
/// decompose a Value into multiple Value's.
///
/// The reason we need this inverse is easily understood by looking at what we
/// need to do for decomposing types for a return op. When converting a return
/// op, the dialect conversion framework will give the list of converted
/// operands, and will ensure that each converted operand, even if it expanded
/// into multiple types, is materialized as a single result. We then need to
/// undo that materialization to a single result, which we do with the
/// decomposeValue hooks registered on this object.
///
/// TODO: Eventually, the type conversion infra should have this hook built-in.
/// See
/// https://llvm.discourse.group/t/extending-type-conversion-infrastructure/779/2
class ValueDecomposer {
public:
  /// This method tries to decompose a value of a certain type using provided
  /// decompose callback functions. If it is unable to do so, the original value
  /// is returned.
  void decomposeValue(OpBuilder &, Location, Type, Value,
                      SmallVectorImpl<Value> &);

  /// This method registers a callback function that will be called to decompose
  /// a value of a certain type into 0, 1, or multiple values.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<2>>
  void addDecomposeValueConversion(FnT &&callback) {
    decomposeValueConversions.emplace_back(
        wrapDecomposeValueConversionCallback<T>(std::forward<FnT>(callback)));
  }

private:
  using DecomposeValueConversionCallFn =
      std::function<std::optional<LogicalResult>(
          OpBuilder &, Location, Type, Value, SmallVectorImpl<Value> &)>;

  /// Generate a wrapper for the given decompose value conversion callback.
  template <typename T, typename FnT>
  DecomposeValueConversionCallFn
  wrapDecomposeValueConversionCallback(FnT &&callback) {
    return
        [callback = std::forward<FnT>(callback)](
            OpBuilder &builder, Location loc, Type type, Value value,
            SmallVectorImpl<Value> &newValues) -> std::optional<LogicalResult> {
          if (T derivedType = dyn_cast<T>(type))
            return callback(builder, loc, derivedType, value, newValues);
          return std::nullopt;
        };
  }

  SmallVector<DecomposeValueConversionCallFn, 2> decomposeValueConversions;
};

/// Populates the patterns needed to drive the conversion process for
/// decomposing call graph types with the given `ValueDecomposer`.
void populateDecomposeCallGraphTypesPatterns(MLIRContext *context,
                                             TypeConverter &typeConverter,
                                             ValueDecomposer &decomposer,
                                             RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_DIALECT_FUNC_TRANSFORMS_DECOMPOSECALLGRAPHTYPES_H
