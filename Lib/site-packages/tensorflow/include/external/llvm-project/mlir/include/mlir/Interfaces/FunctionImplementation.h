//===- FunctionImplementation.h - Function-like Op utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility functions for implementing function-like
// operations, in particular, parsing, printing and verification components
// common to function-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTIONIMPLEMENTATION_H_
#define MLIR_IR_FUNCTIONIMPLEMENTATION_H_

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {

namespace function_interface_impl {

/// A named class for passing around the variadic flag.
class VariadicFlag {
public:
  explicit VariadicFlag(bool variadic) : variadic(variadic) {}
  bool isVariadic() const { return variadic; }

private:
  /// Underlying storage.
  bool variadic;
};

/// Adds argument and result attributes, provided as `argAttrs` and
/// `resultAttrs` arguments, to the list of operation attributes in `result`.
/// Internally, argument and result attributes are stored as dict attributes
/// with special names given by getResultAttrName, getArgumentAttrName.
void addArgAndResultAttrs(Builder &builder, OperationState &result,
                          ArrayRef<DictionaryAttr> argAttrs,
                          ArrayRef<DictionaryAttr> resultAttrs,
                          StringAttr argAttrsName, StringAttr resAttrsName);
void addArgAndResultAttrs(Builder &builder, OperationState &result,
                          ArrayRef<OpAsmParser::Argument> args,
                          ArrayRef<DictionaryAttr> resultAttrs,
                          StringAttr argAttrsName, StringAttr resAttrsName);

/// Callback type for `parseFunctionOp`, the callback should produce the
/// type that will be associated with a function-like operation from lists of
/// function arguments and results, VariadicFlag indicates whether the function
/// should have variadic arguments; in case of error, it may populate the last
/// argument with a message.
using FuncTypeBuilder = function_ref<Type(
    Builder &, ArrayRef<Type>, ArrayRef<Type>, VariadicFlag, std::string &)>;

/// Parses a function signature using `parser`. The `allowVariadic` argument
/// indicates whether functions with variadic arguments are supported. The
/// trailing arguments are populated by this function with names, types,
/// attributes and locations of the arguments and those of the results.
ParseResult
parseFunctionSignature(OpAsmParser &parser, bool allowVariadic,
                       SmallVectorImpl<OpAsmParser::Argument> &arguments,
                       bool &isVariadic, SmallVectorImpl<Type> &resultTypes,
                       SmallVectorImpl<DictionaryAttr> &resultAttrs);

/// Parser implementation for function-like operations.  Uses
/// `funcTypeBuilder` to construct the custom function type given lists of
/// input and output types. The parser sets the `typeAttrName` attribute to the
/// resulting function type. If `allowVariadic` is set, the parser will accept
/// trailing ellipsis in the function signature and indicate to the builder
/// whether the function is variadic.  If the builder returns a null type,
/// `result` will not contain the `type` attribute.  The caller can then add a
/// type, report the error or delegate the reporting to the op's verifier.
ParseResult parseFunctionOp(OpAsmParser &parser, OperationState &result,
                            bool allowVariadic, StringAttr typeAttrName,
                            FuncTypeBuilder funcTypeBuilder,
                            StringAttr argAttrsName, StringAttr resAttrsName);

/// Printer implementation for function-like operations.
void printFunctionOp(OpAsmPrinter &p, FunctionOpInterface op, bool isVariadic,
                     StringRef typeAttrName, StringAttr argAttrsName,
                     StringAttr resAttrsName);

/// Prints the signature of the function-like operation `op`. Assumes `op` has
/// is a FunctionOpInterface and has passed verification.
void printFunctionSignature(OpAsmPrinter &p, FunctionOpInterface op,
                            ArrayRef<Type> argTypes, bool isVariadic,
                            ArrayRef<Type> resultTypes);

/// Prints the list of function prefixed with the "attributes" keyword. The
/// attributes with names listed in "elided" as well as those used by the
/// function-like operation internally are not printed. Nothing is printed
/// if all attributes are elided. Assumes `op` is a FunctionOpInterface and
/// has passed verification.
void printFunctionAttributes(OpAsmPrinter &p, Operation *op,
                             ArrayRef<StringRef> elided = {});

} // namespace function_interface_impl

} // namespace mlir

#endif // MLIR_IR_FUNCTIONIMPLEMENTATION_H_
