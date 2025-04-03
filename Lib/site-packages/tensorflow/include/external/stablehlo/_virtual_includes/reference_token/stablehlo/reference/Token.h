/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_REFERENCE_TOKEN_H
#define STABLEHLO_REFERENCE_TOKEN_H

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

/// Class to model a Token.
class Token {
 public:
  /// \name Constructors
  /// @{
  Token(MLIRContext *context);
  /// @}

  /// Returns the type of the Token object.
  TokenType getType() const;

  /// Prints Token object.
  void print(raw_ostream &os) const;
  void dump() const;

 private:
  TokenType type_;
};

/// Print utilities for Token objects.
inline raw_ostream &operator<<(raw_ostream &os, Token Token) {
  Token.print(os);
  return os;
}

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_TOKEN_H
