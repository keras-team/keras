/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_VHLO_OPS_H
#define STABLEHLO_DIALECT_VHLO_OPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloTypes.h"

namespace mlir {
namespace vhlo {

class VhloDialect : public Dialect {
 public:
  explicit VhloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "vhlo"; }

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  // Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

 private:
  // Adds VHLO types to this dialect.
  // See implementation comments for additional details.
  void addVhloTypes();

  // Does the same this as Dialect::addTypes but without calling registerType.
  // See comments for `addVhloTypes` for additional details.
  template <typename... Types>
  void addTypesWithoutRegistering() {
    (addType(Types::getTypeID(), AbstractType::get<Types>(*this)), ...);
  }
};

}  // namespace vhlo
}  // namespace mlir

// Attrs and Enums
#include "stablehlo/dialect/VhloAttrInterfaces.h.inc"
#include "stablehlo/dialect/VhloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/VhloAttrs.h.inc"

// Ops
#include "stablehlo/dialect/VhloOpInterfaces.h.inc"
#define GET_OP_CLASSES
#include "stablehlo/dialect/VhloOps.h.inc"

#endif  // STABLEHLO_DIALECT_VHLO_OPS_H
