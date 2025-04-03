//===- ExtensibleDialect.h - Extensible dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DynamicOpDefinition class, the DynamicTypeDefinition
// class, and the DynamicAttrDefinition class, which represent respectively
// operations, types, and attributes that can be defined at runtime. They can
// be registered at runtime to an extensible dialect, using the
// ExtensibleDialect class defined in this file.
//
// For a more complete documentation, see
// https://mlir.llvm.org/docs/ExtensibleDialects/ .
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_EXTENSIBLEDIALECT_H
#define MLIR_IR_EXTENSIBLEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace mlir {
class AsmParser;
class AsmPrinter;
class DynamicAttr;
class DynamicType;
class ExtensibleDialect;
class MLIRContext;
class OptionalParseResult;

namespace detail {
struct DynamicAttrStorage;
struct DynamicTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// Dynamic attribute
//===----------------------------------------------------------------------===//

/// The definition of a dynamic attribute. A dynamic attribute is an attribute
/// that is defined at runtime, and that can be registered at runtime by an
/// extensible dialect (a dialect inheriting ExtensibleDialect). This class
/// stores the parser, the printer, and the verifier of the attribute. Each
/// dynamic attribute definition refers to one instance of this class.
class DynamicAttrDefinition : public SelfOwningTypeID {
public:
  using VerifierFn = llvm::unique_function<LogicalResult(
      function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) const>;
  using ParserFn = llvm::unique_function<ParseResult(
      AsmParser &parser, llvm::SmallVectorImpl<Attribute> &parsedAttributes)
                                             const>;
  using PrinterFn = llvm::unique_function<void(
      AsmPrinter &printer, ArrayRef<Attribute> params) const>;

  /// Create a new attribute definition at runtime. The attribute is registered
  /// only after passing it to the dialect using registerDynamicAttr.
  static std::unique_ptr<DynamicAttrDefinition>
  get(StringRef name, ExtensibleDialect *dialect, VerifierFn &&verifier);
  static std::unique_ptr<DynamicAttrDefinition>
  get(StringRef name, ExtensibleDialect *dialect, VerifierFn &&verifier,
      ParserFn &&parser, PrinterFn &&printer);

  /// Sets the verifier function for this attribute. It should emits an error
  /// message and returns failure if a problem is detected, or returns success
  /// if everything is ok.
  void setVerifyFn(VerifierFn &&verify) { verifier = std::move(verify); }

  /// Sets the static hook for parsing this attribute assembly.
  void setParseFn(ParserFn &&parse) { parser = std::move(parse); }

  /// Sets the static hook for printing this attribute assembly.
  void setPrintFn(PrinterFn &&print) { printer = std::move(print); }

  /// Check that the attribute parameters are valid.
  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> params) const {
    return verifier(emitError, params);
  }

  /// Return the MLIRContext in which the dynamic attributes are uniqued.
  MLIRContext &getContext() const { return *ctx; }

  /// Return the name of the attribute, in the format 'attrname' and
  /// not 'dialectname.attrname'.
  StringRef getName() const { return name; }

  /// Return the dialect defining the attribute.
  ExtensibleDialect *getDialect() const { return dialect; }

private:
  DynamicAttrDefinition(StringRef name, ExtensibleDialect *dialect,
                        VerifierFn &&verifier, ParserFn &&parser,
                        PrinterFn &&printer);

  /// This constructor should only be used when we need a pointer to
  /// the DynamicAttrDefinition in the verifier, the parser, or the printer.
  /// The verifier, parser, and printer need thus to be initialized after the
  /// constructor.
  DynamicAttrDefinition(ExtensibleDialect *dialect, StringRef name);

  /// Register the concrete attribute in the attribute Uniquer.
  void registerInAttrUniquer();

  /// The name should be prefixed with the dialect name followed by '.'.
  std::string name;

  /// Dialect in which this attribute is defined.
  ExtensibleDialect *dialect;

  /// The attribute verifier. It checks that the attribute parameters satisfy
  /// the invariants.
  VerifierFn verifier;

  /// The attribute parameters parser. It parses only the parameters, and
  /// expects the attribute name to have already been parsed.
  ParserFn parser;

  /// The attribute parameters printer. It prints only the parameters, and
  /// expects the attribute name to have already been printed.
  PrinterFn printer;

  /// Context in which the concrete attributes are uniqued.
  MLIRContext *ctx;

  friend ExtensibleDialect;
  friend DynamicAttr;
};

/// This trait is used to determine if an attribute is a dynamic attribute or
/// not; it should only be implemented by dynamic attributes.
/// Note: This is only required because dynamic attributes do not have a
/// static/single TypeID.
namespace AttributeTrait {
template <typename ConcreteType>
class IsDynamicAttr : public TraitBase<ConcreteType, IsDynamicAttr> {};
} // namespace AttributeTrait

/// A dynamic attribute instance. This is an attribute whose definition is
/// defined at runtime.
/// It is possible to check if an attribute is a dynamic attribute using
/// `my_attr.isa<DynamicAttr>()`, and getting the attribute definition of a
/// dynamic attribute using the `DynamicAttr::getAttrDef` method.
/// All dynamic attributes have the same storage, which is an array of
/// attributes.

class DynamicAttr : public Attribute::AttrBase<DynamicAttr, Attribute,
                                               detail::DynamicAttrStorage,
                                               AttributeTrait::IsDynamicAttr> {
public:
  // Inherit Base constructors.
  using Base::Base;

  /// Return an instance of a dynamic attribute given a dynamic attribute
  /// definition and attribute parameters.
  /// This asserts that the attribute verifier succeeded.
  static DynamicAttr get(DynamicAttrDefinition *attrDef,
                         ArrayRef<Attribute> params = {});

  /// Return an instance of a dynamic attribute given a dynamic attribute
  /// definition and attribute parameters. If the parameters provided are
  /// invalid, errors are emitted using the provided location and a null object
  /// is returned.
  static DynamicAttr getChecked(function_ref<InFlightDiagnostic()> emitError,
                                DynamicAttrDefinition *attrDef,
                                ArrayRef<Attribute> params = {});

  /// Return the attribute definition of the concrete attribute.
  DynamicAttrDefinition *getAttrDef();

  /// Return the attribute parameters.
  ArrayRef<Attribute> getParams();

  /// Check if an attribute is a specific dynamic attribute.
  static bool isa(Attribute attr, DynamicAttrDefinition *attrDef) {
    return attr.getTypeID() == attrDef->getTypeID();
  }

  /// Check if an attribute is a dynamic attribute.
  static bool classof(Attribute attr);

  /// Parse the dynamic attribute parameters and construct the attribute.
  /// The parameters are either empty, and nothing is parsed,
  /// or they are in the format '<>' or '<attr (,attr)*>'.
  static ParseResult parse(AsmParser &parser, DynamicAttrDefinition *attrDef,
                           DynamicAttr &parsedAttr);

  /// Print the dynamic attribute with the format 'attrname' if there is no
  /// parameters, or 'attrname<attr (,attr)*>'.
  void print(AsmPrinter &printer);
};

//===----------------------------------------------------------------------===//
// Dynamic type
//===----------------------------------------------------------------------===//

/// The definition of a dynamic type. A dynamic type is a type that is
/// defined at runtime, and that can be registered at runtime by an
/// extensible dialect (a dialect inheriting ExtensibleDialect). This class
/// stores the parser, the printer, and the verifier of the type. Each dynamic
/// type definition refers to one instance of this class.
class DynamicTypeDefinition : public SelfOwningTypeID {
public:
  using VerifierFn = llvm::unique_function<LogicalResult(
      function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) const>;
  using ParserFn = llvm::unique_function<ParseResult(
      AsmParser &parser, llvm::SmallVectorImpl<Attribute> &parsedAttributes)
                                             const>;
  using PrinterFn = llvm::unique_function<void(
      AsmPrinter &printer, ArrayRef<Attribute> params) const>;

  /// Create a new dynamic type definition. The type is registered only after
  /// passing it to the dialect using registerDynamicType.
  static std::unique_ptr<DynamicTypeDefinition>
  get(StringRef name, ExtensibleDialect *dialect, VerifierFn &&verifier);
  static std::unique_ptr<DynamicTypeDefinition>
  get(StringRef name, ExtensibleDialect *dialect, VerifierFn &&verifier,
      ParserFn &&parser, PrinterFn &&printer);

  /// Sets the verifier function for this type. It should emits an error
  /// message and returns failure if a problem is detected, or returns success
  /// if everything is ok.
  void setVerifyFn(VerifierFn &&verify) { verifier = std::move(verify); }

  /// Sets the static hook for parsing this type assembly.
  void setParseFn(ParserFn &&parse) { parser = std::move(parse); }

  /// Sets the static hook for printing this type assembly.
  void setPrintFn(PrinterFn &&print) { printer = std::move(print); }

  /// Check that the type parameters are valid.
  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> params) const {
    return verifier(emitError, params);
  }

  /// Return the MLIRContext in which the dynamic types is uniqued.
  MLIRContext &getContext() const { return *ctx; }

  /// Return the name of the type, in the format 'typename' and
  /// not 'dialectname.typename'.
  StringRef getName() const { return name; }

  /// Return the dialect defining the type.
  ExtensibleDialect *getDialect() const { return dialect; }

private:
  DynamicTypeDefinition(StringRef name, ExtensibleDialect *dialect,
                        VerifierFn &&verifier, ParserFn &&parser,
                        PrinterFn &&printer);

  /// This constructor should only be used when we need a pointer to
  /// the DynamicTypeDefinition in the verifier, the parser, or the printer.
  /// The verifier, parser, and printer need thus to be initialized after the
  /// constructor.
  DynamicTypeDefinition(ExtensibleDialect *dialect, StringRef name);

  /// Register the concrete type in the type Uniquer.
  void registerInTypeUniquer();

  /// The name should be prefixed with the dialect name followed by '.'.
  std::string name;

  /// Dialect in which this type is defined.
  ExtensibleDialect *dialect;

  /// The type verifier. It checks that the type parameters satisfy the
  /// invariants.
  VerifierFn verifier;

  /// The type parameters parser. It parses only the parameters, and expects the
  /// type name to have already been parsed.
  ParserFn parser;

  /// The type parameters printer. It prints only the parameters, and expects
  /// the type name to have already been printed.
  PrinterFn printer;

  /// Context in which the concrete types are uniqued.
  MLIRContext *ctx;

  friend ExtensibleDialect;
  friend DynamicType;
};

/// This trait is used to determine if a type is a dynamic type or not;
/// it should only be implemented by dynamic types.
/// Note: This is only required because dynamic type do not have a
/// static/single TypeID.
namespace TypeTrait {
template <typename ConcreteType>
class IsDynamicType : public TypeTrait::TraitBase<ConcreteType, IsDynamicType> {
};
} // namespace TypeTrait

/// A dynamic type instance. This is a type whose definition is defined at
/// runtime.
/// It is possible to check if a type is a dynamic type using
/// `my_type.isa<DynamicType>()`, and getting the type definition of a dynamic
/// type using the `DynamicType::getTypeDef` method.
/// All dynamic types have the same storage, which is an array of attributes.
class DynamicType
    : public Type::TypeBase<DynamicType, Type, detail::DynamicTypeStorage,
                            TypeTrait::IsDynamicType> {
public:
  // Inherit Base constructors.
  using Base::Base;

  /// Return an instance of a dynamic type given a dynamic type definition and
  /// type parameters.
  /// This asserts that the type verifier succeeded.
  static DynamicType get(DynamicTypeDefinition *typeDef,
                         ArrayRef<Attribute> params = {});

  /// Return an instance of a dynamic type given a dynamic type definition and
  /// type parameters. If the parameters provided are invalid, errors are
  /// emitted using the provided location and a null object is returned.
  static DynamicType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                DynamicTypeDefinition *typeDef,
                                ArrayRef<Attribute> params = {});

  /// Return the type definition of the concrete type.
  DynamicTypeDefinition *getTypeDef();

  /// Return the type parameters.
  ArrayRef<Attribute> getParams();

  /// Check if a type is a specific dynamic type.
  static bool isa(Type type, DynamicTypeDefinition *typeDef) {
    return type.getTypeID() == typeDef->getTypeID();
  }

  /// Check if a type is a dynamic type.
  static bool classof(Type type);

  /// Parse the dynamic type parameters and construct the type.
  /// The parameters are either empty, and nothing is parsed,
  /// or they are in the format '<>' or '<attr (,attr)*>'.
  static ParseResult parse(AsmParser &parser, DynamicTypeDefinition *typeDef,
                           DynamicType &parsedType);

  /// Print the dynamic type with the format
  /// 'type' or 'type<>' if there is no parameters, or 'type<attr (,attr)*>'.
  void print(AsmPrinter &printer);
};

//===----------------------------------------------------------------------===//
// Dynamic operation
//===----------------------------------------------------------------------===//

/// The definition of a dynamic op. A dynamic op is an op that is defined at
/// runtime, and that can be registered at runtime by an extensible dialect (a
/// dialect inheriting ExtensibleDialect). This class implements the method
/// exposed by the OperationName class, and in addition defines the TypeID of
/// the op that will be defined. Each dynamic operation definition refers to one
/// instance of this class.
class DynamicOpDefinition : public OperationName::Impl {
public:
  using GetCanonicalizationPatternsFn =
      llvm::unique_function<void(RewritePatternSet &, MLIRContext *) const>;

  /// Create a new op at runtime. The op is registered only after passing it to
  /// the dialect using registerDynamicOp.
  static std::unique_ptr<DynamicOpDefinition>
  get(StringRef name, ExtensibleDialect *dialect,
      OperationName::VerifyInvariantsFn &&verifyFn,
      OperationName::VerifyRegionInvariantsFn &&verifyRegionFn);
  static std::unique_ptr<DynamicOpDefinition>
  get(StringRef name, ExtensibleDialect *dialect,
      OperationName::VerifyInvariantsFn &&verifyFn,
      OperationName::VerifyRegionInvariantsFn &&verifyRegionFn,
      OperationName::ParseAssemblyFn &&parseFn,
      OperationName::PrintAssemblyFn &&printFn);
  static std::unique_ptr<DynamicOpDefinition>
  get(StringRef name, ExtensibleDialect *dialect,
      OperationName::VerifyInvariantsFn &&verifyFn,
      OperationName::VerifyRegionInvariantsFn &&verifyRegionFn,
      OperationName::ParseAssemblyFn &&parseFn,
      OperationName::PrintAssemblyFn &&printFn,
      OperationName::FoldHookFn &&foldHookFn,
      GetCanonicalizationPatternsFn &&getCanonicalizationPatternsFn,
      OperationName::PopulateDefaultAttrsFn &&populateDefaultAttrsFn);

  /// Returns the op typeID.
  TypeID getTypeID() { return typeID; }

  /// Sets the verifier function for this operation. It should emits an error
  /// message and returns failure if a problem is detected, or returns success
  /// if everything is ok.
  void setVerifyFn(OperationName::VerifyInvariantsFn &&verify) {
    verifyFn = std::move(verify);
  }

  /// Sets the region verifier function for this operation. It should emits an
  /// error message and returns failure if a problem is detected, or returns
  /// success if everything is ok.
  void setVerifyRegionFn(OperationName::VerifyRegionInvariantsFn &&verify) {
    verifyRegionFn = std::move(verify);
  }

  /// Sets the static hook for parsing this op assembly.
  void setParseFn(OperationName::ParseAssemblyFn &&parse) {
    parseFn = std::move(parse);
  }

  /// Sets the static hook for printing this op assembly.
  void setPrintFn(OperationName::PrintAssemblyFn &&print) {
    printFn = std::move(print);
  }

  /// Sets the hook implementing a generalized folder for the op. See
  /// `RegisteredOperationName::foldHook` for more details
  void setFoldHookFn(OperationName::FoldHookFn &&foldHook) {
    foldHookFn = std::move(foldHook);
  }

  /// Set the hook returning any canonicalization pattern rewrites that the op
  /// supports, for use by the canonicalization pass.
  void setGetCanonicalizationPatternsFn(
      GetCanonicalizationPatternsFn &&getCanonicalizationPatterns) {
    getCanonicalizationPatternsFn = std::move(getCanonicalizationPatterns);
  }

  /// Set the hook populating default attributes.
  void setPopulateDefaultAttrsFn(
      OperationName::PopulateDefaultAttrsFn &&populateDefaultAttrs) {
    populateDefaultAttrsFn = std::move(populateDefaultAttrs);
  }

  LogicalResult foldHook(Operation *op, ArrayRef<Attribute> attrs,
                         SmallVectorImpl<OpFoldResult> &results) final {
    return foldHookFn(op, attrs, results);
  }
  void getCanonicalizationPatterns(RewritePatternSet &set,
                                   MLIRContext *context) final {
    getCanonicalizationPatternsFn(set, context);
  }
  bool hasTrait(TypeID id) final { return false; }
  OperationName::ParseAssemblyFn getParseAssemblyFn() final {
    return [&](OpAsmParser &parser, OperationState &state) {
      return parseFn(parser, state);
    };
  }
  void populateDefaultAttrs(const OperationName &name,
                            NamedAttrList &attrs) final {
    populateDefaultAttrsFn(name, attrs);
  }
  void printAssembly(Operation *op, OpAsmPrinter &printer,
                     StringRef name) final {
    printFn(op, printer, name);
  }
  LogicalResult verifyInvariants(Operation *op) final { return verifyFn(op); }
  LogicalResult verifyRegionInvariants(Operation *op) final {
    return verifyRegionFn(op);
  }

  /// Implementation for properties (unsupported right now here).
  std::optional<Attribute> getInherentAttr(Operation *op,
                                           StringRef name) final {
    llvm::report_fatal_error("Unsupported getInherentAttr on Dynamic dialects");
  }
  void setInherentAttr(Operation *op, StringAttr name, Attribute value) final {
    llvm::report_fatal_error("Unsupported setInherentAttr on Dynamic dialects");
  }
  void populateInherentAttrs(Operation *op, NamedAttrList &attrs) final {}
  LogicalResult
  verifyInherentAttrs(OperationName opName, NamedAttrList &attributes,
                      function_ref<InFlightDiagnostic()> emitError) final {
    return success();
  }
  int getOpPropertyByteSize() final { return 0; }
  void initProperties(OperationName opName, OpaqueProperties storage,
                      OpaqueProperties init) final {}
  void deleteProperties(OpaqueProperties prop) final {}
  void populateDefaultProperties(OperationName opName,
                                 OpaqueProperties properties) final {}

  LogicalResult
  setPropertiesFromAttr(OperationName opName, OpaqueProperties properties,
                        Attribute attr,
                        function_ref<InFlightDiagnostic()> emitError) final {
    emitError() << "extensible Dialects don't support properties";
    return failure();
  }
  Attribute getPropertiesAsAttr(Operation *op) final { return {}; }
  void copyProperties(OpaqueProperties lhs, OpaqueProperties rhs) final {}
  bool compareProperties(OpaqueProperties, OpaqueProperties) final { return false; }
  llvm::hash_code hashProperties(OpaqueProperties prop) final { return {}; }

private:
  DynamicOpDefinition(
      StringRef name, ExtensibleDialect *dialect,
      OperationName::VerifyInvariantsFn &&verifyFn,
      OperationName::VerifyRegionInvariantsFn &&verifyRegionFn,
      OperationName::ParseAssemblyFn &&parseFn,
      OperationName::PrintAssemblyFn &&printFn,
      OperationName::FoldHookFn &&foldHookFn,
      GetCanonicalizationPatternsFn &&getCanonicalizationPatternsFn,
      OperationName::PopulateDefaultAttrsFn &&populateDefaultAttrsFn);

  /// Dialect defining this operation.
  ExtensibleDialect *getdialect();

  OperationName::VerifyInvariantsFn verifyFn;
  OperationName::VerifyRegionInvariantsFn verifyRegionFn;
  OperationName::ParseAssemblyFn parseFn;
  OperationName::PrintAssemblyFn printFn;
  OperationName::FoldHookFn foldHookFn;
  GetCanonicalizationPatternsFn getCanonicalizationPatternsFn;
  OperationName::PopulateDefaultAttrsFn populateDefaultAttrsFn;

  friend ExtensibleDialect;
};

//===----------------------------------------------------------------------===//
// Extensible dialect
//===----------------------------------------------------------------------===//

/// A dialect that can be extended with new operations/types/attributes at
/// runtime.
class ExtensibleDialect : public mlir::Dialect {
public:
  ExtensibleDialect(StringRef name, MLIRContext *ctx, TypeID typeID);

  /// Add a new type defined at runtime to the dialect.
  void registerDynamicType(std::unique_ptr<DynamicTypeDefinition> &&type);

  /// Add a new attribute defined at runtime to the dialect.
  void registerDynamicAttr(std::unique_ptr<DynamicAttrDefinition> &&attr);

  /// Add a new operation defined at runtime to the dialect.
  void registerDynamicOp(std::unique_ptr<DynamicOpDefinition> &&type);

  /// Check if the dialect is an extensible dialect.
  static bool classof(const Dialect *dialect);

  /// Returns nullptr if the definition was not found.
  DynamicTypeDefinition *lookupTypeDefinition(StringRef name) const {
    return nameToDynTypes.lookup(name);
  }

  /// Returns nullptr if the definition was not found.
  DynamicTypeDefinition *lookupTypeDefinition(TypeID id) const {
    auto it = dynTypes.find(id);
    if (it == dynTypes.end())
      return nullptr;
    return it->second.get();
  }

  /// Returns nullptr if the definition was not found.
  DynamicAttrDefinition *lookupAttrDefinition(StringRef name) const {
    return nameToDynAttrs.lookup(name);
  }

  /// Returns nullptr if the definition was not found.
  DynamicAttrDefinition *lookupAttrDefinition(TypeID id) const {
    auto it = dynAttrs.find(id);
    if (it == dynAttrs.end())
      return nullptr;
    return it->second.get();
  }

protected:
  /// Parse the dynamic type 'typeName' in the dialect 'dialect'.
  /// typename should not be prefixed with the dialect name.
  /// If the dynamic type does not exist, return no value.
  /// Otherwise, parse it, and return the parse result.
  /// If the parsing succeed, put the resulting type in 'resultType'.
  OptionalParseResult parseOptionalDynamicType(StringRef typeName,
                                               AsmParser &parser,
                                               Type &resultType) const;

  /// If 'type' is a dynamic type, print it.
  /// Returns success if the type was printed, and failure if the type was not a
  /// dynamic type.
  static LogicalResult printIfDynamicType(Type type, AsmPrinter &printer);

  /// Parse the dynamic attribute 'attrName' in the dialect 'dialect'.
  /// attrname should not be prefixed with the dialect name.
  /// If the dynamic attribute does not exist, return no value.
  /// Otherwise, parse it, and return the parse result.
  /// If the parsing succeed, put the resulting attribute in 'resultAttr'.
  OptionalParseResult parseOptionalDynamicAttr(StringRef attrName,
                                               AsmParser &parser,
                                               Attribute &resultAttr) const;

  /// If 'attr' is a dynamic attribute, print it.
  /// Returns success if the attribute was printed, and failure if the
  /// attribute was not a dynamic attribute.
  static LogicalResult printIfDynamicAttr(Attribute attr, AsmPrinter &printer);

private:
  /// The set of all dynamic types registered.
  DenseMap<TypeID, std::unique_ptr<DynamicTypeDefinition>> dynTypes;

  /// This structure allows to get in O(1) a dynamic type given its name.
  llvm::StringMap<DynamicTypeDefinition *> nameToDynTypes;

  /// The set of all dynamic attributes registered.
  DenseMap<TypeID, std::unique_ptr<DynamicAttrDefinition>> dynAttrs;

  /// This structure allows to get in O(1) a dynamic attribute given its name.
  llvm::StringMap<DynamicAttrDefinition *> nameToDynAttrs;

  /// Give DynamicOpDefinition access to allocateTypeID.
  friend DynamicOpDefinition;

  /// Allocates a type ID to uniquify operations.
  TypeID allocateTypeID() { return typeIDAllocator.allocate(); }

  /// Owns the TypeID generated at runtime for operations.
  TypeIDAllocator typeIDAllocator;
};

//===----------------------------------------------------------------------===//
// Dynamic dialect
//===----------------------------------------------------------------------===//

/// A dialect that can be defined at runtime. It can be extended with new
/// operations, types, and attributes at runtime.
class DynamicDialect : public SelfOwningTypeID, public ExtensibleDialect {
public:
  DynamicDialect(StringRef name, MLIRContext *ctx);

  TypeID getTypeID() { return SelfOwningTypeID::getTypeID(); }

  /// Check if the dialect is an extensible dialect.
  static bool classof(const Dialect *dialect);

  virtual Type parseType(DialectAsmParser &parser) const override;
  virtual void printType(Type type, DialectAsmPrinter &printer) const override;

  virtual Attribute parseAttribute(DialectAsmParser &parser,
                                   Type type) const override;
  virtual void printAttribute(Attribute attr,
                              DialectAsmPrinter &printer) const override;
};
} // namespace mlir

namespace llvm {
/// Provide isa functionality for ExtensibleDialect.
/// This is to override the isa functionality for Dialect.
template <>
struct isa_impl<mlir::ExtensibleDialect, mlir::Dialect> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return mlir::ExtensibleDialect::classof(&dialect);
  }
};

/// Provide isa functionality for DynamicDialect.
/// This is to override the isa functionality for Dialect.
template <>
struct isa_impl<mlir::DynamicDialect, mlir::Dialect> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return mlir::DynamicDialect::classof(&dialect);
  }
};
} // namespace llvm

#endif // MLIR_IR_EXTENSIBLEDIALECT_H
