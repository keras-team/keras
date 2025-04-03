/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class OpAsmOpInterface;
namespace detail {
struct OpAsmOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*getAsmResultNames)(const Concept *impl, ::mlir::Operation *, ::mlir::OpAsmSetValueNameFn);
    void (*getAsmBlockArgumentNames)(const Concept *impl, ::mlir::Operation *, ::mlir::Region&, ::mlir::OpAsmSetValueNameFn);
    void (*getAsmBlockNames)(const Concept *impl, ::mlir::Operation *, ::mlir::OpAsmSetBlockNameFn);
    ::llvm::StringRef (*getDefaultDialect)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::OpAsmOpInterface;
    Model() : Concept{getAsmResultNames, getAsmBlockArgumentNames, getAsmBlockNames, getDefaultDialect} {}

    static inline void getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn);
    static inline ::llvm::StringRef getDefaultDialect();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::OpAsmOpInterface;
    FallbackModel() : Concept{getAsmResultNames, getAsmBlockArgumentNames, getAsmBlockNames, getDefaultDialect} {}

    static inline void getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn);
    static inline ::llvm::StringRef getDefaultDialect();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) const;
    void getAsmBlockArgumentNames(::mlir::Operation *tablegen_opaque_val, ::mlir::Region&region, ::mlir::OpAsmSetValueNameFn setNameFn) const;
    void getAsmBlockNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) const;
    static ::llvm::StringRef getDefaultDialect();
  };
};
template <typename ConcreteOp>
struct OpAsmOpInterfaceTrait;

} // namespace detail
class OpAsmOpInterface : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OpAsmOpInterfaceTrait<ConcreteOp> {};
  /// Get a special name to use when printing the results of this operation.
  /// The given callback is invoked with a specific result value that starts a
  /// result "pack", and the name to give this result pack. To signal that a
  /// result pack should use the default naming scheme, a None can be passed
  /// in instead of the name.
  /// 
  /// For example, if you have an operation that has four results and you want
  /// to split these into three distinct groups you could do the following:
  /// 
  /// ```c++
  ///   setNameFn(getResult(0), "first_result");
  ///   setNameFn(getResult(1), "middle_results");
  ///   setNameFn(getResult(3), ""); // use the default numbering.
  /// ```
  /// 
  /// This would print the operation as follows:
  /// 
  /// ```mlir
  ///   %first_result, %middle_results:2, %0 = "my.op" ...
  /// ```
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  /// Get a special name to use when printing the block arguments for a region
  /// immediately nested under this operation.
  void getAsmBlockArgumentNames(::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn);
  /// Get the name to use for a given block inside a region attached to this
  /// operation.
  /// 
  /// For example if this operation has multiple blocks:
  /// 
  /// ```mlir
  ///   some.op() ({
  ///     ^bb0:
  ///       ...
  ///     ^bb1:
  ///       ...
  ///   })
  /// ```
  /// 
  /// the method will be invoked on each of the blocks allowing the op to
  /// print:
  /// 
  /// ```mlir
  ///   some.op() ({
  ///     ^custom_foo_name:
  ///       ...
  ///     ^custom_bar_name:
  ///       ...
  ///   })
  /// ```
  void getAsmBlockNames(::mlir::OpAsmSetBlockNameFn setNameFn);
  /// Return the default dialect used when printing/parsing operations in
  /// regions nested under this operation. This allows for eliding the dialect
  /// prefix from the operation name, for example it would be possible to omit
  /// the `spirv.` prefix from all operations within a SpirV module if this method
  /// returned `spv`. The default implementation returns an empty string which
  /// is ignored.
  ::llvm::StringRef getDefaultDialect();
};
namespace detail {
  template <typename ConcreteOp>
  struct OpAsmOpInterfaceTrait : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Get a special name to use when printing the results of this operation.
    /// The given callback is invoked with a specific result value that starts a
    /// result "pack", and the name to give this result pack. To signal that a
    /// result pack should use the default naming scheme, a None can be passed
    /// in instead of the name.
    /// 
    /// For example, if you have an operation that has four results and you want
    /// to split these into three distinct groups you could do the following:
    /// 
    /// ```c++
    ///   setNameFn(getResult(0), "first_result");
    ///   setNameFn(getResult(1), "middle_results");
    ///   setNameFn(getResult(3), ""); // use the default numbering.
    /// ```
    /// 
    /// This would print the operation as follows:
    /// 
    /// ```mlir
    ///   %first_result, %middle_results:2, %0 = "my.op" ...
    /// ```
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
      return;
    }
    /// Get a special name to use when printing the block arguments for a region
    /// immediately nested under this operation.
    void getAsmBlockArgumentNames(::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn) {
      return;
    }
    /// Get the name to use for a given block inside a region attached to this
    /// operation.
    /// 
    /// For example if this operation has multiple blocks:
    /// 
    /// ```mlir
    ///   some.op() ({
    ///     ^bb0:
    ///       ...
    ///     ^bb1:
    ///       ...
    ///   })
    /// ```
    /// 
    /// the method will be invoked on each of the blocks allowing the op to
    /// print:
    /// 
    /// ```mlir
    ///   some.op() ({
    ///     ^custom_foo_name:
    ///       ...
    ///     ^custom_bar_name:
    ///       ...
    ///   })
    /// ```
    void getAsmBlockNames(::mlir::OpAsmSetBlockNameFn setNameFn) {
      ;
    }
    /// Return the default dialect used when printing/parsing operations in
    /// regions nested under this operation. This allows for eliding the dialect
    /// prefix from the operation name, for example it would be possible to omit
    /// the `spirv.` prefix from all operations within a SpirV module if this method
    /// returned `spv`. The default implementation returns an empty string which
    /// is ignored.
    static ::llvm::StringRef getDefaultDialect() {
      return "";
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmResultNames(setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmBlockArgumentNames(region, setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmBlockNames(setNameFn);
}
template<typename ConcreteOp>
::llvm::StringRef detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDefaultDialect() {
  return ConcreteOp::getDefaultDialect();
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return static_cast<const ConcreteOp *>(impl)->getAsmResultNames(tablegen_opaque_val, setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return static_cast<const ConcreteOp *>(impl)->getAsmBlockArgumentNames(tablegen_opaque_val, region, setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) {
  return static_cast<const ConcreteOp *>(impl)->getAsmBlockNames(tablegen_opaque_val, setNameFn);
}
template<typename ConcreteOp>
::llvm::StringRef detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDefaultDialect() {
  return ConcreteOp::getDefaultDialect();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) const {
return;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsmBlockArgumentNames(::mlir::Operation *tablegen_opaque_val, ::mlir::Region&region, ::mlir::OpAsmSetValueNameFn setNameFn) const {
return;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsmBlockNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) const {
;
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::StringRef detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDefaultDialect() {
return "";
}
} // namespace mlir
