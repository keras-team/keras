/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

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
void mlir::OpAsmOpInterface::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
      return getImpl()->getAsmResultNames(getImpl(), getOperation(), setNameFn);
  }
/// Get a special name to use when printing the block arguments for a region
/// immediately nested under this operation.
void mlir::OpAsmOpInterface::getAsmBlockArgumentNames(::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn) {
      return getImpl()->getAsmBlockArgumentNames(getImpl(), getOperation(), region, setNameFn);
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
void mlir::OpAsmOpInterface::getAsmBlockNames(::mlir::OpAsmSetBlockNameFn setNameFn) {
      return getImpl()->getAsmBlockNames(getImpl(), getOperation(), setNameFn);
  }
/// Return the default dialect used when printing/parsing operations in
/// regions nested under this operation. This allows for eliding the dialect
/// prefix from the operation name, for example it would be possible to omit
/// the `spirv.` prefix from all operations within a SpirV module if this method
/// returned `spv`. The default implementation returns an empty string which
/// is ignored.
::llvm::StringRef mlir::OpAsmOpInterface::getDefaultDialect() {
      return getImpl()->getDefaultDialect();
  }
