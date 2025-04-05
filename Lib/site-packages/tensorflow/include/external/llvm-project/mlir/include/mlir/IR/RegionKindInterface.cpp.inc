/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return the kind of the region with the given index inside this operation.
::mlir::RegionKind mlir::RegionKindInterface::getRegionKind(unsigned index) {
      return getImpl()->getRegionKind(index);
  }
/// Return true if the kind of the given region requires the SSA-Dominance property
bool mlir::RegionKindInterface::hasSSADominance(unsigned index) {
      return getImpl()->hasSSADominance(index);
  }
