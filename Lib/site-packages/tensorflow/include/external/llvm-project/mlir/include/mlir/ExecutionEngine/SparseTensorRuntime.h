//===- SparseTensorRuntime.h - SparseTensor runtime support lib -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file provides the functions which comprise the public API of the
// sparse tensor runtime support library for the SparseTensor dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSORRUNTIME_H
#define MLIR_EXECUTIONENGINE_SPARSETENSORRUNTIME_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/Float16bits.h"

#include <cinttypes>
#include <complex>
#include <vector>

using namespace mlir::sparse_tensor;

extern "C" {

//===----------------------------------------------------------------------===//
//
// Public functions which operate on MLIR buffers (memrefs) to interact
// with sparse tensors (which are only visible as opaque pointers externally).
// Because these functions deal with memrefs, they should only be used
// by MLIR compiler-generated code (or code that is in sync with MLIR).
//
//===----------------------------------------------------------------------===//

/// This is the "swiss army knife" method for materializing sparse
/// tensors into the computation.  The types of the `ptr` argument and
/// the result depend on the action, as explained in the following table,
/// where "STS" means a sparse-tensor-storage object.
///
/// Action:         `ptr`:          Returns:
/// ---------------------------------------------------------------------------
/// kEmpty          -               STS, empty
/// kFromReader     reader          STS, input from reader
/// kPack           buffers         STS, from level buffers
/// kSortCOOInPlace STS             STS, sorted in place
MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_newSparseTensor( // NOLINT
    StridedMemRefType<index_type, 1> *dimSizesRef,
    StridedMemRefType<index_type, 1> *lvlSizesRef,
    StridedMemRefType<LevelType, 1> *lvlTypesRef,
    StridedMemRefType<index_type, 1> *dim2lvlRef,
    StridedMemRefType<index_type, 1> *lvl2dimRef, OverheadType posTp,
    OverheadType crdTp, PrimaryType valTp, Action action, void *ptr);

/// Tensor-storage method to obtain direct access to the values array.
#define DECL_SPARSEVALUES(VNAME, V)                                            \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparseValues##VNAME(              \
      StridedMemRefType<V, 1> *out, void *tensor);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_SPARSEVALUES)
#undef DECL_SPARSEVALUES

/// Tensor-storage method to obtain direct access to the positions array
/// for the given level.
#define DECL_SPARSEPOSITIONS(PNAME, P)                                         \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparsePositions##PNAME(           \
      StridedMemRefType<P, 1> *out, void *tensor, index_type lvl);
MLIR_SPARSETENSOR_FOREVERY_O(DECL_SPARSEPOSITIONS)
#undef DECL_SPARSEPOSITIONS

/// Tensor-storage method to obtain direct access to the coordinates array
/// for the given level.
#define DECL_SPARSECOORDINATES(CNAME, C)                                       \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparseCoordinates##CNAME(         \
      StridedMemRefType<C, 1> *out, void *tensor, index_type lvl);
MLIR_SPARSETENSOR_FOREVERY_O(DECL_SPARSECOORDINATES)
#undef DECL_SPARSECOORDINATES

/// Tensor-storage method to obtain direct access to the coordinates array
/// buffer for the given level (provides an AoS view into the library).
#define DECL_SPARSECOORDINATES(CNAME, C)                                       \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparseCoordinatesBuffer##CNAME(   \
      StridedMemRefType<C, 1> *out, void *tensor, index_type lvl);
MLIR_SPARSETENSOR_FOREVERY_O(DECL_SPARSECOORDINATES)
#undef DECL_SPARSECOORDINATES

/// Tensor-storage method to insert elements in lexicographical
/// level-coordinate order.
#define DECL_LEXINSERT(VNAME, V)                                               \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_lexInsert##VNAME(                 \
      void *tensor, StridedMemRefType<index_type, 1> *lvlCoordsRef,            \
      StridedMemRefType<V, 0> *vref);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

/// Tensor-storage method to insert using expansion.
#define DECL_EXPINSERT(VNAME, V)                                               \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_expInsert##VNAME(                 \
      void *tensor, StridedMemRefType<index_type, 1> *lvlCoordsRef,            \
      StridedMemRefType<V, 1> *vref, StridedMemRefType<bool, 1> *fref,         \
      StridedMemRefType<index_type, 1> *aref, index_type count);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_EXPINSERT)
#undef DECL_EXPINSERT

/// Constructs a new SparseTensorReader object, opens the file, reads the
/// header, and validates that the actual contents of the file match
/// the expected `dimShapeRef` and `valTp`.
MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_createCheckedSparseTensorReader(
    char *filename, StridedMemRefType<index_type, 1> *dimShapeRef,
    PrimaryType valTp);

/// SparseTensorReader method to obtain direct access to the
/// dimension-sizes array.
MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_getSparseTensorReaderDimSizes(
    StridedMemRefType<index_type, 1> *out, void *p);

/// Reads the sparse tensor, stores the coordinates and values to the given
/// memrefs of a COO in AoS format. Returns a boolean to indicate whether
/// the COO elements are sorted.
#define DECL_READTOBUFFERS(VNAME, V, CNAME, C)                                 \
  MLIR_CRUNNERUTILS_EXPORT bool                                                \
      _mlir_ciface_getSparseTensorReaderReadToBuffers##CNAME##VNAME(           \
          void *p, StridedMemRefType<index_type, 1> *dim2lvlRef,               \
          StridedMemRefType<index_type, 1> *lvl2dimRef,                        \
          StridedMemRefType<C, 1> *cref, StridedMemRefType<V, 1> *vref)        \
          MLIR_SPARSETENSOR_FOREVERY_V_O(DECL_READTOBUFFERS)
#undef DECL_READTOBUFFERS

/// Outputs the sparse tensor dim-rank, nse, and dim-shape.
MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_outSparseTensorWriterMetaData(
    void *p, index_type dimRank, index_type nse,
    StridedMemRefType<index_type, 1> *dimSizesRef);

/// Outputs an element for the sparse tensor.
#define DECL_OUTNEXT(VNAME, V)                                                 \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_outSparseTensorWriterNext##VNAME( \
      void *p, index_type dimRank,                                             \
      StridedMemRefType<index_type, 1> *dimCoordsRef,                          \
      StridedMemRefType<V, 0> *vref);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_OUTNEXT)
#undef DECL_OUTNEXT

//===----------------------------------------------------------------------===//
//
// Public functions which accept only C-style data structures to interact
// with sparse tensors (which are only visible as opaque pointers externally).
// These functions can be used both by MLIR compiler-generated code
// as well as by any external runtime that wants to interact with MLIR
// compiler-generated code.
//
//===----------------------------------------------------------------------===//

/// Tensor-storage method to get the size of the given level.
MLIR_CRUNNERUTILS_EXPORT index_type sparseLvlSize(void *tensor, index_type l);

/// Tensor-storage method to get the size of the given dimension.
MLIR_CRUNNERUTILS_EXPORT index_type sparseDimSize(void *tensor, index_type d);

/// Tensor-storage method to finalize lexicographic insertions.
MLIR_CRUNNERUTILS_EXPORT void endLexInsert(void *tensor);

/// Releases the memory for the tensor-storage object.
MLIR_CRUNNERUTILS_EXPORT void delSparseTensor(void *tensor);

/// Helper function to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
MLIR_CRUNNERUTILS_EXPORT char *getTensorFilename(index_type id);

/// Returns the number of stored elements for the sparse tensor being read.
MLIR_CRUNNERUTILS_EXPORT index_type getSparseTensorReaderNSE(void *p);

/// Releases the SparseTensorReader and closes the associated file.
MLIR_CRUNNERUTILS_EXPORT void delSparseTensorReader(void *p);

/// Creates a SparseTensorWriter for outputting a sparse tensor to a file
/// with the given file name. When the file name is empty, std::cout is used.
/// Only the extended FROSTT format is supported currently.
MLIR_CRUNNERUTILS_EXPORT void *createSparseTensorWriter(char *filename);

/// Finalizes the outputing of a sparse tensor to a file and releases the
/// SparseTensorWriter.
MLIR_CRUNNERUTILS_EXPORT void delSparseTensorWriter(void *p);

} // extern "C"

#endif // MLIR_EXECUTIONENGINE_SPARSETENSORRUNTIME_H
