/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: SparseTensorAttrDefs.td                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace sparse_tensor {
::llvm::StringRef stringifyCrdTransDirectionKind(CrdTransDirectionKind val) {
  switch (val) {
    case CrdTransDirectionKind::dim2lvl: return "dim_to_lvl";
    case CrdTransDirectionKind::lvl2dim: return "lvl_to_dim";
  }
  return "";
}

::std::optional<CrdTransDirectionKind> symbolizeCrdTransDirectionKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CrdTransDirectionKind>>(str)
      .Case("dim_to_lvl", CrdTransDirectionKind::dim2lvl)
      .Case("lvl_to_dim", CrdTransDirectionKind::lvl2dim)
      .Default(::std::nullopt);
}
::std::optional<CrdTransDirectionKind> symbolizeCrdTransDirectionKind(uint32_t value) {
  switch (value) {
  case 0: return CrdTransDirectionKind::dim2lvl;
  case 1: return CrdTransDirectionKind::lvl2dim;
  default: return ::std::nullopt;
  }
}

} // namespace sparse_tensor
} // namespace mlir

namespace mlir {
namespace sparse_tensor {
::llvm::StringRef stringifySparseTensorSortKind(SparseTensorSortKind val) {
  switch (val) {
    case SparseTensorSortKind::HybridQuickSort: return "hybrid_quick_sort";
    case SparseTensorSortKind::InsertionSortStable: return "insertion_sort_stable";
    case SparseTensorSortKind::QuickSort: return "quick_sort";
    case SparseTensorSortKind::HeapSort: return "heap_sort";
  }
  return "";
}

::std::optional<SparseTensorSortKind> symbolizeSparseTensorSortKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SparseTensorSortKind>>(str)
      .Case("hybrid_quick_sort", SparseTensorSortKind::HybridQuickSort)
      .Case("insertion_sort_stable", SparseTensorSortKind::InsertionSortStable)
      .Case("quick_sort", SparseTensorSortKind::QuickSort)
      .Case("heap_sort", SparseTensorSortKind::HeapSort)
      .Default(::std::nullopt);
}
::std::optional<SparseTensorSortKind> symbolizeSparseTensorSortKind(uint32_t value) {
  switch (value) {
  case 0: return SparseTensorSortKind::HybridQuickSort;
  case 1: return SparseTensorSortKind::InsertionSortStable;
  case 2: return SparseTensorSortKind::QuickSort;
  case 3: return SparseTensorSortKind::HeapSort;
  default: return ::std::nullopt;
  }
}

} // namespace sparse_tensor
} // namespace mlir

namespace mlir {
namespace sparse_tensor {
::llvm::StringRef stringifyStorageSpecifierKind(StorageSpecifierKind val) {
  switch (val) {
    case StorageSpecifierKind::LvlSize: return "lvl_sz";
    case StorageSpecifierKind::PosMemSize: return "pos_mem_sz";
    case StorageSpecifierKind::CrdMemSize: return "crd_mem_sz";
    case StorageSpecifierKind::ValMemSize: return "val_mem_sz";
    case StorageSpecifierKind::DimOffset: return "dim_offset";
    case StorageSpecifierKind::DimStride: return "dim_stride";
  }
  return "";
}

::std::optional<StorageSpecifierKind> symbolizeStorageSpecifierKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<StorageSpecifierKind>>(str)
      .Case("lvl_sz", StorageSpecifierKind::LvlSize)
      .Case("pos_mem_sz", StorageSpecifierKind::PosMemSize)
      .Case("crd_mem_sz", StorageSpecifierKind::CrdMemSize)
      .Case("val_mem_sz", StorageSpecifierKind::ValMemSize)
      .Case("dim_offset", StorageSpecifierKind::DimOffset)
      .Case("dim_stride", StorageSpecifierKind::DimStride)
      .Default(::std::nullopt);
}
::std::optional<StorageSpecifierKind> symbolizeStorageSpecifierKind(uint32_t value) {
  switch (value) {
  case 0: return StorageSpecifierKind::LvlSize;
  case 1: return StorageSpecifierKind::PosMemSize;
  case 2: return StorageSpecifierKind::CrdMemSize;
  case 3: return StorageSpecifierKind::ValMemSize;
  case 4: return StorageSpecifierKind::DimOffset;
  case 5: return StorageSpecifierKind::DimStride;
  default: return ::std::nullopt;
  }
}

} // namespace sparse_tensor
} // namespace mlir

