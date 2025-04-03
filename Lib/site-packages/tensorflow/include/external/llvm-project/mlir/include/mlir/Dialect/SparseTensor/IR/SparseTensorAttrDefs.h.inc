/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace sparse_tensor {
class SparseTensorDimSliceAttr;
class SparseTensorEncodingAttr;
class StorageSpecifierKindAttr;
class SparseTensorSortKindAttr;
class CrdTransDirectionKindAttr;
namespace detail {
struct SparseTensorDimSliceAttrStorage;
} // namespace detail
class SparseTensorDimSliceAttr : public ::mlir::Attribute::AttrBase<SparseTensorDimSliceAttr, ::mlir::Attribute, detail::SparseTensorDimSliceAttrStorage> {
public:
  using Base::Base;
  void print(llvm::raw_ostream &os) const;

  /// Special value for dynamic offset/size/stride.
  static constexpr int64_t kDynamic = -1;
  static constexpr bool isDynamic(int64_t v) { return v == kDynamic; }
  static std::optional<uint64_t> getStatic(int64_t v);
  static std::string getStaticString(int64_t v);

  std::optional<uint64_t> getStaticOffset() const;
  std::optional<uint64_t> getStaticStride() const;
  std::optional<uint64_t> getStaticSize() const;
  bool isCompletelyDynamic() const;
  static constexpr ::llvm::StringLiteral name = "sparse_tensor.slice";
  static constexpr ::llvm::StringLiteral dialectName = "sparse_tensor";
  using Base::getChecked;
  static SparseTensorDimSliceAttr get(::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride);
  static SparseTensorDimSliceAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride);
  static SparseTensorDimSliceAttr get(::mlir::MLIRContext *context);
  static SparseTensorDimSliceAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int64_t offset, int64_t size, int64_t stride);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int64_t offset, int64_t size, int64_t stride);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"slice"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getOffset() const;
  int64_t getSize() const;
  int64_t getStride() const;
};
namespace detail {
struct SparseTensorEncodingAttrStorage;
} // namespace detail
class SparseTensorEncodingAttr : public ::mlir::Attribute::AttrBase<SparseTensorEncodingAttr, ::mlir::Attribute, detail::SparseTensorEncodingAttrStorage, ::mlir::VerifiableTensorEncoding::Trait> {
public:
  using Base::Base;
  //
  // Factory methods.
  //

  /// Constructs a new encoding with the given dimToLvl mapping,
  /// and all other fields inherited from `this`.
  SparseTensorEncodingAttr withDimToLvl(AffineMap dimToLvl) const;
  SparseTensorEncodingAttr withDimToLvl(SparseTensorEncodingAttr enc) const;

  /// Constructs a new encoding with dimToLvl reset to the default/identity,
  /// and all other fields inherited from `this`.
  SparseTensorEncodingAttr withoutDimToLvl() const;

  /// Constructs a new encoding with the given pointer and index
  /// bitwidths, and all other fields inherited from `this`.
  SparseTensorEncodingAttr withBitWidths(unsigned posWidth, unsigned crdWidth) const;

  /// Constructs a new encoding with the pointer and index bitwidths
  /// reset to the default, and all other fields inherited from `this`.
  SparseTensorEncodingAttr withoutBitWidths() const;

  /// Constructs a new encoding with the given explicit value
  /// and all other fields inherited from `this`.
  SparseTensorEncodingAttr withExplicitVal(Attribute explicitVal) const;

  /// Constructs a new encoding with the explicit value
  /// reset to the default, and all other fields inherited from `this`.
  SparseTensorEncodingAttr withoutExplicitVal() const;

  /// Constructs a new encoding with the given implicit value
  /// and all other fields inherited from `this`.
  SparseTensorEncodingAttr withImplicitVal(Attribute implicitVal) const;

  /// Constructs a new encoding with the implicit value
  /// reset to the default, and all other fields inherited from `this`.
  SparseTensorEncodingAttr withoutImplicitVal() const;

  /// Constructs a new encoding with the given dimSlices, and all
  /// other fields inherited from `this`.
  SparseTensorEncodingAttr withDimSlices(ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) const;

  /// Constructs a new encoding with the dimSlices reset to the default,
  /// and all other fields inherited from `this`.
  SparseTensorEncodingAttr withoutDimSlices() const;

  //
  // Rank methods.
  //

  /// Returns the expected number of tensor dimensions.  Asserts that
  /// the encoding is non-null (since no fixed result is valid for every
  /// dense-tensor).
  ::mlir::sparse_tensor::Dimension getDimRank() const;

  /// Returns the number of storage levels.  Asserts that the encoding
  /// is non-null (since no fixed result is valid for every dense-tensor).
  ::mlir::sparse_tensor::Level getLvlRank() const;

  uint64_t getBatchLvlRank() const;

  //
  // lvlTypes methods.
  //

  /// Safely looks up the level-type for the requested level.  (Returns
  /// `LevelType::Dense` for the null encoding, since dense-tensors
  /// are always all-dense.)
  ::mlir::sparse_tensor::LevelType getLvlType(::mlir::sparse_tensor::Level l) const;

  bool isDenseLvl(::mlir::sparse_tensor::Level l) const { return isDenseLT(getLvlType(l)); }
  bool isCompressedLvl(::mlir::sparse_tensor::Level l) const { return isCompressedLT(getLvlType(l)); }
  bool isSingletonLvl(::mlir::sparse_tensor::Level l) const { return isSingletonLT(getLvlType(l)); }
  bool isLooseCompressedLvl(::mlir::sparse_tensor::Level l) const { return isLooseCompressedLT(getLvlType(l)); }
  bool isNOutOfMLvl(::mlir::sparse_tensor::Level l) const { return isNOutOfMLT(getLvlType(l)); }
  bool isOrderedLvl(::mlir::sparse_tensor::Level l) const { return isOrderedLT(getLvlType(l)); }
  bool isUniqueLvl(::mlir::sparse_tensor::Level l) const { return isUniqueLT(getLvlType(l)); }

  /// Returns true if every level is dense.  Also returns true for
  /// the null encoding (since dense-tensors are always all-dense).
  bool isAllDense() const;

  /// Returns true if every level is ordered.  Also returns true for
  /// the null encoding (since dense-tensors are always all-ordered).
  bool isAllOrdered() const;

  //
  // Storage type methods.
  //

  /// Returns the coordinate-overhead MLIR type, defaulting to `IndexType`.
  Type getCrdElemType() const;

  /// Returns the position-overhead MLIR type, defaulting to `IndexType`.
  Type getPosElemType() const;

  /// Returns the coordinate-memref MLIR type, an optional tensorDimShape is
  /// used to refine the leading batch dimensions (if any).
  MemRefType getCrdMemRefType(
    std::optional<ArrayRef<int64_t>> tensorDimShape = std::nullopt) const;

  /// Returns the position-memref MLIR type, an optional tensorDimShape is
  /// used to refine the leading batch dimensions (if any).
  MemRefType getPosMemRefType(
    std::optional<ArrayRef<int64_t>> tensorDimShape = std::nullopt) const;

  //
  // dimToLvl methods.
  //

  /// Returns true if the dimToLvl mapping is the identity.
  /// Also returns true for the null encoding (since dense-tensors
  /// always have the identity mapping).
  bool isIdentity() const;

  /// Returns true if the dimToLvl mapping is a permutation.
  /// Also returns true for the null encoding (since dense-tensors
  /// always have the identity mapping).
  bool isPermutation() const;

  //
  // dimSlices methods.
  //

  bool isSlice() const;

  ::mlir::sparse_tensor::SparseTensorDimSliceAttr getDimSlice(::mlir::sparse_tensor::Dimension dim) const;

  std::optional<uint64_t> getStaticDimSliceOffset(::mlir::sparse_tensor::Dimension dim) const;
  std::optional<uint64_t> getStaticDimSliceStride(::mlir::sparse_tensor::Dimension dim) const;
  std::optional<uint64_t> getStaticLvlSliceOffset(::mlir::sparse_tensor::Level lvl) const;
  std::optional<uint64_t> getStaticLvlSliceStride(::mlir::sparse_tensor::Level lvl) const;

  //
  // Helper function to translate between level/dimension space.
  //

  SmallVector<int64_t> translateShape(::mlir::ArrayRef<int64_t> srcShape, ::mlir::sparse_tensor::CrdTransDirectionKind) const;
  ValueRange translateCrds(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange crds, ::mlir::sparse_tensor::CrdTransDirectionKind) const;

  //
  // COO methods.
  //

  /// Returns the starting level of this sparse tensor type for a
  /// trailing COO region that spans **at least** two levels. If
  /// no such COO region is found, then returns the level-rank.
  ///
  /// DEPRECATED: use getCOOSegment instead;
  Level getAoSCOOStart() const;

  /// Returns a list of COO segments in the sparse tensor types.
  SmallVector<COOSegment> getCOOSegments() const;

  //
  // Printing methods.
  //

  void printSymbols(AffineMap &map, AsmPrinter &printer) const;
  void printDimensions(AffineMap &map, AsmPrinter &printer, ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) const;
  void printLevels(AffineMap &map, AsmPrinter &printer, ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes) const;
  static constexpr ::llvm::StringLiteral name = "sparse_tensor.encoding";
  static constexpr ::llvm::StringLiteral dialectName = "sparse_tensor";
  using Base::getChecked;
  static SparseTensorEncodingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices);
  static SparseTensorEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices);
  static SparseTensorEncodingAttr get(::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl = {}, AffineMap lvlToDim = {}, unsigned posWidth = 0, unsigned crdWidth = 0, ::mlir::Attribute explicitVal = {}, ::mlir::Attribute implicitVal = {});
  static SparseTensorEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl = {}, AffineMap lvlToDim = {}, unsigned posWidth = 0, unsigned crdWidth = 0, ::mlir::Attribute explicitVal = {}, ::mlir::Attribute implicitVal = {});
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> getLvlTypes() const;
  AffineMap getDimToLvl() const;
  AffineMap getLvlToDim() const;
  unsigned getPosWidth() const;
  unsigned getCrdWidth() const;
  ::mlir::Attribute getExplicitVal() const;
  ::mlir::Attribute getImplicitVal() const;
  ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> getDimSlices() const;
  ::llvm::LogicalResult verifyEncoding(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
};
namespace detail {
struct StorageSpecifierKindAttrStorage;
} // namespace detail
class StorageSpecifierKindAttr : public ::mlir::Attribute::AttrBase<StorageSpecifierKindAttr, ::mlir::Attribute, detail::StorageSpecifierKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "sparse_tensor.kind";
  static constexpr ::llvm::StringLiteral dialectName = "sparse_tensor";
  static StorageSpecifierKindAttr get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::StorageSpecifierKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::sparse_tensor::StorageSpecifierKind getValue() const;
};
namespace detail {
struct SparseTensorSortKindAttrStorage;
} // namespace detail
class SparseTensorSortKindAttr : public ::mlir::Attribute::AttrBase<SparseTensorSortKindAttr, ::mlir::Attribute, detail::SparseTensorSortKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "sparse_tensor.SparseTensorSortAlgorithm";
  static constexpr ::llvm::StringLiteral dialectName = "sparse_tensor";
  static SparseTensorSortKindAttr get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorSortKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"SparseTensorSortAlgorithm"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::sparse_tensor::SparseTensorSortKind getValue() const;
};
namespace detail {
struct CrdTransDirectionKindAttrStorage;
} // namespace detail
class CrdTransDirectionKindAttr : public ::mlir::Attribute::AttrBase<CrdTransDirectionKindAttr, ::mlir::Attribute, detail::CrdTransDirectionKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "sparse_tensor.CrdTransDirection";
  static constexpr ::llvm::StringLiteral dialectName = "sparse_tensor";
  static CrdTransDirectionKindAttr get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::CrdTransDirectionKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"CrdTransDirection"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::sparse_tensor::CrdTransDirectionKind getValue() const;
};
} // namespace sparse_tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorDimSliceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorSortKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::CrdTransDirectionKindAttr)

#endif  // GET_ATTRDEF_CLASSES

