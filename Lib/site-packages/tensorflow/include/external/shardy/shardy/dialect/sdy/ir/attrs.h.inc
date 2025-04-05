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
namespace sdy {
class ManualAxesAttr;
class MeshAxisAttr;
class MeshAttr;
class SubAxisInfoAttr;
class AxisRefAttr;
class DimensionShardingAttr;
class TensorShardingAttr;
class TensorShardingPerValueAttr;
class DimMappingAttr;
class TensorMappingAttr;
class OpShardingRuleAttr;
namespace detail {
struct ManualAxesAttrStorage;
} // namespace detail
class ManualAxesAttr : public ::mlir::Attribute::AttrBase<ManualAxesAttr, ::mlir::Attribute, detail::ManualAxesAttrStorage> {
public:
  using Base::Base;
  auto begin() const { return getValue().begin(); }
  auto end() const { return getValue().end(); }
  bool empty() const { return getValue().empty(); }
  size_t size() const { return getValue().size(); }
  auto &front() const { return getValue().front(); }
  auto &back() const { return getValue().back(); }
  auto &operator[](size_t index) { return getValue()[index]; }
  operator ::llvm::ArrayRef<StringAttr>() const { return getValue(); }
  static constexpr ::llvm::StringLiteral name = "sdy.manual_axes";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static ManualAxesAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<StringAttr> value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"manual_axes"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<StringAttr> getValue() const;
};
namespace detail {
struct MeshAxisAttrStorage;
} // namespace detail
class MeshAxisAttr : public ::mlir::Attribute::AttrBase<MeshAxisAttr, ::mlir::Attribute, detail::MeshAxisAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "sdy.mesh_axis";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  using Base::getChecked;
  static MeshAxisAttr get(::mlir::MLIRContext *context, ::llvm::StringRef name, int64_t size);
  static MeshAxisAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::StringRef name, int64_t size);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::StringRef name, int64_t size);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::StringRef name, int64_t size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mesh_axis"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getName() const;
  int64_t getSize() const;
};
namespace detail {
struct MeshAttrStorage;
} // namespace detail
class MeshAttr : public ::mlir::Attribute::AttrBase<MeshAttr, ::mlir::Attribute, detail::MeshAttrStorage> {
public:
  using Base::Base;
  int64_t getAxisSize(StringRef axisName) const;

  // Returns the total size of the mesh across all axes, as in the total
  // number of devices.
  int64_t getTotalSize() const;

  // Returns whether this mesh is a maximal-sharding mesh with `deviceId`.
  //
  // A maximal-sharding mesh is a mesh with an empty axis list and a single
  // device ID.
  bool isMaximal(int64_t deviceId) const;

  // Returns a comparator that orders axis names w.r.t. their order in this
  // mesh.
  std::function<bool(StringRef lhs, StringRef rhs)> getAxisNameComparator()
  const;
  static constexpr ::llvm::StringLiteral name = "sdy.mesh";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  using Base::getChecked;
  static MeshAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<MeshAxisAttr> axes, ::llvm::ArrayRef<int64_t> device_ids);
  static MeshAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<MeshAxisAttr> axes, ::llvm::ArrayRef<int64_t> device_ids);
  static MeshAttr get(::mlir::MLIRContext *context, mlir::ArrayRef<MeshAxisAttr> axes);
  static MeshAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, mlir::ArrayRef<MeshAxisAttr> axes);
  static MeshAttr get(::mlir::MLIRContext *context, int64_t device_id);
  static MeshAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int64_t device_id);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<MeshAxisAttr> axes, ::llvm::ArrayRef<int64_t> device_ids);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<MeshAxisAttr> axes, ::llvm::ArrayRef<int64_t> device_ids);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mesh"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<MeshAxisAttr> getAxes() const;
  ::llvm::ArrayRef<int64_t> getDeviceIds() const;
};
namespace detail {
struct SubAxisInfoAttrStorage;
} // namespace detail
class SubAxisInfoAttr : public ::mlir::Attribute::AttrBase<SubAxisInfoAttr, ::mlir::Attribute, detail::SubAxisInfoAttrStorage> {
public:
  using Base::Base;
  // Sub-axes of the same full axis are ordered by their pre-size, and then by
  // their size (overlap is only possible for two sub-axes that shard
  // different tensors), e.g. [1(2), 4(2), 4(4)].
  bool operator<(const SubAxisInfoAttr &rhs) const;

  // Returns the pre-size of the next sub-axis (that is minor to this
  // sub-axis), or the size of the full axis if this is the minor-most
  // sub-axis.
  //
  // The next pre-size is equal to `pre-size * size` of this sub-axis.
  int64_t getNextPreSize() const {
    return getPreSize() * getSize();
  }
  static constexpr ::llvm::StringLiteral name = "sdy.sub_axis_info";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static SubAxisInfoAttr get(::mlir::MLIRContext *context, int64_t pre_size, int64_t size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"sub_axis_info"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getPreSize() const;
  int64_t getSize() const;
};
namespace detail {
struct AxisRefAttrStorage;
} // namespace detail
class AxisRefAttr : public ::mlir::Attribute::AttrBase<AxisRefAttr, ::mlir::Attribute, detail::AxisRefAttrStorage> {
public:
  using Base::Base;
  // Returns a comparator that orders axis names w.r.t. their order in the
  // given `mesh`.
  static std::function<bool(AxisRefAttr lhs, AxisRefAttr rhs)>
  getMeshComparator(MeshAttr mesh);

  std::string toString() const;

  // Returns the size of this axis or sub-axis.
  int64_t getSize(MeshAttr mesh) const;

  // If this is a sub-axis, returns its pre-size, otherwise returns 1.
  int64_t getSubAxisPreSize() const;

  // Returns whether this axis or sub-axis contains `other`, i.e., this axis
  // or sub-axis is equal to `other` or can be split into multiple sub-axes
  // such that one of them is `other`.
  //
  // For example:
  //  "a", "a":(2)2       -> true
  //  "a":(1)8, "a":(1)4  -> true
  //  "a":(2)16, "a":(4)2 -> true
  //  "a", "a"            -> true
  //  "a":(2)2, "a":(2)2  -> true
  //  "a":(1)4, "a":(2)4  -> false
  //  "a":(2)4, "a":(1)2  -> false
  //  "a", "b":(1)2       -> false
  bool contains(AxisRefAttr other) const;

  // Returns whether this axis or sub-axis strictly contains `other`.
  // "a.strictlyContains(b)" is equivalent to "a.contains(b) && a != b".
  //
  // For example:
  //  "a", "a":(2)2       -> true
  //  "a":(1)8, "a":(1)4  -> true
  //  "a":(2)16, "a":(4)2 -> true
  //  "a", "a"            -> false
  //  "a":(2)2, "a":(2)2  -> false
  //  "a":(1)4, "a":(2)4  -> false
  //  "a":(2)4, "a":(1)2  -> false
  //  "a", "b":(1)2       -> false
  bool strictlyContains(AxisRefAttr other) const;

  // Returns whether this axis or sub-axis is a prefix of `other`, i.e.,
  // `other` is equal to this axis ref or can be split into two sub-axes such
  // that the major one is this sub-axis.
  //
  // For example:
  //  "a":(1)2, "a"      -> true
  //  "a":(2)2, "a":(2)4 -> true
  //  "a", "a"           -> true
  //  "a":(2)4, "a":(2)4 -> true
  //  "a":(1)4, "a":(1)2 -> false
  //  "a":(1)4, "a":(2)8 -> false
  //  "a":(1)2, "b"      -> false
  bool prefixOf(AxisRefAttr other) const;

  // Returns whether this axis or sub-axis is a strict prefix of `other`.
  // "a.strictPrefixOf(b)" is equivalent to "a.prefixOf(b) && a != b".
  //
  // For example:
  //  "a":(1)2, "a"      -> true
  //  "a":(2)2, "a":(2)4 -> true
  //  "a", "a"           -> false
  //  "a":(2)4, "a":(2)4 -> false
  //  "a":(1)4, "a":(1)2 -> false
  //  "a":(1)4, "a":(2)8 -> false
  //  "a":(1)2, "b"      -> false
  bool strictPrefixOf(AxisRefAttr other) const;

  // Returns whether this axis or sub-axis overlaps with `other`, i.e., they
  // are equal or there is a sub-axis that is contained in both axis refs.
  //
  // For example:
  //  "a", "a":(2)2      -> true
  //  "a":(2)2, "a":(2)2 -> true
  //  "a":(1)4, "a":(2)4 -> true
  //  "a":(2)4, "a":(1)4 -> true
  //  "a":(1)4, "a":(1)2 -> true
  //  "a":(2)8, "a":(4)2 -> true
  //  "a":(1)4, "a":(4)2 -> false
  //  "a":(1)2, "a":(4)2 -> false
  //  "a":(1)4, "b":(2)4 -> false
  bool overlaps(AxisRefAttr other) const;

  // If there is no overlap between this and other axes, return this axis.
  // Otherwise, return the largest prefix of this axis by removing the
  // overlapping suffix with `other`. Return `std::nullopt` if the prefix does
  // not exist.
  //
  // For example:
  //  "a", "a":(2)2      -> "a":(1)2
  //  "a":(2)2, "a":(2)2 -> std::nullopt
  //  "a":(1)4, "a":(2)4 -> "a":(1)2
  //  "a":(2)4, "a":(1)4 -> std::nullopt
  //  "a":(1)4, "a":(1)2 -> std::nullopt
  //  "a":(2)8, "a":(4)2 -> "a":(2)2
  //  "a":(1)4, "a":(4)2 -> "a":(1)4
  //  "a":(1)2, "a":(4)2 -> "a":(1)2
  //  "a":(1)4, "b":(2)4 -> "a":(1)4
  std::optional<AxisRefAttr> getPrefixWithoutOverlap(AxisRefAttr other) const;

  // Returns whether this axis-ref can be merged with `other`, i.e., they are
  // consecutive sub-axes of the same full axis and this sub-axis is major to
  // `other`.
  //
  // For example:
  //  "a":(2)4, "a":(8)2 -> true
  //  "b":(1)2, "b":(2)4 -> true
  //  "c":(1)2, "c":(4)2 -> false
  //  "d":(2)4, "d":(1)2 -> false
  bool canMerge(AxisRefAttr other) const;

  // Merges this axis-ref with the `other`, assuming `canMerge(other)` is
  // true, i.e., they are consecutive sub-axes of the same full axis and this
  // sub-axis is major to `other`.
  //
  // The mesh is needed for the size of the full axis (see 2nd example below).
  //
  // For example:
  //  "a":(2)4, "a":(8)2 ~> "a":(2)8
  //  "b":(1)2, "b":(2)4 ~> "b"
  AxisRefAttr merge(AxisRefAttr other, MeshAttr mesh) const;
  static constexpr ::llvm::StringLiteral name = "sdy.axis_ref";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static AxisRefAttr get(::mlir::MLIRContext *context, ::llvm::StringRef name, SubAxisInfoAttr sub_axis_info);
  static AxisRefAttr get(::mlir::MLIRContext *context, StringRef name);
  static AxisRefAttr get(::mlir::MLIRContext *context, StringRef name, int64_t pre_size, int64_t size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"axis_ref"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getName() const;
  SubAxisInfoAttr getSubAxisInfo() const;
};
namespace detail {
struct DimensionShardingAttrStorage;
} // namespace detail
class DimensionShardingAttr : public ::mlir::Attribute::AttrBase<DimensionShardingAttr, ::mlir::Attribute, detail::DimensionShardingAttrStorage> {
public:
  using Base::Base;
  ArrayRef<AxisRefAttr>::iterator axis_begin() const {
    return getAxes().begin();
  }
  ArrayRef<AxisRefAttr>::iterator axis_end() const {
    return getAxes().end();
  }

  // Returns true if this dimension sharding has no axes.
  bool emptyAxes() const { return getAxes().empty(); }

  // Shards this dimension further along `axisName`.
  //
  // Assumes it is it not closed or already sharded on `axisName`.
  //
  // Attributes are immutable, so we can't update the sharding in place and
  // must return a new instance.
  DimensionShardingAttr getSharded(StringRef axisName) const;

  // Returns the sharded size of this dimension,
  // i.e., the product of sharding axis sizes.
  int64_t getShardedSize(MeshAttr mesh) const;

  // Drops the first `N` sharding axes, and keeps `M` sharding axes.
  DimensionShardingAttr sliceShardingAxes(size_t N, size_t M) const;

  // Drops the first `N` sharding axes.
  DimensionShardingAttr dropFrontShardingAxes(size_t N) const;

  // Takes the first `N` sharding axes.
  DimensionShardingAttr takeFrontShardingAxes(size_t N) const;

  // Drops the priority of this dimension sharding, if present.
  DimensionShardingAttr dropPriority() const;

  // Returns the priority of this dimension sharding, if present, or the
  // default priority otherwise.
  int64_t getPriorityOrDefault() const;
  static constexpr ::llvm::StringLiteral name = "sdy.dimension_sharding";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static DimensionShardingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<AxisRefAttr> axes, bool is_closed, std::optional<int64_t> priority);
  static DimensionShardingAttr get(::mlir::MLIRContext *context, ArrayRef<AxisRefAttr> axes, bool is_closed);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dimension_sharding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<AxisRefAttr> getAxes() const;
  bool getIsClosed() const;
  std::optional<int64_t> getPriority() const;
};
namespace detail {
struct TensorShardingAttrStorage;
} // namespace detail
class TensorShardingAttr : public ::mlir::Attribute::AttrBase<TensorShardingAttr, ::mlir::Attribute, detail::TensorShardingAttrStorage> {
public:
  using Base::Base;
  int64_t getRank() const {
    return getDimShardings().size();
  }

  DimensionShardingAttr getDimSharding(int64_t dim) const {
    return getDimShardings()[dim];
  }

  bool isClosed(int64_t dim) const {
    return getDimSharding(dim).getIsClosed();
  }

  bool isFullyClosed() const {
    return llvm::all_of(getDimShardings(),
                    [](const DimensionShardingAttr dimSharding) {
                       return dimSharding.getIsClosed();
                    });
  }

  bool isFullyReplicated() const {
    return llvm::all_of(getDimShardings(),
                    [](const DimensionShardingAttr dimSharding) {
                       return dimSharding.emptyAxes();
                    });
  }

  StringRef getMeshName() const {
    return getMeshSymName().getValue();
  }

  // Returns true if all dimension shardings are empty and there are no
  // replicated axes.
  bool emptyAxes() const;

  // Like `llvm::any_of` but checks the predicate against all dimension
  // sharding and replicated `AxisRefAttr`s.
  bool anyOfAxisRef(std::function<bool(AxisRefAttr)> predicate) const;

  // Returns true if `axisName` or a sub-axis of it is used to shard any
  // dimension or is replicated.
  bool isBound(StringRef axisName) const;

  // Returns true if dimension `dim` can be further sharded on the full
  // `axisName`.
  bool canShard(int64_t dim, StringRef axisName) const;

  // Returns true if the tensor can be replicated on the full `axisName`.
  bool canReplicate(StringRef axisName) const;

  // Closes sharding dimensions at the specified dimension indices.
  TensorShardingAttr closeShardingDims(ArrayRef<int64_t> dimIndices) const;

  // Opens sharding dimensions at the specified dimension indices.
  TensorShardingAttr openShardingDims(ArrayRef<int64_t> dimIndices) const;

  // Sets the sharding of dimension `dim`.
  //
  // Assumes `dim < getRank()`.
  //
  // Attributes are immutable, so we can't update the sharding in place and
  // must return a new instance.
  TensorShardingAttr replaceDimSharding(
      int64_t dim, DimensionShardingAttr sharding) const;

  // Shards dimension `dim` further along `axisName`.
  //
  // Assumes `canShard(dim, axisName)` is true.
  //
  // Attributes are immutable, so we can't update the sharding in place and
  // must return a new instance.
  TensorShardingAttr getSharded(int64_t dim, StringRef axisName) const;

  // Replicates the tensor along `axisName`.
  //
  // Assumes `canReplicate(axisName)` is true. The `mesh` is needed to keep
  // the replicated axes sorted.
  //
  // Attributes are immutable, so we can't update the sharding in place and
  // must return a new instance.
  TensorShardingAttr getReplicated(StringRef axisName, MeshAttr mesh) const;


  // Verifies that this `TensorShardingAttr` is valid w.r.t the given
  // tensor type and mesh.
  //
  // If `type` isn't a `ShapedType`, the sharding must have rank 0
  // and no replicated axes. Otherwise, the `ShapedType` must have a static
  // shape.

  //
  // If `checkDivisibility` is true, verifies that each dimension size
  // is divisible by its sharded size.
  mlir::LogicalResult verifyForType(
      Type type, MeshAttr mesh,
      std::function<InFlightDiagnostic(StringRef)> emitError,
      bool checkDivisibility = true);

  // Builds a `TensorShardingAttr` with all dim shardings and replicated axes
  // being marked closed (cannot be further replicated/sharded).
  static TensorShardingAttr getFullyClosed(
      MLIRContext* context, int64_t rank, StringRef meshName);

  // Builds a `TensorShardingAttr` with all dim shardings and replicated axes
  // being marked open (can be further replicated/sharded).
  static TensorShardingAttr getFullyOpen(
      MLIRContext* context, int64_t rank, StringRef meshName);

  // Builds a fully open `TensorShardingAttr` matching `sharding` in
  // `mesh_sym_name` and rank.
  static TensorShardingAttr getFullyOpenLike(TensorShardingAttr sharding);
  static constexpr ::llvm::StringLiteral name = "sdy.sharding";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static TensorShardingAttr get(::mlir::MLIRContext *context, FlatSymbolRefAttr mesh_sym_name, ::llvm::ArrayRef<DimensionShardingAttr> dim_shardings, ::llvm::ArrayRef<AxisRefAttr> replicated_axes);
  static TensorShardingAttr get(::mlir::MLIRContext *context, StringRef mesh_name, ArrayRef<DimensionShardingAttr> dim_shardings, ArrayRef<AxisRefAttr> replicated_axes);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"sharding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  FlatSymbolRefAttr getMeshSymName() const;
  ::llvm::ArrayRef<DimensionShardingAttr> getDimShardings() const;
  ::llvm::ArrayRef<AxisRefAttr> getReplicatedAxes() const;
};
namespace detail {
struct TensorShardingPerValueAttrStorage;
} // namespace detail
class TensorShardingPerValueAttr : public ::mlir::Attribute::AttrBase<TensorShardingPerValueAttr, ::mlir::Attribute, detail::TensorShardingPerValueAttrStorage> {
public:
  using Base::Base;
  // Builds a `TensorShardingPerValue` for each type in `types`, with all
  // dimension shardings marked open (can be further replicated/sharded).
  static TensorShardingPerValueAttr getFullyOpen(
      MLIRContext* context, TypeRange types, StringRef meshName);

  // Returns whether there are no values.
  bool empty() const { return getShardings().empty(); }

  // Returns the number of values.
  int64_t size() const { return getShardings().size(); }

  // Returns the sharding of a value at `operandIndex`.
  //
  // Assumes `operandIndex < size()`.
  TensorShardingAttr getSharding(int64_t operandIndex) const {
    assert(operandIndex < size());
    return getShardings()[operandIndex];
  }

  // Sets the sharding of a value at `index`.
  //
  // Assumes `index < size()`.
  //
  // Attributes are immutable, so we can't update the sharding in place and
  // must return a new instance.
  TensorShardingPerValueAttr replaceValueSharding(
      int64_t index, TensorShardingAttr sharding) const;
  static constexpr ::llvm::StringLiteral name = "sdy.sharding_per_value";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static TensorShardingPerValueAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<TensorShardingAttr> shardings);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"sharding_per_value"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<TensorShardingAttr> getShardings() const;
};
namespace detail {
struct DimMappingAttrStorage;
} // namespace detail
class DimMappingAttr : public ::mlir::Attribute::AttrBase<DimMappingAttr, ::mlir::Attribute, detail::DimMappingAttrStorage> {
public:
  using Base::Base;
  // Returns whether the given `factorIndex` is the minor-most factor.
  bool isMinorMost(int64_t factorIndex) const {
    return !getFactorIndices().empty() &&
            getFactorIndices().back() == factorIndex;
  }
  static constexpr ::llvm::StringLiteral name = "sdy.dim_mapping";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static DimMappingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> factor_indices);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dim_mapping"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getFactorIndices() const;
};
namespace detail {
struct TensorMappingAttrStorage;
} // namespace detail
class TensorMappingAttr : public ::mlir::Attribute::AttrBase<TensorMappingAttr, ::mlir::Attribute, detail::TensorMappingAttrStorage> {
public:
  using Base::Base;
  int64_t getRank() const { return getDimMappings().size(); }
  static constexpr ::llvm::StringLiteral name = "sdy.tensor_mapping";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static TensorMappingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DimMappingAttr> dim_mappings);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_mapping"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<DimMappingAttr> getDimMappings() const;
};
namespace detail {
struct OpShardingRuleAttrStorage;
} // namespace detail
class OpShardingRuleAttr : public ::mlir::Attribute::AttrBase<OpShardingRuleAttr, ::mlir::Attribute, detail::OpShardingRuleAttrStorage> {
public:
  using Base::Base;
  int64_t getNumFactors() const { return getFactorSizes().size(); }
  int64_t getNumOperands() const { return getOperandMappings().size(); }
  int64_t getNumResults() const { return getResultMappings().size(); }

  int64_t getFactorSize(int64_t factorIndex) const {
    return getFactorSizes()[factorIndex];
  }
  TensorMappingAttr getOperandMapping(int64_t operandNum) const {
    return getOperandMappings()[operandNum];
  }
  TensorMappingAttr getResultMapping(int64_t resultNum) const {
    return getResultMappings()[resultNum];
  }

  bool isCustom() const { return getIsCustomRule(); }
  static constexpr ::llvm::StringLiteral name = "sdy.op_sharding_rule";
  static constexpr ::llvm::StringLiteral dialectName = "sdy";
  static OpShardingRuleAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> factor_sizes, ::llvm::ArrayRef<TensorMappingAttr> operand_mappings, ::llvm::ArrayRef<TensorMappingAttr> result_mappings, bool is_custom_rule);
  static OpShardingRuleAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> factor_sizes, ArrayRef<TensorMappingAttr> operand_mappings, ArrayRef<TensorMappingAttr> result_mappings);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"op_sharding_rule"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getFactorSizes() const;
  ::llvm::ArrayRef<TensorMappingAttr> getOperandMappings() const;
  ::llvm::ArrayRef<TensorMappingAttr> getResultMappings() const;
  bool getIsCustomRule() const;
};
} // namespace sdy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::ManualAxesAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::MeshAxisAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::MeshAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::SubAxisInfoAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::AxisRefAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::DimensionShardingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::TensorShardingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::TensorShardingPerValueAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::DimMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::TensorMappingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sdy::OpShardingRuleAttr)

#endif  // GET_ATTRDEF_CLASSES

