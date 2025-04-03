/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class VectorUnrollOpInterface;
namespace detail {
struct VectorUnrollOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::std::optional<::llvm::SmallVector<int64_t, 4>> (*getShapeForUnroll)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::VectorUnrollOpInterface;
    Model() : Concept{getShapeForUnroll} {}

    static inline ::std::optional<::llvm::SmallVector<int64_t, 4>> getShapeForUnroll(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::VectorUnrollOpInterface;
    FallbackModel() : Concept{getShapeForUnroll} {}

    static inline ::std::optional<::llvm::SmallVector<int64_t, 4>> getShapeForUnroll(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::std::optional<::llvm::SmallVector<int64_t, 4>> getShapeForUnroll(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct VectorUnrollOpInterfaceTrait;

} // namespace detail
class VectorUnrollOpInterface : public ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::VectorUnrollOpInterfaceTrait<ConcreteOp> {};
  /// Return the shape ratio of unrolling to the target vector shape
  /// `targetShape`. Return `std::nullopt` if the op cannot be unrolled to the
  /// target vector shape.
  ::std::optional<::llvm::SmallVector<int64_t, 4>> getShapeForUnroll();
};
namespace detail {
  template <typename ConcreteOp>
  struct VectorUnrollOpInterfaceTrait : public ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the shape ratio of unrolling to the target vector shape
    /// `targetShape`. Return `std::nullopt` if the op cannot be unrolled to the
    /// target vector shape.
    ::std::optional<::llvm::SmallVector<int64_t, 4>> getShapeForUnroll() {
      assert((*static_cast<ConcreteOp *>(this))->getNumResults() == 1);
        auto vt =
            ::llvm::dyn_cast<::mlir::VectorType>((*static_cast<ConcreteOp *>(this)).getResult().getType());
        if (!vt)
          return ::std::nullopt;
        ::llvm::SmallVector<int64_t, 4> res(
            vt.getShape().begin(), vt.getShape().end());
        return res;
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class VectorTransferOpInterface;
namespace detail {
struct VectorTransferOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::StringRef (*getInBoundsAttrName)(const Concept *impl, ::mlir::Operation *);
    ::mlir::StringRef (*getPermutationMapAttrName)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayAttr (*getInBounds)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getSource)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getVector)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OperandRange (*getIndices)(const Concept *impl, ::mlir::Operation *);
    ::mlir::AffineMap (*getPermutationMap)(const Concept *impl, ::mlir::Operation *);
    Value (*getMask)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::VectorTransferOpInterface;
    Model() : Concept{getInBoundsAttrName, getPermutationMapAttrName, getInBounds, getSource, getVector, getIndices, getPermutationMap, getMask} {}

    static inline ::mlir::StringRef getInBoundsAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::StringRef getPermutationMapAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getInBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getVector(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange getIndices(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::AffineMap getPermutationMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::VectorTransferOpInterface;
    FallbackModel() : Concept{getInBoundsAttrName, getPermutationMapAttrName, getInBounds, getSource, getVector, getIndices, getPermutationMap, getMask} {}

    static inline ::mlir::StringRef getInBoundsAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::StringRef getPermutationMapAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getInBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getVector(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange getIndices(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::AffineMap getPermutationMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct VectorTransferOpInterfaceTrait;

} // namespace detail
class VectorTransferOpInterface : public ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::VectorTransferOpInterfaceTrait<ConcreteOp> {};
  /// Return the `in_bounds` attribute name.
  ::mlir::StringRef getInBoundsAttrName();
  /// Return the `permutation_map` attribute name.
  ::mlir::StringRef getPermutationMapAttrName();
  /// Return the optional in_bounds attribute that specifies for each vector
  /// dimension whether it is in-bounds or not. (Broadcast dimensions are
  /// always in-bounds).
  ::mlir::ArrayAttr getInBounds();
  /// Return the memref or ranked tensor operand that this operation operates
  /// on. In case of a "read" operation, that's the source from which the
  /// operation reads. In case of a "write" operation, that's the destination
  /// into which the operation writes.
  /// TODO: Change name of operand, which is not accurate for xfer_write.
  ::mlir::Value getSource();
  /// Return the vector that this operation operates on. In case of a "read",
  /// that's the vector OpResult. In case of a "write", that's the vector
  /// operand value that is written by the op.
  ::mlir::Value getVector();
  /// Return the indices that specify the starting offsets into the source
  /// operand. The starting offsets are guaranteed to be in-bounds.
  ::mlir::OperandRange getIndices();
  /// Return the permutation map that describes the mapping of vector
  /// dimensions to source dimensions, as well as broadcast dimensions.
  /// 
  /// The permutation result has one result per vector transfer dimension.
  /// Each result is either a dim expression, indicating the corresponding
  /// dimension in the source operand, or a constant "0" expression,
  /// indicating a broadcast dimension.
  /// 
  /// Note: Nested vector dimensions that are flattened by this op are not
  /// accounted for in the permutation map. E.g.:
  /// ```
  /// // Vector type has rank 4, but permutation map has only 2 results. That
  /// // is because there are only 2 transfer dimensions.
  /// %0 = vector.transfer_read %arg1[%c3, %c3], %vf0
  ///     {permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
  ///     : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  /// ```
  ::mlir::AffineMap getPermutationMap();
  /// Return the mask operand if the op has a mask. Otherwise, return an
  /// empty value.
  Value getMask();

    /// Return a vector of all in_bounds values as booleans (one per vector
    /// transfer dimension).
    ::llvm::SmallVector<bool> getInBoundsValues() {
      ::llvm::SmallVector<bool> inBounds;
      for (int64_t i = 0, e = (*this).getTransferRank(); i < e; ++i)
        inBounds.push_back((*this).isDimInBounds(i));
      return inBounds;
    }

    /// Return the number of leading shaped dimensions (of the "source" operand)
    /// that do not participate in the permutation map.
    unsigned getLeadingShapedRank() {
      return (*this).getShapedType().getRank() - (*this).getTransferRank();
    }

    /// Return the mask type if the op has a mask. Otherwise, return an empty
    /// VectorType.
    ::mlir::VectorType getMaskType() {
      return (*this).getMask()
        ? ::llvm::cast<::mlir::VectorType>((*this).getMask().getType())
        : ::mlir::VectorType();
    }

    /// Return the shaped type of the "source" operand value.
    ::mlir::ShapedType getShapedType() {
      return ::llvm::cast<::mlir::ShapedType>((*this).getSource().getType());
    }

    /// Return the number of dimensions that participate in the permutation map.
    unsigned getTransferRank() {
      return (*this).getPermutationMap().getNumResults();
    }

    /// Return the type of the vector that this operation operates on.
    ::mlir::VectorType getVectorType() {
      return ::llvm::cast<::mlir::VectorType>((*this).getVector().getType());
    }

    /// Return "true" if at least one of the vector dimensions is a broadcasted
    /// dimension.
    bool hasBroadcastDim() {
      for (unsigned i = 0, e = (*this).getTransferRank(); i < e; ++i) {
        if ((*this).isBroadcastDim(i))
          return true;
      }
      return false;
    }

    /// Return "true" if at least one of the vector dimensions may be
    /// out-of-bounds.
    bool hasOutOfBoundsDim() {
      for (unsigned idx = 0, e = (*this).getTransferRank(); idx < e; ++idx)
        if (!(*this).isDimInBounds(idx))
          return true;
      return false;
    }

    /// Return "true" if the specified vector transfer dimension is a
    /// broadcasted dimension.
    bool isBroadcastDim(unsigned dim) {
      auto expr = (*this).getPermutationMap().getResult(dim);
      auto constExpr = ::llvm::dyn_cast<::mlir::AffineConstantExpr>(expr);
      return constExpr && constExpr.getValue() == 0;
    }

    /// Return "true" if the vector transfer dimension `dim` is in-bounds. Also
    /// return "true" if the dimension is a broadcast dimension. Return "false"
    /// otherwise.
    bool isDimInBounds(unsigned dim) {
      if ((*this).isBroadcastDim(dim))
        return true;
      auto inBounds = (*this).getInBounds();
      return ::llvm::cast<::mlir::BoolAttr>(inBounds[dim]).getValue();
    }

    /// Helper function to account for the fact that `permutationMap` results
    /// and `op.getIndices` sizes may not match and may not be aligned. The
    /// first `getLeadingShapedRank()` indices may just be indexed and not
    /// transferred from/into the vector.
    /// For example:
    /// ```
    /// vector.transfer %0[%i, %j, %k, %c0]
    ///     : memref<?x?x?x?xf32>, vector<2x4xf32>
    /// ```
    /// with `permutation_map = (d0, d1, d2, d3) -> (d2, d3)`.
    /// Provide a zip function to coiterate on 2 running indices: `resultIdx`
    /// and `indicesIdx` which accounts for this misalignment.
    void zipResultAndIndexing(
        ::llvm::function_ref<void(int64_t, int64_t)> fun) {
      for (int64_t resultIdx = 0,
                   indicesIdx = (*this).getLeadingShapedRank(),
                   eResult = (*this).getTransferRank();
          resultIdx < eResult;
          ++resultIdx, ++indicesIdx)
      fun(resultIdx, indicesIdx);
    }

    /// Return the shape of the hyperrectangular slice within the tensor/memref
    /// operand that is accessed by the transfer op.
    /// For example:
    /// ```
    /// vector.transfer %w0[%i, %j, %k] {
    ///     permutation_map = affine_map<(d0, d1, d2) -> (d1, d0, 0)>} :
    ///     : tensor<?x?x?xf32>, vector<4x2x6xf32>
    /// ```
    /// returns a shape [2, 4, 1].
    SmallVector<int64_t> getTransferChunkAccessed() {
      SmallVector<int64_t> dimSizes((*this).getPermutationMap().getNumDims(), 1);
      for (auto vecDims : llvm::zip((*this).getPermutationMap().getResults(),
                                    (*this).getVectorType().getShape())) {
        AffineExpr dim = std::get<0>(vecDims);
        int64_t size = std::get<1>(vecDims);
        // Skip broadcast.
        if (::llvm::isa<AffineConstantExpr>(dim))
          continue;
        dimSizes[::llvm::cast<AffineDimExpr>(dim).getPosition()] = size;
      }
      return dimSizes;
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct VectorTransferOpInterfaceTrait : public ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {

    /// Return a vector of all in_bounds values as booleans (one per vector
    /// transfer dimension).
    ::llvm::SmallVector<bool> getInBoundsValues() {
      ::llvm::SmallVector<bool> inBounds;
      for (int64_t i = 0, e = (*static_cast<ConcreteOp *>(this)).getTransferRank(); i < e; ++i)
        inBounds.push_back((*static_cast<ConcreteOp *>(this)).isDimInBounds(i));
      return inBounds;
    }

    /// Return the number of leading shaped dimensions (of the "source" operand)
    /// that do not participate in the permutation map.
    unsigned getLeadingShapedRank() {
      return (*static_cast<ConcreteOp *>(this)).getShapedType().getRank() - (*static_cast<ConcreteOp *>(this)).getTransferRank();
    }

    /// Return the mask type if the op has a mask. Otherwise, return an empty
    /// VectorType.
    ::mlir::VectorType getMaskType() {
      return (*static_cast<ConcreteOp *>(this)).getMask()
        ? ::llvm::cast<::mlir::VectorType>((*static_cast<ConcreteOp *>(this)).getMask().getType())
        : ::mlir::VectorType();
    }

    /// Return the shaped type of the "source" operand value.
    ::mlir::ShapedType getShapedType() {
      return ::llvm::cast<::mlir::ShapedType>((*static_cast<ConcreteOp *>(this)).getSource().getType());
    }

    /// Return the number of dimensions that participate in the permutation map.
    unsigned getTransferRank() {
      return (*static_cast<ConcreteOp *>(this)).getPermutationMap().getNumResults();
    }

    /// Return the type of the vector that this operation operates on.
    ::mlir::VectorType getVectorType() {
      return ::llvm::cast<::mlir::VectorType>((*static_cast<ConcreteOp *>(this)).getVector().getType());
    }

    /// Return "true" if at least one of the vector dimensions is a broadcasted
    /// dimension.
    bool hasBroadcastDim() {
      for (unsigned i = 0, e = (*static_cast<ConcreteOp *>(this)).getTransferRank(); i < e; ++i) {
        if ((*static_cast<ConcreteOp *>(this)).isBroadcastDim(i))
          return true;
      }
      return false;
    }

    /// Return "true" if at least one of the vector dimensions may be
    /// out-of-bounds.
    bool hasOutOfBoundsDim() {
      for (unsigned idx = 0, e = (*static_cast<ConcreteOp *>(this)).getTransferRank(); idx < e; ++idx)
        if (!(*static_cast<ConcreteOp *>(this)).isDimInBounds(idx))
          return true;
      return false;
    }

    /// Return "true" if the specified vector transfer dimension is a
    /// broadcasted dimension.
    bool isBroadcastDim(unsigned dim) {
      auto expr = (*static_cast<ConcreteOp *>(this)).getPermutationMap().getResult(dim);
      auto constExpr = ::llvm::dyn_cast<::mlir::AffineConstantExpr>(expr);
      return constExpr && constExpr.getValue() == 0;
    }

    /// Return "true" if the vector transfer dimension `dim` is in-bounds. Also
    /// return "true" if the dimension is a broadcast dimension. Return "false"
    /// otherwise.
    bool isDimInBounds(unsigned dim) {
      if ((*static_cast<ConcreteOp *>(this)).isBroadcastDim(dim))
        return true;
      auto inBounds = (*static_cast<ConcreteOp *>(this)).getInBounds();
      return ::llvm::cast<::mlir::BoolAttr>(inBounds[dim]).getValue();
    }

    /// Helper function to account for the fact that `permutationMap` results
    /// and `op.getIndices` sizes may not match and may not be aligned. The
    /// first `getLeadingShapedRank()` indices may just be indexed and not
    /// transferred from/into the vector.
    /// For example:
    /// ```
    /// vector.transfer %0[%i, %j, %k, %c0]
    ///     : memref<?x?x?x?xf32>, vector<2x4xf32>
    /// ```
    /// with `permutation_map = (d0, d1, d2, d3) -> (d2, d3)`.
    /// Provide a zip function to coiterate on 2 running indices: `resultIdx`
    /// and `indicesIdx` which accounts for this misalignment.
    void zipResultAndIndexing(
        ::llvm::function_ref<void(int64_t, int64_t)> fun) {
      for (int64_t resultIdx = 0,
                   indicesIdx = (*static_cast<ConcreteOp *>(this)).getLeadingShapedRank(),
                   eResult = (*static_cast<ConcreteOp *>(this)).getTransferRank();
          resultIdx < eResult;
          ++resultIdx, ++indicesIdx)
      fun(resultIdx, indicesIdx);
    }

    /// Return the shape of the hyperrectangular slice within the tensor/memref
    /// operand that is accessed by the transfer op.
    /// For example:
    /// ```
    /// vector.transfer %w0[%i, %j, %k] {
    ///     permutation_map = affine_map<(d0, d1, d2) -> (d1, d0, 0)>} :
    ///     : tensor<?x?x?xf32>, vector<4x2x6xf32>
    /// ```
    /// returns a shape [2, 4, 1].
    SmallVector<int64_t> getTransferChunkAccessed() {
      SmallVector<int64_t> dimSizes((*static_cast<ConcreteOp *>(this)).getPermutationMap().getNumDims(), 1);
      for (auto vecDims : llvm::zip((*static_cast<ConcreteOp *>(this)).getPermutationMap().getResults(),
                                    (*static_cast<ConcreteOp *>(this)).getVectorType().getShape())) {
        AffineExpr dim = std::get<0>(vecDims);
        int64_t size = std::get<1>(vecDims);
        // Skip broadcast.
        if (::llvm::isa<AffineConstantExpr>(dim))
          continue;
        dimSizes[::llvm::cast<AffineDimExpr>(dim).getPosition()] = size;
      }
      return dimSizes;
    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<int64_t, 4>> detail::VectorUnrollOpInterfaceInterfaceTraits::Model<ConcreteOp>::getShapeForUnroll(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapeForUnroll();
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<int64_t, 4>> detail::VectorUnrollOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getShapeForUnroll(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getShapeForUnroll(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::llvm::SmallVector<int64_t, 4>> detail::VectorUnrollOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getShapeForUnroll(::mlir::Operation *tablegen_opaque_val) const {
assert((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumResults() == 1);
        auto vt =
            ::llvm::dyn_cast<::mlir::VectorType>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResult().getType());
        if (!vt)
          return ::std::nullopt;
        ::llvm::SmallVector<int64_t, 4> res(
            vt.getShape().begin(), vt.getShape().end());
        return res;
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::StringRef detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInBoundsAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInBoundsAttrName();
}
template<typename ConcreteOp>
::mlir::StringRef detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPermutationMapAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPermutationMapAttrName();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInBounds();
}
template<typename ConcreteOp>
::mlir::Value detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSource();
}
template<typename ConcreteOp>
::mlir::Value detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getVector(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getVector();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getIndices(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndices();
}
template<typename ConcreteOp>
::mlir::AffineMap detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPermutationMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPermutationMap();
}
template<typename ConcreteOp>
Value detail::VectorTransferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMask();
}
template<typename ConcreteOp>
::mlir::StringRef detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInBoundsAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInBoundsAttrName(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::StringRef detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPermutationMapAttrName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPermutationMapAttrName(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInBounds(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSource(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getVector(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getVector(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIndices(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIndices(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::AffineMap detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPermutationMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPermutationMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::VectorTransferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMask(tablegen_opaque_val);
}
} // namespace mlir
