/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class MemRefElementTypeInterface;
namespace detail {
struct MemRefElementTypeInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::MemRefElementTypeInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::MemRefElementTypeInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct MemRefElementTypeInterfaceTrait;

} // namespace detail
class MemRefElementTypeInterface : public ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::MemRefElementTypeInterfaceTrait<ConcreteType> {};
};
namespace detail {
  template <typename ConcreteType>
  struct MemRefElementTypeInterfaceTrait : public ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class ShapedType;
namespace detail {
struct ShapedTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::ShapedType (*cloneWith)(const Concept *impl, ::mlir::Type , ::std::optional<::llvm::ArrayRef<int64_t>>, ::mlir::Type);
    ::mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
    bool (*hasRank)(const Concept *impl, ::mlir::Type );
    ::llvm::ArrayRef<int64_t> (*getShape)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ShapedType;
    Model() : Concept{cloneWith, getElementType, hasRank, getShape} {}

    static inline ::mlir::ShapedType cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType);
    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline bool hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ShapedType;
    FallbackModel() : Concept{cloneWith, getElementType, hasRank, getShape} {}

    static inline ::mlir::ShapedType cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType);
    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline bool hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct ShapedTypeTrait;

} // namespace detail
class ShapedType : public ::mlir::TypeInterface<ShapedType, detail::ShapedTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<ShapedType, detail::ShapedTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::ShapedTypeTrait<ConcreteType> {};
  /// Returns a clone of this type with the given shape and element type.
  /// 
  /// If no shape is provided, the shape of this type is used. In that case, if
  /// this type is unranked, so is the resulting type.
  /// 
  /// If a shape is provided, the resulting type is always ranked, even if this
  /// type is unranked.
  ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;
  /// Returns the element type of this shaped type.
  ::mlir::Type getElementType() const;
  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;
  /// Returns the shape of this type if it is ranked, otherwise asserts.
  ::llvm::ArrayRef<int64_t> getShape() const;

    static constexpr int64_t kDynamic =
        std::numeric_limits<int64_t>::min();

    /// Whether the given dimension size indicates a dynamic dimension.
    static constexpr bool isDynamic(int64_t dValue) {
      return dValue == kDynamic;
    }

    /// Whether the given shape has any size that indicates a dynamic dimension.
    static bool isDynamicShape(ArrayRef<int64_t> dSizes) {
      return any_of(dSizes, [](int64_t dSize) { return isDynamic(dSize); });
    }

    /// Return the number of elements present in the given shape.
    static int64_t getNumElements(ArrayRef<int64_t> shape);

    /// Return a clone of this type with the given new shape and element type.
    /// The returned type is ranked, even if this type is unranked.
    auto clone(::llvm::ArrayRef<int64_t> shape, Type elementType) {
      return cloneWith(shape, elementType);
    }

    /// Return a clone of this type with the given new shape. The returned type
    /// is ranked, even if this type is unranked.
    auto clone(::llvm::ArrayRef<int64_t> shape) {
      return cloneWith(shape, getElementType());
    }

    /// Return a clone of this type with the given new element type. The
    /// returned type is ranked if and only if this type is ranked. In that
    /// case, the returned type has the same shape as this type.
    auto clone(::mlir::Type elementType) {
      return (*this).cloneWith(/*shape=*/std::nullopt, elementType);
    }

    /// If an element type is an integer or a float, return its width. Otherwise,
    /// abort.
    unsigned getElementTypeBitWidth() const {
      return (*this).getElementType().getIntOrFloatBitWidth();
    }

    /// If this is a ranked type, return the rank. Otherwise, abort.
    int64_t getRank() const {
      assert((*this).hasRank() && "cannot query rank of unranked shaped type");
      return (*this).getShape().size();
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    int64_t getNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::getNumElements((*this).getShape());
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic((*this).getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    bool hasStaticShape() const {
      return (*this).hasRank() &&
             !::mlir::ShapedType::isDynamicShape((*this).getShape());
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && (*this).getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    int64_t getNumDynamicDims() const {
      return llvm::count_if((*this).getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is ranked type, return the size of the specified dimension.
    /// Otherwise, abort.
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return (*this).getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if((*this).getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
};
namespace detail {
  template <typename ConcreteType>
  struct ShapedTypeTrait : public ::mlir::TypeInterface<ShapedType, detail::ShapedTypeInterfaceTraits>::Trait<ConcreteType> {

    /// Return a clone of this type with the given new element type. The
    /// returned type is ranked if and only if this type is ranked. In that
    /// case, the returned type has the same shape as this type.
    auto clone(::mlir::Type elementType) {
      return (*static_cast<const ConcreteType *>(this)).cloneWith(/*shape=*/std::nullopt, elementType);
    }

    /// If an element type is an integer or a float, return its width. Otherwise,
    /// abort.
    unsigned getElementTypeBitWidth() const {
      return (*static_cast<const ConcreteType *>(this)).getElementType().getIntOrFloatBitWidth();
    }

    /// If this is a ranked type, return the rank. Otherwise, abort.
    int64_t getRank() const {
      assert((*static_cast<const ConcreteType *>(this)).hasRank() && "cannot query rank of unranked shaped type");
      return (*static_cast<const ConcreteType *>(this)).getShape().size();
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    int64_t getNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::getNumElements((*static_cast<const ConcreteType *>(this)).getShape());
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic((*static_cast<const ConcreteType *>(this)).getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    bool hasStaticShape() const {
      return (*static_cast<const ConcreteType *>(this)).hasRank() &&
             !::mlir::ShapedType::isDynamicShape((*static_cast<const ConcreteType *>(this)).getShape());
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && (*static_cast<const ConcreteType *>(this)).getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    int64_t getNumDynamicDims() const {
      return llvm::count_if((*static_cast<const ConcreteType *>(this)).getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is ranked type, return the size of the specified dimension.
    /// Otherwise, abort.
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return (*static_cast<const ConcreteType *>(this)).getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if((*static_cast<const ConcreteType *>(this)).getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
} // namespace mlir
namespace mlir {
template<typename ConcreteType>
::mlir::ShapedType detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).cloneWith(shape, elementType);
}
template<typename ConcreteType>
::mlir::Type detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType();
}
template<typename ConcreteType>
bool detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).hasRank();
}
template<typename ConcreteType>
::llvm::ArrayRef<int64_t> detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getShape();
}
template<typename ConcreteType>
::mlir::ShapedType detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) {
  return static_cast<const ConcreteType *>(impl)->cloneWith(tablegen_opaque_val, shape, elementType);
}
template<typename ConcreteType>
::mlir::Type detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
template<typename ConcreteType>
bool detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->hasRank(tablegen_opaque_val);
}
template<typename ConcreteType>
::llvm::ArrayRef<int64_t> detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getShape(tablegen_opaque_val);
}
} // namespace mlir
