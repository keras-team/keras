//===- BuiltinAttributes.h - MLIR Builtin Attribute Classes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINATTRIBUTES_H
#define MLIR_IR_BUILTINATTRIBUTES_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Sequence.h"
#include <complex>
#include <optional>

namespace mlir {
class AffineMap;
class AsmResourceBlob;
class BoolAttr;
class BuiltinDialect;
class DenseIntElementsAttr;
template <typename T>
struct DialectResourceBlobHandle;
class FlatSymbolRefAttr;
class FunctionType;
class IntegerSet;
class IntegerType;
class Location;
class Operation;
class RankedTensorType;

namespace detail {
struct DenseIntOrFPElementsAttrStorage;
struct DenseStringElementsAttrStorage;
struct StringAttrStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// Elements Attributes
//===----------------------------------------------------------------------===//

namespace detail {
/// Pair of raw pointer and a boolean flag of whether the pointer holds a splat,
using DenseIterPtrAndSplat = std::pair<const char *, bool>;

/// Impl iterator for indexed DenseElementsAttr iterators that records a data
/// pointer and data index that is adjusted for the case of a splat attribute.
template <typename ConcreteT, typename T, typename PointerT = T *,
          typename ReferenceT = T &>
class DenseElementIndexedIteratorImpl
    : public llvm::indexed_accessor_iterator<ConcreteT, DenseIterPtrAndSplat, T,
                                             PointerT, ReferenceT> {
protected:
  DenseElementIndexedIteratorImpl(const char *data, bool isSplat,
                                  size_t dataIndex)
      : llvm::indexed_accessor_iterator<ConcreteT, DenseIterPtrAndSplat, T,
                                        PointerT, ReferenceT>({data, isSplat},
                                                              dataIndex) {}

  /// Return the current index for this iterator, adjusted for the case of a
  /// splat.
  ptrdiff_t getDataIndex() const {
    bool isSplat = this->base.second;
    return isSplat ? 0 : this->index;
  }

  /// Return the data base pointer.
  const char *getData() const { return this->base.first; }
};

/// Type trait detector that checks if a given type T is a complex type.
template <typename T>
struct is_complex_t : public std::false_type {};
template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};
} // namespace detail

/// An attribute that represents a reference to a dense vector or tensor
/// object.
class DenseElementsAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const { return cast_if_present<ElementsAttr>(*this); }
  /// Allow implicit conversion to TypedAttr.
  operator TypedAttr() const { return ElementsAttr(*this); }

  /// Type trait used to check if the given type T is a potentially valid C++
  /// floating point type that can be used to access the underlying element
  /// types of a DenseElementsAttr.
  template <typename T>
  struct is_valid_cpp_fp_type {
    /// The type is a valid floating point type if it is a builtin floating
    /// point type, or is a potentially user defined floating point type. The
    /// latter allows for supporting users that have custom types defined for
    /// bfloat16/half/etc.
    static constexpr bool value = llvm::is_one_of<T, float, double>::value ||
                                  (std::numeric_limits<T>::is_specialized &&
                                   !std::numeric_limits<T>::is_integer);
  };

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);

  /// Constructs a dense elements attribute from an array of element values.
  /// Each element attribute value is expected to be an element of 'type'.
  /// 'type' must be a vector or tensor with static shape. If the element of
  /// `type` is non-integer/index/float it is assumed to be a string type.
  static DenseElementsAttr get(ShapedType type, ArrayRef<Attribute> values);

  /// Constructs a dense integer elements attribute from an array of integer
  /// or floating-point values. Each value is expected to be the same bitwidth
  /// of the element type of 'type'. 'type' must be a vector or tensor with
  /// static shape.
  template <typename T,
            typename = std::enable_if_t<std::numeric_limits<T>::is_integer ||
                                        is_valid_cpp_fp_type<T>::value>>
  static DenseElementsAttr get(const ShapedType &type, ArrayRef<T> values) {
    const char *data = reinterpret_cast<const char *>(values.data());
    return getRawIntOrFloat(
        type, ArrayRef<char>(data, values.size() * sizeof(T)), sizeof(T),
        std::numeric_limits<T>::is_integer, std::numeric_limits<T>::is_signed);
  }

  /// Constructs a dense integer elements attribute from a single element.
  template <typename T,
            typename = std::enable_if_t<std::numeric_limits<T>::is_integer ||
                                        is_valid_cpp_fp_type<T>::value ||
                                        detail::is_complex_t<T>::value>>
  static DenseElementsAttr get(const ShapedType &type, T value) {
    return get(type, llvm::ArrayRef(value));
  }

  /// Constructs a dense complex elements attribute from an array of complex
  /// values. Each value is expected to be the same bitwidth of the element type
  /// of 'type'. 'type' must be a vector or tensor with static shape.
  template <
      typename T, typename ElementT = typename T::value_type,
      typename = std::enable_if_t<detail::is_complex_t<T>::value &&
                                  (std::numeric_limits<ElementT>::is_integer ||
                                   is_valid_cpp_fp_type<ElementT>::value)>>
  static DenseElementsAttr get(const ShapedType &type, ArrayRef<T> values) {
    const char *data = reinterpret_cast<const char *>(values.data());
    return getRawComplex(type, ArrayRef<char>(data, values.size() * sizeof(T)),
                         sizeof(T), std::numeric_limits<ElementT>::is_integer,
                         std::numeric_limits<ElementT>::is_signed);
  }

  /// Overload of the above 'get' method that is specialized for boolean values.
  static DenseElementsAttr get(ShapedType type, ArrayRef<bool> values);

  /// Overload of the above 'get' method that is specialized for StringRef
  /// values.
  static DenseElementsAttr get(ShapedType type, ArrayRef<StringRef> values);

  /// Constructs a dense integer elements attribute from an array of APInt
  /// values. Each APInt value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type, ArrayRef<APInt> values);

  /// Constructs a dense complex elements attribute from an array of APInt
  /// values. Each APInt value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type,
                               ArrayRef<std::complex<APInt>> values);

  /// Constructs a dense float elements attribute from an array of APFloat
  /// values. Each APFloat value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type, ArrayRef<APFloat> values);

  /// Constructs a dense complex elements attribute from an array of APFloat
  /// values. Each APFloat value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type,
                               ArrayRef<std::complex<APFloat>> values);

  /// Construct a dense elements attribute for an initializer_list of values.
  /// Each value is expected to be the same bitwidth of the element type of
  /// 'type'. 'type' must be a vector or tensor with static shape.
  template <typename T>
  static DenseElementsAttr get(const ShapedType &type,
                               const std::initializer_list<T> &list) {
    return get(type, ArrayRef<T>(list));
  }

  /// Construct a dense elements attribute from a raw buffer representing the
  /// data for this attribute. Users are encouraged to use one of the
  /// constructors above, which provide more safeties. However, this
  /// constructor is useful for tools which may want to interop and can
  /// follow the precise definition.
  ///
  /// The format of the raw buffer is a densely packed array of values that
  /// can be bitcast to the storage format of the element type specified.
  /// Types that are not byte aligned will be:
  ///   - For bitwidth > 1: Rounded up to the next byte.
  ///   - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to
  ///     the linear order of the shape type from MSB to LSB, padded to on the
  ///     right.
  static DenseElementsAttr getFromRawBuffer(ShapedType type,
                                            ArrayRef<char> rawBuffer);

  /// Returns true if the given buffer is a valid raw buffer for the given type.
  /// `detectedSplat` is set if the buffer is valid and represents a splat
  /// buffer. The definition may be expanded over time, but currently, a
  /// splat buffer is detected if:
  ///   - For >1bit: The buffer consists of a single element.
  ///   - For 1bit: The buffer consists of a single byte with value 0 or 255.
  ///
  /// User code should be prepared for additional, conformant patterns to be
  /// identified as splats in the future.
  static bool isValidRawBuffer(ShapedType type, ArrayRef<char> rawBuffer,
                               bool &detectedSplat);

  //===--------------------------------------------------------------------===//
  // Iterators
  //===--------------------------------------------------------------------===//

  /// The iterator range over the given iterator type T.
  template <typename IteratorT>
  using iterator_range_impl = detail::ElementsAttrRange<IteratorT>;

  /// The iterator for the given element type T.
  template <typename T, typename AttrT = DenseElementsAttr>
  using iterator = decltype(std::declval<AttrT>().template value_begin<T>());
  /// The iterator range over the given element T.
  template <typename T, typename AttrT = DenseElementsAttr>
  using iterator_range =
      decltype(std::declval<AttrT>().template getValues<T>());

  /// A utility iterator that allows walking over the internal Attribute values
  /// of a DenseElementsAttr.
  class AttributeElementIterator
      : public llvm::indexed_accessor_iterator<AttributeElementIterator,
                                               const void *, Attribute,
                                               Attribute, Attribute> {
  public:
    /// Accesses the Attribute value at this iterator position.
    Attribute operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    AttributeElementIterator(DenseElementsAttr attr, size_t index);
  };

  /// Iterator for walking raw element values of the specified type 'T', which
  /// may be any c++ data type matching the stored representation: int32_t,
  /// float, etc.
  template <typename T>
  class ElementIterator
      : public detail::DenseElementIndexedIteratorImpl<ElementIterator<T>,
                                                       const T> {
  public:
    /// Accesses the raw value at this iterator position.
    const T &operator*() const {
      return reinterpret_cast<const T *>(this->getData())[this->getDataIndex()];
    }

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    ElementIterator(const char *data, bool isSplat, size_t dataIndex)
        : detail::DenseElementIndexedIteratorImpl<ElementIterator<T>, const T>(
              data, isSplat, dataIndex) {}
  };

  /// A utility iterator that allows walking over the internal bool values.
  class BoolElementIterator
      : public detail::DenseElementIndexedIteratorImpl<BoolElementIterator,
                                                       bool, bool, bool> {
  public:
    /// Accesses the bool value at this iterator position.
    bool operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    BoolElementIterator(DenseElementsAttr attr, size_t dataIndex);
  };

  /// A utility iterator that allows walking over the internal raw APInt values.
  class IntElementIterator
      : public detail::DenseElementIndexedIteratorImpl<IntElementIterator,
                                                       APInt, APInt, APInt> {
  public:
    /// Accesses the raw APInt value at this iterator position.
    APInt operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    IntElementIterator(DenseElementsAttr attr, size_t dataIndex);

    /// The bitwidth of the element type.
    size_t bitWidth;
  };

  /// A utility iterator that allows walking over the internal raw complex APInt
  /// values.
  class ComplexIntElementIterator
      : public detail::DenseElementIndexedIteratorImpl<
            ComplexIntElementIterator, std::complex<APInt>, std::complex<APInt>,
            std::complex<APInt>> {
  public:
    /// Accesses the raw std::complex<APInt> value at this iterator position.
    std::complex<APInt> operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    ComplexIntElementIterator(DenseElementsAttr attr, size_t dataIndex);

    /// The bitwidth of the element type.
    size_t bitWidth;
  };

  /// Iterator for walking over APFloat values.
  class FloatElementIterator final
      : public llvm::mapped_iterator_base<FloatElementIterator,
                                          IntElementIterator, APFloat> {
  public:
    /// Map the element to the iterator result type.
    APFloat mapElement(const APInt &value) const {
      return APFloat(*smt, value);
    }

  private:
    friend DenseElementsAttr;

    /// Initializes the float element iterator to the specified iterator.
    FloatElementIterator(const llvm::fltSemantics &smt, IntElementIterator it)
        : BaseT(it), smt(&smt) {}

    /// The float semantics to use when constructing the APFloat.
    const llvm::fltSemantics *smt;
  };

  /// Iterator for walking over complex APFloat values.
  class ComplexFloatElementIterator final
      : public llvm::mapped_iterator_base<ComplexFloatElementIterator,
                                          ComplexIntElementIterator,
                                          std::complex<APFloat>> {
  public:
    /// Map the element to the iterator result type.
    std::complex<APFloat> mapElement(const std::complex<APInt> &value) const {
      return {APFloat(*smt, value.real()), APFloat(*smt, value.imag())};
    }

  private:
    friend DenseElementsAttr;

    /// Initializes the float element iterator to the specified iterator.
    ComplexFloatElementIterator(const llvm::fltSemantics &smt,
                                ComplexIntElementIterator it)
        : BaseT(it), smt(&smt) {}

    /// The float semantics to use when constructing the APFloat.
    const llvm::fltSemantics *smt;
  };

  //===--------------------------------------------------------------------===//
  // Value Querying
  //===--------------------------------------------------------------------===//

  /// Returns true if this attribute corresponds to a splat, i.e. if all element
  /// values are the same.
  bool isSplat() const;

  /// Return the splat value for this attribute. This asserts that the attribute
  /// corresponds to a splat.
  template <typename T>
  std::enable_if_t<!std::is_base_of<Attribute, T>::value ||
                       std::is_same<Attribute, T>::value,
                   T>
  getSplatValue() const {
    assert(isSplat() && "expected the attribute to be a splat");
    return *value_begin<T>();
  }
  /// Return the splat value for derived attribute element types.
  template <typename T>
  std::enable_if_t<std::is_base_of<Attribute, T>::value &&
                       !std::is_same<Attribute, T>::value,
                   T>
  getSplatValue() const {
    return llvm::cast<T>(getSplatValue<Attribute>());
  }

  /// Try to get an iterator of the given type to the start of the held element
  /// values. Return failure if the type cannot be iterated.
  template <typename T>
  auto try_value_begin() const {
    auto range = tryGetValues<T>();
    using iterator = decltype(range->begin());
    return failed(range) ? FailureOr<iterator>(failure()) : range->begin();
  }

  /// Try to get an iterator of the given type to the end of the held element
  /// values. Return failure if the type cannot be iterated.
  template <typename T>
  auto try_value_end() const {
    auto range = tryGetValues<T>();
    using iterator = decltype(range->begin());
    return failed(range) ? FailureOr<iterator>(failure()) : range->end();
  }

  /// Return the held element values as a range of the given type.
  template <typename T>
  auto getValues() const {
    auto range = tryGetValues<T>();
    assert(succeeded(range) && "element type cannot be iterated");
    return std::move(*range);
  }

  /// Get an iterator of the given type to the start of the held element values.
  template <typename T>
  auto value_begin() const {
    return getValues<T>().begin();
  }

  /// Get an iterator of the given type to the end of the held element values.
  template <typename T>
  auto value_end() const {
    return getValues<T>().end();
  }

  /// Try to get the held element values as a range of integer or floating-point
  /// values.
  template <typename T>
  using IntFloatValueTemplateCheckT =
      std::enable_if_t<(!std::is_same<T, bool>::value &&
                        std::numeric_limits<T>::is_integer) ||
                       is_valid_cpp_fp_type<T>::value>;
  template <typename T, typename = IntFloatValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<ElementIterator<T>>> tryGetValues() const {
    if (!isValidIntOrFloat(sizeof(T), std::numeric_limits<T>::is_integer,
                           std::numeric_limits<T>::is_signed))
      return failure();
    const char *rawData = getRawData().data();
    bool splat = isSplat();
    return iterator_range_impl<ElementIterator<T>>(
        getType(), ElementIterator<T>(rawData, splat, 0),
        ElementIterator<T>(rawData, splat, getNumElements()));
  }

  /// Try to get the held element values as a range of std::complex.
  template <typename T, typename ElementT>
  using ComplexValueTemplateCheckT =
      std::enable_if_t<detail::is_complex_t<T>::value &&
                       (std::numeric_limits<ElementT>::is_integer ||
                        is_valid_cpp_fp_type<ElementT>::value)>;
  template <typename T, typename ElementT = typename T::value_type,
            typename = ComplexValueTemplateCheckT<T, ElementT>>
  FailureOr<iterator_range_impl<ElementIterator<T>>> tryGetValues() const {
    if (!isValidComplex(sizeof(T), std::numeric_limits<ElementT>::is_integer,
                        std::numeric_limits<ElementT>::is_signed))
      return failure();
    const char *rawData = getRawData().data();
    bool splat = isSplat();
    return iterator_range_impl<ElementIterator<T>>(
        getType(), ElementIterator<T>(rawData, splat, 0),
        ElementIterator<T>(rawData, splat, getNumElements()));
  }

  /// Try to get the held element values as a range of StringRef.
  template <typename T>
  using StringRefValueTemplateCheckT =
      std::enable_if_t<std::is_same<T, StringRef>::value>;
  template <typename T, typename = StringRefValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<ElementIterator<StringRef>>>
  tryGetValues() const {
    auto stringRefs = getRawStringData();
    const char *ptr = reinterpret_cast<const char *>(stringRefs.data());
    bool splat = isSplat();
    return iterator_range_impl<ElementIterator<StringRef>>(
        getType(), ElementIterator<StringRef>(ptr, splat, 0),
        ElementIterator<StringRef>(ptr, splat, getNumElements()));
  }

  /// Try to get the held element values as a range of Attributes.
  template <typename T>
  using AttributeValueTemplateCheckT =
      std::enable_if_t<std::is_same<T, Attribute>::value>;
  template <typename T, typename = AttributeValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<AttributeElementIterator>>
  tryGetValues() const {
    return iterator_range_impl<AttributeElementIterator>(
        getType(), AttributeElementIterator(*this, 0),
        AttributeElementIterator(*this, getNumElements()));
  }

  /// Try to get the held element values a range of T, where T is a derived
  /// attribute type.
  template <typename T>
  using DerivedAttrValueTemplateCheckT =
      std::enable_if_t<std::is_base_of<Attribute, T>::value &&
                       !std::is_same<Attribute, T>::value>;
  template <typename T>
  struct DerivedAttributeElementIterator
      : public llvm::mapped_iterator_base<DerivedAttributeElementIterator<T>,
                                          AttributeElementIterator, T> {
    using llvm::mapped_iterator_base<DerivedAttributeElementIterator<T>,
                                     AttributeElementIterator,
                                     T>::mapped_iterator_base;

    /// Map the element to the iterator result type.
    T mapElement(Attribute attr) const { return llvm::cast<T>(attr); }
  };
  template <typename T, typename = DerivedAttrValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<DerivedAttributeElementIterator<T>>>
  tryGetValues() const {
    using DerivedIterT = DerivedAttributeElementIterator<T>;
    return iterator_range_impl<DerivedIterT>(
        getType(), DerivedIterT(value_begin<Attribute>()),
        DerivedIterT(value_end<Attribute>()));
  }

  /// Try to get the held element values as a range of bool. The element type of
  /// this attribute must be of integer type of bitwidth 1.
  template <typename T>
  using BoolValueTemplateCheckT =
      std::enable_if_t<std::is_same<T, bool>::value>;
  template <typename T, typename = BoolValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<BoolElementIterator>> tryGetValues() const {
    if (!isValidBool())
      return failure();
    return iterator_range_impl<BoolElementIterator>(
        getType(), BoolElementIterator(*this, 0),
        BoolElementIterator(*this, getNumElements()));
  }

  /// Try to get the held element values as a range of APInts. The element type
  /// of this attribute must be of integer type.
  template <typename T>
  using APIntValueTemplateCheckT =
      std::enable_if_t<std::is_same<T, APInt>::value>;
  template <typename T, typename = APIntValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<IntElementIterator>> tryGetValues() const {
    if (!getElementType().isIntOrIndex())
      return failure();
    return iterator_range_impl<IntElementIterator>(getType(), raw_int_begin(),
                                                   raw_int_end());
  }

  /// Try to get the held element values as a range of complex APInts. The
  /// element type of this attribute must be a complex of integer type.
  template <typename T>
  using ComplexAPIntValueTemplateCheckT =
      std::enable_if_t<std::is_same<T, std::complex<APInt>>::value>;
  template <typename T, typename = ComplexAPIntValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<ComplexIntElementIterator>>
  tryGetValues() const {
    return tryGetComplexIntValues();
  }

  /// Try to get the held element values as a range of APFloat. The element type
  /// of this attribute must be of float type.
  template <typename T>
  using APFloatValueTemplateCheckT =
      std::enable_if_t<std::is_same<T, APFloat>::value>;
  template <typename T, typename = APFloatValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<FloatElementIterator>> tryGetValues() const {
    return tryGetFloatValues();
  }

  /// Try to get the held element values as a range of complex APFloat. The
  /// element type of this attribute must be a complex of float type.
  template <typename T>
  using ComplexAPFloatValueTemplateCheckT =
      std::enable_if_t<std::is_same<T, std::complex<APFloat>>::value>;
  template <typename T, typename = ComplexAPFloatValueTemplateCheckT<T>>
  FailureOr<iterator_range_impl<ComplexFloatElementIterator>>
  tryGetValues() const {
    return tryGetComplexFloatValues();
  }

  /// Return the raw storage data held by this attribute. Users should generally
  /// not use this directly, as the internal storage format is not always in the
  /// form the user might expect.
  ArrayRef<char> getRawData() const;

  /// Return the raw StringRef data held by this attribute.
  ArrayRef<StringRef> getRawStringData() const;

  /// Return the type of this ElementsAttr, guaranteed to be a vector or tensor
  /// with static shape.
  ShapedType getType() const;

  /// Return the element type of this DenseElementsAttr.
  Type getElementType() const;

  /// Returns the number of elements held by this attribute.
  int64_t getNumElements() const;

  /// Returns the number of elements held by this attribute.
  int64_t size() const { return getNumElements(); }

  /// Returns if the number of elements held by this attribute is 0.
  bool empty() const { return size() == 0; }

  //===--------------------------------------------------------------------===//
  // Mutation Utilities
  //===--------------------------------------------------------------------===//

  /// Return a new DenseElementsAttr that has the same data as the current
  /// attribute, but has been reshaped to 'newType'. The new type must have the
  /// same total number of elements as well as element type.
  DenseElementsAttr reshape(ShapedType newType);

  /// Return a new DenseElementsAttr that has the same data as the current
  /// attribute, but with a different shape for a splat type. The new type must
  /// have the same element type.
  DenseElementsAttr resizeSplat(ShapedType newType);

  /// Return a new DenseElementsAttr that has the same data as the current
  /// attribute, but has bitcast elements to 'newElType'. The new type must have
  /// the same bitwidth as the current element type.
  DenseElementsAttr bitcast(Type newElType);

  /// Generates a new DenseElementsAttr by mapping each int value to a new
  /// underlying APInt. The new values can represent either an integer or float.
  /// This underlying type must be an DenseIntElementsAttr.
  DenseElementsAttr mapValues(Type newElementType,
                              function_ref<APInt(const APInt &)> mapping) const;

  /// Generates a new DenseElementsAttr by mapping each float value to a new
  /// underlying APInt. the new values can represent either an integer or float.
  /// This underlying type must be an DenseFPElementsAttr.
  DenseElementsAttr
  mapValues(Type newElementType,
            function_ref<APInt(const APFloat &)> mapping) const;

protected:
  /// Iterators to various elements that require out-of-line definition. These
  /// are hidden from the user to encourage consistent use of the
  /// getValues/value_begin/value_end API.
  IntElementIterator raw_int_begin() const {
    return IntElementIterator(*this, 0);
  }
  IntElementIterator raw_int_end() const {
    return IntElementIterator(*this, getNumElements());
  }
  FailureOr<iterator_range_impl<ComplexIntElementIterator>>
  tryGetComplexIntValues() const;
  FailureOr<iterator_range_impl<FloatElementIterator>>
  tryGetFloatValues() const;
  FailureOr<iterator_range_impl<ComplexFloatElementIterator>>
  tryGetComplexFloatValues() const;

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// complex type. This method is used to verify type invariants that the
  /// templatized 'get' method cannot.
  static DenseElementsAttr getRawComplex(ShapedType type, ArrayRef<char> data,
                                         int64_t dataEltSize, bool isInt,
                                         bool isSigned);

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// integer or floating-point type. This method is used to verify type
  /// invariants that the templatized 'get' method cannot.
  static DenseElementsAttr getRawIntOrFloat(ShapedType type,
                                            ArrayRef<char> data,
                                            int64_t dataEltSize, bool isInt,
                                            bool isSigned);

  /// Check the information for a C++ data type, check if this type is valid for
  /// the current attribute. This method is used to verify specific type
  /// invariants that the templatized 'getValues' method cannot.
  bool isValidBool() const { return getElementType().isInteger(1); }
  bool isValidIntOrFloat(int64_t dataEltSize, bool isInt, bool isSigned) const;
  bool isValidComplex(int64_t dataEltSize, bool isInt, bool isSigned) const;
};

/// An attribute that represents a reference to a splat vector or tensor
/// constant, meaning all of the elements have the same value.
class SplatElementsAttr : public DenseElementsAttr {
public:
  using DenseElementsAttr::DenseElementsAttr;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr) {
    auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr);
    return denseAttr && denseAttr.isSplat();
  }
};

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttr
//===----------------------------------------------------------------------===//

using DenseResourceElementsHandle = DialectResourceBlobHandle<BuiltinDialect>;

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinAttributes.h.inc"

//===----------------------------------------------------------------------===//
// C++ Attribute Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
//===----------------------------------------------------------------------===//
// DenseArrayAttr

namespace detail {
/// Base class for DenseArrayAttr that is instantiated and specialized for each
/// supported element type below.
template <typename T>
class DenseArrayAttrImpl : public DenseArrayAttr {
public:
  using DenseArrayAttr::DenseArrayAttr;

  /// Implicit conversion to ArrayRef<T>.
  operator ArrayRef<T>() const;
  ArrayRef<T> asArrayRef() const { return ArrayRef<T>{*this}; }

  /// Random access to elements.
  T operator[](std::size_t index) const { return asArrayRef()[index]; }

  /// Builder from ArrayRef<T>.
  static DenseArrayAttrImpl get(MLIRContext *context, ArrayRef<T> content);

  /// Print the short form `[42, 100, -1]` without any type prefix.
  void print(AsmPrinter &printer) const;
  void print(raw_ostream &os) const;
  /// Print the short form `42, 100, -1` without any braces or type prefix.
  void printWithoutBraces(raw_ostream &os) const;

  /// Parse the short form `[42, 100, -1]` without any type prefix.
  static Attribute parse(AsmParser &parser, Type type);

  /// Parse the short form `42, 100, -1` without any type prefix or braces.
  static Attribute parseWithoutBraces(AsmParser &parser, Type type);

  /// Support for isa<>/cast<>.
  static bool classof(Attribute attr);
};

extern template class DenseArrayAttrImpl<bool>;
extern template class DenseArrayAttrImpl<int8_t>;
extern template class DenseArrayAttrImpl<int16_t>;
extern template class DenseArrayAttrImpl<int32_t>;
extern template class DenseArrayAttrImpl<int64_t>;
extern template class DenseArrayAttrImpl<float>;
extern template class DenseArrayAttrImpl<double>;
} // namespace detail

// Public name for all the supported DenseArrayAttr
using DenseBoolArrayAttr = detail::DenseArrayAttrImpl<bool>;
using DenseI8ArrayAttr = detail::DenseArrayAttrImpl<int8_t>;
using DenseI16ArrayAttr = detail::DenseArrayAttrImpl<int16_t>;
using DenseI32ArrayAttr = detail::DenseArrayAttrImpl<int32_t>;
using DenseI64ArrayAttr = detail::DenseArrayAttrImpl<int64_t>;
using DenseF32ArrayAttr = detail::DenseArrayAttrImpl<float>;
using DenseF64ArrayAttr = detail::DenseArrayAttrImpl<double>;

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttr

namespace detail {
/// Base class for DenseResourceElementsAttr that is instantiated and
/// specialized for each supported element type below.
template <typename T>
class DenseResourceElementsAttrBase : public DenseResourceElementsAttr {
public:
  using DenseResourceElementsAttr::DenseResourceElementsAttr;

  /// A builder that inserts a new resource using the provided blob. The handle
  /// of the inserted blob is used when building the attribute. The provided
  /// `blobName` is used as a hint for the key of the new handle for the `blob`
  /// resource, but may be changed if necessary to ensure uniqueness during
  /// insertion.
  static DenseResourceElementsAttrBase<T>
  get(ShapedType type, StringRef blobName, AsmResourceBlob blob);

  /// Return the data of this attribute as an ArrayRef<T> if it is present,
  /// returns std::nullopt otherwise.
  std::optional<ArrayRef<T>> tryGetAsArrayRef() const;

  /// Support for isa<>/cast<>.
  static bool classof(Attribute attr);
};

extern template class DenseResourceElementsAttrBase<bool>;
extern template class DenseResourceElementsAttrBase<int8_t>;
extern template class DenseResourceElementsAttrBase<int16_t>;
extern template class DenseResourceElementsAttrBase<int32_t>;
extern template class DenseResourceElementsAttrBase<int64_t>;
extern template class DenseResourceElementsAttrBase<uint8_t>;
extern template class DenseResourceElementsAttrBase<uint16_t>;
extern template class DenseResourceElementsAttrBase<uint32_t>;
extern template class DenseResourceElementsAttrBase<uint64_t>;
extern template class DenseResourceElementsAttrBase<float>;
extern template class DenseResourceElementsAttrBase<double>;
} // namespace detail

// Public names for all the supported DenseResourceElementsAttr.

using DenseBoolResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<bool>;
using DenseI8ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int8_t>;
using DenseI16ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int16_t>;
using DenseI32ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int32_t>;
using DenseI64ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int64_t>;
using DenseUI8ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint8_t>;
using DenseUI16ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint16_t>;
using DenseUI32ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint32_t>;
using DenseUI64ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint64_t>;
using DenseF32ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<float>;
using DenseF64ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<double>;

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

/// Special case of IntegerAttr to represent boolean integers, i.e., signless i1
/// integers.
class BoolAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ValueType = bool;

  static BoolAttr get(MLIRContext *context, bool value);

  /// Enable conversion to IntegerAttr and its interfaces. This uses conversion
  /// vs. inheritance to avoid bringing in all of IntegerAttrs methods.
  operator IntegerAttr() const { return IntegerAttr(impl); }
  operator TypedAttr() const { return IntegerAttr(impl); }

  /// Return the boolean value of this attribute.
  bool getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// FlatSymbolRefAttr
//===----------------------------------------------------------------------===//

/// A symbol reference with a reference path containing a single element. This
/// is used to refer to an operation within the current symbol table.
class FlatSymbolRefAttr : public SymbolRefAttr {
public:
  using SymbolRefAttr::SymbolRefAttr;
  using ValueType = StringRef;

  /// Construct a symbol reference for the given value name.
  static FlatSymbolRefAttr get(StringAttr value) {
    return SymbolRefAttr::get(value);
  }
  static FlatSymbolRefAttr get(MLIRContext *ctx, StringRef value) {
    return SymbolRefAttr::get(ctx, value);
  }

  /// Convenience getter for building a SymbolRefAttr based on an operation
  /// that implements the SymbolTrait.
  static FlatSymbolRefAttr get(Operation *symbol) {
    return SymbolRefAttr::get(symbol);
  }

  /// Returns the name of the held symbol reference as a StringAttr.
  StringAttr getAttr() const { return getRootReference(); }

  /// Returns the name of the held symbol reference.
  StringRef getValue() const { return getAttr().getValue(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr) {
    SymbolRefAttr refAttr = llvm::dyn_cast<SymbolRefAttr>(attr);
    return refAttr && refAttr.getNestedReferences().empty();
  }

private:
  using SymbolRefAttr::get;
  using SymbolRefAttr::getNestedReferences;
};

//===----------------------------------------------------------------------===//
// DenseFPElementsAttr
//===----------------------------------------------------------------------===//

/// An attribute that represents a reference to a dense float vector or tensor
/// object. Each element is stored as a double.
class DenseFPElementsAttr : public DenseIntOrFPElementsAttr {
public:
  using iterator = DenseElementsAttr::FloatElementIterator;

  using DenseIntOrFPElementsAttr::DenseIntOrFPElementsAttr;

  /// Get an instance of a DenseFPElementsAttr with the given arguments. This
  /// simply wraps the DenseElementsAttr::get calls.
  template <typename Arg>
  static DenseFPElementsAttr get(const ShapedType &type, Arg &&arg) {
    return llvm::cast<DenseFPElementsAttr>(
        DenseElementsAttr::get(type, llvm::ArrayRef(arg)));
  }
  template <typename T>
  static DenseFPElementsAttr get(const ShapedType &type,
                                 const std::initializer_list<T> &list) {
    return llvm::cast<DenseFPElementsAttr>(DenseElementsAttr::get(type, list));
  }

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr
  mapValues(Type newElementType,
            function_ref<APInt(const APFloat &)> mapping) const;

  /// Iterator access to the float element values.
  iterator begin() const { return tryGetFloatValues()->begin(); }
  iterator end() const { return tryGetFloatValues()->end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// DenseIntElementsAttr
//===----------------------------------------------------------------------===//

/// An attribute that represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr : public DenseIntOrFPElementsAttr {
public:
  /// DenseIntElementsAttr iterates on APInt, so we can use the raw element
  /// iterator directly.
  using iterator = DenseElementsAttr::IntElementIterator;

  using DenseIntOrFPElementsAttr::DenseIntOrFPElementsAttr;

  /// Get an instance of a DenseIntElementsAttr with the given arguments. This
  /// simply wraps the DenseElementsAttr::get calls.
  template <typename Arg>
  static DenseIntElementsAttr get(const ShapedType &type, Arg &&arg) {
    return llvm::cast<DenseIntElementsAttr>(
        DenseElementsAttr::get(type, llvm::ArrayRef(arg)));
  }
  template <typename T>
  static DenseIntElementsAttr get(const ShapedType &type,
                                  const std::initializer_list<T> &list) {
    return llvm::cast<DenseIntElementsAttr>(DenseElementsAttr::get(type, list));
  }

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr mapValues(Type newElementType,
                              function_ref<APInt(const APInt &)> mapping) const;

  /// Iterator access to the integer element values.
  iterator begin() const { return raw_int_begin(); }
  iterator end() const { return raw_int_end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

template <typename T>
auto SparseElementsAttr::try_value_begin_impl(OverloadToken<T>) const
    -> FailureOr<iterator<T>> {
  auto zeroValue = getZeroValue<T>();
  auto valueIt = getValues().try_value_begin<T>();
  if (failed(valueIt))
    return failure();
  const std::vector<ptrdiff_t> flatSparseIndices(getFlattenedSparseIndices());
  std::function<T(ptrdiff_t)> mapFn =
      [flatSparseIndices{flatSparseIndices}, valueIt{std::move(*valueIt)},
       zeroValue{std::move(zeroValue)}](ptrdiff_t index) {
        // Try to map the current index to one of the sparse indices.
        for (unsigned i = 0, e = flatSparseIndices.size(); i != e; ++i)
          if (flatSparseIndices[i] == index)
            return *std::next(valueIt, i);
        // Otherwise, return the zero value.
        return zeroValue;
      };
  return iterator<T>(llvm::seq<ptrdiff_t>(0, getNumElements()).begin(), mapFn);
}

//===----------------------------------------------------------------------===//
// DistinctAttr
//===----------------------------------------------------------------------===//

namespace detail {
struct DistinctAttrStorage;
class DistinctAttributeUniquer;
} // namespace detail

/// An attribute that associates a referenced attribute with a unique
/// identifier. Every call to the create function allocates a new distinct
/// attribute instance. The address of the attribute instance serves as a
/// temporary identifier. Similar to the names of SSA values, the final
/// identifiers are generated during pretty printing. This delayed numbering
/// ensures the printed identifiers are deterministic even if multiple distinct
/// attribute instances are created in-parallel.
///
/// Examples:
///
/// #distinct = distinct[0]<42.0 : f32>
/// #distinct1 = distinct[1]<42.0 : f32>
/// #distinct2 = distinct[2]<array<i32: 10, 42>>
///
/// NOTE: The distinct attribute cannot be defined using ODS since it uses a
/// custom distinct attribute uniquer that cannot be set from ODS.
class DistinctAttr
    : public detail::StorageUserBase<DistinctAttr, Attribute,
                                     detail::DistinctAttrStorage,
                                     detail::DistinctAttributeUniquer> {
public:
  using Base::Base;

  /// Returns the referenced attribute.
  Attribute getReferencedAttr() const;

  /// Creates a distinct attribute that associates a referenced attribute with a
  /// unique identifier.
  static DistinctAttr create(Attribute referencedAttr);

  static constexpr StringLiteral name = "builtin.distinct";
};

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

/// Define comparisons for StringAttr against nullptr and itself to avoid the
/// StringRef overloads from being chosen when not desirable.
inline bool operator==(StringAttr lhs, std::nullptr_t) { return !lhs; }
inline bool operator!=(StringAttr lhs, std::nullptr_t) {
  return static_cast<bool>(lhs);
}
inline bool operator==(StringAttr lhs, StringAttr rhs) {
  return (Attribute)lhs == (Attribute)rhs;
}
inline bool operator!=(StringAttr lhs, StringAttr rhs) { return !(lhs == rhs); }

/// Allow direct comparison with StringRef.
inline bool operator==(StringAttr lhs, StringRef rhs) {
  return lhs.getValue() == rhs;
}
inline bool operator!=(StringAttr lhs, StringRef rhs) { return !(lhs == rhs); }
inline bool operator==(StringRef lhs, StringAttr rhs) {
  return rhs.getValue() == lhs;
}
inline bool operator!=(StringRef lhs, StringAttr rhs) { return !(lhs == rhs); }

} // namespace mlir

//===----------------------------------------------------------------------===//
// Attribute Utilities
//===----------------------------------------------------------------------===//

namespace mlir {

/// Given a list of strides (in which ShapedType::kDynamic
/// represents a dynamic value), return the single result AffineMap which
/// represents the linearized strided layout map. Dimensions correspond to the
/// offset followed by the strides in order. Symbols are inserted for each
/// dynamic dimension in order. A stride is always positive.
///
/// Examples:
/// =========
///
///   1. For offset: 0 strides: ?, ?, 1 return
///         (i, j, k)[M, N]->(M * i + N * j + k)
///
///   2. For offset: 3 strides: 32, ?, 16 return
///         (i, j, k)[M]->(3 + 32 * i + M * j + 16 * k)
///
///   3. For offset: ? strides: ?, ?, ? return
///         (i, j, k)[off, M, N, P]->(off + M * i + N * j + P * k)
AffineMap makeStridedLinearLayoutMap(ArrayRef<int64_t> strides, int64_t offset,
                                     MLIRContext *context);

} // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::StringAttr> : public DenseMapInfo<mlir::Attribute> {
  static mlir::StringAttr getEmptyKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getEmptyKey();
    return mlir::StringAttr::getFromOpaquePointer(pointer);
  }
  static mlir::StringAttr getTombstoneKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getTombstoneKey();
    return mlir::StringAttr::getFromOpaquePointer(pointer);
  }
};
template <>
struct PointerLikeTypeTraits<mlir::StringAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::StringAttr getFromVoidPointer(void *p) {
    return mlir::StringAttr::getFromOpaquePointer(p);
  }
};

template <>
struct PointerLikeTypeTraits<mlir::IntegerAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::IntegerAttr getFromVoidPointer(void *p) {
    return mlir::IntegerAttr::getFromOpaquePointer(p);
  }
};

template <>
struct PointerLikeTypeTraits<mlir::SymbolRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::SymbolRefAttr getFromVoidPointer(void *ptr) {
    return mlir::SymbolRefAttr::getFromOpaquePointer(ptr);
  }
};

} // namespace llvm

#endif // MLIR_IR_BUILTINATTRIBUTES_H
