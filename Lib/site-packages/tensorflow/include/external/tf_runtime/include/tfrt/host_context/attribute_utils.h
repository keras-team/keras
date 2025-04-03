/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Helpers for BEF Attributes
//
// This file declares helper routines for reading BEF Attributes.

#ifndef TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_
#define TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_

#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Alignment.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/support/byte_order.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class StringAttribute;
template <typename T>
class ArrayAttribute;
class AggregateAttribute;

namespace internal {

template <typename>
struct is_array_attribute : std::false_type {};
template <typename T>
struct is_array_attribute<ArrayAttribute<T>> : std::true_type {};

}  // namespace internal

// Kernels should use this so we know they have an attribute input.
template <typename T>
class Attribute {
 public:
  explicit Attribute(const void* value)
      : value_(*reinterpret_cast<const T*>(value)) {
    ASSERT_LITTLE_ENDIAN();
  }

  const T& get() const { return value_; }
  const T* operator->() const { return &value_; }
  const T& operator*() const { return value_; }

 private:
  static_assert(!std::is_same<T, std::string>(),
                "Use StringAttribute instead of Attribute<std::string>");
  static_assert(
      !std::is_same<T, StringAttribute>(),
      "Use StringAttribute directly instead of Attribute<StringAttribute>");
  static_assert(!std::is_same<T, AggregateAttribute>(),
                "Use AggregateAttribute directly instead of "
                "Attribute<AggregateAttribute>");
  static_assert(!internal::is_array_attribute<T>(),
                "Use ArrayAttribute directly instead of "
                "Attribute<ArrayAttribute<T>>");

  const T& value_;
};

// Kernels should use this so we know it has an array attribute.
template <typename T>
class ArrayAttribute {
 public:
  explicit ArrayAttribute(const void* data) {
    auto* ptr = reinterpret_cast<const uint8_t*>(data);
    AttrSizeT element_count;
    std::memcpy(&element_count, ptr, sizeof(AttrSizeT));
    ptr += sizeof(AttrSizeT);
    data_ = llvm::ArrayRef(reinterpret_cast<const T*>(ptr), element_count);
  }

  ArrayRef<T> data() const { return data_; }
  size_t size() const { return data_.size(); }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  ArrayRef<T> data_;
};

// Like Attribute, but specifically for strings. We use this instead of
// Attribute<std::string> because strings are stored as character arrays and we
// don't want unnecessary deep copies.
//
// StringAttribute is equivalent to ArrayAttribute<char>, but
// StringAttribute provides a string_view, while ArrayAttribute<char>
// provides an ArrayRef<char>.
class StringAttribute {
 public:
  StringAttribute() = default;

  explicit StringAttribute(const void* ptr)
      : value_(DecodeLengthPrefixedString(ptr)) {}

  string_view get() const { return value_; }
  operator string_view() const { return value_; }
  std::string str() const { return std::string(value_); }

 private:
  string_view value_;
};

// Compilation unit attribute decodes serialized MLIR module and a compilation
// target symbol (function name).
//
// TODO(ezhulenev): CompilationUnitAttribute in addition to an `id` and `addr`
// should provide a hash (or something like sha-256 fingerprint) of its content
// for cache lookup.
class CompilationUnitAttribute {
 public:
  explicit CompilationUnitAttribute(const void* value)
      : addr_(reinterpret_cast<intptr_t>(value)) {
    ASSERT_LITTLE_ENDIAN();
    const auto* ptr = static_cast<const uint8_t*>(value);

    ptr = ReadVbrInt(ptr, &id_);

    size_t root_symbol_len;
    ptr = ReadVbrInt(ptr, &root_symbol_len);

    size_t num_nested_symbols;
    ptr = ReadVbrInt(ptr, &num_nested_symbols);

    llvm::SmallVector<size_t, 4> nested_symbols_len(num_nested_symbols);
    for (int i = 0; i < num_nested_symbols; ++i) {
      ptr = ReadVbrInt(ptr, &nested_symbols_len[i]);
    }

    size_t serialized_operation_len;
    ptr = ReadVbrInt(ptr, &serialized_operation_len);

    // The base of the attribute payload.
    const char* base = reinterpret_cast<const char*>(ptr);
    root_symbol_ = {base, root_symbol_len};
    size_t offset = root_symbol_len;

    nested_symbols_.reserve(num_nested_symbols);
    for (int i = 0; i < num_nested_symbols; ++i) {
      size_t len = nested_symbols_len[i];
      nested_symbols_.emplace_back(base + offset, len);
      offset += len;
    }

    serialized_operation_ = {base + offset, serialized_operation_len};
  }

  size_t id() const { return id_; }
  intptr_t addr() const { return addr_; }
  string_view root_symbol() const { return root_symbol_; }
  ArrayRef<string_view> nested_symbols() const { return nested_symbols_; }
  string_view serialized_operation() const { return serialized_operation_; }

 private:
  size_t id_;
  intptr_t addr_;
  string_view root_symbol_;
  llvm::SmallVector<string_view> nested_symbols_;
  string_view serialized_operation_;
};

// FunctionAttribute holds the function name. Can be extended in the future.
struct FunctionAttribute {
  string_view func_name;
};

// TypedAttrBase is the base class for all typed attributes below. It provides
// llvm style cast (isa, cast, dyn_cast, etc) for efficient down-casting to
// subclasses.
class TypedAttrBase {
 public:
  TypedAttrBase() = default;

  TypedAttrBase(BEFAttributeType type, const void* data)
      : type_(type), data_(static_cast<const uint8_t*>(data)) {}

  BEFAttributeType type() const { return type_; }

  const uint8_t* data() const { return data_; }

  template <typename T>
  bool isa() const {
    return T::classof(*this);
  }

  template <typename T>
  T dyn_cast() const {
    return isa<T>() ? T(type_, data_) : T();
  }

  template <typename T>
  T cast() const {
    assert(isa<T>());
    return T(type_, data_);
  }

  explicit operator bool() const { return data_ != nullptr; }

 private:
  BEFAttributeType type_ = BEFAttributeType::kUnsupported;
  const uint8_t* data_ = nullptr;
};

namespace internal {

// An intermediate class template for all fixed-width attributes. It provides
// the common GetValue() method for all fixed-width attributes.
template <DType DataTypeEnum, typename DataType>
class DataTypeAttrBase : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;
  explicit DataTypeAttrBase(const void* data)
      : TypedAttrBase(static_cast<BEFAttributeType>(DataTypeEnum), data) {}

  DataType GetValue() const {
    DataType value;
    std::memcpy(&value, data(), sizeof(DataType));
    return value;
  }

  size_t GetByteSize() const { return sizeof(DataType); }

  static bool classof(TypedAttrBase base) {
    const auto attr_type = base.type();
    return IsDataTypeAttribute(attr_type) &&
           GetDataType(attr_type) == DataTypeEnum;
  }
};

}  // namespace internal

using UI8Attr = internal::DataTypeAttrBase<DType::UI8, uint8_t>;
using UI16Attr = internal::DataTypeAttrBase<DType::UI16, uint16_t>;
using UI32Attr = internal::DataTypeAttrBase<DType::UI32, uint32_t>;
using UI64Attr = internal::DataTypeAttrBase<DType::UI64, uint64_t>;
using I8Attr = internal::DataTypeAttrBase<DType::I8, uint8_t>;
using I16Attr = internal::DataTypeAttrBase<DType::I16, int16_t>;
using I32Attr = internal::DataTypeAttrBase<DType::I32, int32_t>;
using I64Attr = internal::DataTypeAttrBase<DType::I64, int64_t>;
using F32Attr = internal::DataTypeAttrBase<DType::F32, float>;
using F64Attr = internal::DataTypeAttrBase<DType::F64, double>;
using BF16Attr = internal::DataTypeAttrBase<DType::BF16, int16_t>;
using F16Attr = internal::DataTypeAttrBase<DType::F16, int16_t>;

class I1Attr : public internal::DataTypeAttrBase<DType::I1, uint8_t> {
 public:
  using DataTypeAttrBase::DataTypeAttrBase;

  bool GetValue() const {
    return static_cast<bool>(DataTypeAttrBase::GetValue());
  }
};

class TypeAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;
  explicit TypeAttr(const void* data)
      : TypedAttrBase(BEFAttributeType::kType, data) {}

  DType GetValue() const {
    DType dtype;
    std::memcpy(&dtype, data(), sizeof(DType));
    return dtype;
  }

  static bool classof(TypedAttrBase base) {
    return base.type() == BEFAttributeType::kType;
  }
};

class ArrayAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;
  explicit ArrayAttr(const void* ptr)
      : ArrayAttr(BEFAttributeType::kArray, ptr) {}

  ArrayAttr(BEFAttributeType type, const void* ptr) : TypedAttrBase(type, ptr) {
    std::memcpy(&element_count_, data(), sizeof(AttrSizeT));
    element_base_ = data() + sizeof(AttrSizeT);
  }

  BEFAttributeType GetElementType() const {
    return GetElementAttributeType(type());
  }

  const void* GetElements() const { return element_base_; }

  template <typename T>
  ArrayRef<T> GetValue() const {
    // For empty arrays, we don't care the element type.
    if (GetNumElements() == 0) return {};
    return llvm::ArrayRef(static_cast<const T*>(GetElements()),
                          GetNumElements());
  }

  size_t GetNumElements() const { return element_count_; }

  static bool classof(TypedAttrBase base) {
    return IsArrayAttribute(base.type());
  }

  size_t GetByteSize() const {
    return sizeof(AttrSizeT) +
           GetAttributeDataTypeByteSize(GetElementAttributeType(type())) *
               element_count_;
  }

 protected:
  const void* element_base_;
  AttrSizeT element_count_;
};

class StringAttr : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;
  explicit StringAttr(const void* ptr)
      : StringAttr(static_cast<BEFAttributeType>(DType::String), ptr) {}

  StringAttr(BEFAttributeType type, const void* ptr)
      : TypedAttrBase(type, ptr), str_(DecodeLengthPrefixedString(data())) {}

  string_view GetValue() const { return str_; }

  static bool classof(TypedAttrBase base) {
    return IsDataTypeAttribute(base.type()) &&
           GetDataType(base.type()) == DType::String;
  }

 private:
  string_view str_;
};

// FuncAttr holds the function names as strings. This attribute is separated
// from StringAttr so that clients (such as TensorFlow runtime fallback)
// can handle separately.
//
// Currently we ignore the attributes in a TensorFlow function op, which is
// different from current TensorFlow runtime. This is acceptable since these
// attributes are unused.
class FuncAttr : public TypedAttrBase {
 public:
  explicit FuncAttr(const void* data)
      : FuncAttr(BEFAttributeType::kFunc, data) {}

  FuncAttr(BEFAttributeType type, const void* ptr)
      : TypedAttrBase(type, ptr), name_(DecodeLengthPrefixedString(data())) {}

  string_view GetFunctionName() const { return name_; }

  static bool classof(TypedAttrBase base) {
    return base.type() == BEFAttributeType::kFunc;
  }

 private:
  string_view name_;
};

template <size_t padding, BEFAttributeType attribute_type, typename AttrStruct>
class StructAttrBase : public TypedAttrBase {
 public:
  using TypedAttrBase::TypedAttrBase;

  StructAttrBase() = default;

  explicit StructAttrBase(const void* data)
      : StructAttrBase(attribute_type, data) {}

  StructAttrBase(BEFAttributeType type, const void* ptr)
      : TypedAttrBase(type, ptr) {
    // handle special case. When the first byte (alignment) has 0 value,
    // it means that it is an empty aggregate or unranked shape.
    if (*data() > 0) {
      header_ = reinterpret_cast<const AttrStruct*>(data() - padding);
    } else {
      header_ = nullptr;
    }
  }

  // Return the peak alignment size.
  size_t Alignment() const {
    return (header_) ? header_->base.alignment : alignof(AttrSizeT);
  }

  // Return the prefix size.
  size_t GetPrefixSize() const {
    return (header_) ? header_->base.prefix_size : 0;
  }

  // Return the total byte size.
  size_t GetByteSize() const {
    return (header_) ? header_->base.byte_size : sizeof(AttrSizeT);
  }

  static bool classof(TypedAttrBase base) {
    return base.type() == attribute_type;
  }

 protected:
  const AttrStruct* header_ = nullptr;
};

class ShapeAttr
    : public StructAttrBase<4, BEFAttributeType::kShape, BefShapeAttr> {
 public:
  using StructAttrBase::StructAttrBase;

  int GetRank() const { return header_->rank; }

  bool HasRank() const { return header_ != nullptr; }

  ArrayRef<AttrShapeT> GetShape() const {
    return llvm::ArrayRef<AttrShapeT>(header_->dims, header_->rank);
  }
};

class DenseAttr
    : public StructAttrBase<4, BEFAttributeType::kDense, BefDenseAttr> {
 public:
  using StructAttrBase::StructAttrBase;

  DType dtype() const { return header_->base.element_type; }

  llvm::ArrayRef<int64_t> shape() const {
    return llvm::ArrayRef(header_->dims, header_->rank);
  }
  size_t GetNumElements() const { return header_->element_count; }

  const void* GetElements() const { return data() + header_->element_offset; }

  ArrayRef<char> GetRawData() const {
    return llvm::ArrayRef<char>(
        reinterpret_cast<const char*>(data() + header_->element_offset),
        header_->base.byte_size - header_->element_offset);
  }

  template <typename T>
  const T& GetElement(size_t index) const {
    assert(GetDType<T>() == dtype());
    return *(reinterpret_cast<const T*>(data() + header_->element_offset) +
             index);
  }
};

class AggregateAttr
    : public StructAttrBase<0, BEFAttributeType::kAggregate, BefAggregateAttr> {
 public:
  using StructAttrBase::StructAttrBase;

  size_t GetNumElements() const {
    return (header_) ? header_->element_count : 0;
  }

  BEFAttributeType GetElementType(int index) const {
    assert(header_ && index < header_->element_count);
    BEFAttributeType element_type;
    std::memcpy(&element_type,
                data() + header_->offsets[index] - sizeof(BEFAttributeType),
                sizeof(BEFAttributeType));
    return element_type;
  }

  size_t GetElementOffset(int index) const {
    assert(header_ && index < header_->element_count);
    return header_->offsets[index];
  }

  TypedAttrBase GetAttribute(int index) const {
    assert(header_ && index < header_->element_count);
    auto ptr = data() + header_->offsets[index];
    BEFAttributeType element_type;
    std::memcpy(&element_type, ptr - sizeof(BEFAttributeType),
                sizeof(BEFAttributeType));
    return TypedAttrBase(element_type, ptr);
  }

  // Usage example;
  //   string_view sv = agg_attr.GetElement<StringAttr>(0).GetValue();
  template <typename AttrClass>
  AttrClass GetAttributeOfType(int index) const {
    assert(header_ && index < header_->element_count);
    return AttrClass(data() + header_->offsets[index]);
  }
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_ATTRIBUTE_UTILS_H_
