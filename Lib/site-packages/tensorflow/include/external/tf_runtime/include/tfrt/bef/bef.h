/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

// This file defines BEF primitives that can be used to build BEF structures.
// This library is C++14 compliant and portable for different platforms. It
// should be also as effcient as plain C++ structs on common platforms.
//
// Usage:
//
// class CustomStruct {
//  public:
//    // The actual storage of this CustomStruct should be defined as a member
//    // struct of this class. Defining storage struct is almost as simple as
//    // defining a plain C++ struct;
//    struct Storage {
//      using Self = Storage;
//      // DEFINE_BEF_FIELD will generate helpers for reading and constructing
//      // the field in BEF.
//      DEFINE_BEF_FIELD(uint32_t, x);
//      DEFINE_BEF_FIELD(bef::Vector<uint32_t>, y);
//    };
//
//    // If the storage involves indirection like std::vector, a member class
//    // Constructor should be also provided.
//    class Constructor {
//      public:
//        // The Constructor will use `allocator` to allocate indirect storage,
//        // though the direct storage is assumed to be already allocated using
//        // the same allocator starting at `address`.
//        explicit Constructor(Allocator* allocator, BefAddr32_t address)
//          : allocator_(allocator), address_(address) {}
//
//      // Setting trivial fields only need to call construct_<field_name>
//      // provided by DEFINE_BEF_FIELD.
//      void set_x(uint32_t x) {
//        Storage::construct_x(allocator_, address_, x);
//      }
//
//      // Setting non-trivial fields only need to call construct_<field_name>
//      // provided by DEFINE_BEF_FIELD and also return the field's constructor.
//      bef::Vector<uint32_t>::Constructor construct_y(size_t y_size) {
//        return Storage::construct_y(allocator_, address_, y_size);
//      }
//
//      BefAddr32_t address() const { return address_; }
//
//      private:
//        bef::Allocator* allocator_;
//        BefAddr32_t address_;
//    };
//    using NonTrivialConstructorType = Constructor;
//
//    explicit CustomStruct(const char* p) : p_(p) {}
//
//    // Reading fields needs only calling read_<field_name> methods provided by
//    // DEFINE_BEF_FIELD.
//    uint32_t x() const { return Storage::read_x(p_); }
//    bef::Vector<uint32_t> y() const { return Storage::read_y(p_); }
//
//    private:
//      // The CustomStruct can contain only the pointer to the actual memory
//      // blob. So fields need not be touched if not necessary, which would
//      // otherwise incurs overhead.
//      const char* p_;
// };

#ifndef TFRT_BEF_BEF_H_
#define TFRT_BEF_BEF_H_

#include <cassert>
#include <cstring>
#include <utility>
#include <vector>

#include "tfrt/support/aligned_buffer.h"
#include "tfrt/support/type_traits.h"

namespace tfrt {
namespace bef {

using BefAddr32_t = uint32_t;

class Buffer {
 public:
  char* Get(BefAddr32_t address) { return &buffer_.at(address); }

  size_t size() const { return buffer_.size(); }
  bool empty() const { return buffer_.empty(); }

 private:
  std::vector<char, internal::AlignedAllocator<char, 8>> buffer_;

  friend class Allocator;
};

class Allocator {
 public:
  explicit Allocator(Buffer* buffer) : buffer_(buffer) {
    assert(buffer != nullptr);
  }

  BefAddr32_t Allocate(size_t size, size_t alignment) {
    assert(alignment <= 8);

    // Calculate the next buffer size that is greater or equal to the previous
    // buffer size, and is also aligned to `alignment`.
    size_t next_align =
        (buffer_->buffer_.size() + alignment - 1) / alignment * alignment;

    buffer_->buffer_.resize(next_align + size);

    return next_align;
  }

  template <typename T>
  BefAddr32_t Allocate() {
    static_assert(std::is_trivial<T>::value, "T must be trivial.");
    return Allocate(sizeof(T), alignof(T));
  }

  size_t size() const { return buffer_->size(); }

  char* raw(BefAddr32_t address) { return buffer_->Get(address); }

 private:
  Buffer* buffer_;
};

// AccessTraits encapsulates the fundamental Read() and Construct() methods for
// reading and constructing BEF data structures.

// AccessTraits specialized for trivial types.
template <typename T, typename Enable = void>
struct AccessTraits {
  using StorageType = T;
  static_assert(std::is_trivial<StorageType>::value,
                "StorageType must be trivial.");

  using ConstructorType = void;

  static T Read(const char* p) {
    // To be compliant with C++ standard on object lifetime and strict aliasing
    // rules, we have to copy the data from memory to construct a new object.
    // This is fine on most platforms as the copy can be optimized away,
    // assuming `p` is sufficiently aligned.
    T value;
    std::memcpy(&value, p, sizeof(T));
    return value;
  }

  template <typename... Args>
  static BefAddr32_t Construct(Allocator* allocator, BefAddr32_t address,
                               Args&&... args) {
    // Similar to Read(), memcpy is used to serialize data to BEF.
    T value(std::forward<Args>(args)...);
    std::memcpy(allocator->raw(address), &value, sizeof(T));
    return address;
  }
};

// AccessTraits specialized for non-trivial types.
template <typename T>
struct AccessTraits<T, void_t<typename T::NonTrivialConstructorType>> {
  // Non-trivial types should provide a member struct `StorageType` to
  // specify the storage layout.
  using StorageType = typename T::StorageType;
  static_assert(std::is_trivial<StorageType>::value,
                "StorageType must be trivial.");

  // Non-trivial types should provide a member type `NonTrivialConstructorType`
  // for contructing storages.
  using ConstructorType = typename T::NonTrivialConstructorType;

  static T Read(const char* p) {
    // Reading non-trivial types is simply constructing the BEF type with the
    // pointer to the memory blob. All reading methods are encapsulated in `T`.
    return T(p);
  }

  template <typename... Args>
  static ConstructorType Construct(Allocator* allocator, BefAddr32_t address,
                                   Args&&... args) {
    // Constructing non-trivial types is simply creating the corresponding
    // constructor.
    return ConstructorType(allocator, address, std::forward<Args>(args)...);
  }
};

// The BEF counterparts of malloc() and operator new() are also provided.
template <typename T>
BefAddr32_t Allocate(Allocator* allocator) {
  return allocator->Allocate<typename AccessTraits<T>::StorageType>();
}
template <typename T, typename... Args>
auto New(Allocator* allocator, Args&&... args) {
  auto address = Allocate<T>(allocator);
  return AccessTraits<T>::Construct(allocator, address,
                                    std::forward<Args>(args)...);
}

// The iterator for reading BEF data. It uses AccessTraits<T>::Read() for
// reading the data. It is an input iterator as we cannot return the type-safe
// reference to the data in BEF in a C++ compliant way due to object lifetime
// and strict aliasing rule.
template <typename T>
class ReadIterator {
  using StorageType = typename AccessTraits<T>::StorageType;

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_cv_t<T>;
  using pointer = void;
  using reference = value_type;
  using iterator_category = std::input_iterator_tag;

  explicit ReadIterator(const char* data) : data_(data) {}

  const char* data() const { return data_; }

  value_type operator*() const { return AccessTraits<T>::Read(data_); }

  ReadIterator& operator++() {
    data_ += sizeof(StorageType);
    return *this;
  }

  ReadIterator operator++(int) {
    ReadIterator r = *this;
    data_ += sizeof(StorageType);
    return r;
  }

  ReadIterator& operator+=(int offset) {
    data_ += offset * sizeof(StorageType);
    return *this;
  }

  friend bool operator==(const ReadIterator& a, const ReadIterator& b) {
    return a.data_ == b.data_;
  }

  friend bool operator!=(const ReadIterator& a, const ReadIterator& b) {
    return !(a == b);
  }

 private:
  const char* data_ = nullptr;
};

// DEFINE_BEF_FIELD provides helper functions for reading and constructing
// member fields in BEF.
#define DEFINE_BEF_FIELD(Type, name)                                        \
  ::tfrt::bef::AccessTraits<Type>::StorageType name;                        \
  static const char* name##_pointer(const char* base) {                     \
    return base + offsetof(Self, name);                                     \
  }                                                                         \
  static ::tfrt::bef::BefAddr32_t name##_address(                           \
      ::tfrt::bef::BefAddr32_t base) {                                      \
    return base + offsetof(Self, name);                                     \
  }                                                                         \
  static Type read_##name(const char* base) {                               \
    return ::tfrt::bef::AccessTraits<Type>::Read(name##_pointer(base));     \
  }                                                                         \
  template <typename... Args>                                               \
  static auto construct_##name(::tfrt::bef::Allocator* allocator,           \
                               ::tfrt::bef::BefAddr32_t base,               \
                               Args&&... args) {                            \
    return ::tfrt::bef::AccessTraits<Type>::Construct(                      \
        allocator, name##_address(base), std::forward<Args>(args)...);      \
  }                                                                         \
  static_assert(                                                            \
      std::is_trivial<::tfrt::bef::AccessTraits<Type>::StorageType>::value, \
      "BEF storage types must be trivial.")

// Defines a BEF vector.
template <typename T>
class Vector {
 public:
  struct Storage {
    using Self = Storage;
    DEFINE_BEF_FIELD(uint32_t, size);
    DEFINE_BEF_FIELD(uint32_t, offset);
  };
  static_assert(std::is_trivial<Storage>::value, "StorageType is trivial");
  static_assert(std::is_standard_layout<Storage>::value,
                "StorageType has standard layout");
  static_assert(sizeof(Storage) == 8,
                "The size of the inline storage of Vector is 8");
  static_assert(alignof(Storage) == 4, "Vector storage is aligned to 4 bytes");

  using StorageType = Storage;
  using ElementStorageType = typename AccessTraits<T>::StorageType;

  using value_type = T;
  using iterator = ReadIterator<T>;
  using const_iterator = iterator;

  class Constructor {
   public:
    Constructor(Allocator* allocator, BefAddr32_t address, size_t size)
        : allocator_(allocator), address_(address) {
      assert(allocator->size() >= address + sizeof(StorageType));
      size_t data_start = allocator->Allocate(size * sizeof(ElementStorageType),
                                              alignof(ElementStorageType));

      storage_.size = size;
      storage_.offset = data_start - address;
      AccessTraits<StorageType>::Construct(allocator, address, storage_);
    }

    template <typename... Args>
    auto ConstructAt(size_t index, Args&&... args) {
      assert(index < size());
      return AccessTraits<T>::Construct(allocator_, GetElementAddress(index),
                                        std::forward<Args>(args)...);
    }

    template <typename V>
    void Assign(std::initializer_list<V> ilist) {
      assert(ilist.size() == size());
      Assign(ilist.begin(), ilist.end());
    }

    template <typename Iter>
    void Assign(Iter begin, Iter end) {
      size_t i = 0;
      for (; begin != end; ++begin) {
        ConstructAt(i++, *begin);
      }
      assert(i == size());
    }

    // TODO(chky): Implement iterators for construction.

    size_t size() const { return storage_.size; }
    BefAddr32_t address() const { return address_; }

   private:
    BefAddr32_t GetElementAddress(size_t index) const {
      return address_ + storage_.offset + index * sizeof(ElementStorageType);
    }

    Allocator* allocator_;
    BefAddr32_t address_;
    Vector::Storage storage_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit Vector(const char* p) : p_(p) { assert(p_ != nullptr); }
  Vector() {
    static Storage kEmptyStorage{0, 0};
    p_ = reinterpret_cast<const char*>(&kEmptyStorage);
  }

  const char* data() const { return p_ + offset(); }

  size_t size() const { return StorageType::read_size(p_); }
  bool empty() const { return size() == 0; }

  iterator begin() const { return iterator(data()); }
  iterator end() const {
    return iterator(data() + size() * sizeof(ElementStorageType));
  }

  T operator[](size_t index) const {
    assert(index < size());
    auto iter = begin();
    iter += index;
    return *iter;
  }

 private:
  uint32_t offset() const { return StorageType::read_offset(p_); }

  const char* p_;
};

}  // namespace bef
}  // namespace tfrt

#endif  // TFRT_BEF_BEF_H_
