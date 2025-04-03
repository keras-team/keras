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
#ifndef TFRT_BEF_KERNEL_H_
#define TFRT_BEF_KERNEL_H_

#include "tfrt/bef/bef.h"

namespace tfrt {
namespace bef {

// The register index of the result and the kernel ids of its users in a BEF
// kernel.
//
// Example usage:
//
// auto ctor = bef::New<bef::ResultAndUsers>(&allocator);
// ctor.set_result(100);
//
// auto users_ctor = ctor.construct_users(/*num_users=*/2);
// users_ctor.ConstructAt(0, 200);
// users_ctor.ConstructAt(1, 300);
//
// bef::ResultAndUsers view(buffer.Get(ctor.address()));
//
class ResultAndUsers {
 public:
  struct Storage {
    using Self = Storage;
    DEFINE_BEF_FIELD(uint32_t, result);
    DEFINE_BEF_FIELD(Vector<uint32_t>, users);
  };
  static_assert(std::is_standard_layout<Storage>::value, "");
  static_assert(sizeof(Storage) == 12, "");
  static_assert(alignof(Storage) == 4, "");

  using StorageType = Storage;

  class Constructor {
   public:
    explicit Constructor(Allocator* allocator, BefAddr32_t address)
        : allocator_(allocator), address_(address) {}

    void set_result(uint32_t result) {
      StorageType::construct_result(allocator_, address_, result);
    }

    Vector<uint32_t>::Constructor construct_users(size_t num_users) {
      return StorageType::construct_users(allocator_, address_, num_users);
    }

    BefAddr32_t address() const { return address_; }

   private:
    Allocator* allocator_;
    BefAddr32_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit ResultAndUsers(const char* p) : p_(p) {}

  uint32_t result() const { return Storage::read_result(p_); }
  Vector<uint32_t> users() const { return Storage::read_users(p_); }

 private:
  const char* p_;
};

// The BEF kernel implemented using BEF primitives.
//
// Example usage:
//
// bef::Kernel::Constructor ctor = bef::New<bef::Kernel>(&allocator);
//
// ctor.set_code(100);
// ctor.set_location(200);
//
// ctor.construct_arguments(/*size=*/2).Assign({400, 500});
// ctor.construct_attributes(/*size=*/1).Assign({1400});
// ctor.construct_functions(/*size=*/0);
//
// auto results_ctor = ctor.construct_results(/*size=*/2);
//
// for (uint32_t i = 0; i < 2; ++i) {
//   bef::ResultAndUsers::Constructor ru_ctor = results_ctor.ConstructAt(i);
//   ru_ctor.set_result(i);
//   ru_ctor.construct_users(/*size=*/2).Assign({100 + i, 200 + i});
// }
//
// bef::Kernel view(buffer.Get(ctor.address()));
//
class Kernel {
 public:
  struct Storage {
    using Self = Storage;
    DEFINE_BEF_FIELD(uint32_t, code);
    DEFINE_BEF_FIELD(uint32_t, location);
    DEFINE_BEF_FIELD(Vector<uint32_t>, arguments);
    DEFINE_BEF_FIELD(Vector<uint32_t>, attributes);
    DEFINE_BEF_FIELD(Vector<uint32_t>, functions);
    DEFINE_BEF_FIELD(Vector<ResultAndUsers>, results);
  };

  using StorageType = Storage;

  class Constructor {
   public:
    Constructor(Allocator* allocator, BefAddr32_t address)
        : allocator_(allocator), address_(address) {}

    void set_code(uint32_t value) {
      StorageType::construct_code(allocator_, address_, value);
    }
    void set_location(uint32_t value) {
      StorageType::construct_location(allocator_, address_, value);
    }

    template <typename... Args>
    auto construct_arguments(Args&&... args) {
      return StorageType::construct_arguments(allocator_, address_,
                                              std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_attributes(Args&&... args) {
      return StorageType::construct_attributes(allocator_, address_,
                                               std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_functions(Args&&... args) {
      return StorageType::construct_functions(allocator_, address_,
                                              std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_results(Args&&... args) {
      return StorageType::construct_results(allocator_, address_,
                                            std::forward<Args>(args)...);
    }

    BefAddr32_t address() const { return address_; }

   private:
    Allocator* allocator_;
    BefAddr32_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit Kernel(const char* p) : p_(p) {}

  uint32_t code() const { return Storage::read_code(p_); }
  uint32_t location() const { return Storage::read_location(p_); }
  Vector<uint32_t> arguments() const { return Storage::read_arguments(p_); }
  Vector<uint32_t> attributes() const { return Storage::read_attributes(p_); }
  Vector<uint32_t> functions() const { return Storage::read_functions(p_); }
  Vector<ResultAndUsers> results() const { return Storage::read_results(p_); }

 private:
  const char* p_;
};

}  // namespace bef
}  // namespace tfrt

#endif  // TFRT_BEF_KERNEL_H_
