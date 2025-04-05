//===- in_place.h - Tag types for in place construction ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file provides tag types for in-place constructions. It is adapted from
//  absl/utility/utility.h to be used for LLVM and TFRT support libraries.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LLVM_DERIVED_IN_PLACE_H_
#define TFRT_LLVM_DERIVED_IN_PLACE_H_

namespace tfrt {

//===----------------------------------------------------------------------===//
// Tag types such as in_place_t, in_place_type_t.
//===----------------------------------------------------------------------===//

namespace internal {
template <typename T>
struct InPlaceTypeTag {
  InPlaceTypeTag() = delete;
  InPlaceTypeTag(const InPlaceTypeTag&) = delete;
  InPlaceTypeTag& operator=(const InPlaceTypeTag&) = delete;
};
}  // namespace internal

struct in_place_t {};

// pre-C++17, inline variables are not supported
// The following trick is adapted from ABSL_INTERNAL_INLINE_CONSTEXPR.
// ABSL_INTERNAL_INLINE_CONSTEXPR(var_type=in_place_t, name=in_place, init={})
// inline const in_place_t in_place = {};
template <class /*InternalDummy*/ = void>
struct InternalInlineVariableHolderName {
  static constexpr in_place_t kInstance = {};
};

template <class InternalDummy>
constexpr in_place_t InternalInlineVariableHolderName<InternalDummy>::kInstance;

static constexpr const in_place_t& in_place =
    InternalInlineVariableHolderName<>::kInstance;
static_assert(sizeof(void (*)(decltype(in_place))) != 0,
              "Silence unused variable warning.");

template <typename T>
using in_place_type_t = void (*)(internal::InPlaceTypeTag<T>);

template <typename T>
void in_place_type(internal::InPlaceTypeTag<T>) {}

}  // namespace tfrt

#endif  // TFRT_LLVM_DERIVED_IN_PLACE_H_
