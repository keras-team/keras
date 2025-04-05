// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"

namespace ov {
template <element::Type_t>
struct element_type_traits {};

template <element::Type_t Type>
using fundamental_type_for = typename element_type_traits<Type>::value_type;

template <>
struct element_type_traits<element::Type_t::boolean> {
    using value_type = char;
};

template <>
struct element_type_traits<element::Type_t::bf16> {
    using value_type = bfloat16;
};

template <>
struct element_type_traits<element::Type_t::f16> {
    using value_type = float16;
};

template <>
struct element_type_traits<element::Type_t::f32> {
    using value_type = float;
};

template <>
struct element_type_traits<element::Type_t::f64> {
    using value_type = double;
};

template <>
struct element_type_traits<element::Type_t::i4> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::i8> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::i16> {
    using value_type = int16_t;
};

template <>
struct element_type_traits<element::Type_t::i32> {
    using value_type = int32_t;
};

template <>
struct element_type_traits<element::Type_t::i64> {
    using value_type = int64_t;
};

template <>
struct element_type_traits<element::Type_t::u1> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::u2> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::u3> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::u4> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::u6> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::u8> {
    using value_type = uint8_t;
};

template <>
struct element_type_traits<element::Type_t::u16> {
    using value_type = uint16_t;
};

template <>
struct element_type_traits<element::Type_t::u32> {
    using value_type = uint32_t;
};

template <>
struct element_type_traits<element::Type_t::u64> {
    using value_type = uint64_t;
};

template <>
struct element_type_traits<element::Type_t::nf4> {
    using value_type = int8_t;
};

template <>
struct element_type_traits<element::Type_t::f8e4m3> {
    using value_type = ov::float8_e4m3;
};

template <>
struct element_type_traits<element::Type_t::f8e5m2> {
    using value_type = ov::float8_e5m2;
};

template <>
struct element_type_traits<element::Type_t::string> {
    using value_type = std::string;
};

template <>
struct element_type_traits<element::Type_t::f4e2m1> {
    using value_type = ov::float4_e2m1;
};

template <>
struct element_type_traits<element::Type_t::f8e8m0> {
    using value_type = ov::float8_e8m0;
};

}  // namespace ov
