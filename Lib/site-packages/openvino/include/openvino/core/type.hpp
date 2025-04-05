// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {

/**
 * @brief Type information for a type system without inheritance; instances have exactly one type not
 * related to any other type.
 *
 * Supports three functions, ov::is_type<Type>, ov::as_type<Type>, and ov::as_type_ptr<Type> for type-safe
 * dynamic conversions via static_cast/static_ptr_cast without using C++ RTTI.
 * Type must have a static type_info member and a virtual get_type_info() member that
 * returns a reference to its type_info member.
 * @ingroup ov_model_cpp_api
 */
struct OPENVINO_API DiscreteTypeInfo {
    const char* name;
    const char* version_id;
    // A pointer to a parent type info; used for casting and inheritance traversal, not for
    // exact type identification
    const DiscreteTypeInfo* parent;

    DiscreteTypeInfo() = default;
    DiscreteTypeInfo(const DiscreteTypeInfo&) = default;
    DiscreteTypeInfo(DiscreteTypeInfo&&) = default;
    DiscreteTypeInfo& operator=(const DiscreteTypeInfo&) = default;

    explicit constexpr DiscreteTypeInfo(const char* _name,
                                        const char* _version_id,
                                        const DiscreteTypeInfo* _parent = nullptr)
        : name(_name),
          version_id(_version_id),
          parent(_parent),
          hash_value(0) {}

    constexpr DiscreteTypeInfo(const char* _name, const DiscreteTypeInfo* _parent = nullptr)
        : name(_name),
          version_id(nullptr),
          parent(_parent),
          hash_value(0) {}

    bool is_castable(const DiscreteTypeInfo& target_type) const;

    std::string get_version() const;

    // For use as a key
    bool operator<(const DiscreteTypeInfo& b) const;
    bool operator<=(const DiscreteTypeInfo& b) const;
    bool operator>(const DiscreteTypeInfo& b) const;
    bool operator>=(const DiscreteTypeInfo& b) const;
    bool operator==(const DiscreteTypeInfo& b) const;
    bool operator!=(const DiscreteTypeInfo& b) const;

    operator std::string() const;

    size_t hash() const;
    size_t hash();

private:
    size_t hash_value;
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const DiscreteTypeInfo& info);

#if defined(__ANDROID__) || defined(ANDROID)
#    define OPENVINO_DYNAMIC_CAST
#endif

/// \brief Tests if value is a pointer/shared_ptr that can be statically cast to a
/// Type*/shared_ptr<Type>
template <typename Type, typename Value>
typename std::enable_if<
    std::is_convertible<decltype(std::declval<Value>()->get_type_info().is_castable(Type::get_type_info_static())),
                        bool>::value,
    bool>::type
is_type(Value value) {
    return value && value->get_type_info().is_castable(Type::get_type_info_static());
}

/// Casts a Value* to a Type* if it is of type Type, nullptr otherwise
template <typename Type, typename Value>
typename std::enable_if<std::is_convertible<decltype(static_cast<Type*>(std::declval<Value>())), Type*>::value,
                        Type*>::type
as_type(Value value) {
#ifdef OPENVINO_DYNAMIC_CAST
    return ov::is_type<Type>(value) ? static_cast<Type*>(value) : nullptr;
#else
    return dynamic_cast<Type*>(value);
#endif
}

namespace util {
template <typename T>
struct AsTypePtr;
/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename In>
struct AsTypePtr<std::shared_ptr<In>> {
    template <typename Type>
    static std::shared_ptr<Type> call(const std::shared_ptr<In>& value) {
        return ov::is_type<Type>(value) ? std::static_pointer_cast<Type>(value) : std::shared_ptr<Type>();
    }
};
}  // namespace util

/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename T, typename U>
auto as_type_ptr(const U& value) -> decltype(::ov::util::AsTypePtr<U>::template call<T>(value)) {
#ifdef OPENVINO_DYNAMIC_CAST
    return ::ov::util::AsTypePtr<U>::template call<T>(value);
#else
    return std::dynamic_pointer_cast<T>(value);
#endif
}
}  // namespace ov

namespace std {
template <>
struct OPENVINO_API hash<ov::DiscreteTypeInfo> {
    size_t operator()(const ov::DiscreteTypeInfo& k) const;
};
}  // namespace std
