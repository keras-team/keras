/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_CPP_COMPAT_HPP
#define COMMON_CPP_COMPAT_HPP

#include <exception>
#include <type_traits>

namespace dnnl {
namespace impl {
namespace cpp_compat {

// oneDNN relies on C++11 standard. However, for DPCPP runtime the standard we
// use to build oneDNN must be C++17 per requirements. Some C++11 features have
// been deprecated in C++17, which triggers deprecations warnings. This file
// contains a compatibility layer for such C++ features.

// Older than C++17.
#if defined(__cplusplus) && __cplusplus < 201703L
inline int uncaught_exceptions() {
    return (int)std::uncaught_exception();
}

template <class F, class... ArgTypes>
using invoke_result = std::result_of<F(ArgTypes...)>;
#else

inline int uncaught_exceptions() {
    return std::uncaught_exceptions();
}

template <class F, class... ArgTypes>
using invoke_result = std::invoke_result<F, ArgTypes...>;

#endif
} // namespace cpp_compat
} // namespace impl
} // namespace dnnl

#endif
