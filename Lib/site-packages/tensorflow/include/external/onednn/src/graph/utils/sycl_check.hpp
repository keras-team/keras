/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GRAPH_UTILS_SYCL_CHECK_HPP
#define GRAPH_UTILS_SYCL_CHECK_HPP

#ifdef DNNL_WITH_SYCL
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#if defined(__INTEL_LLVM_COMPILER)
#if (__INTEL_LLVM_COMPILER < 20230000)
#define DNNL_USE_SYCL121_API 1
#else
#define DNNL_USE_SYCL121_API 0
#endif
#elif defined(__LIBSYCL_MAJOR_VERSION)
#if (__LIBSYCL_MAJOR_VERSION < 6)
#define DNNL_USE_SYCL121_API 1
#else
#define DNNL_USE_SYCL121_API 0
#endif
#else
#error "Unsupported compiler"
#endif
#endif

#endif
