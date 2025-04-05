/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef ONEAPI_DNNL_DNNL_CONFIG_H
#define ONEAPI_DNNL_DNNL_CONFIG_H

/// @cond DO_NOT_DOCUMENT_THIS

// All symbols shall be internal unless marked as DNNL_API
#if defined _WIN32 || defined __CYGWIN__
#define DNNL_HELPER_DLL_IMPORT __declspec(dllimport)
#define DNNL_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define DNNL_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define DNNL_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#else
#define DNNL_HELPER_DLL_IMPORT
#define DNNL_HELPER_DLL_EXPORT
#endif
#endif

#ifdef DNNL_DLL
#ifdef DNNL_DLL_EXPORTS
#define DNNL_API DNNL_HELPER_DLL_EXPORT
#else
#define DNNL_API DNNL_HELPER_DLL_IMPORT
#endif
#else
#define DNNL_API
#endif

#if defined(__GNUC__)
#define DNNL_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DNNL_DEPRECATED __declspec(deprecated)
#else
#define DNNL_DEPRECATED
#endif

/// @endcond

// clang-format off

/// @addtogroup dnnl_api_service
/// @{

/// No runtime (disabled)
#define DNNL_RUNTIME_NONE 0u

/// Sequential runtime (CPU only)
#define DNNL_RUNTIME_SEQ 1u

/// OpenMP runtime (CPU only)
#define DNNL_RUNTIME_OMP 2u

/// TBB runtime (CPU only)
#define DNNL_RUNTIME_TBB 4u

/// Threadpool runtime (CPU only)
#define DNNL_RUNTIME_THREADPOOL 8u

/// OpenCL runtime
#define DNNL_RUNTIME_OCL 256u

/// SYCL runtime
#define DNNL_RUNTIME_SYCL 512u

/// DPC++ runtime
#define DNNL_RUNTIME_DPCPP DNNL_RUNTIME_SYCL

/// @} dnnl_api_service

// oneDNN CPU threading runtime
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL

// oneDNN CPU engine runtime
#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL

// oneDNN GPU engine runtime
#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE

// clang-format on

#if defined(DNNL_CPU_RUNTIME) && defined(DNNL_GPU_RUNTIME)
#if (DNNL_CPU_RUNTIME == DNNL_RUNTIME_OCL)
#error "Unexpected DNNL_CPU_RUNTIME"
#endif
#if (DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE) \
        && (DNNL_GPU_RUNTIME != DNNL_RUNTIME_OCL) \
        && (DNNL_GPU_RUNTIME != DNNL_RUNTIME_SYCL)
#error "Unexpected DNNL_GPU_RUNTIME"
#endif
#if (DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE \
        && DNNL_GPU_RUNTIME == DNNL_RUNTIME_NONE)
#error "At least one runtime must be specified"
#endif
#else
#error "BOTH DNNL_CPU_RUNTIME and DNNL_GPU_RUNTIME must be defined"
#endif

// For SYCL CPU, a primitive may be created and executed in different threads
// hence the global scratchpad does not work. This enables concurrent execution
// when CPU runtime is SYCL to avoid the issue.
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
#ifndef DNNL_ENABLE_CONCURRENT_EXEC
#define DNNL_ENABLE_CONCURRENT_EXEC
#endif
#endif

// When defined, primitive cache stores runtime objects.
#undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE

// When defined, DPCPP is supported.
#undef DNNL_WITH_SYCL

// When defined, Level Zero is supported.
#undef DNNL_WITH_LEVEL_ZERO

// When defined, SYCL CUDA backend is used.
#undef DNNL_SYCL_CUDA

// When defined, SYCL HIP backend is used.
#undef DNNL_SYCL_HIP

// When defined, stack checker is enabled.
#undef DNNL_ENABLE_STACK_CHECKER

// When defined, experimental features are enabled.
#undef DNNL_EXPERIMENTAL

// When defined, experimental functionality for sparse domain is enabled.
#define DNNL_EXPERIMENTAL_SPARSE

// When defined, experimental functionality for ukernels is enabled.
#undef DNNL_EXPERIMENTAL_UKERNEL

// When defined, graph component is enabled.
#define ONEDNN_BUILD_GRAPH

// When defined, experimental profiling capabilities are enabled.
#undef DNNL_EXPERIMENTAL_PROFILING

// List of configurating build controls
// Workload controls
#define BUILD_TRAINING 1
#define BUILD_INFERENCE 0
// Primitive controls
#define BUILD_PRIMITIVE_ALL 1
#define BUILD_BATCH_NORMALIZATION 0
#define BUILD_BINARY 0
#define BUILD_CONCAT 0
#define BUILD_CONVOLUTION 0
#define BUILD_DECONVOLUTION 0
#define BUILD_ELTWISE 0
#define BUILD_GROUP_NORMALIZATION 1
#define BUILD_INNER_PRODUCT 0
#define BUILD_LAYER_NORMALIZATION 0
#define BUILD_LRN 0
#define BUILD_MATMUL 0
#define BUILD_POOLING 0
#define BUILD_PRELU 0
#define BUILD_REDUCTION 0
#define BUILD_REORDER 0
#define BUILD_RESAMPLING 0
#define BUILD_RNN 0
#define BUILD_SHUFFLE 0
#define BUILD_SOFTMAX 0
#define BUILD_SUM 0
// Primitives CPU ISA controls
#define BUILD_PRIMITIVE_CPU_ISA_ALL 1
#define BUILD_SSE41 0
#define BUILD_AVX2 0
#define BUILD_AVX512 0
#define BUILD_AMX 0
// Primitives GPU ISA controls
#define BUILD_PRIMITIVE_GPU_ISA_ALL 0
#define BUILD_GEN9 0
#define BUILD_GEN11 0
#define BUILD_XELP 0
#define BUILD_XEHP 0
#define BUILD_XEHPG 0
#define BUILD_XEHPC 0
#define BUILD_XE2 0
// GeMM kernels ISA controls
#define BUILD_GEMM_KERNELS_ALL 1
#define BUILD_GEMM_KERNELS_NONE 0
#define BUILD_GEMM_SSE41 1
#define BUILD_GEMM_AVX2 1
#define BUILD_GEMM_AVX512 1
#endif
