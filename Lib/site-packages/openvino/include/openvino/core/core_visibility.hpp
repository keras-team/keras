// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

// Now we use the generic helper definitions above to define OPENVINO_API
// OPENVINO_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

/**
 * @defgroup ov_cpp_api OpenVINO Runtime C++ API
 * OpenVINO Runtime C++ API
 *
 * @defgroup ov_model_cpp_api Basics
 * @ingroup ov_cpp_api
 * OpenVINO Core C++ API to work with ov::Model, dynamic and static shapes, types
 *
 * @defgroup ov_ops_cpp_api Operations
 * @ingroup ov_cpp_api
 * OpenVINO C++ API to create operations from different opsets. Such API is used to
 * creation models from code, write transformations and traverse the model graph
 *
 * @defgroup ov_opset_cpp_api Operation sets
 * @ingroup ov_cpp_api
 * OpenVINO C++ API to work with operation sets
 *
 * @defgroup ov_pass_cpp_api Transformation passes
 * @ingroup ov_cpp_api
 * OpenVINO C++ API to work with OpenVINO transformations
 *
 * @defgroup ov_runtime_cpp_api Inference
 * @ingroup ov_cpp_api
 * OpenVINO Inference C++ API provides ov::Core, ov::CompiledModel, ov::InferRequest
 * and ov::Tensor classes
 */

/**
 * @brief OpenVINO C++ API
 * @ingroup ov_cpp_api
 */
namespace ov {}  // namespace ov

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#ifdef OPENVINO_STATIC_LIBRARY  // defined if we are building or calling OpenVINO as a static library
#    define OPENVINO_API
#    define OPENVINO_API_C(...) __VA_ARGS__
#else
#    ifdef IMPLEMENT_OPENVINO_API  // defined if we are building the OpenVINO DLL (instead of using it)
#        define OPENVINO_API        OPENVINO_CORE_EXPORTS
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS __VA_ARGS__ OPENVINO_CDECL
#    else
#        define OPENVINO_API        OPENVINO_CORE_IMPORTS
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS __VA_ARGS__ OPENVINO_CDECL
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY
