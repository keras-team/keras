// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a common header file for the C API
 *
 * @file ov_common.h
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined(__INTEL_COMPILER) || defined(_MSC_VER) || defined(__GNUC__)
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2))
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <wchar.h>
#endif

#ifdef __cplusplus
#    define OPENVINO_C_API_EXTERN extern "C"
#else
#    define OPENVINO_C_API_EXTERN
#endif

#if defined(OPENVINO_STATIC_LIBRARY) || defined(__GNUC__) && (__GNUC__ < 4)
#    define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __VA_ARGS__
#    define OPENVINO_C_VAR(...) OPENVINO_C_API_EXTERN __VA_ARGS__
#    define OV_NODISCARD
#else
#    if defined(_WIN32) || defined(__CYGWIN__)
#        define OPENVINO_C_API_CALLBACK __cdecl
#        ifdef openvino_c_EXPORTS
#            define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __declspec(dllexport) __VA_ARGS__ __cdecl
#            define OPENVINO_C_VAR(...) OPENVINO_C_API_EXTERN __declspec(dllexport) __VA_ARGS__
#        else
#            define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __declspec(dllimport) __VA_ARGS__ __cdecl
#            define OPENVINO_C_VAR(...) OPENVINO_C_API_EXTERN __declspec(dllimport) __VA_ARGS__
#        endif
#        define OV_NODISCARD
#    else
#        define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __attribute__((visibility("default"))) __VA_ARGS__
#        define OPENVINO_C_VAR(...) OPENVINO_C_API_EXTERN __attribute__((visibility("default"))) __VA_ARGS__
#        define OV_NODISCARD        __attribute__((warn_unused_result))
#    endif
#endif

#ifndef OPENVINO_C_API_CALLBACK
#    define OPENVINO_C_API_CALLBACK
#endif

/**
 * @defgroup ov_c_api OpenVINO Runtime C API
 * OpenVINO Runtime C API
 *
 * @defgroup ov_base_c_api Basics
 * @ingroup ov_c_api
 * @brief The basic definitions & interfaces of OpenVINO C API to work with other components
 *
 * @defgroup ov_compiled_model_c_api Compiled Model
 * @ingroup ov_c_api
 * @brief The operations about compiled model
 *
 * @defgroup ov_core_c_api Core
 * @ingroup ov_c_api
 * @brief The definitions & operations about core
 *
 * @defgroup ov_dimension_c_api Dimension
 * @ingroup ov_c_api
 * @brief The definitions & operations about dimension
 *
 * @defgroup ov_infer_request_c_api Infer Request
 * @ingroup ov_c_api
 * @brief The definitions & operations about infer request
 *
 * @defgroup ov_layout_c_api Layout
 * @ingroup ov_c_api
 * @brief The definitions & operations about layout
 *
 * @defgroup ov_model_c_api Model
 * @ingroup ov_c_api
 * @brief The definitions & operations about model
 *
 * @defgroup ov_node_c_api Node
 * @ingroup ov_c_api
 * @brief The definitions & operations about node
 *
 * @defgroup ov_partial_shape_c_api Partial Shape
 * @ingroup ov_c_api
 * @brief The definitions & operations about partial shape
 *
 * @defgroup ov_prepostprocess_c_api Pre Post Process
 * @ingroup ov_c_api
 * @brief The definitions & operations about prepostprocess
 *
 * @defgroup ov_property_c_api Property
 * @ingroup ov_c_api
 * @brief The definitions & operations about property
 *
 * @defgroup ov_rank_c_api Rank
 * @ingroup ov_c_api
 * @brief The definitions & operations about rank
 *
 * @defgroup ov_shape_c_api Shape
 * @ingroup ov_c_api
 * @brief The definitions & operations about shape
 *
 * @defgroup ov_tensor_c_api Tensor
 * @ingroup ov_c_api
 * @brief The definitions & operations about tensor
 *
 * @defgroup ov_remote_context_c_api Remote Context
 * @ingroup ov_c_api
 * @brief Set of functions representing of RemoteContext
 */

/**
 * @enum ov_status_e
 * @ingroup ov_base_c_api
 * @brief This enum contains codes for all possible return values of the interface functions
 */
typedef enum {
    OK = 0,  //!< SUCCESS
    /*
     * @brief map exception to C++ interface
     */
    GENERAL_ERROR = -1,       //!< GENERAL_ERROR
    NOT_IMPLEMENTED = -2,     //!< NOT_IMPLEMENTED
    NETWORK_NOT_LOADED = -3,  //!< NETWORK_NOT_LOADED
    PARAMETER_MISMATCH = -4,  //!< PARAMETER_MISMATCH
    NOT_FOUND = -5,           //!< NOT_FOUND
    OUT_OF_BOUNDS = -6,       //!< OUT_OF_BOUNDS
    /*
     * @brief exception not of std::exception derived type was thrown
     */
    UNEXPECTED = -7,          //!< UNEXPECTED
    REQUEST_BUSY = -8,        //!< REQUEST_BUSY
    RESULT_NOT_READY = -9,    //!< RESULT_NOT_READY
    NOT_ALLOCATED = -10,      //!< NOT_ALLOCATED
    INFER_NOT_STARTED = -11,  //!< INFER_NOT_STARTED
    NETWORK_NOT_READ = -12,   //!< NETWORK_NOT_READ
    INFER_CANCELLED = -13,    //!< INFER_CANCELLED
    /*
     * @brief exception in C wrapper
     */
    INVALID_C_PARAM = -14,         //!< INVALID_C_PARAM
    UNKNOWN_C_ERROR = -15,         //!< UNKNOWN_C_ERROR
    NOT_IMPLEMENT_C_METHOD = -16,  //!< NOT_IMPLEMENT_C_METHOD
    UNKNOW_EXCEPTION = -17,        //!< UNKNOW_EXCEPTION
} ov_status_e;

/**
 * @enum ov_element_type_e
 * @ingroup ov_base_c_api
 * @brief This enum contains codes for element type, which is aligned with ov::element::Type_t in
 * src/core/include/openvino/core/type/element_type.hpp
 */
typedef enum {
    UNDEFINED = 0U,  //!< Undefined element type
    DYNAMIC,         //!< Dynamic element type
    BOOLEAN,         //!< boolean element type
    BF16,            //!< bf16 element type
    F16,             //!< f16 element type
    F32,             //!< f32 element type
    F64,             //!< f64 element type
    I4,              //!< i4 element type
    I8,              //!< i8 element type
    I16,             //!< i16 element type
    I32,             //!< i32 element type
    I64,             //!< i64 element type
    U1,              //!< binary element type
    U2,              //!< u2 element type
    U3,              //!< u3 element type
    U4,              //!< u4 element type
    U6,              //!< u6 element type
    U8,              //!< u8 element type
    U16,             //!< u16 element type
    U32,             //!< u32 element type
    U64,             //!< u64 element type
    NF4,             //!< nf4 element type
    F8E4M3,          //!< f8e4m3 element type
    F8E5M3,          //!< f8e5m2 element type
    STRING,          //!< string element type
    F4E2M1,          //!< f4e2m1 element type
    F8E8M0,          //!< f8e8m0 element type
} ov_element_type_e;

/**
 * @brief encryption_func is a function pointer that encrypt or decrypt the input memory, example of this function is
 * codec(const char* input, const size_t in_size, const char* output, size_t* out_size)
 * This function needs to be called twice,
 * the first call to obtain out_size (the size of output buffer), the second call to obtain output buffer.
 * The first call output is nullptr, before the second call, the caller needs to apply for output
 * memory based on the out_size returned by the first call.
 * the memory of parameter output is allocated and released by the caller.
 * @param input The pointer to the input buffer.
 * @param in_size The size of input.
 * @param output The pointer to the encrypted/decrypted buffer.
 * @param out_size The size of output.
 */
typedef void (*encryption_func)(const char*, const size_t, char*, size_t*);
typedef struct {
    encryption_func encrypt_func;  // encryption function pointer
    encryption_func decrypt_func;  // decryption function pointer
} ov_encryption_callbacks;

/**
 * @brief Print the error info.
 * @ingroup ov_base_c_api
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(const char*)
ov_get_error_info(ov_status_e status);

/**
 * @brief free char
 * @ingroup ov_base_c_api
 * @param content The pointer to the char to free.
 */
OPENVINO_C_API(void)
ov_free(const char* content);

/**
 * @brief Get the last error msg.
 * @ingroup ov_base_c_api
 */
OPENVINO_C_API(const char*)
ov_get_last_err_msg();
