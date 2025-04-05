// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#define ONNX_UNUSED_PARAMETER(x) (void)(x)

#ifdef ONNX_NO_EXCEPTIONS
#include <iostream>
#define ONNX_THROW(...)                                   \
  do {                                                    \
    std::cerr << ONNX_NAMESPACE::MakeString(__VA_ARGS__); \
    abort();                                              \
  } while (false)

#define ONNX_THROW_EX(ex)                \
  do {                                   \
    std::cerr << ex.what() << std::endl; \
    abort();                             \
  } while (false)

#define ONNX_TRY if (true)
#define ONNX_CATCH(x) else if (false)
#define ONNX_HANDLE_EXCEPTION(func)

#else
#define ONNX_THROW(...) throw std::runtime_error(ONNX_NAMESPACE::MakeString(__VA_ARGS__))
#define ONNX_THROW_EX(ex) throw ex

#define ONNX_TRY try
#define ONNX_CATCH(x) catch (x)
#define ONNX_HANDLE_EXCEPTION(func) func()
#endif

// Macros to disable the copy and/or assignment methods
// These are usually placed in the private: declarations for a class.

#define ONNX_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define ONNX_DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define ONNX_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  ONNX_DISALLOW_COPY(TypeName);                     \
  ONNX_DISALLOW_ASSIGNMENT(TypeName)

#define ONNX_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;     \
  TypeName& operator=(TypeName&&) = delete

#define ONNX_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  ONNX_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  ONNX_DISALLOW_MOVE(TypeName)
