//===--- Float16bits.h - supports 2-byte floats ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements f16 and bf16 to support the compilation and execution
// of programs using these types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_FLOAT16BITS_H_
#define MLIR_EXECUTIONENGINE_FLOAT16BITS_H_

#include <cstdint>
#include <iostream>

#ifdef _WIN32
#ifdef mlir_float16_utils_EXPORTS // We are building this library
#define MLIR_FLOAT16_EXPORT __declspec(dllexport)
#define MLIR_FLOAT16_DEFINE_FUNCTIONS
#else // We are using this library
#define MLIR_FLOAT16_EXPORT __declspec(dllimport)
#endif // mlir_float16_utils_EXPORTS
#else  // Non-windows: use visibility attributes.
#define MLIR_FLOAT16_EXPORT __attribute__((visibility("default")))
#define MLIR_FLOAT16_DEFINE_FUNCTIONS
#endif // _WIN32

// Implements half precision and bfloat with f16 and bf16, using the MLIR type
// names. These data types are also used for c-interface runtime routines.
extern "C" {
struct MLIR_FLOAT16_EXPORT f16 {
  f16(float f = 0);
  uint16_t bits;
};

struct MLIR_FLOAT16_EXPORT bf16 {
  bf16(float f = 0);
  uint16_t bits;
};
}

// Outputs a half precision value.
MLIR_FLOAT16_EXPORT std::ostream &operator<<(std::ostream &os, const f16 &f);
// Outputs a bfloat value.
MLIR_FLOAT16_EXPORT std::ostream &operator<<(std::ostream &os, const bf16 &d);

MLIR_FLOAT16_EXPORT bool operator==(const f16 &f1, const f16 &f2);
MLIR_FLOAT16_EXPORT bool operator==(const bf16 &bf1, const bf16 &bf2);

extern "C" MLIR_FLOAT16_EXPORT void printF16(uint16_t bits);
extern "C" MLIR_FLOAT16_EXPORT void printBF16(uint16_t bits);

#undef MLIR_FLOAT16_EXPORT
#endif // MLIR_EXECUTIONENGINE_FLOAT16BITS_H_
