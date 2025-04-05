//===- Msan.h - Utils related to the memory sanitizer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares and defines macros related to msan.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_MSAN_H
#define MLIR_EXECUTIONENGINE_MSAN_H

// Memory sanitizer currently can't be enabled for the jit-compiled code, and
// to suppress msan warnings we need to unpoison pointers and pointed-to
// datastructures before they can be accessed.

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(memory_sanitizer) && !defined(MLIR_MEMORY_SANITIZER)
#define MLIR_MEMORY_SANITIZER
#endif

#if defined(MLIR_MEMORY_SANITIZER)
#include <sanitizer/msan_interface.h>
#define MLIR_MSAN_MEMORY_IS_INITIALIZED(p, s) __msan_unpoison((p), (s))
#else // Memory sanitizer: OFF
#define MLIR_MSAN_MEMORY_IS_INITIALIZED(p, s)
#endif // MLIR_MEMORY_SANITIZER

#endif // MLIR_EXECUTIONENGINE_MSAN_H
