/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===- msan.h ---------------------------------------------------*- C++ -*-===//
//
// This file declares and defines macros related to msan.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_SUPPORT_MSAN_H_
#define TFRT_SUPPORT_MSAN_H_

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(memory_sanitizer) && !defined(MEMORY_SANITIZER)
#define MEMORY_SANITIZER
#endif

#if defined(MEMORY_SANITIZER)
#include <sanitizer/msan_interface.h>  // NOLINT

// Marks a memory range as uninitialized, as if it was allocated here.
#define TFRT_MSAN_ALLOCATED_UNINITIALIZED_MEMORY(p, s) \
  __msan_allocated_memory((p), (s))
// Marks a memory range as initialized.
#define TFRT_MSAN_MEMORY_IS_INITIALIZED(p, s) __msan_unpoison((p), (s))

#else

#define TFRT_MSAN_ALLOCATED_UNINITIALIZED_MEMORY(p, s)
#define TFRT_MSAN_MEMORY_IS_INITIALIZED(p, s)
#endif

#endif  // TFRT_SUPPORT_MSAN_H_
