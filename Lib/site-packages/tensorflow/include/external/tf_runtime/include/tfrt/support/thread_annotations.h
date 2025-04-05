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

//===- thread_annotations.h -------------------------------------*- C++ -*-===//
//
// Defines thread safety annotations on platforms that support them. Copied from
// http://clang.llvm.org/docs/ThreadSafetyAnalysis.html#mutex-h
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_THREAD_ANNOTATIONS_H_
#define TFRT_HOST_CONTEXT_THREAD_ANNOTATIONS_H_

// Enable thread safety attributes only with clang.
// The attributes can be safely erased when compiling with other compilers.
#if defined(__clang__) && (!defined(SWIG))
#define TFRT_THREAD_ANNOTATION_ATTRIBUTE__(x) __attribute__((x))
#else
#define TFRT_THREAD_ANNOTATION_ATTRIBUTE__(x)  // no-op
#endif

#define TFRT_CAPABILITY(x) TFRT_THREAD_ANNOTATION_ATTRIBUTE__(capability(x))

#define TFRT_SCOPED_CAPABILITY \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

#define TFRT_GUARDED_BY(x) TFRT_THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))

#define TFRT_PT_GUARDED_BY(x) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_by(x))

#define TFRT_ACQUIRED_BEFORE(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(acquired_before(__VA_ARGS__))

#define TFRT_ACQUIRED_AFTER(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(acquired_after(__VA_ARGS__))

#define TFRT_REQUIRES(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(requires_capability(__VA_ARGS__))

#define TFRT_REQUIRES_SHARED(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))

#define TFRT_ACQUIRE(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(acquire_capability(__VA_ARGS__))

#define TFRT_ACQUIRE_SHARED(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

#define TFRT_RELEASE(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(release_capability(__VA_ARGS__))

#define TFRT_RELEASE_SHARED(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(release_shared_capability(__VA_ARGS__))

#define TFRT_TRY_ACQUIRE(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_capability(__VA_ARGS__))

#define TFRT_TRY_ACQUIRE_SHARED(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_shared_capability(__VA_ARGS__))

#define TFRT_EXCLUDES(...) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

#define TFRT_ASSERT_CAPABILITY(x) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(assert_capability(x))

#define TFRT_ASSERT_SHARED_CAPABILITY(x) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_capability(x))

#define TFRT_RETURN_CAPABILITY(x) \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

#define TFRT_NO_THREAD_SAFETY_ANALYSIS \
  TFRT_THREAD_ANNOTATION_ATTRIBUTE__(no_thread_safety_analysis)

#endif  // TFRT_HOST_CONTEXT_THREAD_ANNOTATIONS_H_
