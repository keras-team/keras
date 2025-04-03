// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file defines dynamic annotations for use with dynamic analysis tool
// such as valgrind, PIN, etc.
//
// Dynamic annotation is a source code annotation that affects the generated
// code (that is, the annotation is not a comment). Each such annotation is
// attached to a particular instruction and/or to a particular object (address)
// in the program.
//
// The annotations that should be used by users are macros in all upper-case
// (e.g., ANNOTATE_THREAD_NAME).
//
// Actual implementation of these macros may differ depending on the dynamic
// analysis tool being used.
//
// This file supports the following configurations:
// - Dynamic Annotations enabled (with static thread-safety warnings disabled).
//   In this case, macros expand to functions implemented by Thread Sanitizer,
//   when building with TSan. When not provided an external implementation,
//   dynamic_annotations.cc provides no-op implementations.
//
// - Static Clang thread-safety warnings enabled.
//   When building with a Clang compiler that supports thread-safety warnings,
//   a subset of annotations can be statically-checked at compile-time. We
//   expand these macros to static-inline functions that can be analyzed for
//   thread-safety, but afterwards elided when building the final binary.
//
// - All annotations are disabled.
//   If neither Dynamic Annotations nor Clang thread-safety warnings are
//   enabled, then all annotation-macros expand to empty.

#ifndef ABSL_BASE_INTERNAL_DYNAMIC_ANNOTATIONS_H_
#define ABSL_BASE_INTERNAL_DYNAMIC_ANNOTATIONS_H_

#include <stddef.h>

#include "absl/base/config.h"

// -------------------------------------------------------------------------
// Decide which features are enabled

#ifndef DYNAMIC_ANNOTATIONS_ENABLED
#define DYNAMIC_ANNOTATIONS_ENABLED 0
#endif

#if defined(__clang__) && !defined(SWIG)
#define ABSL_INTERNAL_IGNORE_READS_ATTRIBUTE_ENABLED 1
#endif

#if DYNAMIC_ANNOTATIONS_ENABLED != 0

#define ABSL_INTERNAL_RACE_ANNOTATIONS_ENABLED 1
#define ABSL_INTERNAL_READS_ANNOTATIONS_ENABLED 1
#define ABSL_INTERNAL_WRITES_ANNOTATIONS_ENABLED 1
#define ABSL_INTERNAL_ANNOTALYSIS_ENABLED 0
#define ABSL_INTERNAL_READS_WRITES_ANNOTATIONS_ENABLED 1

#else

#define ABSL_INTERNAL_RACE_ANNOTATIONS_ENABLED 0
#define ABSL_INTERNAL_READS_ANNOTATIONS_ENABLED 0
#define ABSL_INTERNAL_WRITES_ANNOTATIONS_ENABLED 0

// Clang provides limited support for static thread-safety analysis through a
// feature called Annotalysis. We configure macro-definitions according to
// whether Annotalysis support is available. When running in opt-mode, GCC
// will issue a warning, if these attributes are compiled. Only include them
// when compiling using Clang.

// ANNOTALYSIS_ENABLED == 1 when IGNORE_READ_ATTRIBUTE_ENABLED == 1
#define ABSL_INTERNAL_ANNOTALYSIS_ENABLED \
  defined(ABSL_INTERNAL_IGNORE_READS_ATTRIBUTE_ENABLED)
// Read/write annotations are enabled in Annotalysis mode; disabled otherwise.
#define ABSL_INTERNAL_READS_WRITES_ANNOTATIONS_ENABLED \
  ABSL_INTERNAL_ANNOTALYSIS_ENABLED
#endif

// Memory annotations are also made available to LLVM's Memory Sanitizer
#if defined(ABSL_HAVE_MEMORY_SANITIZER) && !defined(__native_client__)
#define ABSL_INTERNAL_MEMORY_ANNOTATIONS_ENABLED 1
#endif

#ifndef ABSL_INTERNAL_MEMORY_ANNOTATIONS_ENABLED
#define ABSL_INTERNAL_MEMORY_ANNOTATIONS_ENABLED 0
#endif

#ifdef __cplusplus
#define ABSL_INTERNAL_BEGIN_EXTERN_C extern "C" {
#define ABSL_INTERNAL_END_EXTERN_C }  // extern "C"
#define ABSL_INTERNAL_GLOBAL_SCOPED(F) ::F
#define ABSL_INTERNAL_STATIC_INLINE inline
#else
#define ABSL_INTERNAL_BEGIN_EXTERN_C  // empty
#define ABSL_INTERNAL_END_EXTERN_C    // empty
#define ABSL_INTERNAL_GLOBAL_SCOPED(F) F
#define ABSL_INTERNAL_STATIC_INLINE static inline
#endif

// -------------------------------------------------------------------------
// Define race annotations.

#if ABSL_INTERNAL_RACE_ANNOTATIONS_ENABLED == 1

// -------------------------------------------------------------
// Annotations that suppress errors. It is usually better to express the
// program's synchronization using the other annotations, but these can be used
// when all else fails.

// Report that we may have a benign race at `pointer`, with size
// "sizeof(*(pointer))". `pointer` must be a non-void* pointer. Insert at the
// point where `pointer` has been allocated, preferably close to the point
// where the race happens. See also ANNOTATE_BENIGN_RACE_STATIC.
#define ANNOTATE_BENIGN_RACE(pointer, description)     \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateBenignRaceSized) \
  (__FILE__, __LINE__, pointer, sizeof(*(pointer)), description)

// Same as ANNOTATE_BENIGN_RACE(`address`, `description`), but applies to
// the memory range [`address`, `address`+`size`).
#define ANNOTATE_BENIGN_RACE_SIZED(address, size, description) \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateBenignRaceSized)         \
  (__FILE__, __LINE__, address, size, description)

// Enable (`enable`!=0) or disable (`enable`==0) race detection for all threads.
// This annotation could be useful if you want to skip expensive race analysis
// during some period of program execution, e.g. during initialization.
#define ANNOTATE_ENABLE_RACE_DETECTION(enable)             \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateEnableRaceDetection) \
  (__FILE__, __LINE__, enable)

// -------------------------------------------------------------
// Annotations useful for debugging.

// Report the current thread `name` to a race detector.
#define ANNOTATE_THREAD_NAME(name) \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateThreadName)(__FILE__, __LINE__, name)

// -------------------------------------------------------------
// Annotations useful when implementing locks. They are not normally needed by
// modules that merely use locks. The `lock` argument is a pointer to the lock
// object.

// Report that a lock has been created at address `lock`.
#define ANNOTATE_RWLOCK_CREATE(lock) \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateRWLockCreate)(__FILE__, __LINE__, lock)

// Report that a linker initialized lock has been created at address `lock`.
#ifdef ABSL_HAVE_THREAD_SANITIZER
#define ANNOTATE_RWLOCK_CREATE_STATIC(lock)               \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateRWLockCreateStatic) \
  (__FILE__, __LINE__, lock)
#else
#define ANNOTATE_RWLOCK_CREATE_STATIC(lock) ANNOTATE_RWLOCK_CREATE(lock)
#endif

// Report that the lock at address `lock` is about to be destroyed.
#define ANNOTATE_RWLOCK_DESTROY(lock) \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateRWLockDestroy)(__FILE__, __LINE__, lock)

// Report that the lock at address `lock` has been acquired.
// `is_w`=1 for writer lock, `is_w`=0 for reader lock.
#define ANNOTATE_RWLOCK_ACQUIRED(lock, is_w)          \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateRWLockAcquired) \
  (__FILE__, __LINE__, lock, is_w)

// Report that the lock at address `lock` is about to be released.
// `is_w`=1 for writer lock, `is_w`=0 for reader lock.
#define ANNOTATE_RWLOCK_RELEASED(lock, is_w)          \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateRWLockReleased) \
  (__FILE__, __LINE__, lock, is_w)

// Apply ANNOTATE_BENIGN_RACE_SIZED to a static variable `static_var`.
#define ANNOTATE_BENIGN_RACE_STATIC(static_var, description)      \
  namespace {                                                     \
  class static_var##_annotator {                                  \
   public:                                                        \
    static_var##_annotator() {                                    \
      ANNOTATE_BENIGN_RACE_SIZED(&static_var, sizeof(static_var), \
                                 #static_var ": " description);   \
    }                                                             \
  };                                                              \
  static static_var##_annotator the##static_var##_annotator;      \
  }  // namespace

#else  // ABSL_INTERNAL_RACE_ANNOTATIONS_ENABLED == 0

#define ANNOTATE_RWLOCK_CREATE(lock)                            // empty
#define ANNOTATE_RWLOCK_CREATE_STATIC(lock)                     // empty
#define ANNOTATE_RWLOCK_DESTROY(lock)                           // empty
#define ANNOTATE_RWLOCK_ACQUIRED(lock, is_w)                    // empty
#define ANNOTATE_RWLOCK_RELEASED(lock, is_w)                    // empty
#define ANNOTATE_BENIGN_RACE(address, description)              // empty
#define ANNOTATE_BENIGN_RACE_SIZED(address, size, description)  // empty
#define ANNOTATE_THREAD_NAME(name)                              // empty
#define ANNOTATE_ENABLE_RACE_DETECTION(enable)                  // empty
#define ANNOTATE_BENIGN_RACE_STATIC(static_var, description)    // empty

#endif  // ABSL_INTERNAL_RACE_ANNOTATIONS_ENABLED

// -------------------------------------------------------------------------
// Define memory annotations.

#if ABSL_INTERNAL_MEMORY_ANNOTATIONS_ENABLED == 1

#include <sanitizer/msan_interface.h>

#define ANNOTATE_MEMORY_IS_INITIALIZED(address, size) \
  __msan_unpoison(address, size)

#define ANNOTATE_MEMORY_IS_UNINITIALIZED(address, size) \
  __msan_allocated_memory(address, size)

#else  // ABSL_INTERNAL_MEMORY_ANNOTATIONS_ENABLED == 0

#if DYNAMIC_ANNOTATIONS_ENABLED == 1
#define ANNOTATE_MEMORY_IS_INITIALIZED(address, size) \
  do {                                                \
    (void)(address);                                  \
    (void)(size);                                     \
  } while (0)
#define ANNOTATE_MEMORY_IS_UNINITIALIZED(address, size) \
  do {                                                  \
    (void)(address);                                    \
    (void)(size);                                       \
  } while (0)
#else
#define ANNOTATE_MEMORY_IS_INITIALIZED(address, size)    // empty
#define ANNOTATE_MEMORY_IS_UNINITIALIZED(address, size)  // empty
#endif

#endif  // ABSL_INTERNAL_MEMORY_ANNOTATIONS_ENABLED

// -------------------------------------------------------------------------
// Define IGNORE_READS_BEGIN/_END attributes.

#if defined(ABSL_INTERNAL_IGNORE_READS_ATTRIBUTE_ENABLED)

#define ABSL_INTERNAL_IGNORE_READS_BEGIN_ATTRIBUTE \
  __attribute((exclusive_lock_function("*")))
#define ABSL_INTERNAL_IGNORE_READS_END_ATTRIBUTE \
  __attribute((unlock_function("*")))

#else  // !defined(ABSL_INTERNAL_IGNORE_READS_ATTRIBUTE_ENABLED)

#define ABSL_INTERNAL_IGNORE_READS_BEGIN_ATTRIBUTE  // empty
#define ABSL_INTERNAL_IGNORE_READS_END_ATTRIBUTE    // empty

#endif  // defined(ABSL_INTERNAL_IGNORE_READS_ATTRIBUTE_ENABLED)

// -------------------------------------------------------------------------
// Define IGNORE_READS_BEGIN/_END annotations.

#if ABSL_INTERNAL_READS_ANNOTATIONS_ENABLED == 1

// Request the analysis tool to ignore all reads in the current thread until
// ANNOTATE_IGNORE_READS_END is called. Useful to ignore intentional racey
// reads, while still checking other reads and all writes.
// See also ANNOTATE_UNPROTECTED_READ.
#define ANNOTATE_IGNORE_READS_BEGIN() \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateIgnoreReadsBegin)(__FILE__, __LINE__)

// Stop ignoring reads.
#define ANNOTATE_IGNORE_READS_END() \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateIgnoreReadsEnd)(__FILE__, __LINE__)

#elif defined(ABSL_INTERNAL_ANNOTALYSIS_ENABLED)

// When Annotalysis is enabled without Dynamic Annotations, the use of
// static-inline functions allows the annotations to be read at compile-time,
// while still letting the compiler elide the functions from the final build.
//
// TODO(delesley) -- The exclusive lock here ignores writes as well, but
// allows IGNORE_READS_AND_WRITES to work properly.

#define ANNOTATE_IGNORE_READS_BEGIN() \
  ABSL_INTERNAL_GLOBAL_SCOPED(AbslInternalAnnotateIgnoreReadsBegin)()

#define ANNOTATE_IGNORE_READS_END() \
  ABSL_INTERNAL_GLOBAL_SCOPED(AbslInternalAnnotateIgnoreReadsEnd)()

#else

#define ANNOTATE_IGNORE_READS_BEGIN()  // empty
#define ANNOTATE_IGNORE_READS_END()    // empty

#endif

// -------------------------------------------------------------------------
// Define IGNORE_WRITES_BEGIN/_END annotations.

#if ABSL_INTERNAL_WRITES_ANNOTATIONS_ENABLED == 1

// Similar to ANNOTATE_IGNORE_READS_BEGIN, but ignore writes instead.
#define ANNOTATE_IGNORE_WRITES_BEGIN() \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateIgnoreWritesBegin)(__FILE__, __LINE__)

// Stop ignoring writes.
#define ANNOTATE_IGNORE_WRITES_END() \
  ABSL_INTERNAL_GLOBAL_SCOPED(AnnotateIgnoreWritesEnd)(__FILE__, __LINE__)

#else

#define ANNOTATE_IGNORE_WRITES_BEGIN()  // empty
#define ANNOTATE_IGNORE_WRITES_END()    // empty

#endif

// -------------------------------------------------------------------------
// Define the ANNOTATE_IGNORE_READS_AND_WRITES_* annotations using the more
// primitive annotations defined above.
//
//     Instead of doing
//        ANNOTATE_IGNORE_READS_BEGIN();
//        ... = x;
//        ANNOTATE_IGNORE_READS_END();
//     one can use
//        ... = ANNOTATE_UNPROTECTED_READ(x);

#if defined(ABSL_INTERNAL_READS_WRITES_ANNOTATIONS_ENABLED)

// Start ignoring all memory accesses (both reads and writes).
#define ANNOTATE_IGNORE_READS_AND_WRITES_BEGIN() \
  do {                                           \
    ANNOTATE_IGNORE_READS_BEGIN();               \
    ANNOTATE_IGNORE_WRITES_BEGIN();              \
  } while (0)

// Stop ignoring both reads and writes.
#define ANNOTATE_IGNORE_READS_AND_WRITES_END() \
  do {                                         \
    ANNOTATE_IGNORE_WRITES_END();              \
    ANNOTATE_IGNORE_READS_END();               \
  } while (0)

#ifdef __cplusplus
// ANNOTATE_UNPROTECTED_READ is the preferred way to annotate racey reads.
#define ANNOTATE_UNPROTECTED_READ(x) \
  absl::base_internal::AnnotateUnprotectedRead(x)

#endif

#else

#define ANNOTATE_IGNORE_READS_AND_WRITES_BEGIN()  // empty
#define ANNOTATE_IGNORE_READS_AND_WRITES_END()    // empty
#define ANNOTATE_UNPROTECTED_READ(x) (x)

#endif

// -------------------------------------------------------------------------
// Address sanitizer annotations

#ifdef ABSL_HAVE_ADDRESS_SANITIZER
// Describe the current state of a contiguous container such as e.g.
// std::vector or std::string. For more details see
// sanitizer/common_interface_defs.h, which is provided by the compiler.
#include <sanitizer/common_interface_defs.h>

#define ANNOTATE_CONTIGUOUS_CONTAINER(beg, end, old_mid, new_mid) \
  __sanitizer_annotate_contiguous_container(beg, end, old_mid, new_mid)
#define ADDRESS_SANITIZER_REDZONE(name)    \
  struct {                                 \
    char x[8] __attribute__((aligned(8))); \
  } name

#else

#define ANNOTATE_CONTIGUOUS_CONTAINER(beg, end, old_mid, new_mid)
#define ADDRESS_SANITIZER_REDZONE(name) static_assert(true, "")

#endif  // ABSL_HAVE_ADDRESS_SANITIZER

// -------------------------------------------------------------------------
// Undefine the macros intended only for this file.

#undef ABSL_INTERNAL_RACE_ANNOTATIONS_ENABLED
#undef ABSL_INTERNAL_MEMORY_ANNOTATIONS_ENABLED
#undef ABSL_INTERNAL_READS_ANNOTATIONS_ENABLED
#undef ABSL_INTERNAL_WRITES_ANNOTATIONS_ENABLED
#undef ABSL_INTERNAL_ANNOTALYSIS_ENABLED
#undef ABSL_INTERNAL_READS_WRITES_ANNOTATIONS_ENABLED
#undef ABSL_INTERNAL_BEGIN_EXTERN_C
#undef ABSL_INTERNAL_END_EXTERN_C
#undef ABSL_INTERNAL_STATIC_INLINE

#endif  // ABSL_BASE_INTERNAL_DYNAMIC_ANNOTATIONS_H_
