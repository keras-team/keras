// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// internal/platform.h: a place to put platform specific code

#ifndef GEMMLOWP_INTERNAL_PLATFORM_H_
#define GEMMLOWP_INTERNAL_PLATFORM_H_

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/time.h>
#endif

#if defined ANDROID || defined __ANDROID__
#include <malloc.h>
#include <android/api-level.h>
// The 18 here should be 16, but has to be 18 for now due
// to a Google-internal issue.
#if __ANDROID_API__ < 18
#define GEMMLOWP_USE_MEMALIGN
#endif
// posix_memalign is missing on some 4.1 x86 devices
#if __ANDROID_API__ == 18
#ifdef GEMMLOWP_X86_32
#define GEMMLOWP_USE_MEMALIGN
#endif
#endif
#endif

// Needed by chrome native builds
#ifndef _SC_NPROCESSORS_CONF
#define _SC_NPROCESSORS_CONF _SC_NPROCESSORS_ONLN
#endif

namespace gemmlowp {

#ifdef _WIN32
inline void *aligned_alloc(size_t alignment, size_t size) {
  return _aligned_malloc(size, alignment);
}

inline void aligned_free(void *memptr) { _aligned_free(memptr); }

inline int GetHardwareConcurrency(int max_threads) {
  if (max_threads == 0) {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
  }
  return max_threads;
}

inline double real_time_in_seconds() {
  __int64 wintime;
  GetSystemTimeAsFileTime((FILETIME *)&wintime);
  wintime -= 116444736000000000LL;  // 1jan1601 to 1jan1970
  return wintime / 10000000LL + wintime % 10000000LL * 100 * 1e-9;
}

#else
inline void *aligned_alloc(size_t alignment, size_t size) {
#ifdef GEMMLOWP_USE_MEMALIGN
  return memalign(alignment, size);
#else
  void *memptr;
  if (posix_memalign(&memptr, alignment, size)) {
    memptr = nullptr;
  }
  return memptr;
#endif
}

inline int GetHardwareConcurrency(int max_threads) {
  if (max_threads == 0) {
    static const int hardware_threads_count =
        static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    return hardware_threads_count;
  }
  return max_threads;
}

inline void aligned_free(void *memptr) { free(memptr); }

inline double real_time_in_seconds() {
#ifdef __APPLE__
  timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + 1e-6 * t.tv_usec;
#else
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
}

#endif
}  // namespace gemmlowp
#endif  // GEMMLOWP_INTERNAL_PLATFORM_H_
