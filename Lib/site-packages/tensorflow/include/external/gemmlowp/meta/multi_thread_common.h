// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
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

#ifndef GEMMLOWP_META_MULTI_THREAD_COMMON_H_
#define GEMMLOWP_META_MULTI_THREAD_COMMON_H_

#include "../internal/multi_thread_gemm.h"

namespace gemmlowp {
namespace meta {

inline int ResolveMaxThreads(int max_threads) {
  if (max_threads == 0) {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    static const int hardware_threads_count =
        static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    return hardware_threads_count;
#endif
  }
  return max_threads;
}

template <typename WorkersPool>
class SimpleContext {
 public:
  SimpleContext(int max_num_threads, WorkersPool* pool)
      : max_num_threads_(max_num_threads), pool_(pool) {}

  WorkersPool* workers_pool() { return pool_; }

  int max_num_threads() { return max_num_threads_; }

 private:
  int max_num_threads_;
  WorkersPool* pool_;
};

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_MULTI_THREAD_COMMON_H_
