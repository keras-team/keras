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

// ThreadingEnvironment defines how to start, join and detatch threads in
// the blocking and non-blocking work queues.
//
// Uses std::thread implementation.

#ifndef TFRT_SUPPORT_THREAD_ENVIRONMENT_STD_H_
#define TFRT_SUPPORT_THREAD_ENVIRONMENT_STD_H_

#include <thread>

#include "llvm/ADT/StringRef.h"

namespace tfrt {
namespace internal {

class StdThread {
 public:
  explicit StdThread(std::thread thread) : thread_(std::move(thread)) {}
  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

struct StdThreadingEnvironment {
  using Thread = ::tfrt::internal::StdThread;

  template <class Function, class... Args>
  static std::unique_ptr<Thread> StartThread(llvm::StringRef name_prefix,
                                             Function&& f, Args&&... args) {
    return std::make_unique<Thread>(
        std::thread(std::forward<Function>(f), std::forward<Args>(args)...));
  }

  static uint64_t ThisThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }
};
}  // namespace internal

using ThreadingEnvironment = internal::StdThreadingEnvironment;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_THREAD_ENVIRONMENT_STD_H_
