// Copyright 2017 The Gemmlowp Authors. All Rights Reserved.
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

// pthread_everywhere.h: Either includes <pthread.h> or implements a
// subset of pthread functionality on top of C++11 <thread> for portability.

#ifndef GEMMLOWP_PROFILING_PTHREAD_EVERYWHERE_H_
#define GEMMLOWP_PROFILING_PTHREAD_EVERYWHERE_H_

#ifndef _WIN32
#define GEMMLOWP_USE_PTHREAD
#endif

#if defined GEMMLOWP_USE_PTHREAD
#include <pthread.h>
#else
// Implement a small subset of pthread on top of C++11 threads.
// The function signatures differ from true pthread functions in two ways:
//  - True pthread functions return int error codes, ours return void.
//    Rationale: the c++11 <thread> equivalent functions return void
//    and use exceptions to report errors; we don't want to deal with
//    exceptions in this code, so we couldn't meaningfully return errors
//    in the polyfill. Also, the gemmlowp code using these pthread functions
//    never checks their return values anyway.
//  - True pthread *_create/*_init functions take pointers to 'attribute'
//    structs; ours take nullptr_t. That is because gemmlowp always passes
//    nullptr at the moment, so any support we would code for non-null
//    attribs would be unused.
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
namespace gemmlowp {
using pthread_t = std::thread *;
using pthread_mutex_t = std::mutex *;
using pthread_cond_t = std::condition_variable *;
inline void pthread_create(pthread_t *thread, std::nullptr_t,
                           void *(*start_routine)(void *), void *arg) {
  *thread = new std::thread(start_routine, arg);
}
inline void pthread_join(pthread_t thread, std::nullptr_t) { thread->join(); }
inline void pthread_mutex_init(pthread_mutex_t *mutex, std::nullptr_t) {
  *mutex = new std::mutex;
}
inline void pthread_mutex_lock(pthread_mutex_t *mutex) { (*mutex)->lock(); }
inline void pthread_mutex_unlock(pthread_mutex_t *mutex) { (*mutex)->unlock(); }
inline void pthread_mutex_destroy(pthread_mutex_t *mutex) { delete *mutex; }
inline void pthread_cond_init(pthread_cond_t *cond, std::nullptr_t) {
  *cond = new std::condition_variable;
}
inline void pthread_cond_signal(pthread_cond_t *cond) { (*cond)->notify_one(); }
inline void pthread_cond_broadcast(pthread_cond_t *cond) {
  (*cond)->notify_all();
}
inline void pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
  std::unique_lock<std::mutex> lock(**mutex, std::adopt_lock);
  (*cond)->wait(lock);
  // detach lock from mutex so when we leave this conext
  // the lock is not released
  lock.release();
}
inline void pthread_cond_destroy(pthread_cond_t *cond) { delete *cond; }
}  // end namespace gemmlowp
#endif

#endif  // GEMMLOWP_PROFILING_PTHREAD_EVERYWHERE_H_
