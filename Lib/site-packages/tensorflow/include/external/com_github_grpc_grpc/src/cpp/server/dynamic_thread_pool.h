/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_INTERNAL_CPP_DYNAMIC_THREAD_POOL_H
#define GRPC_INTERNAL_CPP_DYNAMIC_THREAD_POOL_H

#include <list>
#include <memory>
#include <queue>

#include <grpcpp/support/config.h>

#include "src/core/lib/gprpp/sync.h"
#include "src/core/lib/gprpp/thd.h"
#include "src/cpp/server/thread_pool_interface.h"

namespace grpc {

class DynamicThreadPool final : public ThreadPoolInterface {
 public:
  explicit DynamicThreadPool(int reserve_threads);
  ~DynamicThreadPool();

  void Add(const std::function<void()>& callback) override;

 private:
  class DynamicThread {
   public:
    DynamicThread(DynamicThreadPool* pool);
    ~DynamicThread();

   private:
    DynamicThreadPool* pool_;
    grpc_core::Thread thd_;
    void ThreadFunc();
  };
  grpc_core::Mutex mu_;
  grpc_core::CondVar cv_;
  grpc_core::CondVar shutdown_cv_;
  bool shutdown_;
  std::queue<std::function<void()>> callbacks_;
  int reserve_threads_;
  int nthreads_;
  int threads_waiting_;
  std::list<DynamicThread*> dead_threads_;

  void ThreadFunc();
  static void ReapThreads(std::list<DynamicThread*>* tlist);
};

}  // namespace grpc

#endif  // GRPC_INTERNAL_CPP_DYNAMIC_THREAD_POOL_H
