/*
 *
 * Copyright 2019 gRPC authors.
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

#include <grpc/support/port_platform.h>

#include <functional>

#include "src/core/lib/debug/trace.h"
#include "src/core/lib/gprpp/atomic.h"
#include "src/core/lib/gprpp/debug_location.h"
#include "src/core/lib/gprpp/mpscq.h"
#include "src/core/lib/gprpp/ref_counted.h"

#ifndef GRPC_CORE_LIB_IOMGR_LOGICAL_THREAD_H
#define GRPC_CORE_LIB_IOMGR_LOGICAL_THREAD_H

namespace grpc_core {
extern DebugOnlyTraceFlag grpc_logical_thread_trace;

// LogicalThread is a mechanism to schedule callbacks in a synchronized manner.
// All callbacks scheduled on a LogicalThread instance will be executed serially
// in a borrowed thread. The API provides a FIFO guarantee to the execution of
// callbacks scheduled on the thread.
class LogicalThread : public RefCounted<LogicalThread> {
 public:
  void Run(std::function<void()> callback,
           const grpc_core::DebugLocation& location);

 private:
  void DrainQueue();

  Atomic<size_t> size_{0};
  MultiProducerSingleConsumerQueue queue_;
};
} /* namespace grpc_core */

#endif /* GRPC_CORE_LIB_IOMGR_LOGICAL_THREAD_H */
