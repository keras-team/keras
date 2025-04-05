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

// This file declares RequestDeadlineTracker.

#ifndef TFRT_HOST_CONTEXT_REQUEST_DEADLINE_TRACKER_H_
#define TFRT_HOST_CONTEXT_REQUEST_DEADLINE_TRACKER_H_

#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/timer_queue.h"

namespace tfrt {

class RequestDeadlineTracker {
 public:
  explicit RequestDeadlineTracker(HostContext* host_context) {
    timer_queue_ = host_context->GetTimerQueue();
  }

  // Enqueue a timer that tracks the dealine of the request to the TimerQueue.
  void CancelRequestOnDeadline(std::chrono::system_clock::time_point deadline,
                               const RCReference<RequestContext>& req_ctx) {
    timer_queue_->ScheduleTimerAt(
        deadline, [cancellation_context = req_ctx->cancellation_context()] {
          cancellation_context->Cancel();
        });
  }

 private:
  TimerQueue* timer_queue_;
};
}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_REQUEST_DEADLINE_TRACKER_H_
