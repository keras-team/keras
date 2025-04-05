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

// Timer Queue
//
// This file declares TimerQueue, a priority queue to keep track of pending
// timers. The queue is keyed by timer deadline (the sooner it expires, the
// higher the priority). On timer expiration, it calls the associated callback.

#ifndef TFRT_HOST_CONTEXT_TIMER_QUEUE_H_
#define TFRT_HOST_CONTEXT_TIMER_QUEUE_H_

#include <chrono>
#include <queue>
#include <thread>

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class TimerQueue {
  using Clock = std::chrono::system_clock;
  using TimeDuration = Clock::duration;
  using TimerCallback = llvm::unique_function<void()>;
  using TimePoint =
      std::chrono::time_point<std::chrono::system_clock, TimeDuration>;

  class TimerEntry;

 public:
  using TimerHandle = RCReference<TimerEntry>;

  // On creation, starts the timer thread for TimerQueue monitoring.
  TimerQueue();
  // On destruction, cancel every timer in the queue.
  ~TimerQueue();

  // Enqueue a new timer.
  TimerHandle ScheduleTimerAt(TimePoint deadline, TimerCallback callback);

  // Enqueue a timer. Deadline is `timeout` microseconds from now.
  TimerHandle ScheduleTimer(TimeDuration timeout, TimerCallback callback);

  void CancelTimer(const TimerHandle& timer_handle);

 private:
  // A reference counted timer, which has a deadline and a callback function.
  class TimerEntry : public ReferenceCounted<TimerEntry> {
   public:
    TimerEntry(TimePoint deadline, TimerCallback timer_callback)
        : deadline_(deadline), timer_callback_(std::move(timer_callback)) {}

    // Factory method to create a timer. Meant for internal usage only.
    static RCReference<TimerEntry> Create(TimePoint deadline,
                                          TimerCallback timer_callback) {
      return MakeRef<TimerEntry>(deadline, std::move(timer_callback));
    }

    struct TimerEntryCompare {
      bool operator()(const RCReference<TimerEntry>& a,
                      const RCReference<TimerEntry>& b) const {
        return a->deadline_ > b->deadline_;
      }
    };

   private:
    friend class TimerQueue;
    TimePoint deadline_;
    TimerCallback timer_callback_;
    std::atomic<bool> cancelled_{false};
  };
  // Timer thread. If a timeout goes off, it calls the callback.
  void TimerThreadRun();

  // Obtain the top timer (timer with the closest deadline) from the queue.
  // Returns NULL if no active (not cancelled) timer found.
  TimerEntry* getTopTimer();

  mutable mutex mu_;
  condition_variable cv_;
  std::thread timer_thread_;
  std::atomic<bool> stop_{false};
  std::priority_queue<RCReference<TimerEntry>,
                      std::vector<RCReference<TimerEntry>>,
                      TimerEntry::TimerEntryCompare>
      timers_ TFRT_GUARDED_BY(mu_);
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_TIMER_QUEUE_H_
