/*******************************************************************************
* Copyright 2020 Intel Corporation
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
*******************************************************************************/

#ifndef COMMON_COUNTING_BARRIER_HPP
#define COMMON_COUNTING_BARRIER_HPP

#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace dnnl {
namespace impl {

// Similar to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/blocking_counter.h
struct counting_barrier_t {
    counting_barrier_t(unsigned size = 0) { init(size); }

    void init(unsigned size) {
        assert(size < waiter_mask_);
        notified_ = false;
        state_ = size;
    }

    void notify() {
        auto s = state_.fetch_sub(1) - 1;
        if (s != waiter_mask_) {
            assert(((s + 1) & ~waiter_mask_) != 0);
            return;
        }
        std::unique_lock<std::mutex> l(m_);
        notified_ = true;
        cv_.notify_all();
    }

    void wait() {
        auto s = state_.fetch_or(waiter_mask_);
        if (s == 0) return;
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this]() { return notified_; });
    }

private:
    static constexpr unsigned waiter_mask_ = 1u << (sizeof(unsigned) * 8 - 1);

    std::atomic<unsigned> state_;
    bool notified_;

    std::condition_variable cv_;
    std::mutex m_;
};

} // namespace impl
} // namespace dnnl

#endif
