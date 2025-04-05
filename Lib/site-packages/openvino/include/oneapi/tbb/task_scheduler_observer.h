/*
    Copyright (c) 2005-2021 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_task_scheduler_observer_H
#define __TBB_task_scheduler_observer_H

#include "detail/_namespace_injection.h"
#include "task_arena.h"
#include <atomic>

namespace tbb {
namespace detail {

namespace d1 {
class task_scheduler_observer;
}

namespace r1 {
class observer_proxy;
class observer_list;

//! Enable or disable observation
/** For local observers the method can be used only when the current thread
has the task scheduler initialized or is attached to an arena.
Repeated calls with the same state are no-ops. **/
void __TBB_EXPORTED_FUNC observe(d1::task_scheduler_observer&, bool state = true);
}

namespace d1 {
class task_scheduler_observer {
    friend class r1::observer_proxy;
    friend class r1::observer_list;
    friend void r1::observe(d1::task_scheduler_observer&, bool);

    //! Pointer to the proxy holding this observer.
    /** Observers are proxied by the scheduler to maintain persistent lists of them. **/
    std::atomic<r1::observer_proxy*> my_proxy{ nullptr };

    //! Counter preventing the observer from being destroyed while in use by the scheduler.
    /** Valid only when observation is on. **/
    std::atomic<intptr_t> my_busy_count{ 0 };

    //! Contains task_arena pointer
    task_arena* my_task_arena{ nullptr };
public:
    //! Returns true if observation is enabled, false otherwise.
    bool is_observing() const { return my_proxy.load(std::memory_order_relaxed) != nullptr; }

    //! Entry notification
    /** Invoked from inside observe(true) call and whenever a worker enters the arena
        this observer is associated with. If a thread is already in the arena when
        the observer is activated, the entry notification is called before it
        executes the first stolen task. **/
    virtual void on_scheduler_entry( bool /*is_worker*/ ) {}

    //! Exit notification
    /** Invoked from inside observe(false) call and whenever a worker leaves the
        arena this observer is associated with. **/
    virtual void on_scheduler_exit( bool /*is_worker*/ ) {}

    //! Construct local or global observer in inactive state (observation disabled).
    /** For a local observer entry/exit notifications are invoked whenever a worker
        thread joins/leaves the arena of the observer's owner thread. If a thread is
        already in the arena when the observer is activated, the entry notification is
        called before it executes the first stolen task. **/
    explicit task_scheduler_observer() = default;

    //! Construct local observer for a given arena in inactive state (observation disabled).
    /** entry/exit notifications are invoked whenever a thread joins/leaves arena.
        If a thread is already in the arena when the observer is activated, the entry notification
        is called before it executes the first stolen task. **/
    explicit task_scheduler_observer(task_arena& a) : my_task_arena(&a) {}

    /** Destructor protects instance of the observer from concurrent notification.
       It is recommended to disable observation before destructor of a derived class starts,
       otherwise it can lead to concurrent notification callback on partly destroyed object **/
    virtual ~task_scheduler_observer() {
        if (my_proxy.load(std::memory_order_relaxed)) {
            observe(false);
        }
    }

    //! Enable or disable observation
    /** Warning: concurrent invocations of this method are not safe.
        Repeated calls with the same state are no-ops. **/
    void observe(bool state = true) {
        if( state && !my_proxy.load(std::memory_order_relaxed) ) {
            __TBB_ASSERT( my_busy_count.load(std::memory_order_relaxed) == 0, "Inconsistent state of task_scheduler_observer instance");
        }
        r1::observe(*this, state);
    }
};

} // namespace d1
} // namespace detail

inline namespace v1 {
    using detail::d1::task_scheduler_observer;
}
} // namespace tbb


#endif /* __TBB_task_scheduler_observer_H */
