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

#ifndef __TBB_queuing_mutex_H
#define __TBB_queuing_mutex_H

#include "detail/_namespace_injection.h"
#include "detail/_assert.h"
#include "detail/_utils.h"

#include "profiling.h"

#include <atomic>

namespace tbb {
namespace detail {
namespace d1 {

//! Queuing mutex with local-only spinning.
/** @ingroup synchronization */
class queuing_mutex {
public:
    //! Construct unacquired mutex.
    queuing_mutex() noexcept  {
        create_itt_sync(this, "tbb::queuing_mutex", "");
    };

    queuing_mutex(const queuing_mutex&) = delete;
    queuing_mutex& operator=(const queuing_mutex&) = delete;

    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    class scoped_lock {
        //! Reset fields to mean "no lock held".
        void reset() {
            m_mutex = nullptr;
        }

    public:
        //! Construct lock that has not acquired a mutex.
        /** Equivalent to zero-initialization of *this. */
        scoped_lock() = default;

        //! Acquire lock on given mutex.
        scoped_lock(queuing_mutex& m) {
            acquire(m);
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if (m_mutex) release();
        }

        //! No Copy
        scoped_lock( const scoped_lock& ) = delete;
        scoped_lock& operator=( const scoped_lock& ) = delete;

        //! Acquire lock on given mutex.
        void acquire( queuing_mutex& m ) {
            __TBB_ASSERT(!m_mutex, "scoped_lock is already holding a mutex");

            // Must set all fields before the exchange, because once the
            // exchange executes, *this becomes accessible to other threads.
            m_mutex = &m;
            m_next.store(nullptr, std::memory_order_relaxed);
            m_going.store(0U, std::memory_order_relaxed);

            // x86 compare exchange operation always has a strong fence
            // "sending" the fields initialized above to other processors.
            scoped_lock* pred = m.q_tail.exchange(this);
            if (pred) {
                call_itt_notify(prepare, &m);
                __TBB_ASSERT(pred->m_next.load(std::memory_order_relaxed) == nullptr, "the predecessor has another successor!");

                pred->m_next.store(this, std::memory_order_relaxed);
                spin_wait_while_eq(m_going, 0U);
            }
            call_itt_notify(acquired, &m);

            // Force acquire so that user's critical section receives correct values
            // from processor that was previously in the user's critical section.
            atomic_fence(std::memory_order_acquire);
        }

        //! Acquire lock on given mutex if free (i.e. non-blocking)
        bool try_acquire( queuing_mutex& m ) {
            __TBB_ASSERT(!m_mutex, "scoped_lock is already holding a mutex");

            // Must set all fields before the compare_exchange_strong, because once the
            // compare_exchange_strong executes, *this becomes accessible to other threads.
            m_next.store(nullptr, std::memory_order_relaxed);
            m_going.store(0U, std::memory_order_relaxed);

            scoped_lock* expected = nullptr;
            // The compare_exchange_strong must have release semantics, because we are
            // "sending" the fields initialized above to other processors.
            // x86 compare exchange operation always has a strong fence
            if (!m.q_tail.compare_exchange_strong(expected, this))
                return false;

            m_mutex = &m;

            // Force acquire so that user's critical section receives correct values
            // from processor that was previously in the user's critical section.
            atomic_fence(std::memory_order_acquire);
            call_itt_notify(acquired, &m);
            return true;
        }

        //! Release lock.
        void release()
        {
            __TBB_ASSERT(this->m_mutex, "no lock acquired");

            call_itt_notify(releasing, this->m_mutex);

            if (m_next.load(std::memory_order_relaxed) == nullptr) {
                scoped_lock* expected = this;
                if (m_mutex->q_tail.compare_exchange_strong(expected, nullptr)) {
                    // this was the only item in the queue, and the queue is now empty.
                    reset();
                    return;
                }
                // Someone in the queue
                spin_wait_while_eq(m_next, nullptr);
            }
            m_next.load(std::memory_order_relaxed)->m_going.store(1U, std::memory_order_release);

            reset();
        }

    private:
        //! The pointer to the mutex owned, or NULL if not holding a mutex.
        queuing_mutex* m_mutex{nullptr};

        //! The pointer to the next competitor for a mutex
        std::atomic<scoped_lock*> m_next{nullptr};

        //! The local spin-wait variable
        /** Inverted (0 - blocked, 1 - acquired the mutex) for the sake of
            zero-initialization.  Defining it as an entire word instead of
            a byte seems to help performance slightly. */
        std::atomic<uintptr_t> m_going{0U};
    };

    // Mutex traits
    static constexpr bool is_rw_mutex = false;
    static constexpr bool is_recursive_mutex = false;
    static constexpr bool is_fair_mutex = true;

private:
    //! The last competitor requesting the lock
    std::atomic<scoped_lock*> q_tail{nullptr};

};

#if TBB_USE_PROFILING_TOOLS
inline void set_name(queuing_mutex& obj, const char* name) {
    itt_set_sync_name(&obj, name);
}
#if (_WIN32||_WIN64) && !__MINGW32__
inline void set_name(queuing_mutex& obj, const wchar_t* name) {
    itt_set_sync_name(&obj, name);
}
#endif //WIN
#else
inline void set_name(queuing_mutex&, const char*) {}
#if (_WIN32||_WIN64) && !__MINGW32__
inline void set_name(queuing_mutex&, const wchar_t*) {}
#endif //WIN
#endif
} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::queuing_mutex;
} // namespace v1
namespace profiling {
    using detail::d1::set_name;
}
} // namespace tbb

#endif /* __TBB_queuing_mutex_H */
