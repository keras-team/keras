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

#ifndef __TBB_spin_rw_mutex_H
#define __TBB_spin_rw_mutex_H

#include "detail/_namespace_injection.h"

#include "profiling.h"

#include "detail/_assert.h"
#include "detail/_utils.h"

#include <atomic>

namespace tbb {
namespace detail {
namespace d1 {

#if __TBB_TSX_INTRINSICS_PRESENT
class rtm_rw_mutex;
#endif

//! Fast, unfair, spinning reader-writer lock with backoff and writer-preference
/** @ingroup synchronization */
class spin_rw_mutex {
public:
    //! Constructors
    spin_rw_mutex() noexcept : m_state(0) {
       create_itt_sync(this, "tbb::spin_rw_mutex", "");
    }

    //! Destructor
    ~spin_rw_mutex() {
        __TBB_ASSERT(!m_state, "destruction of an acquired mutex");
    }

    //! No Copy
    spin_rw_mutex(const spin_rw_mutex&) = delete;
    spin_rw_mutex& operator=(const spin_rw_mutex&) = delete;

    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    class scoped_lock {
    public:
        //! Construct lock that has not acquired a mutex.
        /** Equivalent to zero-initialization of *this. */
        constexpr scoped_lock() noexcept : m_mutex(nullptr), m_is_writer(false) {}

        //! Acquire lock on given mutex.
        scoped_lock(spin_rw_mutex& m, bool write = true) : m_mutex(nullptr) {
            acquire(m, write);
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if (m_mutex) {
                release();
            }
        }

        //! No Copy
        scoped_lock(const scoped_lock&) = delete;
        scoped_lock& operator=(const scoped_lock&) = delete;

        //! Acquire lock on given mutex.
        void acquire(spin_rw_mutex& m, bool write = true) {
            m_is_writer = write;
            m_mutex = &m;
            if (write) {
                m_mutex->lock();
            } else {
                m_mutex->lock_shared();
            }
        }

        //! Try acquire lock on given mutex.
        bool try_acquire(spin_rw_mutex& m, bool write = true) {
            m_is_writer = write;
            bool result = write ? m.try_lock() : m.try_lock_shared();
            if (result) {
                m_mutex = &m;
            }
            return result;
        }

        //! Release lock.
        void release() {
            spin_rw_mutex* m = m_mutex;
            m_mutex = nullptr;

            if (m_is_writer) {
                m->unlock();
            } else {
                m->unlock_shared();
            }
        }

        //! Upgrade reader to become a writer.
        /** Returns whether the upgrade happened without releasing and re-acquiring the lock */
        bool upgrade_to_writer() {
            if (m_is_writer) return true; // Already a writer
            m_is_writer = true;
            return m_mutex->upgrade();
        }

        //! Downgrade writer to become a reader.
        bool downgrade_to_reader() {
            if (!m_is_writer) return true; // Already a reader
            m_mutex->downgrade();
            m_is_writer = false;
            return true;
        }

    protected:
        //! The pointer to the current mutex that is held, or nullptr if no mutex is held.
        spin_rw_mutex* m_mutex;

        //! If mutex != nullptr, then is_writer is true if holding a writer lock, false if holding a reader lock.
        /** Not defined if not holding a lock. */
        bool m_is_writer;
    };

    //! Mutex traits
    static constexpr bool is_rw_mutex = true;
    static constexpr bool is_recursive_mutex = false;
    static constexpr bool is_fair_mutex = false;

    //! Acquire lock
    void lock() {
        call_itt_notify(prepare, this);
        for (atomic_backoff backoff; ; backoff.pause()) {
            state_type s = m_state.load(std::memory_order_relaxed);
            if (!(s & BUSY)) { // no readers, no writers
                if (m_state.compare_exchange_strong(s, WRITER))
                    break; // successfully stored writer flag
                backoff.reset(); // we could be very close to complete op.
            } else if (!(s & WRITER_PENDING)) { // no pending writers
                m_state |= WRITER_PENDING;
            }
        }
        call_itt_notify(acquired, this);
    }

    //! Try acquiring lock (non-blocking)
    /** Return true if lock acquired; false otherwise. */
    bool try_lock() {
        // for a writer: only possible to acquire if no active readers or writers
        state_type s = m_state.load(std::memory_order_relaxed);
        if (!(s & BUSY)) { // no readers, no writers; mask is 1..1101
            if (m_state.compare_exchange_strong(s, WRITER)) {
                call_itt_notify(acquired, this);
                return true; // successfully stored writer flag
            }
        }
        return false;
    }

    //! Release lock
    void unlock() {
        call_itt_notify(releasing, this);
        m_state &= READERS;
    }

    //! Lock shared ownership mutex
    void lock_shared() {
        call_itt_notify(prepare, this);
        for (atomic_backoff b; ; b.pause()) {
            state_type s = m_state.load(std::memory_order_relaxed);
            if (!(s & (WRITER | WRITER_PENDING))) { // no writer or write requests
                state_type prev_state = m_state.fetch_add(ONE_READER);
                if (!(prev_state & WRITER)) {
                    break; // successfully stored increased number of readers
                }
                // writer got there first, undo the increment
                m_state -= ONE_READER;
            }
        }
        call_itt_notify(acquired, this);
        __TBB_ASSERT(m_state & READERS, "invalid state of a read lock: no readers");
    }

    //! Try lock shared ownership mutex
    bool try_lock_shared() {
        // for a reader: acquire if no active or waiting writers
        state_type s = m_state.load(std::memory_order_relaxed);
        if (!(s & (WRITER | WRITER_PENDING))) { // no writers
            state_type prev_state = m_state.fetch_add(ONE_READER);
            if (!(prev_state & WRITER)) {  // got the lock
                call_itt_notify(acquired, this);
                return true; // successfully stored increased number of readers
            }
            // writer got there first, undo the increment
            m_state -= ONE_READER;
        }
        return false;
    }

    //! Unlock shared ownership mutex
    void unlock_shared() {
        __TBB_ASSERT(m_state & READERS, "invalid state of a read lock: no readers");
        call_itt_notify(releasing, this);
        m_state -= ONE_READER;
    }

protected:
    /** Internal non ISO C++ standard API **/
    //! This API is used through the scoped_lock class

    //! Upgrade reader to become a writer.
    /** Returns whether the upgrade happened without releasing and re-acquiring the lock */
    bool upgrade() {
        state_type s = m_state.load(std::memory_order_relaxed);
        __TBB_ASSERT(s & READERS, "invalid state before upgrade: no readers ");
        // Check and set writer-pending flag.
        // Required conditions: either no pending writers, or we are the only reader
        // (with multiple readers and pending writer, another upgrade could have been requested)
        while ((s & READERS) == ONE_READER || !(s & WRITER_PENDING)) {
            if (m_state.compare_exchange_strong(s, s | WRITER | WRITER_PENDING)) {
                atomic_backoff backoff;
                while ((m_state.load(std::memory_order_relaxed) & READERS) != ONE_READER) backoff.pause();
                __TBB_ASSERT((m_state & (WRITER_PENDING|WRITER)) == (WRITER_PENDING | WRITER), "invalid state when upgrading to writer");
                // Both new readers and writers are blocked at this time
                m_state -= (ONE_READER + WRITER_PENDING);
                return true; // successfully upgraded
            }
        }
        // Slow reacquire
        unlock_shared();
        lock();
        return false;
    }

    //! Downgrade writer to a reader
    void downgrade() {
        call_itt_notify(releasing, this);
        m_state += (ONE_READER - WRITER);
        __TBB_ASSERT(m_state & READERS, "invalid state after downgrade: no readers");
    }

    using state_type = std::intptr_t;
    static constexpr state_type WRITER = 1;
    static constexpr state_type WRITER_PENDING = 2;
    static constexpr state_type READERS = ~(WRITER | WRITER_PENDING);
    static constexpr state_type ONE_READER = 4;
    static constexpr state_type BUSY = WRITER | READERS;
    //! State of lock
    /** Bit 0 = writer is holding lock
        Bit 1 = request by a writer to acquire lock (hint to readers to wait)
        Bit 2..N = number of readers holding lock */
    std::atomic<state_type> m_state;
}; // class spin_rw_mutex

#if TBB_USE_PROFILING_TOOLS
inline void set_name(spin_rw_mutex& obj, const char* name) {
    itt_set_sync_name(&obj, name);
}
#if (_WIN32||_WIN64) && !__MINGW32__
inline void set_name(spin_rw_mutex& obj, const wchar_t* name) {
    itt_set_sync_name(&obj, name);
}
#endif // WIN
#else
inline void set_name(spin_rw_mutex&, const char*) {}
#if (_WIN32||_WIN64) && !__MINGW32__
inline void set_name(spin_rw_mutex&, const wchar_t*) {}
#endif // WIN
#endif
} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::spin_rw_mutex;
} // namespace v1
namespace profiling {
    using detail::d1::set_name;
}
} // namespace tbb

#include "detail/_rtm_rw_mutex.h"

namespace tbb {
inline namespace v1 {
#if __TBB_TSX_INTRINSICS_PRESENT
    using speculative_spin_rw_mutex = detail::d1::rtm_rw_mutex;
#else
    using speculative_spin_rw_mutex = detail::d1::spin_rw_mutex;
#endif
}
}

#endif /* __TBB_spin_rw_mutex_H */

