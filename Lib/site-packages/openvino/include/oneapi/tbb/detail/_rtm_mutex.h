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

#ifndef __TBB__rtm_mutex_impl_H
#define __TBB__rtm_mutex_impl_H

#include "_assert.h"
#include "_utils.h"
#include "../spin_mutex.h"

#include "../profiling.h"

namespace tbb {
namespace detail {
namespace r1 {
struct rtm_mutex_impl;
}
namespace d1 {

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress warning: structure was padded due to alignment specifier
    #pragma warning (push)
    #pragma warning (disable: 4324)
#endif

/** A rtm_mutex is an speculation-enabled spin mutex.
    It should be used for locking short critical sections where the lock is
    contended but the data it protects are not.  If zero-initialized, the
    mutex is considered unheld.
    @ingroup synchronization */
class alignas(max_nfs_size) rtm_mutex : private spin_mutex {
private:
    enum class rtm_state {
        rtm_none,
        rtm_transacting,
        rtm_real
    };
public:
    //! Constructors
    rtm_mutex() noexcept {
        create_itt_sync(this, "tbb::speculative_spin_mutex", "");
    }

    //! Destructor
    ~rtm_mutex() = default;

    //! Represents acquisition of a mutex.
    class scoped_lock {
    public:
        friend class rtm_mutex;
        //! Construct lock that has not acquired a mutex.
        /** Equivalent to zero-initialization of *this. */
        constexpr scoped_lock() : m_mutex(nullptr), m_transaction_state(rtm_state::rtm_none) {}

        //! Acquire lock on given mutex.
        scoped_lock(rtm_mutex& m) : m_mutex(nullptr), m_transaction_state(rtm_state::rtm_none) {
            acquire(m);
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if(m_transaction_state != rtm_state::rtm_none) {
                release();
            }
        }

        //! No Copy
        scoped_lock(const scoped_lock&) = delete;
        scoped_lock& operator=(const scoped_lock&) = delete;

        //! Acquire lock on given mutex.
        void acquire(rtm_mutex& m);

        //! Try acquire lock on given mutex.
        bool try_acquire(rtm_mutex& m);

        //! Release lock
        void release();

    private:
        rtm_mutex* m_mutex;
        rtm_state m_transaction_state;
        friend r1::rtm_mutex_impl;
    };

    //! Mutex traits
    static constexpr bool is_rw_mutex = false;
    static constexpr bool is_recursive_mutex = false;
    static constexpr bool is_fair_mutex = false;
private:
    friend r1::rtm_mutex_impl;
}; // end of rtm_mutex
} // namespace d1

namespace r1 {
    //! Internal acquire lock.
    // only_speculate == true if we're doing a try_lock, else false.
    void __TBB_EXPORTED_FUNC acquire(d1::rtm_mutex&, d1::rtm_mutex::scoped_lock&, bool only_speculate = false);
    //! Internal try_acquire lock.
    bool __TBB_EXPORTED_FUNC try_acquire(d1::rtm_mutex&, d1::rtm_mutex::scoped_lock&);
    //! Internal release lock.
    void __TBB_EXPORTED_FUNC release(d1::rtm_mutex::scoped_lock&);
} // namespace r1

namespace d1 {
//! Acquire lock on given mutex.
inline void rtm_mutex::scoped_lock::acquire(rtm_mutex& m) {
    __TBB_ASSERT(!m_mutex, "lock is already acquired");
    r1::acquire(m, *this);
}

//! Try acquire lock on given mutex.
inline bool rtm_mutex::scoped_lock::try_acquire(rtm_mutex& m) {
    __TBB_ASSERT(!m_mutex, "lock is already acquired");
    return r1::try_acquire(m, *this);
}

//! Release lock
inline void rtm_mutex::scoped_lock::release() {
    __TBB_ASSERT(m_mutex, "lock is not acquired");
    __TBB_ASSERT(m_transaction_state != rtm_state::rtm_none, "lock is not acquired");
    return r1::release(*this);
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning (pop) // 4324 warning
#endif

#if TBB_USE_PROFILING_TOOLS
inline void set_name(rtm_mutex& obj, const char* name) {
    itt_set_sync_name(&obj, name);
}
#if (_WIN32||_WIN64) && !__MINGW32__
inline void set_name(rtm_mutex& obj, const wchar_t* name) {
    itt_set_sync_name(&obj, name);
}
#endif // WIN
#else
inline void set_name(rtm_mutex&, const char*) {}
#if (_WIN32||_WIN64) && !__MINGW32__
inline void set_name(rtm_mutex&, const wchar_t*) {}
#endif // WIN
#endif

} // namespace d1
} // namespace detail
} // namespace tbb

#endif /* __TBB__rtm_mutex_impl_H */
