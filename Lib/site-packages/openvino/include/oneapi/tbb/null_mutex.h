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

#ifndef __TBB_null_mutex_H
#define __TBB_null_mutex_H

#include "detail/_config.h"
#include "detail/_namespace_injection.h"

namespace tbb {
namespace detail {
namespace d1 {

//! A mutex which does nothing
/** A null_mutex does no operation and simulates success.
    @ingroup synchronization */
class null_mutex {
public:
    //! Constructors
    constexpr null_mutex() noexcept = default;

    //! Destructor
    ~null_mutex() = default;

    //! No Copy
    null_mutex(const null_mutex&) = delete;
    null_mutex& operator=(const null_mutex&) = delete;

    //! Represents acquisition of a mutex.
    class scoped_lock {
    public:
        //! Constructors
        constexpr scoped_lock() noexcept = default;
        scoped_lock(null_mutex&) {}

        //! Destructor
        ~scoped_lock() = default;

        //! No Copy
        scoped_lock(const scoped_lock&) = delete;
        scoped_lock& operator=(const scoped_lock&) = delete;

        void acquire(null_mutex&) {}
        bool try_acquire(null_mutex&) { return true; }
        void release() {}
    };

    //! Mutex traits
    static constexpr bool is_rw_mutex = false;
    static constexpr bool is_recursive_mutex = true;
    static constexpr bool is_fair_mutex = true;

    void lock() {}
    bool try_lock() { return true; }
    void unlock() {}
}; // class null_mutex

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::null_mutex;
} // namespace v1
} // namespace tbb

#endif /* __TBB_null_mutex_H */
