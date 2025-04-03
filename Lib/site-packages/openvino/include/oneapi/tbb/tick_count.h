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

#ifndef __TBB_tick_count_H
#define __TBB_tick_count_H

#include <chrono>

#include "detail/_namespace_injection.h"

namespace tbb {
namespace detail {
namespace d1 {


//! Absolute timestamp
/** @ingroup timing */
class tick_count {
public:
    using clock_type = typename std::conditional<std::chrono::high_resolution_clock::is_steady,
        std::chrono::high_resolution_clock, std::chrono::steady_clock>::type;

    //! Relative time interval.
    class interval_t : public clock_type::duration {
    public:
        //! Construct a time interval representing zero time duration
        interval_t() : clock_type::duration(clock_type::duration::zero()) {}

        //! Construct a time interval representing sec seconds time duration
        explicit interval_t( double sec )
            : clock_type::duration(std::chrono::duration_cast<clock_type::duration>(std::chrono::duration<double>(sec))) {}

        //! Return the length of a time interval in seconds
        double seconds() const {
            return std::chrono::duration_cast<std::chrono::duration<double>>(*this).count();
        }

        //! Extract the intervals from the tick_counts and subtract them.
        friend interval_t operator-( const tick_count& t1, const tick_count& t0 );

        //! Add two intervals.
        friend interval_t operator+( const interval_t& i, const interval_t& j ) {
            return interval_t(std::chrono::operator+(i, j));
        }

        //! Subtract two intervals.
        friend interval_t operator-( const interval_t& i, const interval_t& j ) {
            return interval_t(std::chrono::operator-(i, j));
        }

    private:
        explicit interval_t( clock_type::duration value_ ) : clock_type::duration(value_) {}
    };

    tick_count() = default;

    //! Return current time.
    static tick_count now() {
        return clock_type::now();
    }

    //! Subtract two timestamps to get the time interval between
    friend interval_t operator-( const tick_count& t1, const tick_count& t0 ) {
        return tick_count::interval_t(t1.my_time_point - t0.my_time_point);
    }

    //! Return the resolution of the clock in seconds per tick.
    static double resolution() {
        return static_cast<double>(interval_t::period::num) / interval_t::period::den;
    }

private:
    clock_type::time_point my_time_point;
    tick_count( clock_type::time_point tp ) : my_time_point(tp) {}
};

} // namespace d1
} // namespace detail

inline namespace v1 {
    using detail::d1::tick_count;
} // namespace v1

} // namespace tbb

#endif /* __TBB_tick_count_H */
