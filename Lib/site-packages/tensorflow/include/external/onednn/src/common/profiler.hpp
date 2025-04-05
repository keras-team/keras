/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
#ifndef COMMON_PROFILER_HPP
#define COMMON_PROFILER_HPP

#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace dnnl {
namespace impl {

static double get_msec() {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    // In case the hardware does not support high-resolution perf counter
    if (frequency.QuadPart == 0) return 0.0;
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return 1e+3 * now.QuadPart / frequency.QuadPart;
#else
    struct timeval time;
    gettimeofday(&time, nullptr);
    return 1e+3 * static_cast<double>(time.tv_sec)
            + 1e-3 * static_cast<double>(time.tv_usec);
#endif
}

// Record custom profiling information within a single thread.
//
// Basic Usage:
//        profiler_t profile("Profile Name"); // Start profiling
//        ...                                 // Some code
//        profile.stamp("Stamp Name 1");      // Record data
//        ...                                 // Some code
//        profile.stamp("Stamp Name 2");      // Record data
//        ...                                 // Some code
//        profile.stop("Stamp Name 3");       // Record data
//        std::cout << profile << "\n";       // Print recorded results
//
// The profiler_t structure can be used over multiple calls to a function, and
// all time stamps will be collated into a total time.
// Alternative Usage:
//        profiler_t profile("Profile Name");
//        function() {
//           profile.start()                     // Start Profiling
//           ...                                 // Some code
//           profile.stamp("Stamp Name 1");      // Record data
//           ...                                 // Some code
//           profile.stamp("Stamp Name 2");      // Record data
//           ...                                 // Some code
//           profile.stop("Stamp Name 3");       // Record data
//        }
//
//        high_level_function() {
//           profile.reset()                     // Clear unwanted profiling data
//           ...                                 // Some code
//           std::cout << profile << "\n";       // Print recorded results
//        }

// Note: To reduce overhead, the stamp names are initially recorded as pointers.
// All pointers need to be valid when stop() is called, at which point stamp
// names are copied into long term storage.

struct profiler_t {
    profiler_t(const std::string &profile_name)
        : _profile_name(profile_name), _run_data(), _data() {
        // Reserve data on construction to reduce chance of recording
        // reallocation
        _run_data.reserve(128);
        start();
    }

    // Start recording timing data
    void start() {
        _run_data.clear();
        _state = RUNNING;
        _start_time = get_msec();
        optimization_barrier();
    }

    // Recording data
    void stamp(const char *name) {
        optimization_barrier();
        _run_data.emplace_back(record_t<const char *>(name, get_msec()));
        assert(_state == RUNNING);
        optimization_barrier();
    }

    void stop(const char *name) {
        optimization_barrier();
        _run_data.emplace_back(record_t<const char *>(name, get_msec()));
        stop();
    }

    void stop() {
        assert(_state == RUNNING);
        _state = STOPPED;
        collate();
    }

    void reset() {
        _data.clear();
        start();
    }

    std::string str() const {
        std::ostringstream oss;
        std::vector<record_t<std::string>> print_data(
                _data.begin(), _data.end());

        std::sort(print_data.begin(), print_data.end());

        prof_time_t total_time = 0;
        for (auto &record : print_data) {
            total_time += record.time;
        }

        const int max_name_width = 20;
        oss << _profile_name << ":\n";
        oss << std::setw(max_name_width) << "Total Time" << std::setw(0)
            << ":          " << std::setw(8) << std::fixed
            << std::setprecision(3) << total_time << std::setw(0) << " ms\n";
        for (const auto &record : print_data) {
            oss << std::setw(max_name_width)
                << record.name.substr(0, max_name_width) << std::setw(0)
                << ": interval " << std::setw(8) << std::fixed
                << std::setprecision(3) << record.time << std::setw(0)
                << " ms, total " << std::setw(8)
                << 100.0 * record.time / total_time << " % recorded time\n";
        }

        return oss.str();
    }

private:
    using prof_time_t = double;

    static void inline optimization_barrier() {
        atomic_signal_fence(std::memory_order_seq_cst);
    }

    template <typename T>
    struct record_t {
        T name;
        prof_time_t time;
        record_t(T name, prof_time_t time) : name(name), time(time) {}
        record_t(std::pair<T, prof_time_t> record)
            : name(record.first), time(record.second) {}
        // Reversed time ordering
        bool operator<(const record_t &b) const { return this->time > b.time; }
    };

    enum state_t {
        RUNNING,
        STOPPED,
    };

    // Record data in a vector to minimize overhead associated with recording
    // data. Data is flushed into a separate long term storage once time
    // recording has stopped.
    std::string _profile_name;
    std::vector<record_t<const char *>> _run_data;
    std::unordered_map<std::string, prof_time_t> _data;

    state_t _state = STOPPED;
    prof_time_t _start_time = 0;

    void collate() {
        assert(_state == STOPPED);
        prof_time_t last_stamp = _start_time;
        for (const auto &record : _run_data) {
            _data[std::string(record.name)] += record.time - last_stamp;
            last_stamp = record.time;
        }
        _run_data.clear();
    }
};

inline std::ostream &operator<<(std::ostream &out, const profiler_t &profile) {
    out << profile.str();
    return out;
}

} // namespace impl
} // namespace dnnl
#endif
