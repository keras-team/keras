/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#ifndef COMMON_DNNL_THREAD_HPP
#define COMMON_DNNL_THREAD_HPP

#include <algorithm>
#include <functional>
#include <mutex>

#include "utils.hpp"
#include "z_magic.hpp"

// IMPORTANT NOTICE:
// This header is special in the library since it enables all threading
// functionality in the product including tests.
// tests/test_thread.{c,h}pp files rely on this header file by:
// * Substituting `threadpool_utils` namespace to re-use threadpool functions
//   and enable a second threadpool different from the library;
// * Re-defining `DNNL_CPU_THREADING_RUNTIME` macro value when it is supposed
//   to be `DNNL_RUNTIME_SEQ`, e.g., for CPU_NONE configuration.
//      1. It implies all parts of code relying on this macro should stay in the
//         file.
//      2. It implies there are no function bodies in the translation units
//         related to the library. Tests threading layer uses dnnl::impl::func
//         signature, and if library has symbols defined, regardless of
//         redefinition, it will take those that were compiled with original
//         macro value.
//
// Potential drawback could be increased binary size but it doesn't happen much
// due to linker optimizations. The newer compiler and C++ standard, the less
// binary size will be achieved.

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "counting_barrier.hpp"
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "common/ittnotify.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
#define DNNL_THR_SYNC 1
inline int dnnl_get_max_threads() {
    return 1;
}
inline int dnnl_in_parallel() {
    return 0;
}
inline void dnnl_thr_barrier() {}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#include "omp.h"
#define DNNL_THR_SYNC 1
inline int dnnl_get_max_threads() {
    return omp_get_max_threads();
}
inline int dnnl_in_parallel() {
    return omp_in_parallel();
}
inline void dnnl_thr_barrier() {
#pragma omp barrier
}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"
#define DNNL_THR_SYNC 0
inline int dnnl_get_max_threads() {
    return tbb::this_task_arena::max_concurrency();
}
inline int dnnl_in_parallel() {
    return 0;
}
inline void dnnl_thr_barrier() {
    assert(!"no barrier in TBB");
}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include <thread>
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#define DNNL_THR_SYNC 0

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace threadpool_utils {

// Each thread maintains a thread-local pointer to a threadpool which is
// 'active' for the current thread. If this pointer is a nullptr, all the work
// is executed sequentially.

// Sets `tp` to be the active threadpool for the calling thread. This will
// make all calls to `get_active_threadpool()` to return `tp` thus enabling
// `parallel()` and `parallel_nd()` to submit work to `tp`.
void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp);

// Resets the active threadpool for the calling thread to nullptr. After this
// call `parallel()` and `parallel_nd()` would execute work sequentially.
void deactivate_threadpool();

// Returns the active threadpool for the calling thread.
dnnl::threadpool_interop::threadpool_iface *get_active_threadpool();

// returns the maximum concurrency available in the given global context
int get_max_concurrency();

int &get_threadlocal_max_concurrency();

} // namespace threadpool_utils
} // namespace impl
} // namespace dnnl

inline int dnnl_get_max_threads() {
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();

    // This is the maximum number of threads oneDNN will use by default
    int max_concurrency = dnnl::impl::threadpool_utils::get_max_concurrency();

    // Use the default max_concurrency only when no tp is passed by
    // user (e.g. primitive creation).
    return tp ? std::max(1, tp->get_num_threads()) : max_concurrency;
}
inline int dnnl_in_parallel() {
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    return tp ? tp->get_in_parallel() : 0;
}
inline void dnnl_thr_barrier() {
    assert(!"no barrier with THREADPOOL");
}
#endif

/* The purpose of this function is to provide the number of threads the library
 * is aware of when this function is invoked. Since oneDNN does not allow nested
 * parallelism, inside a parallel region the number of available threads is 1.
 * Otherwise, the number of current threads varies between threading runtimes:
 * - for OpenMP and TBB, return the max number of threads since the number of
 *   threads is held in a global object throughout the entire execution.
 * - for Threadpool, since the global object in oneDNN changes throughout
 *   execution, two situations can occur:
 *   a) if the library *is* aware of a threadpool when this function is invoked,
 *   return the number of available threads in the threadpool;
 *   b) if the library *is not* aware of a threadpool when this function is
 *   invoked, return 1 since the main thread will do the work.
 */
inline int dnnl_get_current_num_threads() {
    if (dnnl_in_parallel()) return 1;
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    return omp_get_max_threads();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    return tbb::this_task_arena::max_concurrency();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    return (tp) ? dnnl_get_max_threads() : 1;
#else
    return 1;
#endif
}

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))
#define OMP_GET_THREAD_NUM() omp_get_thread_num()
#define OMP_GET_NUM_THREADS() omp_get_num_threads()
#else
#define PRAGMA_OMP(...)
#define OMP_GET_THREAD_NUM() 0
#define OMP_GET_NUM_THREADS() 1
#endif

// MSVC still supports omp 2.0 only
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#define PRAGMA_OMP_SIMD(...)
#else
#define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)

// process simdlen; it is supported for Clang >= 3.9; ICC >= 17.0; GCC >= 6.1
// No support on Windows.
#if (defined(__clang_major__) \
        && (__clang_major__ < 3 \
                || (__clang_major__ == 3 && __clang_minor__ < 9))) \
        || (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700) \
        || (!defined(__INTEL_COMPILER) && !defined(__clang__) \
                && (defined(_MSC_VER) || __GNUC__ < 6 \
                        || (__GNUC__ == 6 && __GNUC_MINOR__ < 1)))
#define simdlen(x)
#endif // long simdlen if

namespace dnnl {
namespace impl {

inline bool dnnl_thr_syncable() {
    return DNNL_THR_SYNC == 1;
}

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = utils::div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? (T)tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end, T nx, T &nx_start,
        T &nx_end, T nx_divider) {
    const T grp_count = nstl::min(nx_divider, static_cast<T>(nthr));
    const int grp_size_big = nthr / static_cast<int>(grp_count) + 1;
    const int grp_size_small = nthr / static_cast<int>(grp_count);
    const int n_grp_big = nthr % static_cast<int>(grp_count);
    const int threads_in_big_groups = n_grp_big * grp_size_big;

    const int ithr_bound_distance = ithr - threads_in_big_groups;
    T grp, grp_ithr, grp_nthr;
    if (ithr_bound_distance < 0) { // ithr in first groups
        grp = ithr / grp_size_big;
        grp_ithr = ithr % grp_size_big;
        grp_nthr = grp_size_big;
    } else { // ithr in last groups
        grp = n_grp_big + ithr_bound_distance / grp_size_small;
        grp_ithr = ithr_bound_distance % grp_size_small;
        grp_nthr = grp_size_small;
    }

    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}

/* Functions:
 *  - parallel(nthr, f)                  - executes f in parallel using at
 *                                         most nthr threads. If nthr equals
 *                                         0 dnnl_get_current_num_threads() threads
 *                                         is used
 *  - for_nd(ithr, nthr, dims..., f)     - multidimensional for loop for
 *                                         already created threads
 *  - for_nd_ext(ithr, nthr, dims..., f) - multidimensional for loop for
 *                                         already created threads that passes
 *                                         ithr and nthr
 *  - parallel_nd(dims..., f)            - creates a parallel section and then
 *                                         calls for_nd
 *  - parallel_nd_ext(nthr, dims..., f)  - creates a parallel section and then
 *                                         calls for_nd_ext
 */

/* general parallelization */
inline int adjust_num_threads(int nthr, dim_t work_amount) {
    if (nthr == 0) nthr = dnnl_get_current_num_threads();
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    return (work_amount == 1 || omp_in_parallel()) ? 1 : nthr;
#else
    return (int)std::min((dim_t)nthr, work_amount);
#endif
}

static inline void parallel(int nthr, const std::function<void(int, int)> &f) {
    nthr = adjust_num_threads(nthr, INT64_MAX);
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    for (int i = 0; i < nthr; ++i) {
        f(i, nthr);
    }
#else
#if defined(DNNL_ENABLE_ITT_TASKS)
    auto task_primitive_kind = itt::primitive_task_get_current_kind();
    bool itt_enable = itt::get_itt(itt::__itt_task_level_high);
#endif
    if (nthr == 1) {
        f(0, 1);
        return;
    }
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#pragma omp parallel num_threads(nthr)
    {
        int nthr_ = omp_get_num_threads();
        int ithr_ = omp_get_thread_num();
        assert(nthr_ == nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
        if (ithr_ && itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
        f(ithr_, nthr_);
#if defined(DNNL_ENABLE_ITT_TASKS)
        if (ithr_ && itt_enable) itt::primitive_task_end();
#endif
    }
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    tbb::parallel_for(
            0, nthr,
            [&](int ithr) {
#if defined(DNNL_ENABLE_ITT_TASKS)
                bool mark_task = itt::primitive_task_get_current_kind()
                        == primitive_kind::undefined;
                if (mark_task && itt_enable)
                    itt::primitive_task_start(task_primitive_kind);
#endif
                f(ithr, nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (mark_task && itt_enable) itt::primitive_task_end();
#endif
            },
            tbb::static_partitioner());
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    if (!tp || dnnl_in_parallel()) {
        threadpool_utils::deactivate_threadpool();
        for (int ithr = 0; ithr < nthr; ithr++) {
            f(ithr, nthr);
        }
        threadpool_utils::activate_threadpool(tp);
    } else {
        bool async = tp->get_flags()
                & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
        counting_barrier_t b;
        if (async) b.init(nthr);
        tp->parallel_for(nthr, [&, tp](int ithr, int nthr) {
            bool is_master = threadpool_utils::get_active_threadpool() == tp;
            if (!is_master) {
                threadpool_utils::activate_threadpool(tp);
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
            }
            f(ithr, nthr);
            if (!is_master) {
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (itt_enable) itt::primitive_task_end();
#endif
                threadpool_utils::deactivate_threadpool();
            }
            if (async) b.notify();
        });
        if (async) b.wait();
    }
#endif
#endif
}

// XXX: IMPORTANT!!!
// Keep the functions below static.
//
// The threading file is included in gtests and benchdnn. When
// the functions are not static it can cause a crash in gtests and
// benchdnn on macOS with Intel 2021 compiler.

/* for_nd section */
static inline void for_nd(const int ithr, const int nthr, dim_t D0,
        const std::function<void(dim_t)> &f) {
    dim_t start {0}, end {0};
    balance211(D0, nthr, ithr, start, end);
    for (dim_t d0 = start; d0 < end; ++d0)
        f(d0);
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
        const std::function<void(dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
        dim_t D2, const std::function<void(dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
        dim_t D2, dim_t D3,
        const std::function<void(dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
        dim_t D2, dim_t D3, dim_t D4,
        const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
        dim_t D2, dim_t D3, dim_t D4, dim_t D5,
        const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)>
                &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0}, d5 {0};
    utils::nd_iterator_init(
            start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

/* for_nd_ext section */
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
        const std::function<void(int, int, dim_t)> &f) {
    dim_t start {0}, end {0};
    balance211(D0, nthr, ithr, start, end);
    for (dim_t d0 = start; d0 < end; ++d0)
        f(ithr, nthr, d0);
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
        dim_t D1, const std::function<void(int, int, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
        dim_t D1, dim_t D2,
        const std::function<void(int, int, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
        dim_t D1, dim_t D2, dim_t D3,
        const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
        dim_t D1, dim_t D2, dim_t D3, dim_t D4,
        const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t, dim_t)>
                &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
        dim_t D1, dim_t D2, dim_t D3, dim_t D4, dim_t D5,
        const std::function<void(
                int, int, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0}, d5 {0};
    utils::nd_iterator_init(
            start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

/* parallel_nd_ext section */
static inline void parallel_nd_ext(
        int nthr, dim_t D0, const std::function<void(int, int, dim_t)> &f) {
    const dim_t work_amount = D0;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, f); });
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1,
        const std::function<void(int, int, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, D1, f); });
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
        const std::function<void(int, int, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, f);
        });
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3,
        const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, f);
        });
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, dim_t D4,
        const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t, dim_t)>
                &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, f);
        });
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, dim_t D4, dim_t D5,
        const std::function<void(
                int, int, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
        });
}

/* parallel_nd section */
static inline void parallel_nd(dim_t D0, const std::function<void(dim_t)> &f) {
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), D0);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, f); });
}
static inline void parallel_nd(
        dim_t D0, dim_t D1, const std::function<void(dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, f); });
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2,
        const std::function<void(dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, D2, f); });
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3,
        const std::function<void(dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, f);
        });
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4,
        const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, D4, f);
        });
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4,
        dim_t D5,
        const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)>
                &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
        });
}

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
