/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#if defined(DNNL_ENABLE_STACK_CHECKER)

#ifndef __linux__
#error "Stack checker is supported only on Linux"
#endif

#ifndef DNNL_ENABLE_CONCURRENT_EXEC
#error "Stack checker requires using concurrent scratchpad"
#endif

#ifndef COMMON_STACK_CHECKER_HPP
#define COMMON_STACK_CHECKER_HPP

#include <cassert>
#include <tuple>
#include <type_traits>

#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>

#include "common/cpp_compat.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace stack_checker {

/* Stack checker
 *
 * The purpose of the stack checker is to get information about stack
 * consumption per call stack.
 *
 * Motivation for introducing such a capability was excessive stack consumption
 * for `dnnl_primitive_create`, `dnnl_primitive_execute` and GEMM APIs that
 * resulted in a crash on the customer side.
 *
 * The stack checker is represented as `stack_checker_t` class. The class
 * provides an interface called `check(...)` that is used to get the information
 * about stack consumption.
 * The stack checker has a capability to issue an error when the obtained
 * stack consumption exceeds a specified limit.
 *
 * The stack checker can be configured with the following environment variables:
 * - DNNL_SC_STACK_SIZE: specifies the size of the stack in bytes for the thread
 *   that runs a function that needs to be checked.
 *   The default is 8388608 bytes (8 MiB).
 *
 * - DNNL_SC_SOFT_STACK_LIMIT: specifies a soft limit in memory pages. When
 *   stack consumption exceeds the limit the stack checker prints an error
 *   message that contains the obtained stack consumption. The default is 5
 *   pages (20480 bytes).
 *
 * - DNNL_SC_HARD_STACK_LIMIT: specifies a hard limit in memory pages. When
 *   the limit is exceeded the SIGSEGV signal is raised. This can be used for
 *   debug purposes. For example, it can be used to get a place within the call
 *   stack where the limit is exceeded. By default, the limit is equal to the
 *   `stack size` / `page size` - all memory is available.
 *   for debug purposes.
 *
 * - DNNL_SC_TRACE: enables tracing. If the soft limit is exceeded and the
 *   tracing is enabled the stack checker prints an error message. The tracing
 *   is enabled by default.
 *
 * The `stack_checker_t` class has one constructor that takes an `std::string`
 * which is printed out as part of the error message when soft limit is
 * exceeded. This can be useful to give a context about the function that is
 * being checked.
 *
 * Implementation details
 *
 * The stack checker populates started thread with a particular pattern before
 * calling function to be checked.  Once the thread completed execution of the
 * function being checked, it sweeps the stack for the pattern to compute how
 * much stack memory was actually used.
 *
 * The stack checker is disabled in the default build configuration. It can
 * be enabled via CMake option `DNNL_ENABLE_STACK_CHECKER=ON` at the build time.
 *
 * Usage example
 *
 * ```cpp
 * #include "common/stack_checker.hpp"
 *
 * void bar() {
 *     volatile char arr[1024] = {};
 * }
 *
 * int foo(int *a, int &b, int c) {
 *     bar();
 *     return 0;
 * }
 *
 * int main() {
 *    int x = 5;
 *    stack_checker::stack_checker_t sc("main");
 *    return sc.check(foo, &x, std::ref(x), x);
 * }
 * ```
 * If the soft limit is 3 pages then the output of this code will be the
 * following:
 *  === Stack checker: ERROR: 'main' consumed 14824 bytes of stack while the limit is 12288 bytes. ===
 *
 * Limitations:
 *  - There is only Linux support
 *  - The functions being checked should be non-member functions
 *  - Works only with the concurrent scratchpad because the global scratchpad is
 *    global per thread (thread local).
 */

template <typename F, typename... Targs>
struct thread_args_t {
    thread_args_t() = delete;
    thread_args_t(const F &func, const Targs &...func_args)
        : func(func)
        , func_args(std::forward<Targs>(func_args)...)
        , func_retval {} {}
    const F &func;
    std::tuple<Targs...> func_args;
    typename cpp_compat::invoke_result<F *, Targs...>::type func_retval;
};

template <typename T>
constexpr size_t get_number_args() {
    return std::tuple_size<typename std::remove_reference<T>::type> {};
}

// The executor_t is a helper class that is used to prepare arguments for
// the function and call it.
template <size_t i>
struct executor_t {
    template <typename T, typename... Targs>
    static void execute(T &thread_args, Targs &...unpacked_func_args) {
        const auto &func_args = thread_args.func_args;
        constexpr size_t idx = get_number_args<decltype(func_args)>() - i;
        executor_t<i - 1>::execute(thread_args,
                std::forward<Targs>(unpacked_func_args)...,
                std::get<idx>(func_args));
    }
};

template <>
struct executor_t<0> {
    template <typename T, typename... Targs>
    static void execute(T &thread_args, Targs &...unpacked_func_args) {
        thread_args.func_retval
                = thread_args.func(std::forward<Targs>(unpacked_func_args)...);
    }
};

struct stack_checker_t {
    stack_checker_t(const std::string &context) : context_(context) {}

    template <typename F, typename... Targs>
    typename cpp_compat::invoke_result<F *, Targs...>::type check(
            const F &func, const Targs &...func_args) {

        auto thread_args = utils::make_unique<thread_args_t<F, const Targs...>>(
                func, std::forward<const Targs>(func_args)...);

        pthread_t thread;
        pthread_attr_t attr;
        int res = pthread_attr_init(&attr);
        assert(res == 0);

        // Use full stack size with no guard area as there seems to be some
        // variation in pthreads implementation of guard area. Instead, call
        // mprotect later on to guard an area within the stack.
        res = pthread_attr_setstacksize(&attr, get_stack_size());
        assert(res == 0);

        res = pthread_attr_setguardsize(&attr, 0);
        assert(res == 0);

        res = pthread_create(
                &thread, &attr, worker<F, Targs...>, (void *)thread_args.get());
        assert(res == 0);

        void *stack_consumption_ptr = nullptr;
        res = pthread_join(thread, &stack_consumption_ptr);
        assert(res == 0);

        auto stack_consumption
                = reinterpret_cast<size_t>(stack_consumption_ptr);

        if (is_trace_enabled()) {
            size_t soft_stack_limit_in_bytes
                    = get_soft_stack_limit() * get_page_size();
            if (stack_consumption > soft_stack_limit_in_bytes) {
                VERROR(common, stack_checker,
                        "'%s' consumed %lu bytes of "
                        "stack while the limit is %lu bytes",
                        context_.c_str(), stack_consumption,
                        soft_stack_limit_in_bytes);
            }
        }

        res = pthread_attr_destroy(&attr);
        assert(res == 0);
        MAYBE_UNUSED(res);

        return thread_args->func_retval;
    }

private:
    std::string context_;
    static constexpr int8_t pattern_ = INT8_MAX;

    // The worker function is a wrapper for the function being checked.
    // The worker starts when a new thread is created.
    template <typename F, typename... Types>
    static void *worker(void *args) {
        auto &thread_args
                = *reinterpret_cast<thread_args_t<F, Types...> *>(args);
        constexpr size_t n_args
                = get_number_args<decltype(thread_args.func_args)>();

        pthread_attr_t attr;
        int res = pthread_getattr_np(pthread_self(), &attr);
        assert(res == 0);

        void *stack_base;
        size_t stack_size;
        res = pthread_attr_getstack(&attr, &stack_base, &stack_size);
        assert(res == 0);

        size_t protected_region
                = get_stack_size() - get_page_size() * get_hard_stack_limit();

        // Stack grows downwards, so protected region is at the bottom.
        mprotect(stack_base, protected_region, PROT_NONE);

        // Only write _above_ the protected region to avoid segfault.
        write_pattern(
                static_cast<int8_t *>(stack_base) + protected_region, pattern_);

        executor_t<n_args>::execute(thread_args);

        res = pthread_attr_destroy(&attr);
        assert(res == 0);
        MAYBE_UNUSED(res);

        // Only check _above_ the protected region to avoid segfault.
        size_t stack_consumption = 0;
        for (size_t i = protected_region; i < stack_size; i++) {
            if (((const int8_t *)stack_base)[i] != pattern_) {
                stack_consumption = stack_size - i;
                break;
            }
        }
        // OS can reserve a space of size up to 4096 (page size) in the
        // beginning of stack buffer. We shouldn't take the reserved space into
        // account when calculating stack consumption.
        if (stack_consumption >= get_page_size())
            stack_consumption -= get_page_size();
        return reinterpret_cast<void *>(stack_consumption);
    }

    static size_t get_stack_size() {
        static const size_t stack_size
                = getenv_int_user("SC_STACK_SIZE", 1024 * 1024 * 8);
        if (stack_size % get_page_size() != 0) {
            VERROR(common, stack_checker,
                    "DNNL_SC_STACK_SIZE is expected to be "
                    "multiple of page size (%lu)",
                    get_page_size());
            std::terminate();
        }
        return stack_size;
    }

    static size_t get_hard_stack_limit() {
        static const size_t hard_stack_limit = getenv_int_user(
                "SC_HARD_STACK_LIMIT", get_stack_size() / get_page_size());
        return hard_stack_limit;
    }

    static size_t get_soft_stack_limit() {
        // Set up the default limit of 5 pages (20480 bytes).
        static const size_t soft_stack_limit
                = getenv_int_user("SC_SOFT_STACK_LIMIT", 5);
        return soft_stack_limit;
    }

    static bool is_trace_enabled() {
        static const bool is_trace_enabled = getenv_int_user("SC_TRACE", 1);
        return is_trace_enabled;
    }

    static size_t get_page_size() {
        static const size_t page_size = ::getpagesize();
        return page_size;
    }

#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

    // Computes frame size of its caller.
    static NOINLINE size_t get_frame_size(int8_t *base_addr) {
        volatile int8_t rough_stack_top = 0;
        MAYBE_UNUSED(rough_stack_top);
        assert(base_addr > &rough_stack_top);

#ifdef __GNUC__
        return base_addr - (int8_t *)__builtin_frame_address(0);
#else
        return base_addr - &rough_stack_top;
#endif
    }

    // This function writes on its own stack.
    static NOINLINE void write_pattern(int8_t *stack_base, int8_t pattern) {
        volatile int8_t rough_stack_top = 0;

        int8_t *base_addr = nullptr;
#ifdef __GNUC__
        base_addr = (int8_t *)__builtin_frame_address(0);
#else
        base_addr = (int8_t *)&rough_stack_top;
#endif
        size_t frame_sz = get_frame_size(base_addr);

        // Write pattern without overwriting its locals variables on the stack.
        // NOTE: To use memset, one would have to account for the frame size of
        // memset.
        int8_t *p = stack_base;
        while (p + frame_sz < &rough_stack_top) {
            *p = pattern;
            p++;
        }
    }

#undef NOINLINE
};

} // namespace stack_checker
} // namespace impl
} // namespace dnnl

#endif
#endif
