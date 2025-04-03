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

#ifndef __TBB_parallel_invoke_H
#define __TBB_parallel_invoke_H

#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "detail/_exception.h"
#include "detail/_task.h"
#include "detail/_template_helpers.h"
#include "detail/_small_object_pool.h"

#include "task_group.h"

#include <tuple>
#include <atomic>
#include <utility>

namespace tbb {
namespace detail {
namespace d1 {

//! Simple task object, executing user method
template<typename Function, typename WaitObject>
struct function_invoker : public task {
    function_invoker(const Function& function, WaitObject& wait_ctx) :
        my_function(function),
        parent_wait_ctx(wait_ctx)
    {}

    task* execute(execution_data& ed) override {
        my_function();
        parent_wait_ctx.release(ed);
        call_itt_task_notify(destroy, this);
        return nullptr;
    }

    task* cancel(execution_data& ed) override {
        parent_wait_ctx.release(ed);
        return nullptr;
    }

    const Function& my_function;
    WaitObject& parent_wait_ctx;
}; // struct function_invoker

//! Task object for managing subroots in trinary task trees.
// Endowed with additional synchronization logic (compatible with wait object intefaces) to support
// continuation passing execution. This task spawns 2 function_invoker tasks with first and second functors
// and then executes first functor by itself. But only the last executed functor must destruct and deallocate
// the subroot task.
template<typename F1, typename F2, typename F3>
struct invoke_subroot_task : public task {
    wait_context& root_wait_ctx;
    std::atomic<unsigned> ref_count{0};
    bool child_spawned = false;

    const F1& self_invoked_functor;
    function_invoker<F2, invoke_subroot_task<F1, F2, F3>> f2_invoker;
    function_invoker<F3, invoke_subroot_task<F1, F2, F3>> f3_invoker;

    task_group_context& my_execution_context;
    small_object_allocator my_allocator;

    invoke_subroot_task(const F1& f1, const F2& f2, const F3& f3, wait_context& wait_ctx, task_group_context& context,
                 small_object_allocator& alloc) :
        root_wait_ctx(wait_ctx),
        self_invoked_functor(f1),
        f2_invoker(f2, *this),
        f3_invoker(f3, *this),
        my_execution_context(context),
        my_allocator(alloc)
    {
        root_wait_ctx.reserve();
    }

    void finalize(const execution_data& ed) {
        root_wait_ctx.release();

        my_allocator.delete_object(this, ed);
    }

    void release(const execution_data& ed) {
        __TBB_ASSERT(ref_count > 0, nullptr);
        call_itt_task_notify(releasing, this);
        if( --ref_count == 0 ) {
            call_itt_task_notify(acquired, this);
            finalize(ed);
        }
    }

    task* execute(execution_data& ed) override {
        ref_count.fetch_add(3, std::memory_order_relaxed);
        spawn(f3_invoker, my_execution_context);
        spawn(f2_invoker, my_execution_context);
        self_invoked_functor();

        release(ed);
        return nullptr;
    }

    task* cancel(execution_data& ed) override {
        if( ref_count > 0 ) { // detect children spawn
            release(ed);
        } else {
            finalize(ed);
        }
        return nullptr;
    }
}; // struct subroot_task

class invoke_root_task {
public:
    invoke_root_task(wait_context& wc) : my_wait_context(wc) {}
    void release(const execution_data&) {
        my_wait_context.release();
    }
private:
    wait_context& my_wait_context;
};

template<typename F1>
void invoke_recursive_separation(wait_context& root_wait_ctx, task_group_context& context, const F1& f1) {
    root_wait_ctx.reserve(1);
    invoke_root_task root(root_wait_ctx);
    function_invoker<F1, invoke_root_task> invoker1(f1, root);

    execute_and_wait(invoker1, context, root_wait_ctx, context);
}

template<typename F1, typename F2>
void invoke_recursive_separation(wait_context& root_wait_ctx, task_group_context& context, const F1& f1, const F2& f2) {
    root_wait_ctx.reserve(2);
    invoke_root_task root(root_wait_ctx);
    function_invoker<F1, invoke_root_task> invoker1(f1, root);
    function_invoker<F2, invoke_root_task> invoker2(f2, root);

    spawn(invoker1, context);
    execute_and_wait(invoker2, context, root_wait_ctx, context);
}

template<typename F1, typename F2, typename F3>
void invoke_recursive_separation(wait_context& root_wait_ctx, task_group_context& context, const F1& f1, const F2& f2, const F3& f3) {
    root_wait_ctx.reserve(3);
    invoke_root_task root(root_wait_ctx);
    function_invoker<F1, invoke_root_task> invoker1(f1, root);
    function_invoker<F2, invoke_root_task> invoker2(f2, root);
    function_invoker<F3, invoke_root_task> invoker3(f3, root);

    //TODO: implement sub root for two tasks (measure performance)
    spawn(invoker1, context);
    spawn(invoker2, context);
    execute_and_wait(invoker3, context, root_wait_ctx, context);
}

template<typename F1, typename F2, typename F3, typename... Fs>
void invoke_recursive_separation(wait_context& root_wait_ctx, task_group_context& context,
                                 const F1& f1, const F2& f2, const F3& f3, const Fs&... fs) {
    small_object_allocator alloc{};
    auto sub_root = alloc.new_object<invoke_subroot_task<F1, F2, F3>>(f1, f2, f3, root_wait_ctx, context, alloc);
    spawn(*sub_root, context);

    invoke_recursive_separation(root_wait_ctx, context, fs...);
}

template<typename... Fs>
void parallel_invoke_impl(task_group_context& context, const Fs&... fs) {
    static_assert(sizeof...(Fs) >= 2, "Parallel invoke may be called with at least two callable");
    wait_context root_wait_ctx{0};

    invoke_recursive_separation(root_wait_ctx, context, fs...);
}

template<typename F1, typename... Fs>
void parallel_invoke_impl(const F1& f1, const Fs&... fs) {
    static_assert(sizeof...(Fs) >= 1, "Parallel invoke may be called with at least two callable");
    task_group_context context(PARALLEL_INVOKE);
    wait_context root_wait_ctx{0};

    invoke_recursive_separation(root_wait_ctx, context, fs..., f1);
}

//! Passes last argument of variadic pack as first for handling user provided task_group_context
template <typename Tuple, typename... Fs>
struct invoke_helper;

template <typename... Args, typename T, typename... Fs>
struct invoke_helper<std::tuple<Args...>, T, Fs...> : invoke_helper<std::tuple<Args..., T>, Fs...> {};

template <typename... Fs, typename T/*task_group_context or callable*/>
struct invoke_helper<std::tuple<Fs...>, T> {
    void operator()(Fs&&... args, T&& t) {
        parallel_invoke_impl(std::forward<T>(t), std::forward<Fs>(args)...);
    }
};

//! Parallel execution of several function objects
// We need to pass parameter pack through forwarding reference,
// since this pack may contain task_group_context that must be passed via lvalue non-const reference
template<typename... Fs>
void parallel_invoke(Fs&&... fs) {
    invoke_helper<std::tuple<>, Fs...>()(std::forward<Fs>(fs)...);
}

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::parallel_invoke;
} // namespace v1

} // namespace tbb
#endif /* __TBB_parallel_invoke_H */
