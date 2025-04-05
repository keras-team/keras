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

#ifndef __TBB_task_group_H
#define __TBB_task_group_H

#include "detail/_config.h"
#include "detail/_namespace_injection.h"
#include "detail/_template_helpers.h"
#include "detail/_utils.h"
#include "detail/_exception.h"
#include "detail/_task.h"
#include "detail/_small_object_pool.h"

#include "profiling.h"

#include <functional>

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress warning: structure was padded due to alignment specifier
    #pragma warning(push)
    #pragma warning(disable:4324)
#endif

namespace tbb {
namespace detail {

namespace d1 {
class delegate_base;
class task_arena_base;
class task_group_context;
}

namespace r1 {
// Forward declarations
class tbb_exception_ptr;
class market;
class thread_data;
class task_dispatcher;
template <bool>
class context_guard_helper;
struct task_arena_impl;

void __TBB_EXPORTED_FUNC execute(d1::task_arena_base&, d1::delegate_base&);
void __TBB_EXPORTED_FUNC isolate_within_arena(d1::delegate_base&, std::intptr_t);

void __TBB_EXPORTED_FUNC initialize(d1::task_group_context&);
void __TBB_EXPORTED_FUNC destroy(d1::task_group_context&);
void __TBB_EXPORTED_FUNC reset(d1::task_group_context&);
bool __TBB_EXPORTED_FUNC cancel_group_execution(d1::task_group_context&);
bool __TBB_EXPORTED_FUNC is_group_execution_cancelled(d1::task_group_context&);
void __TBB_EXPORTED_FUNC capture_fp_settings(d1::task_group_context&);

struct task_group_context_impl;
}

namespace d1 {

struct context_list_node {
    std::atomic<context_list_node*> prev{};
    std::atomic<context_list_node*> next{};

    void remove_relaxed() {
        context_list_node* p = prev.load(std::memory_order_relaxed);
        context_list_node* n = next.load(std::memory_order_relaxed);
        p->next.store(n, std::memory_order_relaxed);
        n->prev.store(p, std::memory_order_relaxed);
    }
};

//! Used to form groups of tasks
/** @ingroup task_scheduling
    The context services explicit cancellation requests from user code, and unhandled
    exceptions intercepted during tasks execution. Intercepting an exception results
    in generating internal cancellation requests (which is processed in exactly the
    same way as external ones).

    The context is associated with one or more root tasks and defines the cancellation
    group that includes all the descendants of the corresponding root task(s). Association
    is established when a context object is passed as an argument to the task::allocate_root()
    method. See task_group_context::task_group_context for more details.

    The context can be bound to another one, and other contexts can be bound to it,
    forming a tree-like structure: parent -> this -> children. Arrows here designate
    cancellation propagation direction. If a task in a cancellation group is cancelled
    all the other tasks in this group and groups bound to it (as children) get cancelled too.
**/
class task_group_context : no_copy {
public:
    enum traits_type {
        fp_settings     = 1 << 1,
        concurrent_wait = 1 << 2,
        default_traits  = 0
    };
    enum kind_type {
        isolated,
        bound
    };
private:
    //! Space for platform-specific FPU settings.
    /** Must only be accessed inside TBB binaries, and never directly in user
    code or inline methods. */
    std::uint64_t my_cpu_ctl_env;

    //! Specifies whether cancellation was requested for this task group.
    std::atomic<std::uint32_t> my_cancellation_requested;

    //! Version for run-time checks and behavioral traits of the context.
    std::uint8_t my_version;

    //! The context traits.
    struct context_traits {
        bool fp_settings        : 1;
        bool concurrent_wait    : 1;
        bool bound              : 1;
    } my_traits;

    static_assert(sizeof(context_traits) == 1, "Traits shall fit into one byte.");

    static constexpr std::uint8_t may_have_children = 1;
    //! The context internal state (currently only may_have_children).
    std::atomic<std::uint8_t> my_state;

    enum class lifetime_state : std::uint8_t {
        created,
        locked,
        isolated,
        bound,
        detached,
        dying
    };

    //! The synchronization machine state to manage lifetime.
    std::atomic<lifetime_state> my_lifetime_state;

    //! Pointer to the context of the parent cancellation group. NULL for isolated contexts.
    task_group_context* my_parent;

    //! Thread data instance that registered this context in its list.
    std::atomic<r1::thread_data*> my_owner;

    //! Used to form the thread specific list of contexts without additional memory allocation.
    /** A context is included into the list of the current thread when its binding to
        its parent happens. Any context can be present in the list of one thread only. **/
    context_list_node my_node;

    //! Pointer to the container storing exception being propagated across this task group.
    r1::tbb_exception_ptr* my_exception;

    //! Used to set and maintain stack stitching point for Intel Performance Tools.
    void* my_itt_caller;

    //! Description of algorithm for scheduler based instrumentation.
    string_resource_index my_name;

    char padding[max_nfs_size
        - sizeof(std::uint64_t) // my_cpu_ctl_env
        - sizeof(std::atomic<std::uint32_t>) // my_cancellation_requested
        - sizeof(std::uint8_t) // my_version
        - sizeof(context_traits) // my_traits
        - sizeof(std::atomic<std::uint8_t>) // my_state
        - sizeof(std::atomic<lifetime_state>) // my_lifetime_state
        - sizeof(task_group_context*) // my_parent
        - sizeof(std::atomic<r1::thread_data*>) // my_owner
        - sizeof(context_list_node) // my_node
        - sizeof(r1::tbb_exception_ptr*) // my_exception
        - sizeof(void*) // my_itt_caller
        - sizeof(string_resource_index) // my_name
    ];

    task_group_context(context_traits t, string_resource_index name)
        : my_version{}, my_name{ name } {
        my_traits = t; // GCC4.8 issues warning list initialization for bitset (missing-field-initializers)
        r1::initialize(*this);
    }

    static context_traits make_traits(kind_type relation_with_parent, std::uintptr_t user_traits) {
        context_traits ct;
        ct.bound = relation_with_parent == bound;
        ct.fp_settings = (user_traits & fp_settings) == fp_settings;
        ct.concurrent_wait = (user_traits & concurrent_wait) == concurrent_wait;
        return ct;
    }

public:
    //! Default & binding constructor.
    /** By default a bound context is created. That is this context will be bound
        (as child) to the context of the currently executing task . Cancellation
        requests passed to the parent context are propagated to all the contexts
        bound to it. Similarly priority change is propagated from the parent context
        to its children.

        If task_group_context::isolated is used as the argument, then the tasks associated
        with this context will never be affected by events in any other context.

        Creating isolated contexts involve much less overhead, but they have limited
        utility. Normally when an exception occurs in an algorithm that has nested
        ones running, it is desirably to have all the nested algorithms cancelled
        as well. Such a behavior requires nested algorithms to use bound contexts.

        There is one good place where using isolated algorithms is beneficial. It is
        an external thread. That is if a particular algorithm is invoked directly from
        the external thread (not from a TBB task), supplying it with explicitly
        created isolated context will result in a faster algorithm startup.

        VERSIONING NOTE:
        Implementation(s) of task_group_context constructor(s) cannot be made
        entirely out-of-line because the run-time version must be set by the user
        code. This will become critically important for binary compatibility, if
        we ever have to change the size of the context object. **/

    task_group_context(kind_type relation_with_parent = bound,
                       std::uintptr_t t = default_traits)
        : task_group_context(make_traits(relation_with_parent, t), CUSTOM_CTX) {}

    // Custom constructor for instrumentation of oneTBB algorithm
    task_group_context (string_resource_index name )
        : task_group_context(make_traits(bound, default_traits), name) {}

    // Do not introduce any logic on user side since it might break state propagation assumptions
    ~task_group_context() {
        r1::destroy(*this);
    }

    //! Forcefully reinitializes the context after the task tree it was associated with is completed.
    /** Because the method assumes that all the tasks that used to be associated with
        this context have already finished, calling it while the context is still
        in use somewhere in the task hierarchy leads to undefined behavior.

        IMPORTANT: This method is not thread safe!

        The method does not change the context's parent if it is set. **/
    void reset() {
        r1::reset(*this);
    }

    //! Initiates cancellation of all tasks in this cancellation group and its subordinate groups.
    /** \return false if cancellation has already been requested, true otherwise.

        Note that canceling never fails. When false is returned, it just means that
        another thread (or this one) has already sent cancellation request to this
        context or to one of its ancestors (if this context is bound). It is guaranteed
        that when this method is concurrently called on the same not yet cancelled
        context, true will be returned by one and only one invocation. **/
    bool cancel_group_execution() {
        return r1::cancel_group_execution(*this);
    }

    //! Returns true if the context received cancellation request.
    bool is_group_execution_cancelled() {
        return r1::is_group_execution_cancelled(*this);
    }

#if __TBB_FP_CONTEXT
    //! Captures the current FPU control settings to the context.
    /** Because the method assumes that all the tasks that used to be associated with
        this context have already finished, calling it while the context is still
        in use somewhere in the task hierarchy leads to undefined behavior.

        IMPORTANT: This method is not thread safe!

        The method does not change the FPU control settings of the context's parent. **/
    void capture_fp_settings() {
        r1::capture_fp_settings(*this);
    }
#endif

    //! Returns the user visible context trait
    std::uintptr_t traits() const {
        std::uintptr_t t{};
        t |= my_traits.fp_settings ? fp_settings : 0;
        t |= my_traits.concurrent_wait ? concurrent_wait : 0;
        return t;
    }
private:
    //// TODO: cleanup friends
    friend class r1::market;
    friend class r1::thread_data;
    friend class r1::task_dispatcher;
    template <bool>
    friend class r1::context_guard_helper;
    friend struct r1::task_arena_impl;
    friend struct r1::task_group_context_impl;
}; // class task_group_context

static_assert(sizeof(task_group_context) == 128, "Wrong size of task_group_context");

enum task_group_status {
    not_complete,
    complete,
    canceled
};

class task_group;
class structured_task_group;
#if TBB_PREVIEW_ISOLATED_TASK_GROUP
class isolated_task_group;
#endif

template<typename F>
class function_task : public task {
    const F m_func;
    wait_context& m_wait_ctx;
    small_object_allocator m_allocator;

    void finalize(const execution_data& ed) {
        // Make a local reference not to access this after destruction.
        wait_context& wo = m_wait_ctx;
        // Copy allocator to the stack
        auto allocator = m_allocator;
        // Destroy user functor before release wait.
        this->~function_task();
        wo.release();

        allocator.deallocate(this, ed);
    }
    task* execute(execution_data& ed) override {
        m_func();
        finalize(ed);
        return nullptr;
    }
    task* cancel(execution_data& ed) override {
        finalize(ed);
        return nullptr;
    }
public:
    function_task(const F& f, wait_context& wo, small_object_allocator& alloc)
        : m_func(f)
        , m_wait_ctx(wo)
        , m_allocator(alloc) {}

    function_task(F&& f, wait_context& wo, small_object_allocator& alloc)
        : m_func(std::move(f))
        , m_wait_ctx(wo)
        , m_allocator(alloc) {}
};

template <typename F>
class function_stack_task : public task {
    const F& m_func;
    wait_context& m_wait_ctx;

    void finalize() {
        m_wait_ctx.release();
    }
    task* execute(execution_data&) override {
        m_func();
        finalize();
        return nullptr;
    }
    task* cancel(execution_data&) override {
        finalize();
        return nullptr;
    }
public:
    function_stack_task(const F& f, wait_context& wo) : m_func(f), m_wait_ctx(wo) {}
};

class task_group_base : no_copy {
protected:
    wait_context m_wait_ctx;
    task_group_context m_context;

    template<typename F>
    task_group_status internal_run_and_wait(const F& f) {
        function_stack_task<F> t{ f, m_wait_ctx };
        m_wait_ctx.reserve();
        bool cancellation_status = false;
        try_call([&] {
            execute_and_wait(t, m_context, m_wait_ctx, m_context);
        }).on_completion([&] {
            // TODO: the reset method is not thread-safe. Ensure the correct behavior.
            cancellation_status = m_context.is_group_execution_cancelled();
            m_context.reset();
        });
        return cancellation_status ? canceled : complete;
    }

    template<typename F>
    task* prepare_task(F&& f) {
        m_wait_ctx.reserve();
        small_object_allocator alloc{};
        return alloc.new_object<function_task<typename std::decay<F>::type>>(std::forward<F>(f), m_wait_ctx, alloc);
    }

public:
    task_group_base(uintptr_t traits = 0)
        : m_wait_ctx(0)
        , m_context(task_group_context::bound, task_group_context::default_traits | traits)
    {
    }

    ~task_group_base() noexcept(false) {
        if (m_wait_ctx.continue_execution()) {
#if __TBB_CPP17_UNCAUGHT_EXCEPTIONS_PRESENT
            bool stack_unwinding_in_progress = std::uncaught_exceptions() > 0;
#else
            bool stack_unwinding_in_progress = std::uncaught_exception();
#endif
            // Always attempt to do proper cleanup to avoid inevitable memory corruption
            // in case of missing wait (for the sake of better testability & debuggability)
            if (!m_context.is_group_execution_cancelled())
                cancel();
            d1::wait(m_wait_ctx, m_context);
            if (!stack_unwinding_in_progress)
                throw_exception(exception_id::missing_wait);
        }
    }

    task_group_status wait() {
        bool cancellation_status = false;
        try_call([&] {
            d1::wait(m_wait_ctx, m_context);
        }).on_completion([&] {
            // TODO: the reset method is not thread-safe. Ensure the correct behavior.
            cancellation_status = m_context.is_group_execution_cancelled();
            m_context.reset();
        });
        return cancellation_status ? canceled : complete;
    }

    void cancel() {
        m_context.cancel_group_execution();
    }
}; // class task_group_base

class task_group : public task_group_base {
public:
    task_group() : task_group_base(task_group_context::concurrent_wait) {}

    template<typename F>
    void run(F&& f) {
        spawn(*prepare_task(std::forward<F>(f)), m_context);
    }

    template<typename F>
    task_group_status run_and_wait(const F& f) {
        return internal_run_and_wait(f);
    }
}; // class task_group

#if TBB_PREVIEW_ISOLATED_TASK_GROUP
class spawn_delegate : public delegate_base {
    task* task_to_spawn;
    task_group_context& context;
    bool operator()() const override {
        spawn(*task_to_spawn, context);
        return true;
    }
public:
    spawn_delegate(task* a_task, task_group_context& ctx)
        : task_to_spawn(a_task), context(ctx)
    {}
};

class wait_delegate : public delegate_base {
    bool operator()() const override {
        status = tg.wait();
        return true;
    }
protected:
    task_group& tg;
    task_group_status& status;
public:
    wait_delegate(task_group& a_group, task_group_status& tgs)
        : tg(a_group), status(tgs) {}
};

template<typename F>
class run_wait_delegate : public wait_delegate {
    F& func;
    bool operator()() const override {
        status = tg.run_and_wait(func);
        return true;
    }
public:
    run_wait_delegate(task_group& a_group, F& a_func, task_group_status& tgs)
        : wait_delegate(a_group, tgs), func(a_func) {}
};

class isolated_task_group : public task_group {
    intptr_t this_isolation() {
        return reinterpret_cast<intptr_t>(this);
    }
public:
    isolated_task_group () : task_group() {}

    template<typename F>
    void run(F&& f) {
        spawn_delegate sd(prepare_task(std::forward<F>(f)), m_context);
        r1::isolate_within_arena(sd, this_isolation());
    }

    template<typename F>
    task_group_status run_and_wait( const F& f ) {
        task_group_status result = not_complete;
        run_wait_delegate<const F> rwd(*this, f, result);
        r1::isolate_within_arena(rwd, this_isolation());
        __TBB_ASSERT(result != not_complete, "premature exit from wait?");
        return result;
    }

    task_group_status wait() {
        task_group_status result = not_complete;
        wait_delegate wd(*this, result);
        r1::isolate_within_arena(wd, this_isolation());
        __TBB_ASSERT(result != not_complete, "premature exit from wait?");
        return result;
    }
}; // class isolated_task_group
#endif // TBB_PREVIEW_ISOLATED_TASK_GROUP

inline bool is_current_task_group_canceling() {
    task_group_context* ctx = current_context();
    return ctx ? ctx->is_group_execution_cancelled() : false;
}

} // namespace d1
} // namespace detail

inline namespace v1 {
using detail::d1::task_group_context;
using detail::d1::task_group;
#if TBB_PREVIEW_ISOLATED_TASK_GROUP
using detail::d1::isolated_task_group;
#endif

using detail::d1::task_group_status;
using detail::d1::not_complete;
using detail::d1::complete;
using detail::d1::canceled;

using detail::d1::is_current_task_group_canceling;
using detail::r1::missing_wait;
}

} // namespace tbb

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop) // 4324 warning
#endif

#endif // __TBB_task_group_H
