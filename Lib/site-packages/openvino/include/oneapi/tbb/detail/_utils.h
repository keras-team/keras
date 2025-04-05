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

#ifndef __TBB_detail__utils_H
#define __TBB_detail__utils_H

#include <type_traits>
#include <cstdint>
#include <atomic>

#include "_config.h"
#include "_assert.h"
#include "_machine.h"

namespace tbb {
namespace detail {
inline namespace d0 {

//! Utility template function to prevent "unused" warnings by various compilers.
template<typename... T> void suppress_unused_warning(T&&...) {}

//! Compile-time constant that is upper bound on cache line/sector size.
/** It should be used only in situations where having a compile-time upper
  bound is more useful than a run-time exact answer.
  @ingroup memory_allocation */
constexpr size_t max_nfs_size = 128;

//! Class that implements exponential backoff.
class atomic_backoff {
    //! Time delay, in units of "pause" instructions.
    /** Should be equal to approximately the number of "pause" instructions
      that take the same time as an context switch. Must be a power of two.*/
    static constexpr std::int32_t LOOPS_BEFORE_YIELD = 16;
    std::int32_t count;

public:
    // In many cases, an object of this type is initialized eagerly on hot path,
    // as in for(atomic_backoff b; ; b.pause()) { /*loop body*/ }
    // For this reason, the construction cost must be very small!
    atomic_backoff() : count(1) {}
    // This constructor pauses immediately; do not use on hot paths!
    atomic_backoff(bool) : count(1) { pause(); }

    //! No Copy
    atomic_backoff(const atomic_backoff&) = delete;
    atomic_backoff& operator=(const atomic_backoff&) = delete;

    //! Pause for a while.
    void pause() {
        if (count <= LOOPS_BEFORE_YIELD) {
            machine_pause(count);
            // Pause twice as long the next time.
            count *= 2;
        } else {
            // Pause is so long that we might as well yield CPU to scheduler.
            yield();
        }
    }

    //! Pause for a few times and return false if saturated.
    bool bounded_pause() {
        machine_pause(count);
        if (count < LOOPS_BEFORE_YIELD) {
            // Pause twice as long the next time.
            count *= 2;
            return true;
        } else {
            return false;
        }
    }

    void reset() {
        count = 1;
    }
};

//! Spin WHILE the condition is true.
/** T and U should be comparable types. */
template <typename T, typename C>
void spin_wait_while_condition(const std::atomic<T>& location, C comp) {
    atomic_backoff backoff;
    while (comp(location.load(std::memory_order_acquire))) {
        backoff.pause();
    }
}

//! Spin WHILE the value of the variable is equal to a given value
/** T and U should be comparable types. */
template <typename T, typename U>
void spin_wait_while_eq(const std::atomic<T>& location, const U value) {
    spin_wait_while_condition(location, [&value](T t) { return t == value; });
}

//! Spin UNTIL the value of the variable is equal to a given value
/** T and U should be comparable types. */
template<typename T, typename U>
void spin_wait_until_eq(const std::atomic<T>& location, const U value) {
    spin_wait_while_condition(location, [&value](T t) { return t != value; });
}

template <typename T>
std::uintptr_t log2(T in) {
    __TBB_ASSERT(in > 0, "The logarithm of a non-positive value is undefined.");
    return machine_log2(in);
}

template<typename T>
T reverse_bits(T src) {
    return machine_reverse_bits(src);
}

template<typename T>
T reverse_n_bits(T src, std::size_t n) {
    __TBB_ASSERT(n != 0, "Reverse for 0 bits is undefined behavior.");
    return reverse_bits(src) >> (number_of_bits<T>() - n);
}

// A function to check if passed integer is a power of two
template <typename IntegerType>
constexpr bool is_power_of_two( IntegerType arg ) {
    static_assert(std::is_integral<IntegerType>::value,
                  "An argument for is_power_of_two should be integral type");
    return arg && (0 == (arg & (arg - 1)));
}

// A function to determine if passed integer is a power of two
// at least as big as another power of two, i.e. for strictly positive i and j,
// with j being a power of two, determines whether i==j<<k for some nonnegative k
template <typename ArgIntegerType, typename DivisorIntegerType>
constexpr bool is_power_of_two_at_least(ArgIntegerType arg, DivisorIntegerType divisor) {
    // Divisor should be a power of two
    static_assert(std::is_integral<ArgIntegerType>::value,
                  "An argument for is_power_of_two_at_least should be integral type");
    return 0 == (arg & (arg - divisor));
}

// A function to compute arg modulo divisor where divisor is a power of 2.
template<typename ArgIntegerType, typename DivisorIntegerType>
inline ArgIntegerType modulo_power_of_two(ArgIntegerType arg, DivisorIntegerType divisor) {
    __TBB_ASSERT( is_power_of_two(divisor), "Divisor should be a power of two" );
    return arg & (divisor - 1);
}

//! A function to check if passed in pointer is aligned on a specific border
template<typename T>
constexpr bool is_aligned(T* pointer, std::uintptr_t alignment) {
    return 0 == ((std::uintptr_t)pointer & (alignment - 1));
}

#if TBB_USE_ASSERT
static void* const poisoned_ptr = reinterpret_cast<void*>(-1);

//! Set p to invalid pointer value.
template<typename T>
inline void poison_pointer( T* &p ) { p = reinterpret_cast<T*>(poisoned_ptr); }

template<typename T>
inline void poison_pointer(std::atomic<T*>& p) { p.store(reinterpret_cast<T*>(poisoned_ptr), std::memory_order_relaxed); }

/** Expected to be used in assertions only, thus no empty form is defined. **/
template<typename T>
inline bool is_poisoned( T* p ) { return p == reinterpret_cast<T*>(poisoned_ptr); }

template<typename T>
inline bool is_poisoned(const std::atomic<T*>& p) { return is_poisoned(p.load(std::memory_order_relaxed)); }
#else
template<typename T>
inline void poison_pointer(T* &) {/*do nothing*/}

template<typename T>
inline void poison_pointer(std::atomic<T*>&) { /* do nothing */}
#endif /* !TBB_USE_ASSERT */

template <std::size_t alignment = 0, typename T>
bool assert_pointer_valid(T* p, const char* comment = nullptr) {
    suppress_unused_warning(p, comment);
    __TBB_ASSERT(p != nullptr, comment);
    __TBB_ASSERT(!is_poisoned(p), comment);
#if !(_MSC_VER && _MSC_VER <= 1900 && !__INTEL_COMPILER)
    __TBB_ASSERT(is_aligned(p, alignment == 0 ? alignof(T) : alignment), comment);
#endif
    // Returns something to simplify assert_pointers_valid implementation.
    return true;
}

template <typename... Args>
void assert_pointers_valid(Args*... p) {
    // suppress_unused_warning is used as an evaluation context for the variadic pack.
    suppress_unused_warning(assert_pointer_valid(p)...);
}

//! Base class for types that should not be assigned.
class no_assign {
public:
    void operator=(const no_assign&) = delete;
    no_assign(const no_assign&) = default;
    no_assign() = default;
};

//! Base class for types that should not be copied or assigned.
class no_copy: no_assign {
public:
    no_copy(const no_copy&) = delete;
    no_copy() = default;
};

template <typename T>
void swap_atomics_relaxed(std::atomic<T>& lhs, std::atomic<T>& rhs){
    T tmp = lhs.load(std::memory_order_relaxed);
    lhs.store(rhs.load(std::memory_order_relaxed), std::memory_order_relaxed);
    rhs.store(tmp, std::memory_order_relaxed);
}

//! One-time initialization states
enum class do_once_state {
    uninitialized = 0,      ///< No execution attempts have been undertaken yet
    pending,                ///< A thread is executing associated do-once routine
    executed,               ///< Do-once routine has been executed
    initialized = executed  ///< Convenience alias
};

//! One-time initialization function
/** /param initializer Pointer to function without arguments
           The variant that returns bool is used for cases when initialization can fail
           and it is OK to continue execution, but the state should be reset so that
           the initialization attempt was repeated the next time.
    /param state Shared state associated with initializer that specifies its
            initialization state. Must be initially set to #uninitialized value
            (e.g. by means of default static zero initialization). **/
template <typename F>
void atomic_do_once( const F& initializer, std::atomic<do_once_state>& state ) {
    // The loop in the implementation is necessary to avoid race when thread T2
    // that arrived in the middle of initialization attempt by another thread T1
    // has just made initialization possible.
    // In such a case T2 has to rely on T1 to initialize, but T1 may already be past
    // the point where it can recognize the changed conditions.
    do_once_state expected_state;
    while ( state.load( std::memory_order_acquire ) != do_once_state::executed ) {
        if( state.load( std::memory_order_relaxed ) == do_once_state::uninitialized ) {
            expected_state = do_once_state::uninitialized;
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1910
            using enum_type = typename std::underlying_type<do_once_state>::type;
            if( ((std::atomic<enum_type>&)state).compare_exchange_strong( (enum_type&)expected_state, (enum_type)do_once_state::pending ) ) {
#else
            if( state.compare_exchange_strong( expected_state, do_once_state::pending ) ) {
#endif
                run_initializer( initializer, state );
                break;
            }
        }
        spin_wait_while_eq( state, do_once_state::pending );
    }
}

// Run the initializer which can not fail
template<typename Functor>
void run_initializer(const Functor& f, std::atomic<do_once_state>& state ) {
    f();
    state.store(do_once_state::executed, std::memory_order_release);
}

#if __TBB_CPP20_CONCEPTS_PRESENT
template <typename T>
concept boolean_testable_impl = std::convertible_to<T, bool>;

template <typename T>
concept boolean_testable = boolean_testable_impl<T> && requires( T&& t ) {
                               { !std::forward<T>(t) } -> boolean_testable_impl;
                           };

#if __TBB_CPP20_COMPARISONS_PRESENT
struct synthesized_three_way_comparator {
    template <typename T1, typename T2>
    auto operator()( const T1& lhs, const T2& rhs ) const
        requires requires {
            { lhs < rhs } -> boolean_testable;
            { rhs < lhs } -> boolean_testable;
        }
    {
        if constexpr (std::three_way_comparable_with<T1, T2>) {
            return lhs <=> rhs;
        } else {
            if (lhs < rhs) {
                return std::weak_ordering::less;
            }
            if (rhs < lhs) {
                return std::weak_ordering::greater;
            }
            return std::weak_ordering::equivalent;
        }
    }
}; // struct synthesized_three_way_comparator

template <typename T1, typename T2 = T1>
using synthesized_three_way_result = decltype(synthesized_three_way_comparator{}(std::declval<T1&>(),
                                                                                 std::declval<T2&>()));

#endif // __TBB_CPP20_COMPARISONS_PRESENT
#endif // __TBB_CPP20_CONCEPTS_PRESENT

} // namespace d0

namespace d1 {

class delegate_base {
public:
    virtual bool operator()() const = 0;
    virtual ~delegate_base() {}
}; // class delegate_base

}  // namespace d1

} // namespace detail
} // namespace tbb

#endif // __TBB_detail__utils_H
