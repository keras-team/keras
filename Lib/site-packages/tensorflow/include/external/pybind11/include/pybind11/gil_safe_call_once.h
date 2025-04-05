// Copyright (c) 2023 The pybind Community.

#pragma once

#include "detail/common.h"
#include "gil.h"

#include <cassert>
#include <mutex>

#ifdef Py_GIL_DISABLED
#    include <atomic>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// Use the `gil_safe_call_once_and_store` class below instead of the naive
//
//   static auto imported_obj = py::module_::import("module_name"); // BAD, DO NOT USE!
//
// which has two serious issues:
//
//     1. Py_DECREF() calls potentially after the Python interpreter was finalized already, and
//     2. deadlocks in multi-threaded processes (because of missing lock ordering).
//
// The following alternative avoids both problems:
//
//   PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
//   auto &imported_obj = storage // Do NOT make this `static`!
//       .call_once_and_store_result([]() {
//           return py::module_::import("module_name");
//       })
//       .get_stored();
//
// The parameter of `call_once_and_store_result()` must be callable. It can make
// CPython API calls, and in particular, it can temporarily release the GIL.
//
// `T` can be any C++ type, it does not have to involve CPython API types.
//
// The behavior with regard to signals, e.g. `SIGINT` (`KeyboardInterrupt`),
// is not ideal. If the main thread is the one to actually run the `Callable`,
// then a `KeyboardInterrupt` will interrupt it if it is running normal Python
// code. The situation is different if a non-main thread runs the
// `Callable`, and then the main thread starts waiting for it to complete:
// a `KeyboardInterrupt` will not interrupt the non-main thread, but it will
// get processed only when it is the main thread's turn again and it is running
// normal Python code. However, this will be unnoticeable for quick call-once
// functions, which is usually the case.
template <typename T>
class gil_safe_call_once_and_store {
public:
    // PRECONDITION: The GIL must be held when `call_once_and_store_result()` is called.
    template <typename Callable>
    gil_safe_call_once_and_store &call_once_and_store_result(Callable &&fn) {
        if (!is_initialized_) { // This read is guarded by the GIL.
            // Multiple threads may enter here, because the GIL is released in the next line and
            // CPython API calls in the `fn()` call below may release and reacquire the GIL.
            gil_scoped_release gil_rel; // Needed to establish lock ordering.
            std::call_once(once_flag_, [&] {
                // Only one thread will ever enter here.
                gil_scoped_acquire gil_acq;
                ::new (storage_) T(fn()); // fn may release, but will reacquire, the GIL.
                is_initialized_ = true;   // This write is guarded by the GIL.
            });
            // All threads will observe `is_initialized_` as true here.
        }
        // Intentionally not returning `T &` to ensure the calling code is self-documenting.
        return *this;
    }

    // This must only be called after `call_once_and_store_result()` was called.
    T &get_stored() {
        assert(is_initialized_);
        PYBIND11_WARNING_PUSH
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 5
        // Needed for gcc 4.8.5
        PYBIND11_WARNING_DISABLE_GCC("-Wstrict-aliasing")
#endif
        return *reinterpret_cast<T *>(storage_);
        PYBIND11_WARNING_POP
    }

    constexpr gil_safe_call_once_and_store() = default;
    PYBIND11_DTOR_CONSTEXPR ~gil_safe_call_once_and_store() = default;

private:
    alignas(T) char storage_[sizeof(T)] = {};
    std::once_flag once_flag_ = {};
#ifdef Py_GIL_DISABLED
    std::atomic_bool
#else
    bool
#endif
        is_initialized_{false};
    // The `is_initialized_`-`storage_` pair is very similar to `std::optional`,
    // but the latter does not have the triviality properties of former,
    // therefore `std::optional` is not a viable alternative here.
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
