/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

/// @file
/// C++ common API

#ifndef ONEAPI_DNNL_DNNL_COMMON_HPP
#define ONEAPI_DNNL_DNNL_COMMON_HPP

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_common.h"

/// @endcond

// __cpp_exceptions is referred from
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_exceptions.html
// gcc < 5 does not define __cpp_exceptions but __EXCEPTIONS,
// Microsoft C++ Compiler does not provide an option to disable exceptions
#ifndef DNNL_ENABLE_EXCEPTIONS
#if __cpp_exceptions || __EXCEPTIONS \
        || (defined(_MSC_VER) && !defined(__clang__))
#define DNNL_ENABLE_EXCEPTIONS 1
#else
#define DNNL_ENABLE_EXCEPTIONS 0
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DNNL_TRAP() __builtin_trap()
#elif defined(__INTEL_COMPILER) || defined(_MSC_VER)
#define DNNL_TRAP() __debugbreak()
#else
#error "unknown compiler"
#endif

#if DNNL_ENABLE_EXCEPTIONS
#define DNNL_THROW_ERROR(status, msg) throw error(status, msg)
#else
#include <cstdio>
#define DNNL_THROW_ERROR(status, msg) \
    do { \
        fputs(msg, stderr); \
        DNNL_TRAP(); \
    } while (0)
#endif

/// @addtogroup dnnl_api oneDNN API
/// @{

/// oneDNN namespace
namespace dnnl {

/// @addtogroup dnnl_api_common Common API
/// @{

/// @addtogroup dnnl_api_utils Utilities
/// Utility types and definitions.
/// @{

/// oneDNN exception class.
///
/// This class captures the status returned by a failed C API function and
/// the error message from the call site.
struct error : public std::exception {
    dnnl_status_t status;
    const char *message;

    /// Constructs an instance of an exception class.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    error(dnnl_status_t status, const char *message)
        : status(status), message(message) {}

    /// Returns the explanatory string.
    const char *what() const noexcept override { return message; }

    /// A convenience function for wrapping calls to C API functions. Checks
    /// the return status and throws an dnnl::error in case of failure.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    static void wrap_c_api(dnnl_status_t status, const char *message) {
        if (status != dnnl_success) DNNL_THROW_ERROR(status, message);
    }
};

/// A class that provides the destructor for a oneDNN C API handle.
template <typename T>
struct handle_traits {};

/// oneDNN C API handle wrapper class.
///
/// This class is used as the base class for primitive (dnnl::primitive),
/// engine (dnnl::engine), and stream (dnnl::stream) classes, as well as
/// others. An object of the dnnl::handle class can be passed by value.
///
/// A handle can be weak, in which case it follows std::weak_ptr semantics.
/// Otherwise, it follows `std::shared_ptr` semantics.
///
/// @note
///     The implementation stores oneDNN C API handles in a `std::shared_ptr`
///     with deleter set to a dummy function in the weak mode.
///
template <typename T, typename traits = handle_traits<T>>
struct handle {
private:
    static dnnl_status_t dummy_destructor(T) { return dnnl_success; }
    std::shared_ptr<typename std::remove_pointer<T>::type> data_ {0};

protected:
    bool operator==(const T other) const { return other == data_.get(); }
    bool operator!=(const T other) const { return !(*this == other); }

public:
    /// Constructs an empty handle object.
    ///
    /// @warning
    ///     Uninitialized object cannot be used in most library calls and is
    ///     equivalent to a null pointer. Any attempt to use its methods, or
    ///     passing it to the other library function, will cause an exception
    ///     to be thrown.
    handle() = default;

    /// Copy constructor.
    handle(const handle<T, traits> &) = default;
    /// Assignment operator.
    handle<T, traits> &operator=(const handle<T, traits> &) = default;
    /// Move constructor.
    handle(handle<T, traits> &&) = default;
    /// Move assignment operator.
    handle<T, traits> &operator=(handle<T, traits> &&) = default;

    /// Constructs a handle wrapper object from a C API handle.
    ///
    /// @param t The C API handle to wrap.
    /// @param weak A flag specifying whether to construct a weak wrapper;
    ///     defaults to @c false.
    explicit handle(T t, bool weak = false) { reset(t, weak); }

    /// Resets the handle wrapper objects to wrap a new C API handle.
    ///
    /// @param t The new value of the C API handle.
    /// @param weak A flag specifying whether the wrapper should be weak;
    ///     defaults to @c false.
    void reset(T t, bool weak = false) {
        data_.reset(t, weak ? &dummy_destructor : traits::destructor);
    }

    /// Returns the underlying C API handle.
    ///
    /// @param allow_empty A flag signifying whether the method is allowed to
    ///     return an empty (null) object without throwing an exception.
    /// @returns The underlying C API handle.
    T get(bool allow_empty = false) const {
        T result = data_.get();
        if (allow_empty == false && result == nullptr)
            DNNL_THROW_ERROR(
                    dnnl_invalid_arguments, "object is not initialized");
        return result;
    }

    /// Converts a handle to the underlying C API handle type. Does not throw
    /// and returns `nullptr` if the object is empty.
    ///
    /// @returns The underlying C API handle.
    explicit operator T() const { return get(true); }

    /// Checks whether the object is not empty.
    ///
    /// @returns Whether the object is not empty.
    explicit operator bool() const { return get(true) != nullptr; }

    /// Equality operator.
    ///
    /// @param other Another handle wrapper.
    /// @returns @c true if this and the other handle wrapper manage the same
    ///     underlying C API handle, and @c false otherwise. Empty handle
    ///     objects are considered to be equal.
    bool operator==(const handle<T, traits> &other) const {
        return other.data_.get() == data_.get();
    }

    /// Inequality operator.
    ///
    /// @param other Another handle wrapper.
    /// @returns @c true if this and the other handle wrapper manage different
    ///     underlying C API handles, and @c false otherwise. Empty handle
    ///     objects are considered to be equal.
    bool operator!=(const handle &other) const { return !(*this == other); }
};

/// @} dnnl_api_utils

/// @addtogroup dnnl_api_engine Engine
///
/// An abstraction of a computational device: a CPU, a specific GPU
/// card in the system, etc. Most primitives are created to execute
/// computations on one specific engine. The only exceptions are reorder
/// primitives that transfer data between two different engines.
///
/// @sa @ref dev_guide_basic_concepts
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_engine_t> {
    static dnnl_status_t destructor(dnnl_engine_t p) {
        return dnnl_engine_destroy(p);
    }
};
/// @endcond

/// An execution engine.
struct engine : public handle<dnnl_engine_t> {
    friend struct primitive;
    friend struct reorder;

    /// Kinds of engines.
    enum class kind {
        /// An unspecified engine
        any = dnnl_any_engine,
        /// CPU engine
        cpu = dnnl_cpu,
        /// GPU engine
        gpu = dnnl_gpu,
    };

    using handle::handle;

    /// Constructs an empty engine. An empty engine cannot be used in any
    /// operations.
    engine() = default;

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind The kind of engines to count.
    /// @returns The number of engines of the specified kind.
    static size_t get_count(kind akind) {
        return dnnl_engine_get_count(convert_to_c(akind));
    }

    /// Constructs an engine.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///     returned by #get_count() for this particular kind of engine.
    engine(kind akind, size_t index) {
        dnnl_engine_t engine;
        error::wrap_c_api(
                dnnl_engine_create(&engine, convert_to_c(akind), index),
                "could not create an engine");
        reset(engine);
    }

    /// Returns the kind of the engine.
    /// @returns The kind of the engine.
    kind get_kind() const {
        dnnl_engine_kind_t kind;
        error::wrap_c_api(dnnl_engine_get_kind(get(), &kind),
                "could not get kind of an engine");
        return static_cast<engine::kind>(kind);
    }

private:
    static dnnl_engine_kind_t convert_to_c(kind akind) {
        return static_cast<dnnl_engine_kind_t>(akind);
    }
};

/// Converts engine kind enum value from C++ API to C API type.
///
/// @param akind C++ API engine kind enum value.
/// @returns Corresponding C API engine kind enum value.
inline dnnl_engine_kind_t convert_to_c(engine::kind akind) {
    return static_cast<dnnl_engine_kind_t>(akind);
}

/// @} dnnl_api_engine

/// @addtogroup dnnl_api_stream Stream
///
/// An encapsulation of execution context tied to a particular engine.
///
/// @sa @ref dev_guide_basic_concepts
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_stream_t> {
    static dnnl_status_t destructor(dnnl_stream_t p) {
        return dnnl_stream_destroy(p);
    }
};
/// @endcond

/// An execution stream.
struct stream : public handle<dnnl_stream_t> {
    using handle::handle;

    /// Stream flags. Can be combined using the bitwise OR operator.
    enum class flags : unsigned {
        /// In-order execution.
        in_order = dnnl_stream_in_order,
        /// Out-of-order execution.
        out_of_order = dnnl_stream_out_of_order,
        /// Default stream configuration.
        default_flags = dnnl_stream_default_flags,
#ifdef DNNL_EXPERIMENTAL_PROFILING
        /// Enables profiling capabilities.
        profiling = dnnl_stream_profiling,
#endif
    };

    /// Constructs an empty stream. An empty stream cannot be used in any
    /// operations.
    stream() = default;

    /// Constructs a stream for the specified engine and with behavior
    /// controlled by the specified flags.
    ///
    /// @param aengine Engine to create the stream on.
    /// @param aflags Flags controlling stream behavior.
    explicit stream(
            const engine &aengine, flags aflags = flags::default_flags) {
        dnnl_stream_t stream;
        error::wrap_c_api(dnnl_stream_create(&stream, aengine.get(),
                                  static_cast<dnnl_stream_flags_t>(aflags)),
                "could not create a stream");
        reset(stream);
    }

    /// Returns the associated engine.
    engine get_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_stream_get_engine(get(), &c_engine),
                "could not get an engine from a stream object");
        return engine(c_engine, true);
    }

    /// Waits for all primitives executing in the stream to finish.
    /// @returns The stream itself.
    stream &wait() {
        error::wrap_c_api(
                dnnl_stream_wait(get()), "could not wait on a stream");
        return *this;
    }
};

#define DNNL_DEFINE_BITMASK_OPS(enum_name) \
    inline enum_name operator|(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator&(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator^(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name &operator|=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator&=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator^=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name operator~(enum_name rhs) { \
        return static_cast<enum_name>(~static_cast<unsigned>(rhs)); \
    }

DNNL_DEFINE_BITMASK_OPS(stream::flags)

/// @} dnnl_api_stream

/// @addtogroup dnnl_api_fpmath_mode Floating-point Math Mode
/// @{

/// Floating-point math mode
enum class fpmath_mode {
    /// Default behavior, no downconversions allowed
    strict = dnnl_fpmath_mode_strict,
    /// Implicit f32->bf16 conversions allowed
    bf16 = dnnl_fpmath_mode_bf16,
    /// Implicit f32->f16 conversions allowed
    f16 = dnnl_fpmath_mode_f16,
    /// Implicit f32->tf32 conversions allowed
    tf32 = dnnl_fpmath_mode_tf32,
    /// Implicit f32->f16, f32->tf32 or f32->bf16 conversions allowed
    any = dnnl_fpmath_mode_any
};

/// Converts an fpmath mode enum value from C++ API to C API type.
///
/// @param mode C++ API fpmath mode enum value.
/// @returns Corresponding C API fpmath mode enum value.
inline dnnl_fpmath_mode_t convert_to_c(fpmath_mode mode) {
    return static_cast<dnnl_fpmath_mode_t>(mode);
}

/// @} dnnl_api_fpmath_mode

/// @addtogroup dnnl_api_accumulation_mode Accumulation Mode
/// @{

/// Accumulation mode
enum class accumulation_mode {
    /// Default behavior, f32 for floating point computation, s32 for integer
    strict = dnnl_accumulation_mode_strict,
    /// same as strict except some partial accumulators can be rounded to
    /// src/dst datatype in memory.
    relaxed = dnnl_accumulation_mode_relaxed,
    /// uses fastest implementation, could use src/dst datatype or
    /// wider datatype for accumulators
    any = dnnl_accumulation_mode_any,
    /// use s32 accumulators during computation
    s32 = dnnl_accumulation_mode_s32,
    /// use f32 accumulators during computation
    f32 = dnnl_accumulation_mode_f32,
    /// use f16 accumulators during computation
    f16 = dnnl_accumulation_mode_f16
};

/// Converts an accumulation mode enum value from C++ API to C API type.
///
/// @param mode C++ API accumulation mode enum value.
/// @returns Corresponding C API accumulation mode enum value.
inline dnnl_accumulation_mode_t convert_to_c(accumulation_mode mode) {
    return static_cast<dnnl_accumulation_mode_t>(mode);
}

/// @} dnnl_api_accumulation_mode

/// @} dnnl_api_common

} // namespace dnnl

/// @} dnnl_api

#endif
