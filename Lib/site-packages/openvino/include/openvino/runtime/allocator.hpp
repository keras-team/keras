// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides Allocator interface
 *
 * @file openvino/runtime/allocator.hpp
 */
#pragma once

#include <cstddef>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/core_visibility.hpp"

namespace ov {

class Tensor;

/**
 * @brief Wraps allocator implementation to provide safe way to store allocater loaded from shared library
 *        And constructs default based on `new` `delete` c++ calls allocator if created without parameters
 *        Accepts any [std::pmr::memory_resource](https://en.cppreference.com/w/cpp/memory/memory_resource) like
 *        allocator
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_API Allocator {
    /**
     * @brief Constructs Tensor from the initialized std::shared_ptr
     * @param other Initialized allocator
     * @param so Plugin to use. This is required to ensure that Allocator can work properly even if plugin object is
     * destroyed.
     */
    Allocator(const Allocator& other, const std::shared_ptr<void>& so);

    friend class ov::Tensor;

    struct OPENVINO_API Base : public std::enable_shared_from_this<Base> {
        virtual void* addressof() = 0;
        const void* addressof() const {
            return const_cast<Base*>(this)->addressof();
        }
        virtual const std::type_info& type_info() const = 0;
        virtual void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t)) = 0;
        virtual void deallocate(void* handle, const size_t bytes, size_t alignment = alignof(max_align_t)) = 0;
        virtual bool is_equal(const Base& other) const = 0;

    protected:
        virtual ~Base();
    };

    template <typename A>
    struct Impl : public Base {
        template <typename... Args>
        explicit Impl(Args&&... args) : a(std::forward<Args>(args)...) {}
        void* addressof() override {
            return &a;
        }
        const std::type_info& type_info() const override {
            return typeid(a);
        }
        void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t)) override {
            return a.allocate(bytes, alignment);
        }
        void deallocate(void* handle, const size_t bytes, size_t alignment = alignof(max_align_t)) override {
            a.deallocate(handle, bytes, alignment);
        }
        bool is_equal(const Base& other) const override {
            if (util::equal(type_info(), other.type_info())) {
                return a.is_equal(*static_cast<const A*>(other.addressof()));
            }
            return false;
        }
        A a;
    };

    std::shared_ptr<Base> _impl;
    std::shared_ptr<void> _so;

public:
    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~Allocator();

    /// @brief Default constructor
    Allocator();

    /// @brief Default copy constructor
    /// @param other other Allocator object
    Allocator(const Allocator& other) = default;

    /// @brief Default copy assignment operator
    /// @param other other Allocator object
    /// @return reference to the current object
    Allocator& operator=(const Allocator& other) = default;

    /// @brief Default move constructor
    /// @param other other Allocator object
    Allocator(Allocator&& other) = default;

    /// @brief Default move assignment operator
    /// @param other other Allocator object
    /// @return reference to the current object
    Allocator& operator=(Allocator&& other) = default;

    /**
     * @brief Initialize allocator using any allocator like object
     * @tparam A Type of allocator
     * @param a allocator object
     */
    template <
        typename A,
        typename std::enable_if<!std::is_same<typename std::decay<A>::type, Allocator>::value &&
                                    !std::is_abstract<typename std::decay<A>::type>::value &&
                                    !std::is_convertible<typename std::decay<A>::type, std::shared_ptr<Base>>::value,
                                bool>::type = true>
    Allocator(A&& a) : _impl{std::make_shared<Impl<typename std::decay<A>::type>>(std::forward<A>(a))} {}

    /**
     * @brief Allocates memory
     *
     * @param bytes The size in bytes at least to allocate
     * @param alignment The alignment of storage
     * @return Handle to the allocated resource
     * @throw Exception if specified size and alignment is not supported
     */
    void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t));

    /**
     * @brief Releases the handle and all associated memory resources which invalidates the handle.
     * @param ptr The handle to free
     * @param bytes The size in bytes that was passed into allocate() method
     * @param alignment The alignment of storage that was passed into allocate() method
     */
    void deallocate(void* ptr, const size_t bytes = 0, const size_t alignment = alignof(max_align_t));

    /**
     * @brief Compares with other Allocator
     * @param other Other instance of allocator
     * @return `true` if and only if memory allocated from one Allocator can be deallocated from the other and vice
     * versa
     */
    bool operator==(const Allocator& other) const;

    /**
     * @brief Checks if current Allocator object is not initialized
     * @return `true` if current Allocator object is not initialized, `false` - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current Allocator object is initialized
     * @return `true` if current Allocator object is initialized, `false` - otherwise
     */
    explicit operator bool() const noexcept;
};

}  // namespace ov
