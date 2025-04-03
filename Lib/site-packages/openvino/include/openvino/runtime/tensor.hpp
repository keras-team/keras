// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime tensor API
 *
 * @file openvino/runtime/tensor.hpp
 */
#pragma once

#include <type_traits>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"

namespace ov {

class Tensor;
class ITensor;

namespace util {
ov::Tensor make_tensor(const std::shared_ptr<ov::ITensor>& tensor, const std::shared_ptr<void>& so);
void get_tensor_impl(const ov::Tensor& tensor, std::shared_ptr<ov::ITensor>& tensor_impl, std::shared_ptr<void>& so);
}  // namespace util

namespace op {
namespace util {
class VariableValue;
}  // namespace util
}  // namespace op

/**
 * @brief Tensor API holding host memory
 * It can throw exceptions safely for the application, where it is properly handled.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_API Tensor {
protected:
    std::shared_ptr<ITensor> _impl;  //!< Shared pointer to internal tensor representation
    std::shared_ptr<void> _so;       //!< Reference to dynamically loaded library

    /**
     * @brief Constructs Tensor from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Plugin to use. This is required to ensure that Tensor can work properly even if plugin object is
     * destroyed.
     */
    Tensor(const std::shared_ptr<ITensor>& impl, const std::shared_ptr<void>& so);

    friend class ov::op::util::VariableValue;
    friend ov::Tensor ov::util::make_tensor(const std::shared_ptr<ov::ITensor>& tensor,
                                            const std::shared_ptr<void>& so);
    friend void ov::util::get_tensor_impl(const ov::Tensor& tensor,
                                          std::shared_ptr<ov::ITensor>& tensor_impl,
                                          std::shared_ptr<void>& so);

public:
    /// @brief Default constructor
    Tensor() = default;

    /**
     * @brief Copy constructor with adding new shared object
     *
     * @param other Original tensor
     * @param so Shared object
     */
    Tensor(const Tensor& other, const std::shared_ptr<void>& so);

    /// @brief Default copy constructor
    /// @param other other Tensor object
    Tensor(const Tensor& other) = default;

    /// @brief Default copy assignment operator
    /// @param other other Tensor object
    /// @return reference to the current object
    Tensor& operator=(const Tensor& other) = default;

    /// @brief Default move constructor
    /// @param other other Tensor object
    Tensor(Tensor&& other) = default;

    /// @brief Default move assignment operator
    /// @param other other Tensor object
    /// @return reference to the current object
    Tensor& operator=(Tensor&& other) = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~Tensor();

    /**
     * @brief Checks openvino tensor type
     * @param tensor a tensor which type will be checked
     * @throw Exception if type check with specified tensor is not pass
     */
    static void type_check(const Tensor& tensor);

    /**
     * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param allocator allocates memory for internal tensor storage
     */
    Tensor(const element::Type& type, const Shape& shape, const Allocator& allocator = {});

    /**
     * @brief Constructs Tensor using element type and shape. Wraps allocated host memory.
     * @note Does not perform memory allocation internally
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param host_ptr Pointer to pre-allocated host memory with initialized objects
     * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
     * on shape and element size
     */
    Tensor(const element::Type& type, const Shape& shape, void* host_ptr, const Strides& strides = {});

    /**
     * @brief Constructs Tensor using port from node. Allocate internal host storage using default allocator
     * @param port port from node
     * @param allocator allocates memory for internal tensor storage
     */
    Tensor(const ov::Output<const ov::Node>& port, const Allocator& allocator = {});

    /**
     * @brief Constructs Tensor using port from node. Wraps allocated host memory.
     * @note Does not perform memory allocation internally
     * @param port port from node
     * @param host_ptr Pointer to pre-allocated host memory with initialized objects
     * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
     * on shape and element size
     */
    Tensor(const ov::Output<const ov::Node>& port, void* host_ptr, const Strides& strides = {});

    /**
     * @brief Constructs region of interest (ROI) tensor form another tensor.
     * @note Does not perform memory allocation internally
     * @param other original tensor
     * @param begin start coordinate of ROI object inside of the original object.
     * @param end end coordinate of ROI object inside of the original object.
     * @note A Number of dimensions in `begin` and `end` must match number of dimensions in `other.get_shape()`
     */
    Tensor(const Tensor& other, const Coordinate& begin, const Coordinate& end);

    /**
     * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
     * @note Memory allocation may happen
     * @param shape A new shape
     */
    void set_shape(const ov::Shape& shape);

    /**
     * @return A tensor element type
     */
    const element::Type& get_element_type() const;

    /**
     * @return A tensor shape
     */
    const Shape& get_shape() const;

    /**
     * @brief Copy tensor, destination tensor should have the same element type and shape
     *
     * @param dst destination tensor
     */
    void copy_to(ov::Tensor dst) const;

    /**
     * @brief Reports whether the tensor is continuous or not
     *
     * @return true if tensor is continuous
     */
    bool is_continuous() const;

    /**
     * @brief Returns the total number of elements (a product of all the dims or 1 for scalar)
     * @return The total number of elements
     */
    size_t get_size() const;

    /**
     * @brief Returns the size of the current Tensor in bytes.
     * @return Tensor's size in bytes
     */
    size_t get_byte_size() const;

    /**
     * @return Tensor's strides in bytes
     */
    Strides get_strides() const;

    /**
     * @brief Provides an access to the underlaying host memory
     * @param type Optional type parameter.
     * @note If type parameter is specified, the method throws an exception
     * if specified type's fundamental type does not match with tensor element type's fundamental type
     * @return A host pointer to tensor memory
     */
    void* data(const element::Type& type = {}) const;

    /**
     * @brief Provides an access to the underlaying host memory casted to type `T`
     * @return A host pointer to tensor memory casted to specified type `T`.
     * @note Throws exception if specified type does not match with tensor element type
     */
    template <typename T, typename datatype = typename std::decay<T>::type>
    T* data() const {
        return static_cast<T*>(data(element::from<datatype>()));
    }

    /**
     * @brief Checks if current Tensor object is not initialized
     * @return `true` if current Tensor object is not initialized, `false` - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current Tensor object is initialized
     * @return `true` if current Tensor object is initialized, `false` - otherwise
     */
    explicit operator bool() const noexcept;

    /**
     * @brief Checks if the Tensor object can be cast to the type T
     *
     * @tparam T Type to be checked. Must represent a class derived from the Tensor
     * @return true if this object can be dynamically cast to the type const T*. Otherwise, false
     */
    template <typename T>
    typename std::enable_if<std::is_base_of<Tensor, T>::value, bool>::type is() const noexcept {
        try {
            T::type_check(*this);
        } catch (...) {
            return false;
        }
        return true;
    }

    /**
     * @brief Casts this Tensor object to the type T.
     *
     * @tparam T Type to cast to. Must represent a class derived from the Tensor
     * @return T object
     */
    template <typename T>
    const typename std::enable_if<std::is_base_of<Tensor, T>::value, T>::type as() const {
        T::type_check(*this);
        return *static_cast<const T*>(this);
    }
};

/**
 * @brief A vector of Tensor's
 */
using TensorVector = std::vector<Tensor>;

}  // namespace ov
