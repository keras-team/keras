/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef ONEAPI_DNNL_DNNL_GRAPH_HPP
#define ONEAPI_DNNL_DNNL_GRAPH_HPP

#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_graph.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_graph_api Graph API
/// @{

namespace dnnl {
namespace graph {

/// @cond DO_NOT_DOCUMENT_THIS

// Alias for common engine and stream API.
using engine = dnnl::engine;
using stream = dnnl::stream;
using fpmath_mode = dnnl::fpmath_mode;

/// @endcond

/// @addtogroup dnnl_graph_api_utils Utilities
/// Utility types and definitions
/// @{

/// @cond DO_NOT_DOCUMENT_THIS

/// A class that provides the destructor for a oneDNN graph C API handle.
template <typename T>
struct graph_handle_traits : public dnnl::handle_traits<T> {};

template <>
struct graph_handle_traits<dnnl_graph_op_t> {
    static dnnl_status_t destructor(dnnl_graph_op_t p) {
        return dnnl_graph_op_destroy(p);
    }
};

template <>
struct graph_handle_traits<dnnl_graph_graph_t> {
    static dnnl_status_t destructor(dnnl_graph_graph_t p) {
        return dnnl_graph_graph_destroy(p);
    }
};

template <>
struct graph_handle_traits<dnnl_graph_tensor_t> {
    static dnnl_status_t destructor(dnnl_graph_tensor_t p) {
        return dnnl_graph_tensor_destroy(p);
    }
};

template <>
struct graph_handle_traits<dnnl_graph_partition_t> {
    static dnnl_status_t destructor(dnnl_graph_partition_t p) {
        return dnnl_graph_partition_destroy(p);
    }
};

template <>
struct graph_handle_traits<dnnl_graph_compiled_partition_t> {
    static dnnl_status_t destructor(dnnl_graph_compiled_partition_t p) {
        return dnnl_graph_compiled_partition_destroy(p);
    }
};

template <>
struct graph_handle_traits<dnnl_graph_allocator_t> {
    static dnnl_status_t destructor(dnnl_graph_allocator_t p) {
        return dnnl_graph_allocator_destroy(p);
    }
};

#define DNNL_GRAPH_HANDLE_ALIAS(type) \
    using type##_handle = dnnl::handle<dnnl_graph_##type##_t, \
            graph_handle_traits<dnnl_graph_##type##_t>>

DNNL_GRAPH_HANDLE_ALIAS(allocator);
DNNL_GRAPH_HANDLE_ALIAS(graph);
DNNL_GRAPH_HANDLE_ALIAS(op);
DNNL_GRAPH_HANDLE_ALIAS(tensor);
DNNL_GRAPH_HANDLE_ALIAS(compiled_partition);
DNNL_GRAPH_HANDLE_ALIAS(partition);

#undef DNNL_GRAPH_HANDLE_ALIAS

template <bool B>
using req = typename std::enable_if<B, bool>::type;

/// @endcond

/// @} dnnl_graph_api_utils

/// @addtogroup dnnl_graph_api_status Status
/// Definitions of status values returned by the library functions.
///
/// @{

/// Status values returned by the library functions.
enum class status {
    /// The operation was successful
    success = dnnl_success,
    /// The operation failed due to an out-of-memory condition
    out_of_memory = dnnl_out_of_memory,
    /// The operation failed because of incorrect function arguments
    invalid_arguments = dnnl_invalid_arguments,
    /// The operation failed because requested functionality is not implemented
    unimplemented = dnnl_unimplemented,
    /// The last available implementation is reached
    last_impl_reached = dnnl_last_impl_reached,
    /// Primitive or engine failed on execution
    runtime_error = dnnl_runtime_error,
    /// Queried element is not required for given primitive
    not_required = dnnl_not_required,
    /// The graph is not legitimate
    invalid_graph = dnnl_invalid_graph,
    /// The operation is not legitimate according to op schema
    invalid_graph_op = dnnl_invalid_graph_op,
    /// The shape cannot be inferred or compiled
    invalid_shape = dnnl_invalid_shape,
    /// The data type cannot be inferred or compiled
    invalid_data_type = dnnl_invalid_data_type,
};

/// @} dnnl_api_status

/// @addtogroup dnnl_graph_api_allocator Allocator
///
/// Definitions of allocator which is used to acquire memory resources in
/// partition compilation and execution. SYCL allocator
/// (#dnnl::graph::sycl_interop::make_allocator) should be used for SYCL runtime
/// and host allocator should be used for non-SYCL.
///
/// @{

/// Allocator
class allocator : public allocator_handle {
public:
    using allocator_handle::handle;

    /// Constructs an allocator according to given function pointers
    ///
    /// @param host_malloc A pointer to malloc function for CPU
    /// @param host_free A pointer to free function for CPU
    allocator(dnnl_graph_host_allocate_f host_malloc,
            dnnl_graph_host_deallocate_f host_free) {
        dnnl_graph_allocator_t a = nullptr;
        error::wrap_c_api(
                dnnl_graph_allocator_create(&a, host_malloc, host_free),
                "could not create allocator for cpu");
        reset(a);
    }

    /// Default constructor
    allocator() {
        dnnl_graph_allocator_t a = nullptr;
        error::wrap_c_api(dnnl_graph_allocator_create(&a, nullptr, nullptr),
                "could not create allocator");
        reset(a);
    }
};

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_engine Engine
/// @{

/// This API is a supplement for existing onednn engine API.
inline engine make_engine_with_allocator(
        engine::kind kind, size_t index, const allocator &alloc) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(
            dnnl_graph_make_engine_with_allocator(&c_engine,
                    static_cast<dnnl_engine_kind_t>(kind), index, alloc.get()),
            "could not make an engine with allocator");
    return engine(c_engine);
}

/// @} dnnl_graph_api_engine

/// @addtogroup dnnl_graph_api_logical_tensor Logical Tensor
///
/// Logical tensor describes the meta-data of the input or output tensor, like
/// elements data type, number of dimensions, size for each dimension (shape),
/// layout, and the property of the tensor.
///
/// Each logical tensor has an unique ID. The library uses logical tensor IDs to
/// build up the connections between operations if the output of one operation
/// has the same ID as the input of another operation. The meta-data in a
/// logical tensor may be enriched in the framework graph as it progresses
/// toward final execution. For example, the library doesn't require detailed
/// shape information at the operation and graph creation stage. But shape
/// information of input logical tensor will be required at partition
/// compilation stage. Logical tensor is not mutable. Users must create a new
/// logical tensor with the same ID to pass any new additional information to
/// oneDNN Graph API. Please note that the library also has unique IDs for
/// operations. The ID should be unique among different logical tensors, but it
/// can have the same value between a logical tensor and an operation.
///
/// @{

/// Logical tensor object
class logical_tensor {
    friend class op;
    friend class tensor;
    friend class partition;
    friend class compiled_partition;

    dnnl_graph_logical_tensor_t data;

public:
    /// Integer type for representing dimension sizes and indices.
    using dim = dnnl_dim_t;
    /// Vector of dimensions. Implementations are free to force a limit on the
    /// vector's length.
    using dims = std::vector<dim>;

    /// Data Type
    enum class data_type {
        undef = dnnl_data_type_undef,
        /// 16-bit/half-precision floating point.
        f16 = dnnl_f16,
        /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
        bf16 = dnnl_bf16,
        /// 32-bit/single-precision floating point.
        f32 = dnnl_f32,
        /// 32-bit signed integer.
        s32 = dnnl_s32,
        /// 8-bit signed integer.
        s8 = dnnl_s8,
        /// 8-bit unsigned integer.
        u8 = dnnl_u8,
        /// Boolean data type. Size is C++ implementation defined.
        boolean = dnnl_boolean,
        /// [OFP8 standard 8-bit
        /// floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
        /// with a 5-bit exponent and a 2-bit mantissa.
        f8_e5m2 = dnnl_f8_e5m2,
        /// [OFP8 standard 8-bit
        /// floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
        /// with a 4-bit exponent and a 3-bit mantissa.
        f8_e4m3 = dnnl_f8_e4m3,
    };

    /// Layout type
    enum class layout_type {
        /// Undefined layout type.
        undef = dnnl_graph_layout_type_undef,
        /// Any means to let the library to decide the layout for a tensor
        /// during partition compilation.
        any = dnnl_graph_layout_type_any,
        /// Strided means that the layout of a tensor is determined by the
        /// strides field in the logical tensor.
        strided = dnnl_graph_layout_type_strided,
        /// Opaque means that the layout of a tensor is the library specific.
        /// Usually, an opaque layout is generated by a partition which is
        /// compiled with layout type any.
        opaque = dnnl_graph_layout_type_opaque,
    };

    /// Tensor property
    enum class property_type {
        /// Undefined tensor property.
        undef = dnnl_graph_tensor_property_undef,
        /// Variable means the tensor may be changed during computation or
        /// between different iterations.
        variable = dnnl_graph_tensor_property_variable,
        /// Constant means the tensor will keep unchanged during computation and
        /// between different iterations. It's useful for the library to apply
        /// optimizations for constant tensors or cache constant tensors inside
        /// the library. For example, constant weight tensors in inference
        /// scenarios.
        constant = dnnl_graph_tensor_property_constant,
    };

    /// default constructor
    /// construct an empty object
    logical_tensor() = default;

    /// Constructs a logical tensor object
    explicit logical_tensor(const dnnl_graph_logical_tensor_t &c_data)
        : data(c_data) {}

    /// Copy
    logical_tensor(const logical_tensor &other) = default;

    /// Assign
    logical_tensor &operator=(const logical_tensor &other) = default;

    /// Constructs a logical tensor object with ID, data type, ndims, layout
    /// type, and property type.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param ndims Number of dimensions. -1 means unknown (see
    ///     #DNNL_GRAPH_UNKNOWN_NDIMS) and 0 means a scalar tensor.
    /// @param ltype Layout type.
    /// @param ptype Property type.
    logical_tensor(size_t tid, data_type dtype, int32_t ndims,
            layout_type ltype, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        error::wrap_c_api(
                dnnl_graph_logical_tensor_init(&val, tid, convert_to_c(dtype),
                        ndims, convert_to_c(ltype), convert_to_c(ptype)),
                "could not create logical_tensor with property");
        data = val;
    }

    /// Delegated constructor.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param ltype Layout type.
    logical_tensor(
            size_t tid, data_type dtype, layout_type ltype = layout_type::undef)
        : logical_tensor(tid, dtype, DNNL_GRAPH_UNKNOWN_NDIMS, ltype) {}

    /// Constructs a logical tensor object with basic information and detailed
    /// dims.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param adims Logical tensor dimensions. #DNNL_GRAPH_UNKNOWN_DIM means
    ///     the size of that dimension is unknown. 0 is used to define
    ///     zero-dimension tensor.
    /// @param ltype Layout type. If it's strided, the strides field in the
    ///     output logical tensor will be deduced accordingly.
    /// @param ptype Property type.
    logical_tensor(size_t tid, data_type dtype, const dims &adims,
            layout_type ltype, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        // if dimension size equals to 0, it's a scalar
        if (adims.size() == 0)
            error::wrap_c_api(dnnl_graph_logical_tensor_init(&val, tid,
                                      convert_to_c(dtype), 0,
                                      convert_to_c(ltype), convert_to_c(ptype)),
                    "could not create logical_tensor with property");
        else
            error::wrap_c_api(
                    dnnl_graph_logical_tensor_init_with_dims(&val, tid,
                            convert_to_c(dtype),
                            static_cast<int32_t>(adims.size()), adims.data(),
                            convert_to_c(ltype), convert_to_c(ptype)),
                    "could not create logical_tensor with dims and property");
        data = val;
    }

    /// Constructs a logical tensor object with detailed dims and strides. The
    /// layout_type of the output logical tensor object will always be strided.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param adims Logical tensor dimensions. #DNNL_GRAPH_UNKNOWN_DIM means
    ///     the size of that dimension is unknown. 0 is used to define
    ///     zero-dimension tensor.
    /// @param strides Logical tensor strides.  #DNNL_GRAPH_UNKNOWN_DIM means
    ///     the stride of the dimension is unknown. The library currently
    ///      doesn't support other negative stride values.
    /// @param ptype Property type.
    logical_tensor(size_t tid, data_type dtype, const dims &adims,
            const dims &strides, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        // TODO(lvtao): check the size of adims and strides.
        // They should be same.
        error::wrap_c_api(
                dnnl_graph_logical_tensor_init_with_strides(&val, tid,
                        convert_to_c(dtype), static_cast<int32_t>(adims.size()),
                        adims.data(), strides.data(), convert_to_c(ptype)),
                "could not create logical_tensor with strides and property");
        data = val;
    }

    /// Constructs a logical tensor object with detailed dims and an opaque
    /// layout ID. layout_type of the output logical tensor object will always
    /// be opaque.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param adims Logical tensor dimensions. #DNNL_GRAPH_UNKNOWN_DIM means
    ///     the size of that dimension is unknown. 0 is used to define
    ///     zero-dimension tensor.
    /// @param lid Opaque layout id.
    /// @param ptype Property type
    logical_tensor(size_t tid, data_type dtype, const dims &adims, size_t lid,
            property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;

        if (adims.size() == 0) {
            error::wrap_c_api(dnnl_graph_logical_tensor_init(&val, tid,
                                      convert_to_c(dtype), 0,
                                      convert_to_c(layout_type::opaque),
                                      convert_to_c(ptype)),
                    "could not create logical_tensor");
        } else {
            error::wrap_c_api(
                    dnnl_graph_logical_tensor_init_with_dims(&val, tid,
                            convert_to_c(dtype),
                            static_cast<int32_t>(adims.size()), adims.data(),
                            convert_to_c(layout_type::opaque),
                            convert_to_c(ptype)),
                    "could not create logical_tensor with dims");
        }

        val.layout.layout_id = lid;
        data = val;
    }

    /// Returns dimensions of a logical tensor.
    ///
    /// @returns A vector describing the size of each dimension.
    dims get_dims() const {
        if (data.ndims < 0) {
            error::wrap_c_api(dnnl_invalid_arguments,
                    "cannot return dims when ndims < 0");
        }

        return {data.dims, data.dims + data.ndims};
    }

    /// Returns the unique id of a logical tensor.
    ///
    /// @returns An integer value describing the ID.
    size_t get_id() const { return data.id; }

    /// Returns the data type of a logical tensor.
    ///
    /// @returns The data type.
    data_type get_data_type() const {
        return static_cast<data_type>(data.data_type);
    }

    /// Returns the property type of a logical tensor.
    ///
    /// @returns The property type.
    property_type get_property_type() const {
        return static_cast<property_type>(data.property);
    }

    /// Returns the layout type of a logical tensor.
    ///
    /// @returns The layout type.
    layout_type get_layout_type() const {
        return static_cast<layout_type>(data.layout_type);
    }

    /// Returns the layout ID of a logical tensor. The API should be called on a
    /// logical tensor with opaque layout type. Otherwise, an exception will be
    /// raised.
    ///
    /// @returns Layout ID.
    size_t get_layout_id() const {
        if (get_layout_type() != layout_type::opaque) {
            error::wrap_c_api(
                    dnnl_invalid_arguments, "layout type should be opaque");
        }

        return data.layout.layout_id;
    }

    /// Returns the strides of a logical tensor. The API should be called on a
    /// logical tensor with strided layout type. Otherwise, an exception will be
    /// raised.
    ///
    /// @returns A vector describing the stride size of each dimension.
    dims get_strides() const {
        if (get_layout_type() != layout_type::strided) {
            error::wrap_c_api(
                    dnnl_invalid_arguments, "layout type should be strided");
        }

        if (data.ndims < 0) {
            error::wrap_c_api(dnnl_invalid_arguments,
                    "cannot return strides when ndims < 0");
        }

        return {data.layout.strides, data.layout.strides + data.ndims};
    }

    /// Returns memory size in bytes required by this logical tensor.
    ///
    /// @returns The memory size in bytes.
    size_t get_mem_size() const {
        size_t size = 0;
        error::wrap_c_api(dnnl_graph_logical_tensor_get_mem_size(&data, &size),
                "could not get memory size from the logical_tensor");
        return size;
    }

    /// Compares if two logical tenors are equal. Users can decide accordingly
    /// if layout reordering is needed for two logical tensors. The method will
    /// return true for below two circumstances:
    ///
    /// 1. the two logical tensors are equal regarding each field in the struct,
    /// eg. id, ndims, dims, layout type, property, etc.
    /// 2. If all other fields are equal but the layout types in two logical
    /// tensors are different, the method will return true when the underlying
    /// memory layout is the same. For example, one logical tensor has strided
    /// layout type while the other one has opaque layout type, but underneath,
    /// both layouts are NHWC, the method will still return true for this case.
    ///
    /// @param lt The input logical tensor to be compared.
    /// @returns @c true if the two logical tensors are equal. @c false otherwise
    bool is_equal(const logical_tensor &lt) const {
        uint8_t equal = 0;
        error::wrap_c_api(
                dnnl_graph_logical_tensor_is_equal(&data, &lt.data, &equal),
                "could not compare between the two logical tensors");
        return equal != 0;
    }

private:
    static dnnl_data_type_t convert_to_c(data_type dtype) {
        return static_cast<dnnl_data_type_t>(dtype);
    }

    static dnnl_graph_layout_type_t convert_to_c(layout_type ltype) {
        return static_cast<dnnl_graph_layout_type_t>(ltype);
    }

    static dnnl_graph_tensor_property_t convert_to_c(property_type ptype) {
        return static_cast<dnnl_graph_tensor_property_t>(ptype);
    }
};

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_tensor Tensor
///
/// Tensor is an abstraction for multi-dimensional input and output data needed
/// in the execution of a compiled partition. A tensor object encapsulates a
/// handle to a memory buffer allocated on a specific engine and a logical
/// tensor which describes the dimensions, elements data type, and memory
/// layout.
///
/// @{

/// A tensor object
class tensor : public tensor_handle {
public:
    /// Default constructor. Constructs an empty object.
    tensor() = default;

    /// Constructs a tensor object according to a given logical tensor, an
    /// engine, and a memory handle.
    ///
    /// @param lt The given logical tensor
    /// @param aengine Engine to store the data on.
    /// @param handle Handle of memory buffer to use as an underlying storage.
    ///     - A pointer to the user-allocated buffer. In this case the library
    ///       doesn't own the buffer.
    ///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
    ///       allocate the buffer for the tensor. In this case the library
    ///       owns the buffer.
    ///     - DNNL_MEMORY_NONE to create tensor without an underlying buffer.
    tensor(const logical_tensor &lt, const engine &aengine, void *handle) {
        dnnl_graph_tensor_t t = nullptr;
        error::wrap_c_api(
                dnnl_graph_tensor_create(&t, &(lt.data), aengine.get(), handle),
                "could not create tensor object with the logical_tensor, "
                "engine, and handle");
        reset(t);
    }

    /// Constructs a tensor object.
    /// The underlying buffer for the memory will be allocated by the library.
    ///
    /// @param lt The given logical tensor
    /// @param aengine Engine to store the data on.
    tensor(const logical_tensor &lt, const engine &aengine)
        : tensor(lt, aengine, DNNL_MEMORY_ALLOCATE) {}

    /// Returns the underlying memory buffer.
    ///
    /// On the CPU engine, or when using USM, this is a pointer to the
    /// allocated memory.
    void *get_data_handle() const {
        void *handle = nullptr;
        error::wrap_c_api(dnnl_graph_tensor_get_data_handle(get(), &handle),
                "could not get data handle from the tensor");
        return handle;
    }

    /// Sets the underlying memory handle.
    ///
    /// @param handle Memory handle.
    void set_data_handle(void *handle) {
        error::wrap_c_api(dnnl_graph_tensor_set_data_handle(get(), handle),
                "setting data handle to the tensor failed");
    }

    /// Returns the associated engine.
    ///
    /// @returns An engine object
    engine get_engine() const {
        dnnl_engine_t c_engine = nullptr;
        error::wrap_c_api(dnnl_graph_tensor_get_engine(get(), &c_engine),
                "could not get an engine from a tensor object");
        return engine(c_engine, true);
    }
};

/// @} dnnl_graph_api_tensor

/// @addtogroup dnnl_graph_api_compiled_partition Compiled Partition
///
/// A compiled partition represents the generated kernels specialized for a
/// partition on a target hardware (engine) with input and output information
/// specified by the logical tensors.
///
/// @{

/// A compiled partition object.
class compiled_partition : public compiled_partition_handle {
public:
    /// Default constructor. Constructs an empty object.
    compiled_partition() = default;

    /// Constructs a compiled partition object
    compiled_partition(dnnl_graph_compiled_partition_t compiled_partition) {
        reset(compiled_partition, false);
    }

    /// Queries an input or output logical tensor according to tensor ID. If the
    /// tensor ID doesn't belong to any input or output of the compiled
    /// partition, an exception will be raised by the API.
    ///
    /// @param tid The unique id of required tensor.
    /// @returns The logical tensor.
    logical_tensor query_logical_tensor(size_t tid) const {
        dnnl_graph_logical_tensor_t lt;
        error::wrap_c_api(dnnl_graph_compiled_partition_query_logical_tensor(
                                  get(), tid, &lt),
                "query logical tensor from compiled_partition failed");
        return logical_tensor {lt};
    }

    /// Returns the hint of in-place pairs from a compiled partition. It
    /// indicates that an input and an output of the partition can share the
    /// same memory buffer for computation. In-place computation helps to reduce
    /// the memory footprint and improves cache locality. But since the library
    /// may not have a global view of user's application, it's possible that the
    /// input tensor is used at other places in user's computation graph. In
    /// this case, the user should take the in-place pair as a hint and pass a
    /// different memory buffer for output tensor to avoid overwriting the input
    /// memory buffer which will probably cause unexpected incorrect results.
    ///
    /// @returns A list of pairs of input and output IDs.
    std::vector<std::pair<size_t, size_t>> get_inplace_ports() const {
        size_t num = 0;
        const dnnl_graph_inplace_pair_t *inplace_pairs;

        error::wrap_c_api(dnnl_graph_compiled_partition_get_inplace_ports(
                                  get(), &num, &inplace_pairs),
                "could not get the in-place pairs from a compiled partition");
        if (num == 0) return {};

        std::vector<std::pair<size_t, size_t>> inplace_options;
        inplace_options.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            const dnnl_graph_inplace_pair_t *inplace_pair = inplace_pairs + i;
            inplace_options.emplace_back(
                    inplace_pair->input_id, inplace_pair->output_id);
        }
        return inplace_options;
    }

    /// Execute a compiled partition.
    ///
    /// @param astream Stream object to run over.
    /// @param inputs A list of input tensors.
    /// @param outputs A list of output tensors.
    void execute(stream &astream, const std::vector<tensor> &inputs,
            const std::vector<tensor> &outputs) const {
        std::vector<const_dnnl_graph_tensor_t> c_inputs;
        c_inputs.reserve(inputs.size());
        for (auto &in : inputs) {
            c_inputs.push_back(in.get());
        }
        std::vector<const_dnnl_graph_tensor_t> c_outputs;
        c_outputs.reserve(outputs.size());
        for (auto &out : outputs) {
            c_outputs.push_back(out.get());
        }

        error::wrap_c_api(
                dnnl_graph_compiled_partition_execute(get(), astream.get(),
                        c_inputs.size(), c_inputs.data(), c_outputs.size(),
                        c_outputs.data()),
                "could not execute the compiled_partition");
    }
};

/// @} dnnl_graph_api_compiled_partition

/// @addtogroup dnnl_graph_api_op Op
///
/// OP is an abstraction of computation logic for deep neural network
/// operations. An op object encapsulates an operation kind which describes the
/// computation logic, an unique ID which differentiates operations with the
/// same kind, and logical tensors which describes the input and output of the
/// operation and its connections to other operations in the graph.
///
/// @{

/// An op object.
class op : public op_handle {
public:
    /// Kinds of operations
    enum class kind {
        Abs = dnnl_graph_op_abs,
        AbsBackward = dnnl_graph_op_abs_backward,
        Add = dnnl_graph_op_add,
        AvgPool = dnnl_graph_op_avg_pool,
        AvgPoolBackward = dnnl_graph_op_avg_pool_backward,
        BatchNormForwardTraining = dnnl_graph_op_batch_norm_forward_training,
        BatchNormInference = dnnl_graph_op_batch_norm_inference,
        BatchNormTrainingBackward = dnnl_graph_op_batch_norm_backward,
        BiasAdd = dnnl_graph_op_bias_add,
        BiasAddBackward = dnnl_graph_op_bias_add_backward,
        Clamp = dnnl_graph_op_clamp,
        ClampBackward = dnnl_graph_op_clamp_backward,
        Concat = dnnl_graph_op_concat,
        Convolution = dnnl_graph_op_convolution,
        ConvolutionBackwardData = dnnl_graph_op_convolution_backward_data,
        ConvolutionBackwardWeights = dnnl_graph_op_convolution_backward_weights,
        ConvTranspose = dnnl_graph_op_conv_transpose,
        ConvTransposeBackwardData = dnnl_graph_op_conv_transpose_backward_data,
        ConvTransposeBackwardWeights
        = dnnl_graph_op_conv_transpose_backward_weights,
        Dequantize = dnnl_graph_op_dequantize,
        Divide = dnnl_graph_op_divide,
        DynamicDequantize = dnnl_graph_op_dynamic_dequantize,
        DynamicQuantize = dnnl_graph_op_dynamic_quantize,
        Elu = dnnl_graph_op_elu,
        EluBackward = dnnl_graph_op_elu_backward,
        End = dnnl_graph_op_end,
        Exp = dnnl_graph_op_exp,
        GELU = dnnl_graph_op_gelu,
        GELUBackward = dnnl_graph_op_gelu_backward,
        HardSigmoid = dnnl_graph_op_hard_sigmoid,
        HardSigmoidBackward = dnnl_graph_op_hard_sigmoid_backward,
        HardSwish = dnnl_graph_op_hard_swish,
        HardSwishBackward = dnnl_graph_op_hard_swish_backward,
        Interpolate = dnnl_graph_op_interpolate,
        InterpolateBackward = dnnl_graph_op_interpolate_backward,
        LayerNorm = dnnl_graph_op_layer_norm,
        LayerNormBackward = dnnl_graph_op_layer_norm_backward,
        LeakyReLU = dnnl_graph_op_leaky_relu,
        Log = dnnl_graph_op_log,
        LogSoftmax = dnnl_graph_op_log_softmax,
        LogSoftmaxBackward = dnnl_graph_op_log_softmax_backward,
        MatMul = dnnl_graph_op_matmul,
        Maximum = dnnl_graph_op_maximum,
        MaxPool = dnnl_graph_op_max_pool,
        MaxPoolBackward = dnnl_graph_op_max_pool_backward,
        Minimum = dnnl_graph_op_minimum,
        Mish = dnnl_graph_op_mish,
        MishBackward = dnnl_graph_op_mish_backward,
        Multiply = dnnl_graph_op_multiply,
        Pow = dnnl_graph_op_pow,
        PReLU = dnnl_graph_op_prelu,
        PReLUBackward = dnnl_graph_op_prelu_backward,
        Quantize = dnnl_graph_op_quantize,
        Reciprocal = dnnl_graph_op_reciprocal,
        ReduceL1 = dnnl_graph_op_reduce_l1,
        ReduceL2 = dnnl_graph_op_reduce_l2,
        ReduceMax = dnnl_graph_op_reduce_max,
        ReduceMean = dnnl_graph_op_reduce_mean,
        ReduceMin = dnnl_graph_op_reduce_min,
        ReduceProd = dnnl_graph_op_reduce_prod,
        ReduceSum = dnnl_graph_op_reduce_sum,
        ReLU = dnnl_graph_op_relu,
        ReLUBackward = dnnl_graph_op_relu_backward,
        Reorder = dnnl_graph_op_reorder,
        Round = dnnl_graph_op_round,
        Select = dnnl_graph_op_select,
        Sigmoid = dnnl_graph_op_sigmoid,
        SigmoidBackward = dnnl_graph_op_sigmoid_backward,
        SoftMax = dnnl_graph_op_softmax,
        SoftMaxBackward = dnnl_graph_op_softmax_backward,
        SoftPlus = dnnl_graph_op_softplus,
        SoftPlusBackward = dnnl_graph_op_softplus_backward,
        Sqrt = dnnl_graph_op_sqrt,
        SqrtBackward = dnnl_graph_op_sqrt_backward,
        Square = dnnl_graph_op_square,
        SquaredDifference = dnnl_graph_op_squared_difference,
        StaticReshape = dnnl_graph_op_static_reshape,
        StaticTranspose = dnnl_graph_op_static_transpose,
        Subtract = dnnl_graph_op_subtract,
        Tanh = dnnl_graph_op_tanh,
        TanhBackward = dnnl_graph_op_tanh_backward,
        TypeCast = dnnl_graph_op_type_cast,
        Wildcard = dnnl_graph_op_wildcard,
        // Sentinel
        LastSymbol = dnnl_graph_op_last_symbol,
    };

    /// Attributes of operations. Different operations support different
    /// attributes. Check the document of each operation for what attributes are
    /// supported and what are the potential values for them. Missing required
    /// attribute or illegal attribute value may lead to failure when adding the
    /// operation to a graph.
    enum class attr {
        /// Undefined op attribute.
        undef = dnnl_graph_op_attr_undef,

        // float32 attributes. The value of these attributes can be any single
        // float32 number.

        /// Specifies an alpha attribute to an op.
        alpha = dnnl_graph_op_attr_alpha,
        /// Specifies an beta attribute to an op.
        beta = dnnl_graph_op_attr_beta,
        /// Specifies an epsilon attribute to an op.
        epsilon = dnnl_graph_op_attr_epsilon,
        /// Specifies a max attribute to an op.
        max = dnnl_graph_op_attr_max,
        /// Specifies a min attribute to an op.
        min = dnnl_graph_op_attr_min,
        /// Specifies a momentum attribute to an op.
        momentum = dnnl_graph_op_attr_momentum,

        // float32 vector attributes. The value of these attributes can be a
        // vector of float32 numbers.

        /// Specifies a scales attribute to an op.
        scales = dnnl_graph_op_attr_scales,

        // int64_t attributes. The value of these attributes can be any single
        // int64 number.

        /// Specifies an axis attribute to an op.
        axis = dnnl_graph_op_attr_axis,
        /// Specifies a begin_norm_axis attribute to an op.
        begin_norm_axis = dnnl_graph_op_attr_begin_norm_axis,
        /// Specifies a groups attribute to an op.
        groups = dnnl_graph_op_attr_groups,

        // int64_t vector attributes. The value of these attributes can be a
        // vector of int64 numbers.

        /// Specifies an axes attribute to an op.
        axes = dnnl_graph_op_attr_axes,
        /// Specifies a dilations attribute to an op.
        dilations = dnnl_graph_op_attr_dilations,
        /// Specifies an dst_shape attribute to an op.
        dst_shape = dnnl_graph_op_attr_dst_shape,
        /// Specifies a kernel attribute to an op.
        kernel = dnnl_graph_op_attr_kernel,
        /// Specifies an order attribute to an op.
        order = dnnl_graph_op_attr_order,
        /// Specifies an output_padding attribute to an op.
        output_padding = dnnl_graph_op_attr_output_padding,
        /// Specifies a pads_begin attribute to an op.
        pads_begin = dnnl_graph_op_attr_pads_begin,
        /// Specifies a pads_end attribute to an op.
        pads_end = dnnl_graph_op_attr_pads_end,
        /// Specifies a shape attribute to an op.
        shape = dnnl_graph_op_attr_shape,
        /// Specifies a sizes attribute to an op.
        sizes = dnnl_graph_op_attr_sizes,
        /// Specifies an src_shape attribute to an op.
        src_shape = dnnl_graph_op_attr_src_shape,
        /// Specifies a strides attribute to an op.
        strides = dnnl_graph_op_attr_strides,
        /// Specifies a weight_shape attribute to an op.
        weights_shape = dnnl_graph_op_attr_weights_shape,
        /// Specifies a zps attribute to an op.
        zps = dnnl_graph_op_attr_zps,

        // bool attributes. The value of these attributes can be any single bool
        // value.

        /// Specifies an exclude_pad attribute to an op.
        exclude_pad = dnnl_graph_op_attr_exclude_pad,
        /// Specifies a keep_dims attribute to an op.
        keep_dims = dnnl_graph_op_attr_keep_dims,
        /// Specifies a keep_stats attribute to an op.
        keep_stats = dnnl_graph_op_attr_keep_stats,
        /// Specifies a per_channel_broadcast attribute to an op.
        per_channel_broadcast = dnnl_graph_op_attr_per_channel_broadcast,
        /// Specifies a special_zero attribute to an op.
        special_zero = dnnl_graph_op_attr_special_zero,
        /// Specifies a transpose_a attribute to an op.
        transpose_a = dnnl_graph_op_attr_transpose_a,
        /// Specifies a transpose_b attribute to an op.
        transpose_b = dnnl_graph_op_attr_transpose_b,
        /// Specifies an use_affine attribute to an op.
        use_affine = dnnl_graph_op_attr_use_affine,
        /// Specifies an use_dst attribute to an op.
        use_dst = dnnl_graph_op_attr_use_dst,

        // string attributes. The value of these attributes can be a string.

        /// Specifies an auto_broadcast attribute to an op. The value can be
        /// "none" or "numpy".
        auto_broadcast = dnnl_graph_op_attr_auto_broadcast,
        /// Specifies an auto_pad attribute to an op. The value can be "none",
        /// "same_upper", "same_lower", or "valid".
        auto_pad = dnnl_graph_op_attr_auto_pad,
        /// Specifies an coordinate_transformation_mode attribute to an op. The
        /// value can be "half_pixel" or "align_corners". The attribute is
        /// defined for Interpolate operations.
        coordinate_transformation_mode
        = dnnl_graph_op_attr_coordinate_transformation_mode,
        /// Specifies a data_format of an op. The value can be "NCX" or "NXC".
        data_format = dnnl_graph_op_attr_data_format,
        /// Specifies a mode attribute of an op. The value can be "nearest",
        /// "linear", "bilinear", or "trilinear". The attribute is defined for
        /// Interpolate operations.
        mode = dnnl_graph_op_attr_mode,
        /// Specifies a qtype attribute to an op. The value can be "per_channel"
        /// or "per_tensor". The attribute is defined for quantization
        /// operations.
        qtype = dnnl_graph_op_attr_qtype,
        /// Specifies a rounding_type attribute to an op. The value can be
        /// "ceil" or "floor".
        rounding_type = dnnl_graph_op_attr_rounding_type,
        /// Specifies a weights_format of an op. The value can be "OIX", "XIO",
        /// "IOX", or "XOI". Different operations may support different values.
        weights_format = dnnl_graph_op_attr_weights_format,

        /// Specifies the end of all above exteral attributes for check.
        end = dnnl_graph_op_attr_end,
    };

    /// Constructs an op object with an unique ID, an operation kind, and a name
    /// string.
    ///
    /// @param id The unique ID of the op.
    /// @param akind The op kind specifies which computation is represented by
    ///     the op, such as Convolution or ReLU.
    /// @param verbose_name The string added as the op name.
    op(size_t id, kind akind, const std::string &verbose_name = "") {
        dnnl_graph_op_t op = nullptr;
        error::wrap_c_api(dnnl_graph_op_create(&op, id, convert_to_c(akind),
                                  verbose_name.c_str()),
                "could not create op with id and op kind");
        reset(op);
    }

    /// Constructs an op object with an unique ID, an operation kind, and
    /// input/output logical tensors.
    ///
    /// @param id The unique ID of this op.
    /// @param akind The op kind specifies which computation is represented by
    ///     this op, such as Convolution or ReLU.
    /// @param inputs Input logical tensor to be bound to this op.
    /// @param outputs Output logical tensor to be bound to this op.
    /// @param verbose_name The string added as the op name.
    op(size_t id, kind akind, const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs,
            const std::string &verbose_name = "")
        : op(id, akind, verbose_name) {
        for (const auto &input : inputs) {
            error::wrap_c_api(dnnl_graph_op_add_input(get(), &(input.data)),
                    "adding input to the op failed");
        }
        for (const auto &output : outputs) {
            error::wrap_c_api(dnnl_graph_op_add_output(get(), &(output.data)),
                    "adding output to the op failed");
        }
    }

    /// Adds an input logical tensor to the op.
    ///
    /// @param t Input logical tensor.
    void add_input(const logical_tensor &t) {
        error::wrap_c_api(dnnl_graph_op_add_input(get(), &(t.data)),
                "adding input to the op failed");
    }

    /// Adds a vector of input logical tensors to the op.
    ///
    /// @param ts The list of input logical tensors.
    void add_inputs(const std::vector<logical_tensor> &ts) {
        for (const auto &t : ts) {
            error::wrap_c_api(dnnl_graph_op_add_input(get(), &(t.data)),
                    "adding input to the op failed");
        }
    }

    /// Adds an output logical tensor to the op.
    ///
    /// @param t Output logical tensor.
    void add_output(const logical_tensor &t) {
        error::wrap_c_api(dnnl_graph_op_add_output(get(), &(t.data)),
                "adding output to the op failed");
    }

    /// Adds a vector of output logical tensors to the op.
    ///
    /// @param ts The list of output logical tensors.
    void add_outputs(const std::vector<logical_tensor> &ts) {
        for (const auto &t : ts) {
            error::wrap_c_api(dnnl_graph_op_add_output(get(), &(t.data)),
                    "adding output to the op failed");
        }
    }

    /// Sets the attribute according to the name and type (int64_t).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, req<std::is_same<Type, int64_t>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::wrap_c_api(dnnl_graph_op_set_attr_s64(get(), attr, &value, 1),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (float).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, req<std::is_same<Type, float>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::wrap_c_api(dnnl_graph_op_set_attr_f32(get(), attr, &value, 1),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (bool).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, req<std::is_same<Type, bool>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        const uint8_t val = value;
        error::wrap_c_api(dnnl_graph_op_set_attr_bool(get(), attr, &val, 1),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (string).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, req<std::is_same<Type, std::string>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::wrap_c_api(dnnl_graph_op_set_attr_str(
                                  get(), attr, value.c_str(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type
    /// (std::vector<int64_t>).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type,
            req<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::wrap_c_api(dnnl_graph_op_set_attr_s64(
                                  get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (std::vector<float>).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type,
            req<std::is_same<Type, std::vector<float>>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::wrap_c_api(dnnl_graph_op_set_attr_f32(
                                  get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

private:
    dnnl_graph_op_kind_t convert_to_c(kind akind) {
        return static_cast<dnnl_graph_op_kind_t>(akind);
    }

    dnnl_graph_op_attr_t convert_to_c(attr aattr) {
        return static_cast<dnnl_graph_op_attr_t>(aattr);
    }
};

/// @} dnnl_graph_api_op

/// @addtogroup dnnl_graph_api_partition Partition
///
/// Partition represents a collection of operations and their input and output
/// logical tensors identified by library as the basic unit for compilation and
/// execution.
///
/// @{

/// A partition object.
class partition : public partition_handle {
public:
    /// Policy specifications for partitioning.
    enum class policy {
        /// Fusion policy returns partitions with typical post-op fusions, eg.
        /// Convolution + ReLU or other element-wise operations or a chian of
        /// post-ops.
        fusion = dnnl_graph_partition_policy_fusion,
        /// Debug policy doesn't not apply any fusions. It returns partitions
        /// with single operations in each partition. The policy is useful when
        /// users notice any bug or correctness issue in fusion policy.
        debug = dnnl_graph_partition_policy_debug,
    };

    partition() = default;

    /// Constructs a partition object
    ///
    /// @param p A raw pointer to the C API handle
    partition(dnnl_graph_partition_t p) { reset(p, false); }

    /// Creates a new partition with a given operator and engine kind. The API
    /// is used to create a partition from an operation directly without
    /// creating the graph and calling `get_partitions()`. The output partition
    /// contains only one operation.
    ///
    /// @param aop An operation used to create the partition.
    /// @param ekind Engine kind.
    partition(const op &aop, engine::kind ekind) {
        dnnl_graph_partition_t p = nullptr;
        error::wrap_c_api(dnnl_graph_partition_create_with_op(&p, aop.get(),
                                  static_cast<dnnl_engine_kind_t>(ekind)),
                "could not create a partition with the op and engine kind");
        reset(p);
    }

    /// Returns the number of operations contained in the partition.
    ///
    /// @returns Number of operations.
    size_t get_ops_num() const {
        size_t num {0};
        error::wrap_c_api(dnnl_graph_partition_get_op_num(get(), &num),
                "could not get number of ops from the partition");
        return num;
    }

    /// Returns all operation IDs contained in the partition.
    ///
    /// @returns An unordered set of operation IDs.
    std::vector<size_t> get_ops() const {
        auto num = get_ops_num();
        std::vector<size_t> ops(num);

        error::wrap_c_api(dnnl_graph_partition_get_ops(get(), num, ops.data()),
                "could not get op ids from the partition");
        return ops;
    }

    /// Returns the unique ID of the partition. Partition ID is generated by the
    /// library internally. The ID can be used for debugging purpose or verbose.
    ///
    /// @returns ID of the partition.
    size_t get_id() const {
        size_t id {};
        error::wrap_c_api(dnnl_graph_partition_get_id(get(), &id),
                "could not get id of the partition");
        return id;
    }

    /// Compiles a partition with given input and output logical tensors. The
    /// output logical tensors can contain unknown dimensions. For this case,
    /// the compilation will deduce the output shapes according to input shapes.
    /// The output logical tensors can also have layout type `any`. The
    /// compilation will choose the optimal layout for output tensors. The
    /// optimal layout will be represented as an opaque layout ID saved in the
    /// output logical tensor.
    ///
    /// @param inputs A list of input logical tensors.
    /// @param outputs A list of output logical tensors.
    /// @param e The engine used to compile the partition.
    /// @returns A compiled partition.
    compiled_partition compile(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &e) const {
        if (!is_supported()) {
            error::wrap_c_api(dnnl_invalid_arguments,
                    "could not compile an unsupported partition");
        }

        return compile_(inputs, outputs, e);
    }

    /// Returns the supporting status of a partition. Some operations may not be
    /// supported by the library under certain circumstances. During
    /// partitioning stage, unsupported partitions will be returned to users
    /// with each containing an unsupported operation. Users should check the
    /// supporting status of a partition before transforming the computation
    /// graph or compiling the partition.
    ///
    /// @returns @c true if this partition is supported or @c false if this
    ///     partition isn't supported by the library
    bool is_supported() const {
        uint8_t supported {0};
        error::wrap_c_api(dnnl_graph_partition_is_supported(get(), &supported),
                "could not get supporting status of the partition");
        return supported != 0;
    }

    /// Returns a list of input logical tensors from the partition.
    ///
    /// @returns A list of input logical tensors.
    std::vector<logical_tensor> get_input_ports() const {
        size_t num = 0;
        error::wrap_c_api(dnnl_graph_partition_get_input_ports_num(get(), &num),
                "could not get number of inputs of the partition");
        if (num == 0) return {};

        std::vector<dnnl_graph_logical_tensor_t> c_inputs(num);
        error::wrap_c_api(dnnl_graph_partition_get_input_ports(
                                  get(), num, c_inputs.data()),
                "could not get input logical tensors of the partition");

        std::vector<logical_tensor> inputs;
        inputs.reserve(num);
        for (auto &c_lt : c_inputs)
            inputs.emplace_back(c_lt);
        return inputs;
    }

    /// Returns a list of output logical tensors from the partition.
    ///
    /// @returns A list of output logical tensor.
    std::vector<logical_tensor> get_output_ports() const {
        size_t num = 0;
        error::wrap_c_api(
                dnnl_graph_partition_get_output_ports_num(get(), &num),
                "cannot get number of outputs of the partition");
        if (num == 0) return {};

        std::vector<dnnl_graph_logical_tensor_t> c_outputs(num);
        error::wrap_c_api(dnnl_graph_partition_get_output_ports(
                                  get(), num, c_outputs.data()),
                "could not get output logical tensors of the partition");

        std::vector<logical_tensor> outputs;
        outputs.reserve(num);
        for (auto &c_lt : c_outputs)
            outputs.emplace_back(c_lt);
        return outputs;
    }

    /// Returns the engine kind of the partition
    ///
    /// @returns The engine kind
    engine::kind get_engine_kind() const {
        dnnl_engine_kind_t akind;
        error::wrap_c_api(dnnl_graph_partition_get_engine_kind(get(), &akind),
                "cannot get the engine kind from the partition");

        return static_cast<engine::kind>(akind);
    }

private:
    compiled_partition compile_(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &e) const {
        std::vector<const dnnl_graph_logical_tensor_t *> c_inputs;
        std::vector<const dnnl_graph_logical_tensor_t *> c_outputs;

        c_inputs.reserve(inputs.size());
        for (const auto &in : inputs) {
            c_inputs.push_back(&(in.data));
        }

        c_outputs.reserve(outputs.size());
        for (const auto &out : outputs) {
            c_outputs.push_back(&(out.data));
        }

        dnnl_graph_compiled_partition_t cpartitions = nullptr;
        error::wrap_c_api(
                dnnl_graph_compiled_partition_create(&cpartitions, get()),
                "could not create compiled_partition");
        error::wrap_c_api(dnnl_graph_partition_compile(get(), cpartitions,
                                  c_inputs.size(), c_inputs.data(),
                                  c_outputs.size(), c_outputs.data(), e.get()),
                "partition compile failed");

        return compiled_partition(cpartitions);
    }
};

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_graph Graph
///
/// Graph represents a computational DAG with a set of operations.
/// #dnnl::graph::graph::add_op() adds an operation and its input and output
/// logical tensors into a graph. The library accumulates the operations and
/// logical tensors and constructs and validates the graph as an internal state.
/// A graph object is associated to a specific engine kind. The partitions
/// returned from the graph will inherit the engine kind of the graph.
///
/// @{

/// A graph object.
class graph : public graph_handle {
public:
    /// Constructs a graph with an engine kind.
    ///
    /// @param engine_kind Engine kind.
    graph(engine::kind engine_kind) {
        dnnl_graph_graph_t g = nullptr;
        error::wrap_c_api(
                dnnl_graph_graph_create(&g, convert_to_c(engine_kind)),
                "could not create graph with engine kind");
        reset(g);
    }

    /// Creates a new empty graph with an engine kind and a floating-point math
    /// mode. All partitions returned from the graph will inherit the engine
    /// kind and floating-point math mode.
    ///
    /// @param engine_kind Engine kind.
    /// @param mode Floating-point math mode.
    graph(engine::kind engine_kind, fpmath_mode mode) {
        dnnl_graph_graph_t g = nullptr;
        error::wrap_c_api(
                dnnl_graph_graph_create_with_fpmath_mode(
                        &g, convert_to_c(engine_kind), convert_to_c(mode)),
                "could not create graph with engine kind and math mode");
        reset(g);
    }

    /// Adds an op into the graph to construct a computational DAG. The API will
    /// return failure if the operator has already been added to the graph or
    /// the operation cannot pass the schema check in the library (eg. input and
    /// output numbers and data types, the attributes of the operation, etc.).
    ///
    /// @param op An operation to be added.
    /// @param allow_exception A flag indicating whether the method is allowed
    ///     to throw an exception if it fails to add the op to the graph.
    /// @returns #status::success or a status describing the error otherwise.
    status add_op(const op &op, bool allow_exception = true) {
        dnnl_status_t ret = dnnl_graph_add_op(get(), op.get());

        if (allow_exception) {
            error::wrap_c_api(ret, "could not add op to the graph");
        }

        return static_cast<status>(ret);
    }

    /// Finalizes a graph. It means users have finished adding operations into
    /// the graph and the graph is ready for partitioning. Adding a new
    /// operation into a finalized graph will return failures. Similarly,
    /// partitioning on a un-finalized graph will also return failures.
    void finalize() {
        error::wrap_c_api(dnnl_graph_graph_finalize(get()),
                "could not finalize the graph");
    }

    /// Checks if a graph is finalized.
    ///
    /// @return True if the graph is finalized or false if the graph is not
    /// finalized.
    bool is_finalized() const {
        uint8_t ret = 0;
        error::wrap_c_api(dnnl_graph_graph_is_finalized(get(), &ret),
                "could not get the finalization status of the graph");

        return ret != 0;
    }

    /// Gets filtered partitions from a graph. Partitions will be claimed
    /// internally according to the capability of the library, the engine kind
    /// of the graph, and the policy.
    ///
    /// @param policy Partition policy, defaults to policy
    ///     #dnnl::graph::partition::policy::fusion.
    /// @return A vector storing the partitions.
    std::vector<partition> get_partitions(
            partition::policy policy = partition::policy::fusion) {
        if (!is_finalized()) {
            error::wrap_c_api(
                    dnnl_invalid_graph, "the graph is not finalized yet");
        }

        error::wrap_c_api(
                dnnl_graph_graph_filter(get(),
                        static_cast<dnnl_graph_partition_policy_t>(policy)),
                "could not filter the graph");

        size_t num = 0;
        error::wrap_c_api(dnnl_graph_graph_get_partition_num(get(), &num),
                "could not get number of partitions from the graph");

        // return early if there is no partitions in the graph.
        if (num == 0) return {};

        std::vector<partition> out_list;
        out_list.reserve(num);

        std::vector<dnnl_graph_partition_t> partitions(num);
        error::wrap_c_api(
                dnnl_graph_graph_get_partitions(get(), num, partitions.data()),
                "could not get partitions from the graph");

        for (auto p : partitions) {
            out_list.emplace_back(p);
        }

        return out_list;
    }

private:
    static dnnl_fpmath_mode_t convert_to_c(fpmath_mode mode) {
        return static_cast<dnnl_fpmath_mode_t>(mode);
    }

    static dnnl_engine_kind_t convert_to_c(engine::kind akind) {
        return static_cast<dnnl_engine_kind_t>(akind);
    }
};

/// @} dnnl_graph_api_graph

/// @addtogroup dnnl_graph_api_compiled_partition_cache Compiled Partition Cache
///
/// A set of functions that provide compiled partition cache control.
///
/// @{

/// Returns the number of compiled partition that can be held in the compiled
/// partition cache at the same time.
inline int get_compiled_partition_cache_capacity() {
    int result = 0;
    error::wrap_c_api(dnnl_graph_get_compiled_partition_cache_capacity(&result),
            "could not get compiled partition cache capacity");
    return result;
}

/// @copydoc dnnl_graph_set_compiled_partition_cache_capacity(int capacity)
inline void set_compiled_partition_cache_capacity(int capacity) {
    error::wrap_c_api(
            dnnl_graph_set_compiled_partition_cache_capacity(capacity),
            "could not set compiled partition cache capacity");
}

/// @} dnnl_graph_api_compiled_partition_cache

/// @addtogroup dnnl_graph_api_constant_tensor_cache Constant Tensor Cache
///
/// A set of functions that provide constant tensor cache control
///
/// @{

/// Control the enabling or disabling of constant tensor cache. This API must be
/// called once before compilation stage. By default, constant tensor cache is
/// disabled in the library.
/// @note This API is deprecated and will be removed in future release, please
/// use the set_constant_tensor_cache_capacity API to disable
/// constant tensor cache by setting it's capacity to zero.
///
/// @param flag Set to positive value to enable the cache and set to 0 to
/// disable the cache. Negative values are invalid.
inline void set_constant_tensor_cache(int flag) {
    error::wrap_c_api(dnnl_graph_set_constant_tensor_cache(flag),
            "fail to set constant tensor cache");
}

/// Return the enabling status of constant tensor cache.
/// @note This API is deprecated and will be removed in future release, please
/// use the get_constant_tensor_cache_capacity API to check the
/// enabling status by checking it's capacity.
inline int get_constant_tensor_cache() {
    int result = 0;
    error::wrap_c_api(dnnl_graph_get_constant_tensor_cache(&result),
            "fail to get constant tensor cache");
    return result;
}

/// Control the capacity for the constant tensor cache that used for specific
/// engine kind. This API is thread safe and can be called multiple times at
/// runtime. The capacity is set to zero by default which means the cache is
/// disabled. When calling this API, the corresponding cache will be flushed.
/// Setting capacity to 0 means to clear all cached tensors and disable cache.
/// Once the capacity limit is reached, no new tensors will be cached. If there
/// are multiple devices for an engine kind, the capacity set here is for each
/// device.
///
/// @param kind The engine kind that the constant tensor cache used for.
/// @param size The constant tensor cache capacity size to set.
inline void set_constant_tensor_cache_capacity(engine::kind kind, size_t size) {
    error::wrap_c_api(dnnl_graph_set_constant_tensor_cache_capacity(
                              static_cast<dnnl_engine_kind_t>(kind), size),
            "fail to set constant tensor cache capacity");
}

/// Return the current capacity of constant tensor cache.
///
/// @param kind The engine kind that the constant tensor cache used for.
inline size_t get_constant_tensor_cache_capacity(engine::kind kind) {
    size_t size = 0;
    error::wrap_c_api(dnnl_graph_get_constant_tensor_cache_capacity(
                              static_cast<dnnl_engine_kind_t>(kind), &size),
            "fail to get constant tensor cache capacity");
    return size;
}

/// @} dnnl_graph_constant_tensor_cache

} // namespace graph
} // namespace dnnl

/// @cond DO_NOT_DOCUMENT_THIS

/// oneAPI namespace
// Contains the oneapi::dnnl namespace as an alias to the ::dnnl namespace.
namespace oneapi {
// Note: without this guard, doxygen warns of potentially recursive namespace
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// oneDNN alias namespace
namespace dnnl = ::dnnl;
#endif
} // namespace oneapi

/// @endcond

/// @} dnnl_graph_api

/// @} dnnl_api

#endif
