/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
/// C API

#ifndef ONEAPI_DNNL_DNNL_GRAPH_H
#define ONEAPI_DNNL_DNNL_GRAPH_H

#include "oneapi/dnnl/dnnl_common.h"
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_graph_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_allocator
/// @{

/// Creates a host allocator with the given allocation and deallocation
/// call-back function pointers.
///
/// @param allocator Output allocator.
/// @param host_malloc A pointer to malloc function for host.
/// @param host_free A pointer to free function for host.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_allocator_create(
        dnnl_graph_allocator_t *allocator,
        dnnl_graph_host_allocate_f host_malloc,
        dnnl_graph_host_deallocate_f host_free);

/// Destroys an allocator.
///
/// @param allocator The allocator to be destroyed.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_allocator_destroy(
        dnnl_graph_allocator_t allocator);

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_engine
/// @{

/// This API is a supplement for existing onednn engine API.
dnnl_status_t DNNL_API dnnl_graph_make_engine_with_allocator(
        dnnl_engine_t *engine, dnnl_engine_kind_t kind, size_t index,
        const_dnnl_graph_allocator_t alloc);

/// @} dnnl_graph_api_engine

/// @addtogroup dnnl_graph_api_logical_tensor
/// @{

/// Initializes a logical tensor with id, data type, number of dimensions,
/// layout type, and property. The logical tensor's dims are unknown with this
/// interface.
///
/// @param logical_tensor Output logical tensor.
/// @param tid The unique id of the output logical tensor.
/// @param dtype Elements data type.
/// @param ndims Number of dimensions.
/// @param ltype Layout type of the underlying tensor buffer.
/// @param ptype Tensor property type.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_logical_tensor_init(
        dnnl_graph_logical_tensor_t *logical_tensor, size_t tid,
        dnnl_data_type_t dtype, int32_t ndims, dnnl_graph_layout_type_t ltype,
        dnnl_graph_tensor_property_t ptype);

/// Initializes a logical tensor with basic information and dims. The logical
/// tensor's dimensions and layout will be initialized according to the input
/// arguments.
///
/// @note
///     If dims contains all valid values and layout type is
///     #dnnl_graph_layout_type_strided. The strides field in
///     #dnnl_graph_logical_tensor_t will be calculated in a row major and
///     contiguous way. Otherwise, Accessing the strides field is an undefined
///     behavior.
///
///     Eg. dims (2, 3, 4, 5) will get strides (60, 20, 5, 1)
///
/// @param logical_tensor Output logical tensor.
/// @param tid The unique id of output logical tensor.
/// @param dtype Elements data type.
/// @param ndims Number of dimensions.
/// @param dims Array of dimensions.
/// @param ltype Layout type of the underlying tensor memory.
/// @param ptype Tensor property type.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_logical_tensor_init_with_dims(
        dnnl_graph_logical_tensor_t *logical_tensor, size_t tid,
        dnnl_data_type_t dtype, int32_t ndims, const dnnl_dims_t dims,
        dnnl_graph_layout_type_t ltype, dnnl_graph_tensor_property_t ptype);

/// Initializes a logical tensor with dimensions and strides provided by user.
///
/// @note
///     Once strides are explicitly provided through the API, the `layout_type`
///     in #dnnl_graph_logical_tensor_t can only be
///     #dnnl_graph_layout_type_strided or #dnnl_graph_layout_type_any.
///
/// @param logical_tensor Output logical tensor.
/// @param tid The unique id of output logical tensor.
/// @param dtype Elements data type.
/// @param ndims Number of dimensions.
/// @param dims Array of dimensions.
/// @param strides Array of strides.
/// @param ptype Tensor property type.
/// @returns #dnnl_success on success or a status describing the error
/// otherwise.
dnnl_status_t DNNL_API dnnl_graph_logical_tensor_init_with_strides(
        dnnl_graph_logical_tensor_t *logical_tensor, size_t tid,
        dnnl_data_type_t dtype, int32_t ndims, const dnnl_dims_t dims,
        const dnnl_dims_t strides, dnnl_graph_tensor_property_t ptype);

/// Returns the memory size described by the logical tensor. If it's a strided
/// layout, the size will be calculated by `dims` and `strides`. If it's an
/// opaque layout, the size will be decided by `layout_id`.
///
/// @param logical_tensor Logical tensor.
/// @param size Output memory size in bytes.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_logical_tensor_get_mem_size(
        const dnnl_graph_logical_tensor_t *logical_tensor, size_t *size);

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
/// @param lt1 The handle of first logical tensor.
/// @param lt2 The handle of second logical tensor.
/// @param is_equal 1 if these two logical tensors are equal, 0 otherwise.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_logical_tensor_is_equal(
        const dnnl_graph_logical_tensor_t *lt1,
        const dnnl_graph_logical_tensor_t *lt2, uint8_t *is_equal);

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_tensor
/// @{

/// Creates a tensor with logical tensor, engine, and data handle.
///
/// @param tensor Output tensor.
/// @param logical_tensor Description for this tensor.
/// @param engine Engine to use.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the tensor. In this case the library
///       owns the buffer.
///     - DNNL_MEMORY_NONE to create tensor without an underlying buffer.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_tensor_create(dnnl_graph_tensor_t *tensor,
        const dnnl_graph_logical_tensor_t *logical_tensor, dnnl_engine_t engine,
        void *handle);

/// Destroys a tensor.
///
/// @param tensor The tensor to be destroyed.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_tensor_destroy(dnnl_graph_tensor_t tensor);

/// Gets the data handle of a tensor.
///
/// @param tensor The input tensor.
/// @param handle Pointer to the data of input tensor.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_tensor_get_data_handle(
        const_dnnl_graph_tensor_t tensor, void **handle);

/// Set data handle for a tensor.
///
/// @param tensor The input tensor.
/// @param handle New data handle for tensor.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_tensor_set_data_handle(
        dnnl_graph_tensor_t tensor, void *handle);

/// Returns the engine of a tensor object.
///
/// @param tensor The input tensor.
/// @param engine Output engine on which the tensor is located.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_tensor_get_engine(
        const_dnnl_graph_tensor_t tensor, dnnl_engine_t *engine);

/// @} dnnl_graph_api_tensor

/// @addtogroup dnnl_graph_api_op
/// @{

/// Initializes an op with unique id, kind, and name.
///
/// @param op Output op
/// @param id The unique id of the output op.
/// @param kind The op kind.
/// @param verbose_name The string added as the op name.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_create(dnnl_graph_op_t *op, size_t id,
        dnnl_graph_op_kind_t kind, const char *verbose_name);

/// Destroys an op.
///
/// @param op The op to be destroyed.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_destroy(dnnl_graph_op_t op);

/// Adds input logical tensor to the op.
///
/// @param op Input op.
/// @param input The input logical tensor to be added.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_add_input(
        dnnl_graph_op_t op, const dnnl_graph_logical_tensor_t *input);

/// Adds output logical tensor to the op.
///
/// @param op Input op.
/// @param output The output logical tensor to be added.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_add_output(
        dnnl_graph_op_t op, const dnnl_graph_logical_tensor_t *output);

/// Sets floating point attribute to an op.
///
/// @param op Input op.
/// @param name The attribute's name.
/// @param value The attribute's value.
/// @param value_len The number of value element.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_set_attr_f32(dnnl_graph_op_t op,
        dnnl_graph_op_attr_t name, const float *value, size_t value_len);

/// Sets boolean attribute to an op.
///
/// @param op Input op.
/// @param name The attribute's name.
/// @param value The attribute's value.
/// @param value_len The number of value element.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_set_attr_bool(dnnl_graph_op_t op,
        dnnl_graph_op_attr_t name, const uint8_t *value, size_t value_len);

/// Sets integer attribute to an op.
///
/// @param op Input op.
/// @param name The attribute's name.
/// @param value The attribute's value.
/// @param value_len The number of value element.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_set_attr_s64(dnnl_graph_op_t op,
        dnnl_graph_op_attr_t name, const int64_t *value, size_t value_len);

/// Sets string attribute to an op.
///
/// @param op Input op.
/// @param name The attribute's name.
/// @param value The attribute's value.
/// @param value_len The length of the string value.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_set_attr_str(dnnl_graph_op_t op,
        dnnl_graph_op_attr_t name, const char *value, size_t value_len);

/// Returns the unique id of an op.
///
/// @param op Input op.
/// @param id Output the unique id.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_get_id(
        const_dnnl_graph_op_t op, size_t *id);

/// Returns the kind of an op.
///
/// @param op Input op.
/// @param kind Output op kind.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_op_get_kind(
        const_dnnl_graph_op_t op, dnnl_graph_op_kind_t *kind);

/// @} dnnl_graph_api_op

/// @addtogroup dnnl_graph_api_partition
/// @{

/// Creates a new partition with a given operator and engine kind. The API is
/// used to create a partition from an operation directly without creating the
/// graph and calling `get_partitions()`. The output partition contains only one
/// operation specified by the parameter. The output partition instance should
/// be destroyed via #dnnl_graph_partition_destroy after use.
///
/// @param partition The handle of output partition.
/// @param op The operation used to create partition.
/// @param ekind The engine kind used to create partition.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_create_with_op(
        dnnl_graph_partition_t *partition, const_dnnl_graph_op_t op,
        dnnl_engine_kind_t ekind);

/// Destroys a partition.
///
/// @param partition The partition to be destroyed.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_destroy(
        dnnl_graph_partition_t partition);

/// Returns the number of operations in a partition.
///
/// @param partition The target partition.
/// @param num Output the number of operations.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_op_num(
        const_dnnl_graph_partition_t partition, size_t *num);

/// Returns the list of op IDs of the partition.
///
/// @param partition The target partition.
/// @param num The number of ops.
/// @param ids Output the op IDs.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_ops(
        dnnl_graph_partition_t partition, size_t num, size_t *ids);

/// Returns the ID of a partition.
///
/// @param partition The target partition.
/// @param id Output the ID of the partition.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_id(
        const_dnnl_graph_partition_t partition, size_t *id);

/// Compiles a partition with given input and output logical tensors. The output
/// logical tensors can contain unknown dimensions. For this case, the
/// compilation will deduce the output shapes according to input shapes. The
/// output logical tensors can also have layout type `any`. The compilation will
/// choose the optimal layout for output tensors. The optimal layout will be
/// represented as an opaque layout ID saved in the output logical tensor.
///
/// @param partition The target partition.
/// @param compiled_partition Output compiled partition.
/// @param in_num The number of input logical tensors.
/// @param inputs A list of input logical tensors.
/// @param out_num The number of output logical tensors.
/// @param outputs A list of output logical tensors.
/// @param engine The target engine of the compilation.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_compile(
        dnnl_graph_partition_t partition,
        dnnl_graph_compiled_partition_t compiled_partition, size_t in_num,
        const dnnl_graph_logical_tensor_t **inputs, size_t out_num,
        const dnnl_graph_logical_tensor_t **outputs, dnnl_engine_t engine);

/// Returns the number of input logical tensors of a partition.
///
/// @param partition The target partition.
/// @param num Output the number of input logical tensors.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_input_ports_num(
        const_dnnl_graph_partition_t partition, size_t *num);

/// Returns a list of input logical tensors from a partition.
///
/// @param partition The target partition.
/// @param num The number of input logical tensors.
/// @param inputs The list of input logical tensors.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_input_ports(
        const_dnnl_graph_partition_t partition, size_t num,
        dnnl_graph_logical_tensor_t *inputs);

/// Returns the number of output logical tensors of a partition.
///
/// @param partition The target partition.
/// @param num Output the number of output logical tensors.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_output_ports_num(
        const_dnnl_graph_partition_t partition, size_t *num);

/// Returns a list of output logical tensors from a partition.
///
/// @param partition The target partition.
/// @param num The number of output logical tensors.
/// @param outputs The list of output logical tensors.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_output_ports(
        const_dnnl_graph_partition_t partition, size_t num,
        dnnl_graph_logical_tensor_t *outputs);

/// Returns the supporting status of a partition. Some operations may not be
/// supported by the library under certain circumstances. During partitioning
/// stage, unsupported partitions will be returned to users with each containing
/// an unsupported operation. Users should check the supporting status of a
/// partition before transforming the computation graph or compiling the
/// partition.
///
/// @param partition The target partition.
/// @param is_supported Output flag to indicate the supporting status. 0 means
///     unsupported while 1 means supported.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_is_supported(
        const_dnnl_graph_partition_t partition, uint8_t *is_supported);

/// Returns the engine kind of a partition.
///
/// @param partition The target partition.
/// @param kind The output engine kind.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_partition_get_engine_kind(
        const_dnnl_graph_partition_t partition, dnnl_engine_kind_t *kind);

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_compiled_partition
/// @{

/// Creates a new compiled partition handle.
///
/// @param compiled_partition The handle of output compiled partition.
/// @param partition The handle of input partition.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_create(
        dnnl_graph_compiled_partition_t *compiled_partition,
        dnnl_graph_partition_t partition);

/// Executes a compiled partition.
///
/// @param compiled_partition The handle of target compiled partition.
/// @param stream The stream used for execution.
/// @param num_inputs The number of input tensors.
/// @param inputs A list of input tensors.
/// @param num_outputs The number of output tensors.
/// @param outputs A non-empty list of output tensors.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_execute(
        const_dnnl_graph_compiled_partition_t compiled_partition,
        dnnl_stream_t stream, size_t num_inputs,
        const_dnnl_graph_tensor_t *inputs, size_t num_outputs,
        const_dnnl_graph_tensor_t *outputs);

/// Destroys a compiled partition.
///
/// @param compiled_partition The compiled partition to be destroyed.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_destroy(
        dnnl_graph_compiled_partition_t compiled_partition);

/// Queries an input or output logical tensor according to tensor ID. If the
/// tensor ID doesn't belong to any input or output of the compiled partition,
/// an error status #dnnl_invalid_arguments will be returned by the API.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param tid The unique id of required tensor.
/// @param lt The output logical tensor.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_query_logical_tensor(
        const_dnnl_graph_compiled_partition_t compiled_partition, size_t tid,
        dnnl_graph_logical_tensor_t *lt);

/// Returns the hint of in-place pairs from a compiled partition. It indicates
/// that an input and an output of the partition can share the same memory
/// buffer for computation. In-place computation helps to reduce the memory
/// footprint and improves cache locality. But since the library may not have a
/// global view of user's application, it's possible that the tensor with
/// `input_id` is used at other places in user's computation graph. In this
/// case, the user should take the in-place pair as a hint and pass a different
/// memory buffer for output tensor to avoid overwriting the input memory buffer
/// which will probably cause unexpected incorrect results.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param num_inplace_pairs The number of in-place pairs.
/// @param inplace_pairs The handle of in-place pairs.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_get_inplace_ports(
        const_dnnl_graph_compiled_partition_t compiled_partition,
        size_t *num_inplace_pairs,
        const dnnl_graph_inplace_pair_t **inplace_pairs);

/// @} dnnl_graph_api_compiled_partition

/// @addtogroup dnnl_graph_api_graph
/// @{

/// Creates a new empty graph. A graph is associated to a specific engine kind.
/// The partitions returned from the graph will inherit the engine kind of the
/// graph.
///
/// @param graph The handle of output graph.
/// @param engine_kind The target engine kind.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_create(
        dnnl_graph_graph_t *graph, dnnl_engine_kind_t engine_kind);

/// Creates a new empty graph with an engine kind and a floating-point math
/// mode. All partitions returned from the graph will inherit the engine kind
/// and floating-point math mode.
///
/// @param graph The handle of output graph.
/// @param engine_kind The kind for engine.
/// @param mode The floating-point math mode.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_create_with_fpmath_mode(
        dnnl_graph_graph_t *graph, dnnl_engine_kind_t engine_kind,
        dnnl_fpmath_mode_t mode);

/// Destroys a graph.
///
/// @param graph The graph to be destroyed.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_destroy(dnnl_graph_graph_t graph);

/// Adds an operation into a graph. The API will return failure if the operator
/// has already been added to the graph or the operation cannot pass the schema
/// check in the library (eg. input and output numbers and data types, the
/// attributes of the operation, etc.).
///
/// @param graph The target graph.
/// @param op The operation to be added.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_add_op(
        dnnl_graph_graph_t graph, dnnl_graph_op_t op);

/// Finalizes a graph. It means users have finished adding operations into the
/// graph and the graph is ready for partitioning. Adding a new operation into a
/// finalized graph will return failures. Similarly, partitioning on a
/// un-finalized graph will also return failures.
///
/// @param graph The target graph to be finalized.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_finalize(dnnl_graph_graph_t graph);

/// Checks if a graph is finalized.
///
/// @param graph The target graph to be finalized.
/// @param finalized Output the finalization status. 0 means then graph is not
///     finalized. Other values means the graph is finalized.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_is_finalized(
        dnnl_graph_graph_t graph, uint8_t *finalized);

/// Filters a graph. Partitions will be claimed internally according to the
/// capability of the library, the engine kind, and the policy.
///
/// @param graph The target graph.
/// @param policy The partition policy.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_filter(
        dnnl_graph_graph_t graph, dnnl_graph_partition_policy_t policy);

/// Returns the number of partitions of a graph. The API should be called after
/// a partition is already filtered. Otherwise, the output number is zero.
///
/// @param graph The graph.
/// @param num Output the number of partitions.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_get_partition_num(
        const_dnnl_graph_graph_t graph, size_t *num);

/// Returns the partitions from a filtered graph. Output partition instances
/// will be written into the parameter `partitions`. Users need to make sure
/// `partitions` is valid and has enough space to accept the partition
/// instances. Each output partition instance should be destroyed via
/// #dnnl_graph_partition_destroy explicitly after use.
///
/// @param graph The target graph.
/// @param num The number of partitions.
/// @param partitions Output the partitions.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_get_partitions(dnnl_graph_graph_t graph,
        size_t num, dnnl_graph_partition_t *partitions);

/// @} dnnl_graph_api_graph

/// @addtogroup dnnl_graph_api_compiled_partition_cache
/// @{

/// Returns the number of compiled partitions that can be held in the compiled
/// partition cache at the same time.
///
/// @param capacity Compiled partition cache capacity to query. Concurrently
/// accessing @p capacity is safe.
/// @returns #dnnl_invalid_arguments if the @p capacity value
///     is invalid, and #dnnl_success on success.
dnnl_status_t DNNL_API dnnl_graph_get_compiled_partition_cache_capacity(
        int *capacity);

/// Sets a number of compiled partitions that can be held in the compiled
/// partition cache at the same time. The default capacity of compiled partition
/// cache is 1024.
///
/// @param capacity Compiled partition cache capacity to set. The default cache
/// capacity is 1024. If a new @p capacity is less than a number of compiled
/// partition that the compiled partition cache already has, then the excess
/// entries will be evicted. Setting the @p capacity to 0 clears the compiled
/// partition cache and disables it. Concurrently modifying @p capacity is safe.
/// @returns #dnnl_invalid_arguments if the @p capacity value
/// is invalid, and #dnnl_success on success.
dnnl_status_t DNNL_API dnnl_graph_set_compiled_partition_cache_capacity(
        int capacity);

/// @} dnnl_graph_api_compiled_partition_cache

/// @addtogroup dnnl_graph_api_constant_tensor_cache
/// @{

/// Control the enabling or disabling of constant tensor cache. This API must
/// be called once before compilation stage. By default, constant tensor cache is
/// disabled in the library.
///
/// @param flag Set to positive value to enable the cache and set to 0 to
/// disable the cache. Negative values are invalid.
/// @returns #dnnl_invalid_arguments if the @p flag value is
/// invalid, and #dnnl_success on success.
/// @note This API is deprecated and will be removed in future release, please
/// use the dnnl_graph_set_constant_tensor_cache_capacity API to disable
/// constant tensor cache by setting it's capacity to zero.
dnnl_status_t DNNL_API dnnl_graph_set_constant_tensor_cache(int flag);

/// Return the enabling or disabling status of constant tensor cache.
///
/// @param flag The constant tensor cache enabling status to query.
/// @returns #dnnl_invalid_arguments if the @p flag value is
/// nullptr, and #dnnl_success on success.
/// @note This API is deprecated and will be removed in future release, please
/// use the dnnl_graph_get_constant_tensor_cache_capacity API to check the
/// enabling status by checking it's capacity.
dnnl_status_t DNNL_API dnnl_graph_get_constant_tensor_cache(int *flag);

/// Control the capacity for the constant tensor cache that used for specific
/// engine kind. This API is thread safe and can be called multiple times at
/// runtime. The capacity is set to zero by default which means the cache is
/// disabled. When calling this API, the corresponding cache will be flushed.
/// Setting capacity to 0 means to clear all cached tensors and disable cache.
/// Once the capacity limit is reached, no new tensors will be cached. If there
/// are multiple devices for an engine kind, the capacity set here is for each
/// device.
///
/// @param eng_kind The engine kind that the constant tensor cache used for.
/// @param size The constant tensor cache capacity size to set.
/// @returns #dnnl_invalid_arguments if the @p eng_kind value is invalid, and
/// #dnnl_success on success.
dnnl_status_t DNNL_API dnnl_graph_set_constant_tensor_cache_capacity(
        dnnl_engine_kind_t eng_kind, size_t size);

/// Return the current capacity of constant tensor cache.
///
/// @param eng_kind The engine kind that the constant tensor cache used for.
/// @param size The constant tensor cache capacity size to query.
/// @returns #dnnl_invalid_arguments if the @p eng_kind value is
/// nullptr or the @p size is nullptr, and #dnnl_success on success.
dnnl_status_t DNNL_API dnnl_graph_get_constant_tensor_cache_capacity(
        dnnl_engine_kind_t eng_kind, size_t *size);

/// @} dnnl_graph_api_constant_tensor_cache

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif
#endif
