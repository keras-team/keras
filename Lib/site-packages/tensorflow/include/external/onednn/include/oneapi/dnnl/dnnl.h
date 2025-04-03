/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#ifndef ONEAPI_DNNL_DNNL_H
#define ONEAPI_DNNL_DNNL_H

#include "oneapi/dnnl/dnnl_common.h"
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"
#include "oneapi/dnnl/dnnl_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_primitives
/// @{

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Changes the primitive descriptor to point to the next available
/// implementation.
///
/// @param primitive_desc A primitive descriptor to change.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
/// @returns #dnnl_last_impl_reached if no more implementations available,
/// in which case the primitive descriptor itself is kept unchanged.
dnnl_status_t DNNL_API dnnl_primitive_desc_next_impl(
        dnnl_primitive_desc_t primitive_desc);

/// Clones a primitive descriptor. The resulting primitive descriptor must be
/// destroyed separately.
///
/// @param primitive_desc Output primitive descriptor.
/// @param existing_primitive_desc Primitive descriptor to clone.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_desc_clone(
        dnnl_primitive_desc_t *primitive_desc,
        const_dnnl_primitive_desc_t existing_primitive_desc);

/// Returns a constant reference to the attributes of a primitive descriptor.
///
/// @warning
///     It is an error to destroy the resulting @p attr.
///
/// @warning
///     The lifetime of an @p attr is the same as that of a @p
///     primitive_desc, so it is an error to use the @p attr once the @p
///     primitive_desc has been destroyed.
///
/// @param primitive_desc Primitive descriptor.
/// @param attr Output primitive attributes.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_desc_get_attr(
        const_dnnl_primitive_desc_t primitive_desc,
        const_dnnl_primitive_attr_t *attr);

/// Destroys a primitive descriptor.
///
/// @param primitive_desc Primitive descriptor to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_desc_destroy(
        dnnl_primitive_desc_t primitive_desc);

/// Queries a primitive descriptor for various pieces of information.
///
/// The most common use case is to query a primitive descriptor, created with
/// source, weights, and destination memory descriptors with format tags set
/// to #dnnl_format_tag_any, for the corresponding memory descriptors (in this
/// case the @p what is set to #dnnl_query_src_md, #dnnl_query_weights_md, and
/// #dnnl_query_dst_md respectively) so that it is possible to create memory
/// objects and reorder primitives if necessary.
///
/// Another typical use case is to query a primitive descriptor for workspace
/// memory descriptor (with @p what set to #dnnl_query_workspace_md). If this
/// query returns #dnnl_not_required status, then workspace memory is not
/// required.
///
/// @note
///     When querying for a memory descriptor for a scratchpad, a workspace,
///     or an optional parameter, the query will return a pointer to a zero
///     memory descriptor if the parameter is not needed.
///
/// A few other use cases:
///  - query a primitive descriptor for the implementation information string
///    (#dnnl_query_impl_info_str)
///  - query a primitive descriptor for the number of inputs and outputs
///    (#dnnl_query_num_of_inputs_s32 and #dnnl_query_num_of_outputs_s32
///    respectively)
///
/// @sa dnnl_query_t for more options
///
/// @param primitive_desc Primitive descriptor.
/// @param what Parameter to query.
/// @param index Index of the parameter to query for.
/// @param result Output result. The type depends on the query. For example,
///     it must be a @c dnnl_memory_desc_t* if querying for a memory
///     descriptor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_desc_query(
        const_dnnl_primitive_desc_t primitive_desc, dnnl_query_t what,
        int index, void *result);

/// Queries primitive descriptor for a memory descriptor.
///
/// @note
///     This function is a convenience version of
///     #dnnl_primitive_desc_query().
///
/// @param primitive_desc Primitive descriptor.
/// @param what Kind of memory descriptor parameter to query for.
/// @param index Index of the parameter to query.
/// @returns A pointer to the requested memory descriptor.
/// @returns A pointer to a zero memory descriptor if the parameter is not
///          needed.
/// @returns NULL in case of any error.
///
const_dnnl_memory_desc_t DNNL_API dnnl_primitive_desc_query_md(
        const_dnnl_primitive_desc_t primitive_desc, dnnl_query_t what,
        int index);

/// Queries primitive descriptor for a signed 32bit int.
///
/// @note
///     This function is a convenience version of
///     #dnnl_primitive_desc_query().
///
/// @param primitive_desc Primitive descriptor.
/// @param what Kind of the value to query for.
/// @param index Index of the parameter to query.
/// @returns The requested value.
/// @returns 0 in case of any error (in particular if the queried entity is
///     not of type int32_t). Note that 0 may also be the actual returned
///     value.
int DNNL_API dnnl_primitive_desc_query_s32(
        const_dnnl_primitive_desc_t primitive_desc, dnnl_query_t what,
        int index);

/// Creates a primitive.
///
/// @param primitive Output primitive.
/// @param primitive_desc Primitive descriptor used to create the primitive.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_create(dnnl_primitive_t *primitive,
        const_dnnl_primitive_desc_t primitive_desc);

/// Creates a primitive from a cache blob.
///
/// @param primitive Output primitive.
/// @param primitive_desc Primitive descriptor used to create the primitive.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_create_from_cache_blob(
        dnnl_primitive_t *primitive, const_dnnl_primitive_desc_t primitive_desc,
        size_t size, const uint8_t *cache_blob);

/// Executes a primitive.
///
/// @param primitive Primitive to execute.
/// @param stream Stream to use.
/// @param nargs Number of arguments.
/// @param args Array of arguments. Each argument is an
///     <index, #dnnl_memory_t> pair. The index is one of the `DNNL_ARG_*`
///     values such as `DNNL_ARG_SRC`. Unless runtime shapes are used (see
///     #DNNL_RUNTIME_DIM_VAL), the memory object must have the same memory
///     descriptor as that returned by
///     #dnnl_primitive_desc_query_md(#dnnl_query_exec_arg_md, index).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.

/// @note If any argument in @p args is padded (padded_dims >
/// dims), the primitive execution will assume properly zero-padded
/// input arguments, and produce zero-padded output arguments.
dnnl_status_t DNNL_API dnnl_primitive_execute(const_dnnl_primitive_t primitive,
        dnnl_stream_t stream, int nargs, const dnnl_exec_arg_t *args);

/// Retrieves a constant reference to the primitive descriptor of a given
/// primitive.
///
/// @warning
///     It is an error to destroy the returned object. It is owned by the
///     primitive. The @c const qualifier of the returned object prevents
///     such attempts.
///
/// @param primitive Primitive to query for the primitive descriptor.
/// @param primitive_desc Output primitive descriptor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_get_primitive_desc(
        const_dnnl_primitive_t primitive,
        const_dnnl_primitive_desc_t *primitive_desc);

/// Retrieves a cache blob associated with the given primitive.
///
/// @param primitive Primitive to query for the cache blob.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size. If the @p cache_blob is
///     nullptr then the size of the cache blob is returned in @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
///
/// @note The cache blob can be empty. It's the user's responsibility to check
///     whether it's empty prior to passing it to
///     #dnnl_primitive_create_from_cache_blob().
dnnl_status_t DNNL_API dnnl_primitive_get_cache_blob(
        const_dnnl_primitive_t primitive, size_t *size, uint8_t *cache_blob);

/// Destroys a primitive.
///
/// @param primitive The primitive to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_destroy(dnnl_primitive_t primitive);

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_attributes
/// @{

/// Creates an empty (default) primitive attributes with all the parameters
/// set to their default values.
///
/// Empty attributes are implied whenever the respective argument is NULL.
///
/// @param attr Output primitive attributes.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_create(dnnl_primitive_attr_t *attr);

/// Clones primitive attributes.
///
/// @param attr Output primitive attributes.
/// @param existing_attr Primitive attributes to clone.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_clone(
        dnnl_primitive_attr_t *attr, const_dnnl_primitive_attr_t existing_attr);

/// Destroys primitive attributes.
///
/// @param attr Primitive attributes to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_destroy(dnnl_primitive_attr_t attr);

/// Returns the floating-point math mode primitive attribute.
///
/// @param attr Primitive attributes.
/// @param mode Output FP math mode.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_fpmath_mode(
        const_dnnl_primitive_attr_t attr, dnnl_fpmath_mode_t *mode);

/// Sets the floating-point math mode primitive attributes.
///
/// @param attr Primitive attributes.
/// @param mode FP math mode. The possible values are:
///     #dnnl_fpmath_mode_strict (default),
///     #dnnl_fpmath_mode_bf16,
///     #dnnl_fpmath_mode_f16,
///     #dnnl_fpmath_mode_tf32,
///     #dnnl_fpmath_mode_any.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_fpmath_mode(
        dnnl_primitive_attr_t attr, dnnl_fpmath_mode_t mode);

/// Returns the floating-point math mode primitive attribute.
///
/// @param attr Primitive attributes.
/// @param mode Output FP math mode.
/// @param apply_to_int Output use floating-point arithmetic for integer primitives.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_fpmath_mode_v2(
        const_dnnl_primitive_attr_t attr, dnnl_fpmath_mode_t *mode,
        int *apply_to_int);

/// Sets the floating-point math mode primitive attributes.
///
/// @param attr Primitive attributes.
/// @param mode FP math mode. The possible values are:
///     #dnnl_fpmath_mode_strict (default),
///     #dnnl_fpmath_mode_bf16,
///     #dnnl_fpmath_mode_f16,
///     #dnnl_fpmath_mode_tf32,
///     #dnnl_fpmath_mode_any.
/// @param apply_to_int Boolean. Use of floating-point arithmetic for integer primitives.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_fpmath_mode_v2(
        dnnl_primitive_attr_t attr, dnnl_fpmath_mode_t mode, int apply_to_int);

/// Returns the deterministic primitive attribute value.
///
/// @param attr Primitive attributes.
/// @param value Output deterministic attribute value
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_deterministic(
        const_dnnl_primitive_attr_t attr, int *value);

/// Sets the deterministic primitive attribute value.
///
/// @param attr Primitive attributes.
/// @param value Boolean value to set deterministic attribute.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_deterministic(
        dnnl_primitive_attr_t attr, int value);

/// Returns the accumulation mode primitive attribute.
///
/// @param attr Primitive attributes.
/// @param mode Output accumulation mode.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_accumulation_mode(
        const_dnnl_primitive_attr_t attr, dnnl_accumulation_mode_t *mode);

/// Sets the accumulation mode primitive attribute.
///
/// @param attr Primitive attributes.
/// @param mode Accumulation mode. The possible values are:
///     #dnnl_accumulation_mode_strict (default), which is s32 for quantized primitives, f32/f64 otherwise
///     #dnnl_accumulation_mode_relaxed, which is same as strict but allows intermediate accumulators to be in src/dst datatype
///     #dnnl_accumulation_mode_any, which allows accumulators to be src/dst datatype or any wider type.
///     #dnnl_accumulation_mode_f32,
///     #dnnl_accumulation_mode_s32,
///     #dnnl_accumulation_mode_f16.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_accumulation_mode(
        dnnl_primitive_attr_t attr, dnnl_accumulation_mode_t mode);

/// Returns the primitive attributes scratchpad mode.
///
/// @param attr Primitive attributes.
/// @param mode Output scratchpad mode.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_scratchpad_mode(
        const_dnnl_primitive_attr_t attr, dnnl_scratchpad_mode_t *mode);

/// Sets primitive attributes scratchpad mode.
///
/// @param attr Primitive attributes.
/// @param mode Scratchpad mode. The possible values are:
///     #dnnl_scratchpad_mode_library (default) and
///     #dnnl_scratchpad_mode_user.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_scratchpad_mode(
        dnnl_primitive_attr_t attr, dnnl_scratchpad_mode_t mode);

/// Sets primitive attributes scaling factors for primitive operations for a
/// given memory argument. The scaling factors must be passed at execution time
/// as an argument with index #DNNL_ARG_ATTR_SCALES | arg.
///
/// @sa dnnl_primitive_attr_set_scales_mask
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     dnnl_primitive_execute() call.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the tensor dimensions and the @p scales array.
///     The set i-th bit indicates that a dedicated scaling factor is used for
///     each index along that dimension. Set the mask to 0 to use a common
///     scaling factor for the whole output tensor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_scales_mask(
        dnnl_primitive_attr_t attr, int arg, int mask);

/// Sets primitive attributes scaling factors for primitive operations for a
/// given memory argument. The scaling factors must be passed at execution time
/// as an argument with index #DNNL_ARG_ATTR_SCALES | arg.
///
/// @sa dnnl_primitive_attr_set_scales
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     dnnl_primitive_execute() call.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the tensor dimensions and the @p scales array.
///     The set i-th bit indicates that a dedicated scaling factor is used for
///     each index along that dimension. Set the mask to 0 to use a common
///     scaling factor for the whole output tensor.
/// @param ndims Number of group dimensions.
/// @param group_dims Scaling factors correspondence groups that define the
///     correspondence between the tensor dimensions and the scales array.
///     The group dimensions should only be provided for each logical dimension
///     that has correspondence mask @p mask set.
/// @param data_type Scaling factors data_type.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_scales(
        dnnl_primitive_attr_t attr, int arg, int mask, int ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type);

/// Sets primitive attributes zero points for primitive operations for a given
/// memory argument. The zero points must be passed at execution time
/// as an argument with index #DNNL_ARG_ATTR_ZERO_POINTS | arg.
///
/// @sa dnnl_primitive_attr_set_zero_points_mask
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     dnnl_primitive_execute() call.
/// @param mask Zero point correspondence mask that defines the
///     correspondence between the tensor dimensions and the @p
///     zero_points array. The set i-th bit indicates that a dedicated
///     zero point is used for each index along that dimension. Set the
///     mask to 0 to use a common zero point for the whole output tensor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_zero_points_mask(
        dnnl_primitive_attr_t attr, int arg, int mask);

/// Sets primitive attributes zero points for primitive operations for a given
/// memory argument. The zero points must be passed at execution time
/// as an argument with index #DNNL_ARG_ATTR_ZERO_POINTS | arg.
///
/// @sa dnnl_primitive_attr_set_zero_points
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     dnnl_primitive_execute() call.
/// @param mask Zero point correspondence mask that defines the
///     correspondence between the tensor dimensions and the @p
///     zero_points array. The set i-th bit indicates that a dedicated
///     zero point is used for each index along that dimension. Set the
///     mask to 0 to use a common zero point for the whole output tensor.
/// @param ndims Number of group dimensions.
/// @param group_dims Zero point factors correspondence groups that define the
///     correspondence between the tensor dimensions and the zero_points array.
///     The group dimensions should be only provided for each logical dimension
///     that has the bit set correspondence mask @p mask set.
/// @param data_type Zero points factors data_type.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_zero_points(
        dnnl_primitive_attr_t attr, int arg, int mask, int ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type);

/// Returns primitive attributes post-ops.
///
/// @warning
///     The output @p post_ops points to the internal @p attr field, so it is
///     an error to modify or destroy them. The lifetime of @p post_ops is
///     the same as that of the @p attr it belongs to, so it is an error to
///     use @p post_ops after @p attr has been destroyed.
///
/// @param attr Primitive attributes.
/// @param post_ops Output post-ops.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_post_ops(
        const_dnnl_primitive_attr_t attr, const_dnnl_post_ops_t *post_ops);

/// Sets primitive attributes post-ops.
///
/// @note
///     There is no way to check whether the post-ops would be supported by
///     the target primitive. Any error will be reported by the
///     dnnl_<primitive name>_[propagation kind]_primitive_desc_create() function call.
///
/// @param attr Primitive attributes.
/// @param post_ops Post-ops to set.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_post_ops(
        dnnl_primitive_attr_t attr, const_dnnl_post_ops_t post_ops);

/// Creates empty post-ops sequence.
///
/// @param post_ops Output post-ops.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_create(dnnl_post_ops_t *post_ops);

/// Clones post-ops primitive attribute.
///
/// @param post_ops Output post-ops primitive attribute.
/// @param existing_post_ops Post-ops primitive attribute to clone.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_clone(
        dnnl_post_ops_t *post_ops, const_dnnl_post_ops_t existing_post_ops);

/// Destroys post-ops.
///
/// @param post_ops Post-ops to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_destroy(dnnl_post_ops_t post_ops);

/// Returns the length of post-ops.
///
/// @param post_ops Post-ops.
/// @returns The number of post-ops entries.
int DNNL_API dnnl_post_ops_len(const_dnnl_post_ops_t post_ops);

/// Returns the kind of a post-op entry.
///
/// @param post_ops Post-ops.
/// @param index Post-op entry index.
/// @returns The kind of the post-op with the specified index.
/// @returns #dnnl_undefined_primitive if there is no post-op at the specified
///     index.
dnnl_primitive_kind_t DNNL_API dnnl_post_ops_get_kind(
        const_dnnl_post_ops_t post_ops, int index);

/// Appends an accumulation v3 (sum) to post-ops. Prior to accumulating the
/// result, a zero point is subtracted from the previous value and is
/// multiplied by the scale.
///
/// The kind of this post-op is #dnnl_sum.
///
/// This feature may improve performance for cases like dequantize the
/// asymmetrically quantized sum's src1 tensor to f32 domain before performing
/// the sum operation by subtracting the @p zero_point before the scaling.
///
/// In the simplest case where accumulation is the only post-op, the
/// computations will be:
///
///     dst[:] <- scale * (dst[:] - zero_point) + op(...)
///                                             // instead of dst[:] <- op(...)
///
/// If @p data_type is specified, original dst tensor will be reinterpreted
/// as a tensor with provided data type. Since it is reinterpretation,
/// data_type and dst data type should have the same size.
/// As a result, computations will be:
///
///     dst[:] <- scale * (as_data_type(dst[:]) - zero_point) + op(...)
///                                        // instead of dst[:] <- op(...)
/// @note
///     This post-op executes in-place and does not change the
///     destination layout.
///
/// @param post_ops Post-ops.
/// @param scale Accumulation scaling factor.
/// @param zero_point Single scalar int32_t value of zero point.
/// @param data_type Accumulation data_type.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_append_sum(dnnl_post_ops_t post_ops,
        float scale, int32_t zero_point, dnnl_data_type_t data_type);

/// Returns the parameters of an accumulation (sum) post-op with
/// zero point and data type parameter.
///
/// @param post_ops Post-ops.
/// @param index Index of the sum post-op.
/// @param scale Output accumulation scaling factor.
/// @param zero_point Zero point.
/// @param data_type Data type for accumulation.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_get_params_sum(
        const_dnnl_post_ops_t post_ops, int index, float *scale,
        int32_t *zero_point, dnnl_data_type_t *data_type);

/// Appends an elementwise post-op.
///
/// The kind of this post operation is #dnnl_eltwise.
///
/// In the simplest case when the elementwise is the only post operation, the
/// computations would be:
///
///     dst[:] <- eltwise_op (op(...)) // instead of dst[:] <- op(...)
///
/// where eltwise_op is configured with the given parameters.
///
/// @param post_ops Post-ops.
/// @param alg_kind Elementwise algorithm for the post-op.
/// @param alpha Alpha parameter for the elementwise algorithm.
/// @param beta Beta parameter for the elementwise algorithm.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_append_eltwise(dnnl_post_ops_t post_ops,
        dnnl_alg_kind_t alg_kind, float alpha, float beta);

/// Returns the parameters of an elementwise post-op.
///
/// @param post_ops Post-ops.
/// @param index Index of the elementwise post-op.
/// @param alg_kind Output elementwise algorithm kind.
/// @param alpha Output alpha parameter for the elementwise algorithm.
/// @param beta Output beta parameter for the elementwise algorithm.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
/// @returns #dnnl_invalid_arguments if @p index does not refer to an
///     elementwise post-op.
dnnl_status_t DNNL_API dnnl_post_ops_get_params_eltwise(
        const_dnnl_post_ops_t post_ops, int index, dnnl_alg_kind_t *alg_kind,
        float *alpha, float *beta);

/// Appends a depthwise post-op convolution.
///
/// This post-op can only be fused with a 2D 1x1 convolution (convolution with
/// weights spatial dimensions equal to 1 i.e., kh=kw=1).
///
/// The kind of this post-op is #dnnl_convolution.
///
/// The number of outputs for primitive with fusion is one. The output spatial
/// size can be derived as below:
///
/// output_height = ceil(output_height_1x1_convolution, stride)
/// output_width = ceil(output_width_1x1_convolution, stride)
///
/// See @ref dev_guide_attributes_post_ops_depthwise and
/// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
///
/// @param post_ops Post-ops.
/// @param weights_data_type Weights data type of depthwise post-op
/// @param bias_data_type Bias data type of depthwise post-op
/// @param dst_data_type Output data type of depthwise post-op
/// @param kernel_size Size of kernel of depthwise post-op
/// @param stride_size Size of stride of depthwise post-op
/// @param padding_l_size Size of left and top paddings of depthwise post-op
/// @returns #dnnl_success on success and a status describing the error
///     otherwise
dnnl_status_t DNNL_API dnnl_post_ops_append_dw(dnnl_post_ops_t post_ops,
        dnnl_data_type_t weights_data_type, dnnl_data_type_t bias_data_type,
        dnnl_data_type_t dst_data_type, dnnl_dim_t kernel_size,
        dnnl_dim_t stride_size, dnnl_dim_t padding_l_size);

/// Returns the parameters of an depthwise post-op.
///
/// @param post_ops Post-ops.
/// @param index Index of the elementwise post-op.
/// @param weights_data_type Weights data type of depthwise post-op
/// @param bias_data_type Bias data type of depthwise post-op
/// @param dst_data_type Output data type of depthwise post-op
/// @param kernel_size Size of kernel of depthwise post-op
/// @param stride_size Size of stride of depthwise post-op
/// @param padding_l_size Size of left and top paddings of depthwise post-op
/// @returns #dnnl_success on success and a status describing the error
///     otherwise
dnnl_status_t DNNL_API dnnl_post_ops_get_params_dw(
        const_dnnl_post_ops_t post_ops, int index,
        dnnl_data_type_t *weights_data_type, dnnl_data_type_t *bias_data_type,
        dnnl_data_type_t *dst_data_type, dnnl_dim_t *kernel_size,
        dnnl_dim_t *stride_size, dnnl_dim_t *padding_l_size);

/// Appends a binary post-op.
///
/// The kind of this post operation is #dnnl_binary.
///
/// In the simplest case when the binary is the only post operation, the
/// computations would be:
///
///     dst[:] <- binary_op (dst[:], another_input[:])
///
/// where binary_op is configured with the given parameters. binary_op supports
/// broadcast semantics for a second operand.
///
/// @param post_ops Post-ops.
/// @param alg_kind Binary algorithm for the post-op.
/// @param src1_desc Memory descriptor of a second operand.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_append_binary(dnnl_post_ops_t post_ops,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t src1_desc);

/// Returns the parameters of a binary post-op.
///
/// @param post_ops Post-ops.
/// @param index Index of the binary post-op.
/// @param alg_kind Output binary algorithm kind.
/// @param src1_desc Output memory descriptor of a second operand.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
/// @returns #dnnl_invalid_arguments if @p index does not refer to a binary
///     post-op.
dnnl_status_t DNNL_API dnnl_post_ops_get_params_binary(
        const_dnnl_post_ops_t post_ops, int index, dnnl_alg_kind_t *alg_kind,
        const_dnnl_memory_desc_t *src1_desc);

/// Appends a prelu forward post-op.
///
/// The kind of this post-op is #dnnl::primitive::kind::prelu.
///
/// The post-op can be defined as:
///
///      dst[:] <- prelu(dst[:], weights[:])
///      prelu:
///      dst[:] <- dst[:] if dst[:] > 0
///      dst[:] <- dst[:] * weights[:] if dst[:] <= 0
///
///
/// @note
///     The order of dimensions does not depend on how elements are laid
///     out in memory. For example:
///     - for a 2D CNN activations tensor the order is always (n, c)
///     - for a 4D CNN activations tensor the order is always (n, c, h, w)
///     - for a 5D CNN weights tensor the order is always
///        (g, oc, ic, kh, kw)
///
///    Prelu weights tensor is passed in runtime execution phase. Prelu
///    weights tensor data type is implicitly assumed as f32 using plain
///    layout (a, ab, acb, acdb, acdeb)
///
/// @param post_ops Post-ops.
/// @param mask Defines the correspondence between the output tensor
///     dimensions and the prelu weights tensor. The set i-th bit indicates
///     that a dedicated weights value is used for each index along that
///     dimension. Set the mask to 0 to use a common weights value
///     for the whole output tensor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_append_prelu(
        dnnl_post_ops_t post_ops, int mask);

/// Returns the parameters of a prelu post-op.
///
/// @param post_ops Post-ops.
/// @param index Index of the prelu post-op.
/// @param mask Mask of the prelu post-op.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_post_ops_get_params_prelu(
        const_dnnl_post_ops_t post_ops, int index, int *mask);

/// @} dnnl_api_attributes

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_memory
/// @{

/// Destroys a memory descriptor.
///
/// @param memory_desc Memory descriptor to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_destroy(dnnl_memory_desc_t memory_desc);

/// Clones a memory descriptor. The resulting memory descriptor must be
/// destroyed separately.
///
/// @param memory_desc Output memory descriptor.
/// @param existing_memory_desc Memory descriptor to clone.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_clone(dnnl_memory_desc_t *memory_desc,
        const_dnnl_memory_desc_t existing_memory_desc);

/// Retrieves a binary blob associated with the given memory descriptor
///
/// @param blob Output pointer to binary blob.
///     If not nullptr, size bytes of the memory descriptor blob are written.
/// @param size Output pointer to the size of the binary blob in bytes.
///     Size is written if blob is nullptr.
/// @param memory_desc input memory descriptor to serialize
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_get_blob(
        uint8_t *blob, size_t *size, const_dnnl_memory_desc_t memory_desc);

/// Creates a memory descriptor from a memory descriptor binary blob.
///
/// @param memory_desc Output pointer to a newly allocated memory descriptor.
/// @param blob Pointer to a memory descriptor binary blob.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_create_with_blob(
        dnnl_memory_desc_t *memory_desc, const uint8_t *blob);

/// Creates a memory descriptor using dimensions and strides.
///
/// @note
///     As always, the logical order of dimensions corresponds to the `abc...`
///     format tag, and the physical meaning of the dimensions depends on both
///     the primitive that consumes the memory and the context of that
///     consumption.
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param strides Strides in each dimension.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_create_with_strides(
        dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, const dnnl_dims_t strides);

/// Creates a memory descriptor using dimensions and memory format tag.
///
/// @note
///     As always, the logical order of dimensions corresponds to the `abc...`
///     format tag, and the physical meaning of the dimensions depends on both
///     the primitive that consumes the memory and the context of that
///     consumption.
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param tag Memory format tag. Can be #dnnl_format_tag_any which would
///     allow a primitive to chose the final memory format. In this case the
///     format_kind field of the memory descriptor would be set to
///     #dnnl_format_kind_any.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_create_with_tag(
        dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, dnnl_format_tag_t tag);

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Creates a memory descriptor for CSR encoding.
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param nnz Number of non-zero entries.
/// @param indices_dt Data type of indices.
/// @param pointers_dt Data type of pointers.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_create_with_csr_encoding(
        dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, dnnl_dim_t nnz, dnnl_data_type_t indices_dt,
        dnnl_data_type_t pointers_dt);

/// Creates a memory descriptor for packed sparse encoding.
///
/// The created memory descriptor cannot be used to create a memory
/// object. It can only be used to create a primitive descriptor to
/// query the actual memory descriptor (similar to the format tag
/// `any`).
///
/// @warning
///     The meaning and content of the handles of the memory object that
///     is created using the queried memory descriptor are unspecified
///     therefore using the content is an undefined behavior.
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param nnz Number of non-zero entries.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_create_with_packed_encoding(
        dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, dnnl_dim_t nnz);
#endif

/// Creates a memory descriptor for a region inside an area
/// described by an existing memory descriptor.
///
/// @warning
///     Some combinations of physical memory layout and/or offsets or dims may
///     result in a failure to create a submemory.
//
/// @param memory_desc Output memory descriptor.
/// @param parent_memory_desc An existing memory descriptor.
/// @param dims Sizes of the region.
/// @param offsets Offsets to the region from the encompassing
///     memory object in each dimension
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_create_submemory(
        dnnl_memory_desc_t *memory_desc,
        const_dnnl_memory_desc_t parent_memory_desc, const dnnl_dims_t dims,
        const dnnl_dims_t offsets);

/// Creates a memory descriptor by reshaping an existing one. The new
/// memory descriptor inherits the data type. This operation is valid only for
/// memory descriptors that have format_kind #dnnl_blocked or
/// #dnnl_format_kind_any.
///
/// The resulting memory descriptor must be destroyed separately.
///
/// The operation ensures the transformation of the physical memory format
/// corresponds to the transformation of the logical dimensions. If such
/// transformation is impossible, the function returns #dnnl_invalid_arguments.
///
/// The reshape operation can be described as a combination of the following
/// basic operations:
/// 1. Add a dimension of size `1`. This is always possible.
/// 2. Remove a dimension of size `1`. This is possible only if the dimension
///    has no padding (i.e. `padded_dims[dim] == dims[dim] && dims[dim] == 1`).
/// 3. Split a dimension into multiple ones. This is possible only if the size
///    of the dimension is exactly equal to the product of the split ones and
///    the dimension does not have padding (i.e.
///    `padded_dims[dim] = dims[dim]`).
/// 4. Joining multiple consecutive dimensions into a single one. As in the
///    cases above, this requires that the dimensions do not have padding and
///    that the memory format is such that in physical memory these dimensions
///    are dense and have the same order as their logical counterparts. This
///    also assumes that these dimensions are not blocked.
///    - Here, dense means:
///      `stride for dim[i] == (stride for dim[i + 1]) * dim[i + 1]`;
///    - And same order means:
///      `i < j` if and only if `stride for dim[j] <= stride for dim[i]`.
///
/// @warning
///     Some combinations of physical memory layout and/or offsets or
///     dimensions may result in a failure to make a reshape.
///
/// @param out_memory_desc Output memory descriptor.
/// @param in_memory_desc An existing memory descriptor. Must have format_kind
///     set to #dnnl_blocked or #dnnl_format_kind_any.
/// @param ndims Number of dimensions for the output memory descriptor.
/// @param dims Dimensions for the output memory descriptor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_reshape(
        dnnl_memory_desc_t *out_memory_desc,
        const_dnnl_memory_desc_t in_memory_desc, int ndims,
        const dnnl_dims_t dims);

/// Creates a memory descriptor by permuting axes in an existing one.
///
/// The physical memory layout representation is adjusted accordingly to
/// maintain the consistency between the logical and physical parts of the
/// memory descriptor.
///
/// The resulting memory descriptor must be destroyed separately.
///
/// The new memory descriptor inherits the data type. This operation is valid
/// only for memory descriptors that have format_kind set to #dnnl_blocked or
/// #dnnl_format_kind_any.
///
/// The logical axes will be permuted in the following manner:
/// ```
/// for (i: 0 .. in_memory_desc->ndims)
///     out_memory_desc->dims[permutation[i]] = in_memory_desc->dims[i];
/// ```
///
/// Example:
/// @code
///     dnnl_memory_desc_t in_md, out_md, expect_out_md;
///
///     const int permutation[] = {1, 0}; // swap the first and the second axes
///
///     dnnl_dims_t in_dims = {2, 3}, out_dims = {3, 2};
///     dnnl_format_tag_t in_tag = dnnl_ab, out_tag = dnnl_ba;
///
///     dnnl_memory_desc_create_with_tag(
///             &in_md, 2, in_dims, data_type, in_tag);
///     dnnl_memory_desc_create_with_tag(
///             &expect_out_md, 2, out_dims, data_type, out_tag);
///
///     dnnl_memory_desc_permute_axes(&out_md, in_md, permutation);
///     assert(dnnl_memory_desc_equal(out_md, expect_out_md));
///
///     dnnl_memory_desc_destroy(in_md);
///     dnnl_memory_desc_destroy(out_md);
///     dnnl_memory_desc_destroy(expect_out_md);
/// @endcode
///
/// @param out_memory_desc Output memory descriptor.
/// @param in_memory_desc An existing memory descriptor. Must have format_kind
///     set to #dnnl_blocked or #dnnl_format_kind_any.
/// @param permutation Axes permutation (of size `in_memory_desc->ndims`).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_permute_axes(
        dnnl_memory_desc_t *out_memory_desc,
        const_dnnl_memory_desc_t in_memory_desc, const int *permutation);

/// Queries a memory descriptor for various pieces of information.
///
/// The following information can be queried:
///  - Number of dimensions (#dnnl_query_ndims_s32)
///  - Dimensions (#dnnl_query_dims) in the following order:
///    - CNN data tensors: mini-batch, channel, spatial
///      (<code>{N, C, [[D,] H,] W}</code>)
///    - CNN weight tensors: group (optional), output channel, input channel,
///      spatial (<code>{[G,] O, I, [[D,] H,] W}</code>)
///    - RNN data tensors: time, mini-batch, channels (<code>{T, N, C}</code>)
///      or layers, directions, states, mini-batch, channels
///      (<code>{L, D, S, N, C}</code>)
///    - RNN weight tensor: layers, directions, input channel, gates, output
///      channels (<code>{L, D, I, G, O}</code>)
///  - Data type of the tensor elements (#dnnl_query_data_type)
///  - Padded dimensions (#dnnl_query_padded_dims) - size of the data including
///    padding in each dimension
///  - Padded offsets (#dnnl_query_padded_offsets) - per-dimension offset from
///    the padding to actual data, the top-level tensor with offsets applied
///    must lie within the padding area.
///  - Submemory offset (#dnnl_query_submemory_offset_s64) - offset from memory
///    origin to the current block, non-zero only in a description of a memory
///    sub-block.
///  - Format kind (#dnnl_query_format_kind) - memory format kind
///
/// @note
///    The order of dimensions does not depend on the memory format, so
///    whether the data is laid out in #dnnl_nchw or #dnnl_nhwc
///    the dims for 4D CN data tensor would be <code>{N, C, H, W}</code>.
///
/// The following queries are applicable only to format kind #dnnl_blocked.
///  - Strides (#dnnl_query_strides) between the outermost blocks or in case
///    of plain (non-blocked) formats the strides between dimensions
///  - Number of innermost blocks (#dnnl_query_inner_nblks_s32), e.g.
///    `{4, 16, 4}` in case of `OIhw_4i16o4i`
///  - Size of the innermost blocks (#dnnl_query_inner_blks), e.g. 3 in case
///    of `OIhw_4i16o4i_`
///  - Logical indices of the blocks (#dnnl_query_inner_idxs), e.g. `{1, 0, 1}`
///    in case of `4i16o4i`, because `i` is the 1st dim and `o` is the 0st dim
///
/// @param memory_desc Memory descriptor.
/// @param what Parameter to query.
/// @param result Output result. The type depends on the query. For example,
///     it must be a @c dnnl_dims_t** if querying for a strides.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_query(
        const_dnnl_memory_desc_t memory_desc, dnnl_query_t what, void *result);

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Queries a memory descriptor for various pieces of information. This version
/// support additional queries #dnnl_query_sparse_encoding, #dnnl_query_nnz_s64
/// #dnnl_query_num_handles_s32 and #dnnl_query_data_type for a particular
/// buffer.
///
/// The following information can be queried:
///  - Number of dimensions (#dnnl_query_ndims_s32)
///  - Dimensions (#dnnl_query_dims) in the following order:
///    - CNN data tensors: mini-batch, channel, spatial
///      (<code>{N, C, [[D,] H,] W}</code>)
///    - CNN weight tensors: group (optional), output channel, input channel,
///      spatial (<code>{[G,] O, I, [[D,] H,] W}</code>)
///    - RNN data tensors: time, mini-batch, channels (<code>{T, N, C}</code>)
///      or layers, directions, states, mini-batch, channels
///      (<code>{L, D, S, N, C}</code>)
///    - RNN weight tensor: layers, directions, input channel, gates, output
///      channels (<code>{L, D, I, G, O}</code>)
///  - Data type of the tensor elements (#dnnl_query_data_type)
///  - Padded dimensions (#dnnl_query_padded_dims) - size of the data including
///    padding in each dimension
///  - Padded offsets (#dnnl_query_padded_offsets) - per-dimension offset from
///    the padding to actual data, the top-level tensor with offsets applied
///    must lie within the padding area.
///  - Submemory offset (#dnnl_query_submemory_offset_s64) - offset from memory
///    origin to the current block, non-zero only in a description of a memory
///    sub-block.
///  - Format kind (#dnnl_query_format_kind) - memory format kind
///
/// @note
///    The order of dimensions does not depend on the memory format, so
///    whether the data is laid out in #dnnl_nchw or #dnnl_nhwc
///    the dims for 4D CN data tensor would be <code>{N, C, H, W}</code>.
///
/// The following queries are applicable only to format kind #dnnl_blocked.
///  - Strides (#dnnl_query_strides) between the outermost blocks or in case
///    of plain (non-blocked) formats the strides between dimensions
///  - Number of innermost blocks (#dnnl_query_inner_nblks_s32), e.g.
///    `{4, 16, 4}` in case of `OIhw_4i16o4i`
///  - Size of the innermost blocks (#dnnl_query_inner_blks), e.g. 3 in case
///    of `OIhw_4i16o4i_`
///  - Logical indices of the blocks (#dnnl_query_inner_idxs), e.g. `{1, 0, 1}`
///    in case of `4i16o4i`, because `i` is the 1st dim and `o` is the 0st dim
///
/// @param memory_desc Memory descriptor.
/// @param what Parameter to query.
/// @param index Index of the parameter to query for. It is mostly used with
///     #dnnl_query_data_type to specify which data type is being queried.
///     The main data type (data type of values) has always index 0. For other
///     indices please refer to the API for creating a memory descriptor for
///     sparse encoding.
/// @param result Output result. The type depends on the query. For example,
///     it must be a @c dnnl_dims_t** if querying for a strides.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_query_v2(
        const_dnnl_memory_desc_t memory_desc, dnnl_query_t what, int index,
        void *result);
#endif

/// Compares two memory descriptors.
///
/// Use this function to identify whether a reorder is required between the
/// two memories
///
/// @param lhs Left-hand side of the comparison.
/// @param rhs Right-hand side of the comparison.
/// @returns 1 if the descriptors are the same.
/// @returns 0 if the descriptors are different.
int DNNL_API dnnl_memory_desc_equal(
        const_dnnl_memory_desc_t lhs, const_dnnl_memory_desc_t rhs);

/// Returns the size of a memory descriptor.
///
/// @param memory_desc Memory descriptor.
/// @returns The number of bytes required for memory described by a memory
///     descriptor.
size_t DNNL_API dnnl_memory_desc_get_size(const_dnnl_memory_desc_t memory_desc);

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Returns the size of the data that corresponds to the given index.
///
/// @param memory_desc Memory descriptor.
/// @param index Index of the buffer.
///
/// @returns The number of bytes required for the requested data.
size_t DNNL_API dnnl_memory_desc_get_size_v2(
        const_dnnl_memory_desc_t memory_desc, int index);
#endif

/// Returns the size of data type.
///
/// @param data_type Data type.
/// @returns The number of bytes occupied by data type.
size_t DNNL_API dnnl_data_type_size(dnnl_data_type_t data_type);

/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE, the constructed memory
/// object will have the underlying buffer set. In this case, the buffer will
/// be initialized as if dnnl_memory_set_data_handle() had been called.
///
/// @sa dnnl_memory_set_data_handle()
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the memory object. In this case the library
///       owns the buffer.
///     - DNNL_MEMORY_NONE to create dnnl_memory without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_create(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        void *handle);

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Creates a memory object with multiple handles.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param nhandles Number of handles.
/// @param handles Handles of the memory buffers to use as underlying storages.
///     For each element of the @p handles array the following applies:
///     - A pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the memory object. In this case the library
///       owns the buffer.
///     - DNNL_MEMORY_NONE Instructs the library to skip allocation of the
///       memory buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_create_v2(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        int nhandles, void **handles);
#endif

/// Returns the memory descriptor for a memory object.
///
/// @param memory Memory object.
/// @param memory_desc Output memory descriptor (a copy).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_get_memory_desc(
        const_dnnl_memory_t memory, const_dnnl_memory_desc_t *memory_desc);

/// Returns the engine of a memory object.
///
/// @param memory Memory object.
/// @param engine Output engine on which the memory is located.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_get_engine(
        const_dnnl_memory_t memory, dnnl_engine_t *engine);

/// Maps a memory object and returns a host-side pointer to a memory buffer
/// with a copy of its contents.
///
/// Mapping enables explicit direct access to memory contents for the engines
/// that do not support it implicitly.
///
/// Mapping is an exclusive operation - a memory object cannot be used in
/// other operations until this memory object is unmapped.
///
/// @note
///     Any primitives working with @p memory should be completed before
///     the memory is mapped. Use dnnl_stream_wait to synchronize the
///     corresponding execution stream.
///
/// @note
///     The dnnl_memory_map_data() and dnnl_memory_unmap_data() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Output pointer to the mapped buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_map_data(
        const_dnnl_memory_t memory, void **mapped_ptr);

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Maps a memory object and returns a host-side pointer to a memory buffer
/// with a copy of its contents. The memory buffer corresponds to the given
/// index.
///
/// Mapping enables explicit direct access to memory contents for the engines
/// that do not support it implicitly.
///
/// Mapping is an exclusive operation - a memory object cannot be used in
/// other operations until this memory object is unmapped.
///
/// @note
///     Any primitives working with @p memory should be completed before
///     the memory is mapped. Use dnnl_stream_wait to synchronize the
///     corresponding execution stream.
///
/// @note
///     The dnnl_memory_map_data() and dnnl_memory_unmap_data() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Output pointer to the mapped buffer.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_map_data_v2(
        const_dnnl_memory_t memory, void **mapped_ptr, int index);
#endif

/// Unmaps a memory object and writes back any changes made to the previously
/// mapped memory buffer. The pointer to the mapped buffer must be obtained
/// via the dnnl_memory_map_data() call.
///
/// @note
///     The dnnl_memory_map_data() and dnnl_memory_unmap_data() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Pointer to the mapped buffer that must have been
///     obtained using the dnnl_memory_map_data() function.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_unmap_data(
        const_dnnl_memory_t memory, void *mapped_ptr);

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Unmaps a memory object and writes back any changes made to the previously
/// mapped memory buffer. The pointer to the mapped buffer must be obtained
/// via the dnnl_memory_map_data() call. The buffer corresponds to the given
/// index.
///
/// @note
///     The dnnl_memory_map_data() and dnnl_memory_unmap_data() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Pointer to the mapped buffer that must have been
///     obtained using the dnnl_memory_map_data() function.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_unmap_data_v2(
        const_dnnl_memory_t memory, void *mapped_ptr, int index);
#endif

/// Returns memory object's data handle.
///
/// @param memory Memory object.
/// @param handle Output data handle. For the CPU engine, the data handle is a
///     pointer to the actual data. For OpenCL it is a cl_mem.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_get_data_handle(
        const_dnnl_memory_t memory, void **handle);

/// Sets the underlying memory buffer.
///
/// @param memory Memory object.
/// @param handle Data handle. For the CPU engine or when USM is used, the
///     memory buffer is a pointer to the actual data. For OpenCL it is a
///     `cl_mem`.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_set_data_handle(
        dnnl_memory_t memory, void *handle);

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Returns an underlying memory buffer that corresponds to the given index.
///
/// @param memory Memory object.
/// @param handle Data handle. For the CPU engine or when USM is used, the
///     memory buffer is a pointer to the actual data. For OpenCL it is a
///     `cl_mem`.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_get_data_handle_v2(
        const_dnnl_memory_t memory, void **handle, int index);

/// Sets an underlying memory buffer that corresponds to the given index.
///
/// @param memory Memory object.
/// @param handle Data handle. For the CPU engine or when USM is used, the
///     memory buffer is a pointer to the actual data. For OpenCL it is a
///     `cl_mem`.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_set_data_handle_v2(
        dnnl_memory_t memory, void *handle, int index);
#endif

/// Destroys a memory object.
///
/// @param memory Memory object to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_destroy(dnnl_memory_t memory);

/// @} dnnl_api_memory

/// @addtogroup dnnl_api_primitives
/// @{

/// @addtogroup dnnl_api_reorder
/// @{

/// Creates a primitive descriptor for a reorder primitive.
///
/// @param reorder_primitive_desc Output primitive descriptor.
/// @param src_desc Source memory descriptor.
/// @param src_engine Engine on which the source memory object will be
///     located.
/// @param dst_desc Destination memory descriptor.
/// @param dst_engine Engine on which the destination memory object
///     will be located.
/// @param attr Primitive attributes to use (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_reorder_primitive_desc_create(
        dnnl_primitive_desc_t *reorder_primitive_desc,
        const_dnnl_memory_desc_t src_desc, dnnl_engine_t src_engine,
        const_dnnl_memory_desc_t dst_desc, dnnl_engine_t dst_engine,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_reorder

/// @addtogroup dnnl_api_concat
/// @{

/// Creates a primitive descriptor for an out-of-place concatenation
/// primitive.
///
/// @param concat_primitive_desc Output primitive descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param n Number of source parameters.
/// @param concat_dimension Source tensors will be concatenated over
///     dimension with this index. Note that order of dimensions does
///     not depend on memory format.
/// @param src_descs Array of source memory descriptors with @p n elements.
/// @param attr Primitive attributes to use (can be NULL).
/// @param engine Engine to use.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_concat_primitive_desc_create(
        dnnl_primitive_desc_t *concat_primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t dst_desc, int n, int concat_dimension,
        const_dnnl_memory_desc_t const *src_descs,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_concat

/// @addtogroup dnnl_api_sum
/// @{

/// Creates a primitive descriptor for an (out-of-place) sum primitive.
///
/// @param sum_primitive_desc Output primitive descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param n Number of source parameters.
/// @param scales Vector of scales to multiply data in each source
///     memory by.
/// @param src_descs Array of source memory descriptors having @p n elements.
/// @param attr Primitive attributes to use (can be NULL).
/// @param engine Engine to use.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sum_primitive_desc_create(
        dnnl_primitive_desc_t *sum_primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t dst_desc, int n, const float *scales,
        const_dnnl_memory_desc_t const *src_descs,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_sum

/// @addtogroup dnnl_api_binary
/// @{

/// Creates a primitive descriptor for a binary primitive.
///
/// @note
///     Memory descriptors @p src1_desc and @p dst_desc are alloweded to be
///     initialized with #dnnl_format_tag_any or with format_kind set to
///     #dnnl_format_kind_any.
///
/// @note
///     Both memory descriptors must have the same number of dimensions.
///     Element broadcasting is supported for memory descriptor @p src1_desc
///     and are applied to @p src1_desc dimensions that have size equal to 1.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Algorithm kind. Valid values are #dnnl_binary_add,
///     #dnnl_binary_mul, #dnnl_binary_max, #dnnl_binary_min, #dnnl_binary_div,
///     #dnnl_binary_sub, #dnnl_binary_ge, #dnnl_binary_gt, #dnnl_binary_le,
///     #dnnl_binary_lt, #dnnl_binary_eq and #dnnl_binary_ne.
/// @param src0_desc Source 0 memory descriptor.
/// @param src1_desc Source 1 memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_binary_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t src0_desc,
        const_dnnl_memory_desc_t src1_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_binary

/// @addtogroup dnnl_api_convolution
/// @{

/// Creates a primitive descriptor for a convolution forward propagation
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind Convolution algorithm. Possible values are
///     #dnnl_convolution_direct, #dnnl_convolution_winograd,
///     #dnnl_convolution_auto.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #dnnl_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_convolution_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t bias_desc, const_dnnl_memory_desc_t dst_desc,
        const dnnl_dims_t strides, const dnnl_dims_t dilates,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a convolution backward propagation
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Convolution algorithm. Possible values are
///     #dnnl_convolution_direct, #dnnl_convolution_winograd,
///     #dnnl_convolution_auto.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_convolution_backward_data_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t diff_dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t dilates, const dnnl_dims_t padding_l,
        const dnnl_dims_t padding_r, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a convolution weights gradient primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Convolution algorithm. Possible values are
///     #dnnl_convolution_direct, #dnnl_convolution_winograd,
///     #dnnl_convolution_auto.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #dnnl_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_convolution_backward_weights_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t diff_weights_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t dilates, const dnnl_dims_t padding_l,
        const dnnl_dims_t padding_r, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_convolution

/// @addtogroup dnnl_api_deconvolution
/// @{

/// Creates a primitive descriptor for a deconvolution forward propagation
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #dnnl_deconvolution_direct, #dnnl_deconvolution_winograd.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #dnnl_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_deconvolution_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t bias_desc, const_dnnl_memory_desc_t dst_desc,
        const dnnl_dims_t strides, const dnnl_dims_t dilates,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a deconvolution backward propagation
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #dnnl_deconvolution_direct, #dnnl_deconvolution_winograd.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_deconvolution_backward_data_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t diff_dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t dilates, const dnnl_dims_t padding_l,
        const dnnl_dims_t padding_r, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a deconvolution weights gradient
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #dnnl_deconvolution_direct, #dnnl_deconvolution_winograd.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #dnnl_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API
dnnl_deconvolution_backward_weights_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t diff_weights_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t dilates, const dnnl_dims_t padding_l,
        const dnnl_dims_t padding_r, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_deconvolution

/// @addtogroup dnnl_api_shuffle
/// @{

/// Creates a primitive descriptor for a shuffle forward propagation primitive
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param axis The axis along which the data is shuffled.
/// @param group_size Shuffle group size.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_shuffle_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, int axis, dnnl_dim_t group_size,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a shuffle backward propagation primitive
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param axis The axis along which the data is shuffled.
/// @param group_size Shuffle group size.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_shuffle_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc, int axis, dnnl_dim_t group_size,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_shuffle

/// @addtogroup dnnl_api_eltwise
/// @{

/// Creates a primitive descriptor for an eltwise forward propagation primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind Elementwise algorithm kind.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param alpha The alpha parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @param beta The beta parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_eltwise_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t dst_desc,
        float alpha, float beta, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for an eltwise backward propagation
///     primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Elementwise algorithm kind.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param data_desc Destination memory descriptor if one of the
///     "use_dst_for_bwd" algorithms are used (such as
///     #dnnl_eltwise_relu_use_dst_for_bwd), source memory descriptor otherwise.
/// @param alpha The alpha parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @param beta The beta parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_eltwise_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t data_desc, float alpha, float beta,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_eltwise

/// @addtogroup dnnl_api_softmax
/// @{

/// Creates a primitive descriptor for a softmax forward propagation primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind Softmax algorithm kind: either #dnnl_softmax_accurate, or
///     #dnnl_softmax_log.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param softmax_axis Axis over which softmax is computed.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_softmax_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t dst_desc,
        int softmax_axis, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a softmax backward propagation primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Softmax algorithm kind: either #dnnl_softmax_accurate, or
///     #dnnl_softmax_log.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param softmax_axis Axis over which softmax is computed.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_softmax_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t dst_desc, int softmax_axis,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_softmax

/// @addtogroup dnnl_api_pooling
/// @{

/// Creates a primitive descriptor for a pooling forward propagation
///     primitive.
///
/// Arrays @p strides, @p kernel, @p dilation, @p padding_l and @p padding_r
/// contain values for spatial dimensions only and hence must have the same
/// number of elements as there are spatial dimensions. The order of values
/// is the same as in the tensor: depth (for 3D tensors),
/// height (for 3D and 2D tensors), and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind Pooling algorithm kind: either #dnnl_pooling_max,
///     #dnnl_pooling_avg_include_padding, or #dnnl_pooling_avg_exclude_padding.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param kernel Array of kernel spatial dimensions.
/// @param dilation Array of dilations for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_pooling_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t dst_desc,
        const dnnl_dims_t strides, const dnnl_dims_t kernel,
        const dnnl_dims_t dilation, const dnnl_dims_t padding_l,
        const dnnl_dims_t padding_r, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a pooling backward propagation
///     primitive.
///
/// Arrays @p strides, @p kernel, @p dilation, @p padding_l and @p padding_r
/// contain values for spatial dimensions only and hence must have the same
/// number of elements as there are spatial dimensions. The order of values
/// is the same as in the tensor: depth (for 3D tensors),
/// height (for 3D and 2D tensors), and width.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind Pooling algorithm kind: either #dnnl_pooling_max,
///     #dnnl_pooling_avg_include_padding, or #dnnl_pooling_avg_exclude_padding.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param kernel Array of kernel spatial dimensions.
/// @param dilation Array of dilations for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_pooling_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc, const dnnl_dims_t strides,
        const dnnl_dims_t kernel, const dnnl_dims_t dilation,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_pooling

/// @addtogroup dnnl_api_prelu
/// @{

/// Creates a primitive descriptor for a PReLU (leaky ReLU with trainable
///     alpha parameter) forward propagation primitive.
///
/// @note
///     weights descriptor is allowed to be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Alpha parameters memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_prelu_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t dst_desc, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a PReLU (leaky ReLU with trainable
///     alpha parameter) backward propagation primitive.
///
/// @note
///     weights descriptor and diff_weights descriptor are allowed
///     to be initialized with #dnnl_format_tag_any or with format_kind
///     set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Alpha parameters memory descriptor.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_weights_desc Diff alpha parameters memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_prelu_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_weights_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_prelu

/// @addtogroup dnnl_api_lrn
/// @{

/// Creates a primitive descriptor for an LRN forward propagation primitive.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind LRN algorithm kind: either #dnnl_lrn_across_channels or
///     #dnnl_lrn_within_channel.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param local_size Regularization local size.
/// @param alpha The alpha regularization parameter.
/// @param beta The beta regularization parameter.
/// @param k The k regularization parameter.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lrn_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t dst_desc,
        dnnl_dim_t local_size, float alpha, float beta, float k,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for an LRN backward propagation primitive.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param alg_kind LRN algorithm kind: either #dnnl_lrn_across_channels or
///     #dnnl_lrn_within_channel.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param local_size Regularization local size.
/// @param alpha The alpha regularization parameter.
/// @param beta The beta regularization parameter.
/// @param k The k regularization parameter.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lrn_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t src_desc, dnnl_dim_t local_size, float alpha,
        float beta, float k, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_lrn

/// @addtogroup dnnl_api_batch_normalization
/// @{

/// Creates a primitive descriptor for a batch normalization forward propagation
///     primitive.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param epsilon Batch normalization epsilon parameter.
/// @param flags Batch normalization flags (@ref dnnl_normalization_flags_t).
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_batch_normalization_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, float epsilon, unsigned flags,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a batch normalization backward
///     propagation primitive.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_backward_data and #dnnl_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param epsilon Batch normalization epsilon parameter.
/// @param flags Batch normalization flags (@ref dnnl_normalization_flags_t).
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_batch_normalization_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t src_desc, float epsilon, unsigned flags,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_batch_normalization

/// @addtogroup dnnl_api_group_normalization
/// @{

/// Creates a primitive descriptor for a group normalization forward propagation
///     primitive.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param groups Group normalization groups parameter.
/// @param epsilon Group normalization epsilon parameter.
/// @param flags Group normalization flags (@ref dnnl_normalization_flags_t).
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_group_normalization_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, dnnl_dim_t groups, float epsilon,
        unsigned flags, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a group normalization backward
///     propagation primitive.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_backward_data and #dnnl_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param groups Group normalization groups parameter.
/// @param epsilon Group normalization epsilon parameter.
/// @param flags Group normalization flags (@ref dnnl_normalization_flags_t).
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_group_normalization_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t src_desc, dnnl_dim_t groups, float epsilon,
        unsigned flags, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_group_normalization

/// @addtogroup dnnl_api_layer_normalization
/// @{

/// Creates a primitive descriptor for a layer normalization forward propagation
///     primitive.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p src_desc by removing the last
///     dimension.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_layer_normalization_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, const_dnnl_memory_desc_t stat_desc,
        float epsilon, unsigned flags, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a layer normalization backward
///     propagation primitive.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_backward_data and #dnnl_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p src_desc by removing the last
///     dimension.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_layer_normalization_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t stat_desc,
        float epsilon, unsigned flags, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a layer normalization forward propagation
///     primitive with a user-provided data type for the scale and shift
///     memory objects.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p src_desc by removing the last
///     dimension.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API
dnnl_layer_normalization_forward_primitive_desc_create_v2(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, const_dnnl_memory_desc_t stat_desc,
        dnnl_data_type_t scale_shift_data_type, float epsilon, unsigned flags,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a layer normalization backward
///     propagation primitive with a user-provided data type for the
///     scale and shift memory objects.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_backward_data and #dnnl_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p src_desc by removing the last
///     dimension.
/// @param diff_scale_shift_data_type Data type of diff scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API
dnnl_layer_normalization_backward_primitive_desc_create_v2(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t stat_desc,
        dnnl_data_type_t diff_scale_shift_data_type,
        dnnl_data_type_t scale_shift_data_type, float epsilon, unsigned flags,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_layer_normalization

/// @addtogroup dnnl_api_inner_product
/// @{

/// Creates a primitive descriptor for an inner product forward propagation
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #dnnl_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_inner_product_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t bias_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for an inner product backward propagation
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_inner_product_backward_data_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for an inner product  weights gradient
///     primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #dnnl_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API
dnnl_inner_product_backward_weights_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t diff_weights_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_inner_product

/// @addtogroup dnnl_api_attributes
/// @{

/// Set quantization scale and shift parameters for RNN data tensors.
///
/// For performance reasons, the low-precision configuration of the RNN
/// primitives expects input activations to have the unsigned 8-bit integer
/// data type. The scale and shift parameters are used to quantize
/// floating-point data to unsigned integer and must be passed to the RNN
/// primitive using attributes.
///
/// The quantization formula is `scale * data + shift`.
///
/// @note
///     Quantization scale and shift are common for src_layer, src_iter,
///     dst_iter, and dst_layer.
///
/// Example usage:
/// @code
///     // RNN parameters
///     int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
///     // Activations quantization parameters
///     float scale = 63.f, shift = 64.f;
///
///     dnnl_primitive_attr_t rnn_attr;
///     // Create default attributes
///     dnnl_primitive_attr_create(&rnn_attr);
///
///     // Set scale and shift for int8 quantization of activation
///     dnnl_primitive_attr_set_rnn_data_qparams(rnn_attr, scale, shift);
///
///     // Create an RNN primitive descriptor.
///     dnnl_primitive_desc_t rnn_pd;
///     dnnl_vanilla_rnn_forward_primitive_desc_create(&rnn_pd,
///             engine, /* arguments */, attr);
/// @endcode
///
/// @param attr Primitive attributes.
/// @param scale The value to scale the data by.
/// @param shift The value to shift the data by.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_rnn_data_qparams(
        dnnl_primitive_attr_t attr, const float scale, const float shift);

/// Returns the quantization scale and shift parameters for RNN data tensors.
///
/// @note
///     Quantization scale and shift are common for src_layer, src_iter,
///     dst_iter, and dst_layer.
///
/// @param attr Primitive attributes.
/// @param scale The value to scale the data by.
/// @param shift The value to shift the data by.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_rnn_data_qparams(
        const_dnnl_primitive_attr_t attr, float *scale, float *shift);

/// Sets quantization scaling factors for RNN weights tensors. The
/// low-precision configuration of the RNN primitives expects input weights to
/// use the signed 8-bit integer data type. The scaling factors are used to
/// quantize floating-point data to signed integer and must be passed to RNN
/// primitives using attributes.
///
/// @note
///     The dimension order is always native and does not depend on the actual
///     layout used. For example, five-dimensional weights always have (l, d,
///     i, g, o) logical dimension ordering.
///
/// @note
///     Quantization scales are common for weights_layer and weights_iteration
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that must contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
///     Violations can only be detected when the attributes are used to create
///     a primitive descriptor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_rnn_weights_qparams(
        dnnl_primitive_attr_t attr, dnnl_dim_t count, int mask,
        const float *scales);

/// Returns the quantization scaling factors for RNN weights tensors.
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_rnn_weights_qparams(
        const_dnnl_primitive_attr_t attr, dnnl_dim_t *count, int *mask,
        const float **scales);

/// Sets quantization scaling factors for RNN projection weights tensors. The
/// low-precision configuration of the RNN primitives expects input weights to
/// use the signed 8-bit integer data type. The scaling factors are used to
/// quantize floating-point data to signed integer and must be passed to RNN
/// primitives using attributes.
///
/// @note
///     The dimension order is always native and does not depend on the actual
///     layout used. For example, five-dimensional weights always have (l, d,
///     i, g, o) logical dimension ordering.
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that must contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
///     Violations can only be detected when the attributes are used to create
///     a primitive descriptor.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_rnn_weights_projection_qparams(
        dnnl_primitive_attr_t attr, dnnl_dim_t count, int mask,
        const float *scales);

/// Returns the quantization scaling factors for RNN projection weights tensors.
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_rnn_weights_projection_qparams(
        const_dnnl_primitive_attr_t attr, dnnl_dim_t *count, int *mask,
        const float **scales);

/// @} dnnl_api_attributes

/// @addtogroup dnnl_api_rnn
/// @{

/// Creates a primitive descriptor for vanilla RNN forward propagation
///     primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the RNN forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param activation Activation kind. Possible values are #dnnl_eltwise_relu,
///     #dnnl_eltwise_tanh or #dnnl_eltwise_logistic.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @param alpha Negative slope if activation is #dnnl_eltwise_relu.
/// @param beta Unused.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_vanilla_rnn_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const dnnl_alg_kind_t activation,
        const dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc, unsigned flags, float alpha,
        float beta, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for vanilla RNN backward propagation
///     primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the RNN backward propagation primitive should
/// not use the respective data and should use zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Must be #dnnl_backward.
/// @param activation Activation kind. Possible values are #dnnl_eltwise_relu,
///     #dnnl_eltwise_tanh or #dnnl_eltwise_logistic.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @param alpha Negative slope if activation is #dnnl_eltwise_relu.
/// @param beta Unused.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_vanilla_rnn_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const dnnl_alg_kind_t activation,
        const dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc,
        const_dnnl_memory_desc_t diff_src_layer_desc,
        const_dnnl_memory_desc_t diff_src_iter_desc,
        const_dnnl_memory_desc_t diff_weights_layer_desc,
        const_dnnl_memory_desc_t diff_weights_iter_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_layer_desc,
        const_dnnl_memory_desc_t diff_dst_iter_desc, unsigned flags,
        float alpha, float beta, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for an LSTM forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc,
/// - @p weights_peephole_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc.
///
/// This would then indicate that the LSTM forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// The @p weights_projection_desc could either be @c NULL or point to a zero
/// memory descriptor. This would then indicate that the LSTM doesn't have
/// recurrent projection layer.
///
/// @note
///     All memory descriptors can be initialized with #dnnl_format_tag_any or
///     with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param weights_peephole_desc Memory descriptor for the weights applied to
///     the cell states (according to the Peephole LSTM formula).
/// @param weights_projection_desc Memory descriptor for the weights applied to
///     the hidden states to get the recurrent projection (according to the
///     Projection LSTM formula).
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param flags Unused.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lstm_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t src_iter_c_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t weights_peephole_desc,
        const_dnnl_memory_desc_t weights_projection_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc,
        const_dnnl_memory_desc_t dst_iter_c_desc, unsigned flags,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for an LSTM backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc, @p diff_src_iter_desc,
///   and @p diff_src_iter_c_desc,
/// - @p weights_peephole_desc together with @p diff_weights_peephole_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc, @p diff_dst_iter_desc,
///   and @p diff_dst_iter_c_desc.
///
/// This would then indicate that the LSTM backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// The @p weights_projection_desc together with @p
/// diff_weights_projection_desc could either be @c NULL or point to a zero
/// memory descriptor. This would then indicate that the LSTM doesn't have
/// recurrent projection layer.
///
/// @note
///     All memory descriptors can be initialized with #dnnl_format_tag_any or
///     with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Must be #dnnl_backward.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param weights_peephole_desc Memory descriptor for the weights applied to
///     the cell states (according to the Peephole LSTM formula).
/// @param weights_projection_desc Memory descriptor for the weights applied to
///     the hidden states to get the recurrent projection (according to the
///     Projection LSTM formula).
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_src_iter_c_desc Memory descriptor for the diff of input
/// recurrent cell state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_weights_peephole_desc Memory descriptor for the diff of weights
///     applied to the cell states (according to the Peephole LSTM formula).
/// @param diff_weights_projection_desc Memory descriptor for the diff of
///     weights applied to the hidden states to get the recurrent projection
///     (according to the Projection LSTM formula).
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param diff_dst_iter_c_desc Memory descriptor for the diff of output
///     recurrent cell state vector.
/// @param flags Unused.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lstm_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t src_iter_c_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t weights_peephole_desc,
        const_dnnl_memory_desc_t weights_projection_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc,
        const_dnnl_memory_desc_t dst_iter_c_desc,
        const_dnnl_memory_desc_t diff_src_layer_desc,
        const_dnnl_memory_desc_t diff_src_iter_desc,
        const_dnnl_memory_desc_t diff_src_iter_c_desc,
        const_dnnl_memory_desc_t diff_weights_layer_desc,
        const_dnnl_memory_desc_t diff_weights_iter_desc,
        const_dnnl_memory_desc_t diff_weights_peephole_desc,
        const_dnnl_memory_desc_t diff_weights_projection_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_layer_desc,
        const_dnnl_memory_desc_t diff_dst_iter_desc,
        const_dnnl_memory_desc_t diff_dst_iter_c_desc, unsigned flags,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for GRU forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the GRU forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_gru_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc, unsigned flags,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for GRU backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the GRU backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Must be #dnnl_backward.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_gru_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc,
        const_dnnl_memory_desc_t diff_src_layer_desc,
        const_dnnl_memory_desc_t diff_src_iter_desc,
        const_dnnl_memory_desc_t diff_weights_layer_desc,
        const_dnnl_memory_desc_t diff_weights_iter_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_layer_desc,
        const_dnnl_memory_desc_t diff_dst_iter_desc, unsigned flags,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a descriptor for LBR GRU forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the LBR GRU forward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lbr_gru_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc, unsigned flags,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for LBR GRU backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the LBR GRU backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Must be #dnnl_backward.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lbr_gru_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc,
        const_dnnl_memory_desc_t diff_src_layer_desc,
        const_dnnl_memory_desc_t diff_src_iter_desc,
        const_dnnl_memory_desc_t diff_weights_layer_desc,
        const_dnnl_memory_desc_t diff_weights_iter_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_layer_desc,
        const_dnnl_memory_desc_t diff_dst_iter_desc, unsigned flags,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for AUGRU forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the AUGRU forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param attention_desc Memory descriptor for the attention vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_augru_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t attention_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc, unsigned flags,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for AUGRU backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the AUGRU backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Must be #dnnl_backward.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param attention_desc Memory descriptor for the attention vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_attention_desc Memory descriptor for the diff of attention vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_augru_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t attention_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc,
        const_dnnl_memory_desc_t diff_src_layer_desc,
        const_dnnl_memory_desc_t diff_src_iter_desc,
        const_dnnl_memory_desc_t diff_attention_desc,
        const_dnnl_memory_desc_t diff_weights_layer_desc,
        const_dnnl_memory_desc_t diff_weights_iter_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_layer_desc,
        const_dnnl_memory_desc_t diff_dst_iter_desc, unsigned flags,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for LBR AUGRU forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the LBR AUGRU forward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param attention_desc Memory descriptor for the attention vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lbr_augru_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t attention_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc, unsigned flags,
        const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for LBR AUGRU backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the LBR AUGRU backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Must be #dnnl_backward.
/// @param direction RNN direction. See @ref dnnl_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param attention_desc Memory descriptor for the attention vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_attention_desc Memory descriptor for the diff of attention vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_lbr_augru_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const_dnnl_memory_desc_t src_layer_desc,
        const_dnnl_memory_desc_t src_iter_desc,
        const_dnnl_memory_desc_t attention_desc,
        const_dnnl_memory_desc_t weights_layer_desc,
        const_dnnl_memory_desc_t weights_iter_desc,
        const_dnnl_memory_desc_t bias_desc,
        const_dnnl_memory_desc_t dst_layer_desc,
        const_dnnl_memory_desc_t dst_iter_desc,
        const_dnnl_memory_desc_t diff_src_layer_desc,
        const_dnnl_memory_desc_t diff_src_iter_desc,
        const_dnnl_memory_desc_t diff_attention_desc,
        const_dnnl_memory_desc_t diff_weights_layer_desc,
        const_dnnl_memory_desc_t diff_weights_iter_desc,
        const_dnnl_memory_desc_t diff_bias_desc,
        const_dnnl_memory_desc_t diff_dst_layer_desc,
        const_dnnl_memory_desc_t diff_dst_iter_desc, unsigned flags,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_rnn

/// @addtogroup dnnl_api_matmul
/// @{

/// Creates a primitive descriptor for a matrix multiplication primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param src_desc Source memory descriptor (matrix A)
/// @param weights_desc Weights memory descriptor (matrix B)
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #dnnl_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor (matrix C).
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_matmul_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t bias_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_matmul

/// @addtogroup dnnl_api_resampling Resampling
/// @{

/// Creates a primitive descriptor for a resampling forward propagation
///     primitive.
///
/// @note
///     Destination memory descriptor is allowed to be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind resampling algorithm kind: either #dnnl_resampling_nearest,
///     or #dnnl_resampling_linear.
/// @param factors Array of scaling factors for spatial dimension.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_resampling_forward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t alg_kind,
        const float *factors, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, const_dnnl_primitive_attr_t attr);

/// Creates a primitive descriptor for a resampling backward propagation
///     primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind resamplinging algorithm kind: either
///     #dnnl_resampling_nearest, or #dnnl_resampling_linear.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param factors Array of scaling factors for spatial dimension.
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
///
dnnl_status_t DNNL_API dnnl_resampling_backward_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const float *factors,
        const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_resampling

/// @addtogroup dnnl_api_reduction Reduction
/// @{

/// Creates a primitive descriptor for a reduction primitive.
///
/// @note
///     Destination memory descriptor is allowed to be initialized with
///     #dnnl_format_tag_any or with format_kind set to #dnnl_format_kind_any.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param alg_kind reduction algorithm kind. Possible values:
///     #dnnl_reduction_max, #dnnl_reduction_min, #dnnl_reduction_sum,
///     #dnnl_reduction_mul, #dnnl_reduction_mean, #dnnl_reduction_norm_lp_max,
///     #dnnl_reduction_norm_lp_sum, #dnnl_reduction_norm_lp_power_p_max,
///     #dnnl_reduction_norm_lp_power_p_sum.
/// @param p Algorithm specific parameter.
/// @param eps Algorithm specific parameter.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_reduction_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, float p, float eps,
        const_dnnl_primitive_attr_t attr);

/// @} dnnl_api_reduction

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_primitive_cache
/// @{

/// Returns the number of primitives that can be held in the primitive cache
/// at the same time.
///
/// @param capacity Primitive cache capacity to query. Concurrently
/// accessing @p capacity is safe.
/// @returns #dnnl_invalid_arguments/#dnnl::status::invalid_arguments if the
///     @p capacity value is invalid, and #dnnl_success/#dnnl::status::success on
///     success.
dnnl_status_t DNNL_API dnnl_get_primitive_cache_capacity(int *capacity);

/// Sets a number of primitives that can be held in the primitive cache
/// at a time.
///
/// @param capacity Primitive cache capacity to set. If a new @p capacity is
/// less than a number of primitives that the primitive cache already has
/// then the excess entries will be evicted. Setting the @p capacity to 0
/// clears the primitive cache and disables it. Concurrently modifying
/// @p capacity is safe.
/// @returns #dnnl_invalid_arguments/#dnnl::status::invalid_arguments if the
///     @p capacity value is invalid, and #dnnl_success/#dnnl::status::success on
///     success.
dnnl_status_t DNNL_API dnnl_set_primitive_cache_capacity(int capacity);

/// @} dnnl_api_primitive_cache

/// @addtogroup dnnl_api_service
/// @{

/// Configures dumping of JIT-generated code.
///
/// @note
///     This setting overrides the DNNL_JIT_DUMP environment variable.
///
/// @param enable Flag value. Set to 0 to disable and set to 1 to enable.
/// @returns #dnnl_invalid_arguments/#dnnl::status::invalid_arguments if the
///     @p flag value is invalid, and #dnnl_success/#dnnl::status::success on
///     success.
dnnl_status_t DNNL_API dnnl_set_jit_dump(int enable);

/// Sets library profiling flags. The flags define which profilers are
/// supported.
///
/// @note
///     This setting overrides DNNL_JIT_PROFILE environment variable.
///
/// @sa @ref dev_guide_profilers
///
/// @param flags Profiling flags that can contain the following bits:
///     - @ref DNNL_JIT_PROFILE_VTUNE -- integration with VTune Profiler
///         (on by default)
///     - @ref DNNL_JIT_PROFILE_LINUX_JITDUMP -- produce Linux-specific
///         jit-pid.dump output (off by default). The location of the output
///         is controlled via JITDUMPDIR environment variable or via
///         dnnl_set_jit_profiling_jitdumpdir() function.
///     - @ref DNNL_JIT_PROFILE_LINUX_PERFMAP -- produce Linux-specific
///         perf-pid.map output (off by default). The output is always placed
///         into /tmp.
///
///     Passing @ref DNNL_JIT_PROFILE_NONE disables profiling completely.
///
/// @returns #dnnl_invalid_arguments/#dnnl::status::invalid_arguments if the
///     @p flags value is invalid, and #dnnl_success/#dnnl::status::success on
///     success.
dnnl_status_t DNNL_API dnnl_set_jit_profiling_flags(unsigned flags);

/// Sets JIT dump output path. Only applicable to Linux and is only
/// used when profiling flags have DNNL_JIT_PROFILE_LINUX_PERF bit set.
///
/// After the first JIT kernel is generated, the jitdump output will be placed
/// into temporary directory created using the mkdtemp template
/// 'dir/.debug/jit/dnnl.XXXXXX'.
///
/// @sa @ref dev_guide_profilers
///
/// @note
///     This setting overrides JITDUMPDIR environment variable.  If
///     JITDUMPDIR is not set, and this function is never called, the path
///     defaults to HOME. Passing NULL reverts the value to default.
///
/// @note
///     The directory is accessed only when the first JIT kernel is being
///     created. JIT profiling will be disabled in case of any errors
///     accessing or creating this directory.
///
/// @param dir JIT dump output path.
/// @returns #dnnl_success/#dnnl::status::success if the
///     output directory was set correctly and an error status otherwise.
/// @returns #dnnl_unimplemented/#dnnl::status::unimplemented on Windows.
dnnl_status_t DNNL_API dnnl_set_jit_profiling_jitdumpdir(const char *dir);

/// Sets the maximal ISA the library can dispatch to on the CPU. See
/// #dnnl_cpu_isa_t and #dnnl::cpu_isa for the list of the values accepted by
/// the C and C++ API functions respectively.
///
/// This function has effect only once, and returns an error on subsequent
/// calls. It should also be invoked before any other oneDNN API call, otherwise
/// it may return an error.
///
/// This function overrides the DNNL_MAX_CPU_ISA environment variable. The
/// environment variable can be set to the desired maximal ISA name in upper
/// case and with dnnl_cpu_isa prefix removed. For example:
/// `DNNL_MAX_CPU_ISA=AVX2`.
///
/// @note
///     The ISAs are only partially ordered:
///         - SSE41 < AVX < AVX2 < AVX2_VNNI < AVX2_VNNI_2,
///         - AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16
///           < AVX10_1_512 < AVX10_1_512_AMX < AVX10_1_512_AMX_FP16,
///         - AVX2_VNNI < AVX10_1_512.
///     Aliases:
///         - AVX512_CORE_FP16 = AVX10_1_512
///         - AVX512_CORE_AMX = AVX10_1_512_AMX
///         - AVX512_CORE_AMX_FP16 = AVX10_1_512_AMX_FP16
///
/// @sa @ref dev_guide_cpu_dispatcher_control for more details
///
/// @param isa Maximal ISA the library should dispatch to. Pass
///     #dnnl_cpu_isa_default/#dnnl::cpu_isa::isa_default to remove ISA restrictions
///     (except for ISAs with initial support in the library).
/// @returns #dnnl_success/#dnnl::status::success on success and a
///     #dnnl_invalid_arguments/#dnnl::status::invalid_arguments if the @p isa
///     parameter is invalid or the ISA cannot be changed at this time.
/// @returns #dnnl_unimplemented/#dnnl::status::unimplemented if the feature
///     was disabled at build time (see @ref dev_guide_build_options for more
///     details).
dnnl_status_t DNNL_API dnnl_set_max_cpu_isa(dnnl_cpu_isa_t isa);

/// Gets the maximal ISA the library can dispatch to on the CPU. See
/// #dnnl_cpu_isa_t and #dnnl::cpu_isa for the list of the values returned by
/// the C and C++ API functions respectively.
///
/// @sa @ref dev_guide_cpu_dispatcher_control for more details
///
/// @returns #dnnl_cpu_isa_t value reflecting the maximal ISA the library may
///     dispatch to.
dnnl_cpu_isa_t DNNL_API dnnl_get_effective_cpu_isa(void);

/// Sets the hints flag for the CPU ISA. See #dnnl_cpu_isa_hints_t and
/// #dnnl::cpu_isa_hints for the list of the values accepted by the C and C++
/// API functions respectively.
///
/// This function has effect only once, and returns an error on subsequent
/// calls. It should also be invoked before any other oneDNN API call, otherwise
/// it may return an error.
///
/// This function overrides the DNNL_CPU_ISA_HINTS environment variable.
/// @sa @ref dev_guide_cpu_isa_hints for more details
///
/// @param isa_hints CPU ISA hints to be passed over to the implementation.
///     Pass #dnnl_cpu_isa_no_hints/#dnnl::cpu_isa_hints::no_hints to use
///     default features i.e. no hints.
/// @returns #dnnl_success/#dnnl::status::success on success and a
///     #dnnl_runtime_error/#dnnl::status::runtime_error if the ISA hints cannot
///     be specified at the current time.
/// @returns #dnnl_unimplemented/#dnnl::status::unimplemented if the feature
///     was disabled at build time (see @ref dev_guide_build_options for more
///     details).
dnnl_status_t DNNL_API dnnl_set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints);

/// Gets the ISA specific hints that library can follow. See
/// #dnnl_cpu_isa_hints_t and #dnnl::cpu_isa_hints for the list of the values
///  returned by the C and C++ API functions respectively.
///
/// @sa @ref dev_guide_cpu_isa_hints for more details
///
/// @returns #dnnl_cpu_isa_hints_t value reflecting the ISA specific hints the
/// library can follow.
dnnl_cpu_isa_hints_t DNNL_API dnnl_get_cpu_isa_hints(void);

/// @} dnnl_api_service

#ifdef DNNL_EXPERIMENTAL_PROFILING

/// @addtogroup dnnl_api_profiling Profiling
/// @{

/// Resets a profiler's state.
///
/// @param stream Stream associated with the profiler.
///
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_reset_profiling(dnnl_stream_t stream);

/// Queries profiling data. The profiling data accumulates for each primitive
/// execution. The @p num_entries will be equal to the number of executions
/// since the last `dnnl_reset_profiling` call. In order to query the
/// @p num_entries the @p data parameter should be NULL. When @p data is NULL
/// then the @p data_kind parameter is ignored.
///
/// The profiling data can be reset by calling #dnnl_reset_profiling.
///
/// @note
///     It is required to wait for all submitted primitives to complete
///     using #dnnl_stream_wait prior to querying profiling data.
///
/// @param stream Stream that was used for executing a primitive that
/// is being profiled.
/// @param data_kind Profiling data kind to query.
/// @param num_entries Number of profiling data entries.
/// @param data Profiling data.
///
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_query_profiling_data(dnnl_stream_t stream,
        dnnl_profiling_data_kind_t data_kind, int *num_entries, uint64_t *data);

/// @} dnnl_api_profiling
#endif

/// @addtogroup dnnl_api_blas
/// @{

/// Performs single-precision matrix-matrix multiply.
///
/// The operation is defined as:
///
/// `C := alpha * op( A ) * op( B ) + beta * C`
///
/// where
///  - `op( X ) = X` or `op( X ) = X**T`,
///  - `alpha` and `beta` are scalars, and
///  - `A`, `B`, and `C` are matrices:
///     - `op( A )` is an `MxK` matrix,
///     - `op( B )` is an `KxN` matrix,
///     - `C` is an `MxN` matrix.
///
/// The matrices are assumed to be stored in row-major order (the elements in
/// each of the matrix rows are contiguous in memory).
///
/// @note
///     This API does not support XERBLA. Instead, unlike the standard BLAS
///     functions, this one returns a dnnl_status_t value to allow error
///     handling.
///
/// @param transa Transposition flag for matrix A: 'N' or 'n' means A is not
///     transposed, and 'T' or 't' means that A is transposed.
/// @param transb Transposition flag for matrix B: 'N' or 'n' means B is not
///     transposed, and 'T' or 't' means that B is transposed.
/// @param M The M dimension.
/// @param N The N dimension.
/// @param K The K dimension.
/// @param alpha The alpha parameter that is used to scale the product of
///     matrices A and B.
/// @param A A pointer to the A matrix data.
/// @param lda The leading dimension for the matrix A.
/// @param B A pointer to the B matrix data.
/// @param ldb The leading dimension for the matrix B.
/// @param beta The beta parameter that is used to scale the matrix C.
/// @param C A pointer to the C matrix data.
/// @param ldc The leading dimension for the matrix C.
/// @returns #dnnl_success/#dnnl::status::success on success and a status
///     describing the error otherwise.
dnnl_status_t DNNL_API dnnl_sgemm(char transa, char transb, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, float *C, dnnl_dim_t ldc);

/// Performs integer matrix-matrix multiply on 8-bit unsigned matrix A, 8-bit
/// signed matrix B, and 32-bit signed resulting matrix C.
///
/// The operation is defined as:
///
/// `C := alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset`
///
/// where
///  - `op( X ) = X` or `op( X ) = X**T`,
///  - `alpha` and `beta` are scalars, and
///  - `A`, `B`, and `C` are matrices:
///     - `op( A )` is an `MxK` matrix,
///     - `op( B )` is an `KxN` matrix,
///     - `C` is an `MxN` matrix.
///  - `A_offset` is an `MxK` matrix with every element equal the `ao` value,
///  - `B_offset` is an `KxN` matrix with every element equal the `bo` value,
///  - `C_offset` is an `MxN` matrix which is defined by the `co` array of size `len`:
///    - if `offsetc = F`: the `len` must be at least `1`,
///    - if `offsetc = C`: the `len` must be at least `max(1, m)`,
///    - if `offsetc = R`: the `len` must be at least `max(1, n)`,
///
/// The matrices are assumed to be stored in row-major order (the elements in
/// each of the matrix rows are contiguous in memory).
///
/// @note
///     This API does not support XERBLA. Instead, unlike the standard BLAS
///     functions, this one returns a dnnl_status_t value to allow error
///     handling.
///
/// @warning
///     On some architectures saturation may happen during intermediate
///     computations, which would lead to unexpected results. For more
///     details, refer to @ref dev_guide_int8_computations.
///
/// @param transa Transposition flag for matrix A: 'N' or 'n' means A is not
///     transposed, and 'T' or 't' means that A is transposed.
/// @param transb Transposition flag for matrix B: 'N' or 'n' means B is not
///     transposed, and 'T' or 't' means that B is transposed.
/// @param offsetc Flag specifying how offsets should be applied to matrix C:
///     - 'F' means that the same offset will be applied to each element of
///         the matrix C,
///     - 'C' means that individual offset will be applied to each element
///         within each column,
///     - 'R' means that individual offset will be applied to each element
///         within each row.
/// @param M The M dimension.
/// @param N The N dimension.
/// @param K The K dimension.
/// @param alpha The alpha parameter that is used to scale the product of
///     matrices A and B.
/// @param A A pointer to the A matrix data.
/// @param lda The leading dimension for the matrix A.
/// @param ao The offset value for the matrix A.
/// @param B A pointer to the B matrix data.
/// @param ldb The leading dimension for the matrix B.
/// @param bo The offset value for the matrix B.
/// @param beta The beta parameter that is used to scale the matrix C.
/// @param C A pointer to the C matrix data.
/// @param ldc The leading dimension for the matrix C.
/// @param co An array of offset values for the matrix C. The number of
///     elements in the array depends on the value of @p offsetc.
/// @returns #dnnl_success/#dnnl::status::success on success and a status
///     describing the error otherwise.
dnnl_status_t DNNL_API dnnl_gemm_u8s8s32(char transa, char transb, char offsetc,
        dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, float alpha, const uint8_t *A,
        dnnl_dim_t lda, uint8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co);

/// Performs integer matrix-matrix multiply on 8-bit signed matrix A, 8-bit
/// signed matrix B, and 32-bit signed resulting matrix C.
///
/// The operation is defined as:
///
/// `C := alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset`
///
/// where
///  - `op( X ) = X` or `op( X ) = X**T`,
///  - `alpha` and `beta` are scalars, and
///  - `A`, `B`, and `C` are matrices:
///     - `op( A )` is an `MxK` matrix,
///     - `op( B )` is an `KxN` matrix,
///     - `C` is an `MxN` matrix.
///  - `A_offset` is an `MxK` matrix with every element equal the `ao` value,
///  - `B_offset` is an `KxN` matrix with every element equal the `bo` value,
///  - `C_offset` is an `MxN` matrix which is defined by the `co` array of size `len`:
///    - if `offsetc = F`: the `len` must be at least `1`,
///    - if `offsetc = C`: the `len` must be at least `max(1, m)`,
///    - if `offsetc = R`: the `len` must be at least `max(1, n)`,
///
/// The matrices are assumed to be stored in row-major order (the elements in
/// each of the matrix rows are contiguous in memory).
///
/// @note
///     This API does not support XERBLA. Instead, unlike the standard BLAS
///     functions, this one returns a dnnl_status_t value to allow error
///     handling.
///
/// @warning
///     On some architectures saturation may happen during intermediate
///     computations, which would lead to unexpected results. For more
///     details, refer to @ref dev_guide_int8_computations.
///
/// @param transa Transposition flag for matrix A: 'N' or 'n' means A is not
///     transposed, and 'T' or 't' means that A is transposed.
/// @param transb Transposition flag for matrix B: 'N' or 'n' means B is not
///     transposed, and 'T' or 't' means that B is transposed.
/// @param offsetc Flag specifying how offsets should be applied to matrix C:
///     - 'F' means that the same offset will be applied to each element of
///         the matrix C,
///     - 'C' means that individual offset will be applied to each element
///         within each column,
///     - 'R' means that individual offset will be applied to each element
///         within each row.
/// @param M The M dimension.
/// @param N The N dimension.
/// @param K The K dimension.
/// @param alpha The alpha parameter that is used to scale the product of
///     matrices A and B.
/// @param A A pointer to the A matrix data.
/// @param lda The leading dimension for the matrix A.
/// @param ao The offset value for the matrix A.
/// @param B A pointer to the B matrix data.
/// @param ldb The leading dimension for the matrix B.
/// @param bo The offset value for the matrix B.
/// @param beta The beta parameter that is used to scale the matrix C.
/// @param C A pointer to the C matrix data.
/// @param ldc The leading dimension for the matrix C.
/// @param co An array of offset values for the matrix C. The number of
///     elements in the array depends on the value of @p offsetc.
/// @returns #dnnl_success/#dnnl::status::success on success and a status
///     describing the error otherwise.
dnnl_status_t DNNL_API dnnl_gemm_s8s8s32(char transa, char transb, char offsetc,
        dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, float alpha, const int8_t *A,
        dnnl_dim_t lda, int8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co);

/// @} dnnl_api_blas

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif /* ONEAPI_DNNL_DNNL_H */
