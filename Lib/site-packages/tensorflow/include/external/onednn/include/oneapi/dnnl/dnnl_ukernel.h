/*******************************************************************************
* Copyright 2024 Intel Corporation
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
/// ukernel C API

#ifndef ONEAPI_DNNL_DNNL_UKERNEL_H
#define ONEAPI_DNNL_DNNL_UKERNEL_H

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_ukernel_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_ukernel
/// @{

#ifdef DNNL_EXPERIMENTAL_UKERNEL

/// @addtogroup dnnl_api_ukernel_brgemm
/// @{

/// Creates a BRGeMM ukernel object. Operates by the following formula:
/// `C = alpha * [A x B] + beta * C`.
/// `D = post-operations(C)`.
///
/// Post-operations applies if one of the following holds:
/// * Non-empty attributes are specified.
/// * Output data type `d_dt` is different from accumulation data type `c_dt`.
///
/// If any of conditions happens, the final call of the accumulation chain
/// must be `dnnl_brgemm_execute_postops`, and `dnnl_brgemm_execute`, otherwise.
///
/// @param brgemm Output BRGeMM ukernel object.
/// @param M Dimension M of tensor A.
/// @param N Dimension N of tensor B.
/// @param K Dimension K of tensors A and B.
/// @param batch_size Number of batches to process.
/// @param lda Leading dimension of tensor A.
/// @param ldb Leading dimension of tensor B.
/// @param ldc Leading dimension of tensor C.
/// @param ldd Leading dimension of tensor D.
/// @param a_dt Data type of tensor A.
/// @param b_dt Data type of tensor B.
/// @param c_dt Data type of tensor C. Must be dnnl_f32.
/// @param d_dt Data type of tensor D.
/// @param alpha Scale for an accumulation output.
/// @param beta Scale for a tensor C to append on an accumulation output.
/// @param attr Primitive attributes to extend the kernel operations.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_create(dnnl_brgemm_t *brgemm, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t batch_size, dnnl_dim_t lda,
        dnnl_dim_t ldb, dnnl_dim_t ldc, dnnl_dim_t ldd, dnnl_data_type_t a_dt,
        dnnl_data_type_t b_dt, dnnl_data_type_t c_dt, dnnl_data_type_t d_dt,
        float alpha, float beta, const_dnnl_primitive_attr_t attr);

/// Returns the size of a scratchpad memory needed for the BRGeMM ukernel
/// object.
///
/// @param brgemm BRGeMM ukernel object.
/// @param size Output size of a buffer required for the BRGeMM ukernel object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_get_scratchpad_size(
        const_dnnl_brgemm_t brgemm, size_t *size);

/// Initializes the hardware-specific context. If no initialization required,
/// returns the success status.
///
/// @param brgemm BRGeMM ukernel object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_set_hw_context(const_dnnl_brgemm_t brgemm);

/// Releases the hardware-specific context. Must be used after all the execution
/// calls to BRGeMM ukernel objects.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_release_hw_context();

/// Generates an executable part of BRGeMM ukernel object.
/// @param brgemm BRGeMM ukernel object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_generate(dnnl_brgemm_t brgemm);

/// Executes a BRGeMM ukernel object.
///
/// @param brgemm BRGeMM ukernel object.
/// @param A_ptr Base pointer to a tensor A.
/// @param B_ptr Base pointer to a tensor B.
/// @param A_B_offsets Pointer to the set of tensor A and tensor B offsets for
///     each batch; the set must be contiguous in memory. Single batch should
///     supply offsets for both tensors A and B simultaneously. The number of
///     batches must coincide with the `batch_size` value passed at the creation
///     stage.
/// @param C_ptr Pointer to a tensor C (accumulation buffer).
/// @param scratchpad_ptr Pointer to a scratchpad buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_execute(const_dnnl_brgemm_t brgemm,
        const void *A_ptr, const void *B_ptr, const dnnl_dim_t *A_B_offsets,
        void *C_ptr, void *scratchpad_ptr);

/// Executes a BRGeMM ukernel object with post operations.
///
/// @param brgemm BRGeMM ukernel object.
/// @param A Base pointer to a tensor A.
/// @param B Base pointer to a tensor B.
/// @param A_B_offsets Pointer to a set of tensor A and tensor B offsets for
///     each batch. A set must be contiguous in memory. A single batch should
///     supply offsets for both tensors A and B simultaneously. The number of
///     batches must coincide with the `batch_size` value passed at the creation
///     stage.
/// @param C_ptr Pointer to a tensor C (accumulation buffer).
/// @param D_ptr Pointer to a tensor D (output buffer).
/// @param scratchpad_ptr Pointer to a scratchpad buffer.
/// @param binary_po_ptr Pointer to binary post-op data.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_execute_postops(const_dnnl_brgemm_t brgemm,
        const void *A, const void *B, const dnnl_dim_t *A_B_offsets,
        const void *C_ptr, void *D_ptr, void *scratchpad_ptr,
        const void *binary_po_ptr);

/// Destroys a BRGeMM ukernel object.
///
/// @param brgemm BRGeMM ukernel object to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_destroy(dnnl_brgemm_t brgemm);

/// Creates a BRGeMM ukernel packing tensor B object.
///
/// @param brgemm_pack_B Output BRGeMM ukernel packing B object.
/// @param K Dimension K.
/// @param N Dimension N.
/// @param in_ld Input leading dimension.
/// @param out_ld Output leading dimension. Specifies a block by N dimension
///     during data packing.
/// @param in_dt Input data type.
/// @param out_dt Output data type.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_pack_B_create(
        dnnl_brgemm_pack_B_t *brgemm_pack_B, dnnl_dim_t K, dnnl_dim_t N,
        dnnl_dim_t in_ld, dnnl_dim_t out_ld, dnnl_data_type_t in_dt,
        dnnl_data_type_t out_dt);

/// Returns the flag if packing is expected by BRGeMM ukernel kernel.
///
/// @param brgemm_pack_B BRGeMM ukernel packing B object.
/// @param need_pack Output flag specifying if packing is needed.
///     Possible values are 0 (not needed) and 1 (needed).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_pack_B_need_pack(
        const_dnnl_brgemm_pack_B_t brgemm_pack_B, int *need_pack);

/// Generates an executable part of BRGeMM ukernel packing B object.
/// @param brgemm_pack_B BRGeMM ukernel packing B object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_pack_B_generate(
        dnnl_brgemm_pack_B_t brgemm_pack_B);

/// Executes a BRGeMM ukernel packing tensor B object.
///
/// @param brgemm_pack_B BRGeMM ukernel packing B object.
/// @param in_ptr Pointer to an input buffer.
/// @param out_ptr Pointer to an output buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_pack_B_execute(
        const_dnnl_brgemm_pack_B_t brgemm_pack_B, const void *in_ptr,
        void *out_ptr);

/// Destroys a BRGeMM ukernel packing tensor B object.
///
/// @param brgemm_pack_B BRGeMM ukernel packing B object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_pack_B_destroy(
        dnnl_brgemm_pack_B_t brgemm_pack_B);

/// @} dnnl_api_ukernel_brgemm

#endif

/// @} dnnl_api_ukernel

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif /* ONEAPI_DNNL_DNNL_UKERNEL_H */
