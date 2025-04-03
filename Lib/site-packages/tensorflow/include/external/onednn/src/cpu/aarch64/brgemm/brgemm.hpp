/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2023 FUJITSU LIMITED
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
#ifndef CPU_AARCH64_BRGEMM_BRGEMM_HPP
#define CPU_AARCH64_BRGEMM_BRGEMM_HPP

#include "cpu/aarch64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
/// Initializes a BRGEMM descriptor
///
/// @param brg Output BRGEMM descriptor
/// @param isa Target ISA of BRGEMM kernel
///     If isa is equal to 'isa_undef' maximum supported ISA on current
///     hardware will be used for BRGEMM kernel generation
/// @param type Type of batch
/// @param dt_a Data type of A matrix, can be
///     SVE_512: f32
/// @param dt_b Data type of B matrix
///     SVE_512: f32
/// @note
///     Data type of matrix C is f32 data type
/// @param transA Specifies the form of A used in the matrix multiplication
///        'false' - A is not transposed, 'true' - A is transposed
/// @param transB Specifies the form of B used in the matrix multiplication
///        'false' - B is not transposed, 'true' - B is transposed
/// @param layout Specifies whether two-dimensional array storage is row-major
///        (brgemm_row_major) or column-major (brgemm_col_major).
/// @param alpha Specifies the scalar alpha
/// @param beta Specifies the scalar beta
/// @param LDA Specifies the leading dimension of matrix A.
///        LDA must be at least max(1, K)
/// @param LDB Specifies the leading dimension of matrix B.
///        LDB must be at least max(1, N)
/// @param LDC Specifies the leading dimension of matrix C.
///       LDC must be at least max(1, N)
/// @param M Specifies the number of rows of the matrix A and of the matrix C.
/// @param N Specifies the number of columns of the matrix B and
///        the number of columns of the matrix C
/// @param K Specifies the number of columns of the matrix A and
///        the number of rows of the matrix B
/// @param strides Strides between the matrices in the batch. Can be nullptr.
///
status_t DNNL_API brgemm_desc_init(brgemm_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, bool transA, bool transB,
        brgemm_layout_t layout, float alpha, float beta, dim_t LDA, dim_t LDB,
        dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides = nullptr);

/// Initializes a BRGEMM descriptor with B matrix as a diagonal matrix
/// represented in packed vector format.
///
/// @param brg Output BRGEMM descriptor
/// @param isa Target ISA of BRGEMM kernel
///     If isa is equal to 'isa_undef' maximum supported ISA on current
///     hardware will be used for BRGEMM kernel generation
/// @param type Type of batch
/// @param dt_a Data type of A matrix can be: f32
/// @param dt_b Data type of B vector can be: f32
/// @note
///     Data type of matrix C f32 data type
/// @param transA Specifies the form of A used in the matrix multiplication
///        'false' - A is not transposed, 'true' - A is transposed
/// @param layout Specifies whether two-dimensional array storage is row-major
///        (brgemm_row_major) or column-major (brgemm_col_major).
/// @param alpha Specifies the scalar alpha
/// @param beta Specifies the scalar beta
/// @param LDA Specifies the leading dimension of matrix A.
///        LDA must be at least max(1, N)
/// @param LDC Specifies the leading dimension of matrix C.
///       LDC must be at least max(1, N)
/// @param M Specifies the number of rows of the matrix A and C.
/// @param N Specifies the number of columns of the matrix A and C.
///
status_t DNNL_API brdgmm_desc_init(brgemm_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, bool transA, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides = nullptr);

/// Adds post-operations to BRGEMM descriptor
///
/// @param brg Output BRGEMM descriptor
/// @param attr Primitive attributes (can be nullptr). Specifies post-ops
///     operations
/// @param dst_md Specifies the memory descriptor of the destination tensor,
///     needed for binary postops to determine broadcast type, as well as to
///     determine dst data type.
/// @param LDD Specifies the leading dimension of matrix D
///        LDD must be at least max(1, N)
/// @param dt_bias Specifies the data type Bias
///     Can be u8, s8, s32, bf16 or fp32
///
status_t DNNL_API brgemm_desc_set_postops(brgemm_t *brg,
        const primitive_attr_t *attr, const memory_desc_t *dst_md, int LDD,
        impl::data_type_t dt_bias = impl::data_type::undef);

/// Adds BRGEMM attributes to BRGEMM descriptor
///
/// @param brg Output BRGEMM descriptor
/// @param brgattr Specifies kernel attributes and hints: virtual padding,
///     maximum batch size, kernel loop order etc.
///
status_t DNNL_API brgemm_desc_set_attr(
        brgemm_t *brg, const brgemm_attr_t &brgattr);

/// Generates a BRGEMM kernel based on descriptor
///
/// @param brg_kernel Output BRGEMM kernel
/// @param brg BRGEMM descriptor
///
status_t DNNL_API brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_t &brg);

/// Destroys a BRGEMM kernel
///
/// @param brg_kernel BRGEMM kernel
///
status_t DNNL_API brgemm_kernel_destroy(brgemm_kernel_t *brg_kernel);

/// Execute BRGEMM kernel (brgemm_addr version)
///
/// @note
///     Only BRGEMM kernel will be executed even if post-ops are added to BRGEMM
///     descriptor
///
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param batch Array of batch elements containing pointers to matrices
///     A,B and virtual padding for matrices A
/// @param ptr_C Pointer to destination matrix C
/// @param scratch Scratchpad memory needed in several scenarios
///
void DNNL_API brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const brgemm_batch_element_t *batch, void *ptr_C,
        void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_offs and brgemm_strd version)
///
/// @note
///     Only BRGEMM kernel will be executed even if post-ops are added to BRGEMM
///     descriptor
///
/// @note
///     See the second note for `brgemm_kernel_execute` API.
///
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Pointer to first matrix A in the batch
/// @param addr_B Pointer to first matrix B in the batch
/// @param batch Array of batch elements containing offsets to matrices A,B
///     and virtual padding for matrix A. This parameter is ignored when
///     using fixed offsets.
/// @param ptr_C Pointer to destination matrix C
/// @param scratch Scratchpad memory needed in several scenarios
///
void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C,
        void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_addr version)
///
/// @note
///     BRGEMM kernel and post-operations will be executed
///
/// @note
///     See the second note for `brgemm_kernel_execute` API.
///
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param batch Array of batch elements containing pointers to matrices A,B
///     and virtual padding for matrices A
/// @param ptr_C Pointer to matrix C
/// @param ptr_D Pointer to destination matrix D
/// @param post_ops_data Specifies tensors and data used in post processing
///     phase
/// @param scratch Scratchpad memory needed in several scenarios
///
void DNNL_API brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel,
        int bs, const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch = nullptr);

/// Execute BRGEMM kernel (brgemm_offs and brgemm_strd version)
///
/// @note
///     BRGEMM kernel and post-operations will be executed
///
/// @note
///     See the second note for `brgemm_kernel_execute` API.
///
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param addr_A Pointer to first matrix A in the batch
/// @param addr_B Pointer to first matrix B in the batch
/// @param batch Array of batch elements containing offsets to matrices A,B
///     and virtual padding for matrices A. This parameter is ignored when
///     using fixed offsets.
/// @param ptr_C Pointer to destination matrix C
/// @param ptr_D Pointer to destination matrix D
/// @param post_ops_data Specifies tensors and data used in post processing
///     phase
/// @param scratch Scratchpad memory needed in several scenarios
///
void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch = nullptr);

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s