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

#ifndef CPU_X64_BRGEMM_BRGEMM_HPP
#define CPU_X64_BRGEMM_BRGEMM_HPP

#include "cpu/x64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
/// Initializes a BRGEMM descriptor
///
/// @param brg Output BRGEMM descriptor
/// @param isa Target ISA of BRGEMM kernel
///     If isa is equal to 'isa_undef' maximum supported ISA on current
///     hardware will be used for BRGEMM kernel generation
/// @param type Type of batch
/// @param dt_a Data type of A matrix, can be
///     AVX512: f32, u8(row-major layout), s8(column-major layout), bf16, f16
///     AMX: u8, s8, bf16, f16
/// @param dt_b Data type of B matrix
///     AVX512: f32, s8(row-major layout), u8(column-major layout), bf16, f16
///     AMX: u8, s8, bf16, f16
/// @note
///     Data type of matrix C depends on data types of matrices A and B
///     If A and B have integer u8/s8 data type, C has int32 data type
///     If A and B have bf16 or f16 or f32 data type, C has f32 data type
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
///        TODO: what does "Can be nullptr" mean?
///
status_t DNNL_API brgemm_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
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
/// @param dt_a Data type of A matrix can be: f32, u8, bf16, f16
/// @param dt_b Data type of B vector can be: f32, s8, bf16, f16
/// @note
///     Data type of matrix C depends on data types of matrices A and vector B
///     If A and B have integer u8/s8 data type, C has int32 data type
///     If A and B have bf16 or f16 or f32 data type, C has f32 data type
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
/// @param strides - TODO: missing documentation.
///
status_t DNNL_API brdgmm_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
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
///        TODO: why LDD can't be obtained from dst_md directly?
/// @param dt_bias Specifies the data type Bias
///     Can be u8, s8, s32, bf16, f16 or fp32
///
status_t DNNL_API brgemm_desc_set_postops(brgemm_desc_t *brg,
        const primitive_attr_t *attr, const memory_desc_t *dst_md, dim_t LDD,
        impl::data_type_t dt_bias = impl::data_type::undef);

/// Adds BRGEMM attributes to BRGEMM descriptor
///
/// @param brg Output BRGEMM descriptor
/// @param brgattr Specifies kernel attributes and hints: virtual padding,
///     maximum batch size, kernel loop order etc.
///
status_t DNNL_API brgemm_desc_set_attr(
        brgemm_desc_t *brg, const brgemm_attr_t &brgattr);

/// Generates a BRGEMM kernel based on descriptor
///
/// @param brg_kernel Output BRGEMM kernel
/// @param brg BRGEMM descriptor
///
status_t DNNL_API brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_desc_t &brg);

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
/// @note
///     In row major mode matrix B (matrix A for column major) is expected to be
///     in a VNNI-friendly format, which requires 4 consecutive elements of K
///     dimension for int8 data type, 2 elements for bfloat16 data type and no
///     requirements for f32 and f16 data types.
///
/// @param brg_kernel BRGEMM kernel
/// @param bs Specifies the size of batch
/// @param batch Array of batch elements containing pointers to matrices
///     A,B and virtual padding for matrices A
/// @param ptr_C Pointer to destination matrix C
/// @param scratch Scratchpad memory needed in several scenarios:
///     * Where: AMX+ hardware; When: always; For: buffer for tiles store.
///     * In rest scenarios is not used.
/// @param dynamic_values TODO: missing doc
///
void DNNL_API brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const brgemm_batch_element_t *batch, void *ptr_C,
        void *scratch = nullptr,
        const brgemm_dynamic_values_t *dynamic_values = nullptr);

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
/// @param scratch Scratchpad memory needed in several scenarios:
///     * Where: AMX+ hardware; When: always; For: buffer for tiles store.
///     * In rest scenarios is not used.
/// @param dynamic_values TODO: missing doc
///
void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C,
        void *scratch = nullptr,
        const brgemm_dynamic_values_t *dynamic_values = nullptr);

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
/// @param scratch Scratchpad memory needed in several scenarios:
///     * Where: AMX+ hardware; When: always; For: buffer for tiles store.
///     * Where: pre-VNNI hardware; When: s8s8 kernel; For: compensation buffer.
///     * In rest scenarios is not used.
/// @param dynamic_values TODO: missing doc
///
void DNNL_API brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel,
        int bs, const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch = nullptr,
        const brgemm_dynamic_values_t *dynamic_values = nullptr);

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
/// @param scratch Scratchpad memory needed in several scenarios:
///     * Where: AMX+ hardware; When: always; For: buffer for tiles store.
///     * Where: pre-VNNI hardware; When: s8s8 kernel; For: compensation buffer.
///     * In rest scenarios is not used.
/// @param dynamic_values TODO: missing doc
///
void DNNL_API brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel,
        int bs, const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch = nullptr,
        const brgemm_dynamic_values_t *dynamic_values = nullptr);

/// AMX utilities: Creates a palette based on BRGEMM descriptor
///
/// @note
///     This call expects brgemm_desc_t object completely set up, thus, must be
///     used after `brgemm_desc_set_attr` call for non-empty attributes.
///
/// @note
///     Caller is expected to subsequently configure AMX tiles by calling
///     amx_tile_configure(palette).
///
/// @param brg Input BRGeMM descriptor
/// @param palette Output 64 bytes array initialized with tile configuration if
///     returned status is status::success. When any other status is returned,
///     the `palette` is not initialized and can't be used.
///
/// TODO: replace `char[64]` with a proper type that can express itself if it
/// was properly initialized and whether it's empty. Current API is broken in a
/// sense that multiple different scenarios are considered equal, whether
/// it's not AMX, or blocking is completely broken or unsupported.
status_t DNNL_API brgemm_init_tiles(const brgemm_desc_t &brg, char palette[64]);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
