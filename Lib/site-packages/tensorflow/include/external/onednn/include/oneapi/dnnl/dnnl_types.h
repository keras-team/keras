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
/// C API types definitions

#ifndef ONEAPI_DNNL_DNNL_TYPES_H
#define ONEAPI_DNNL_DNNL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/// @cond DO_NOT_DOCUMENT_THIS
#include <stddef.h>
#include <stdint.h>
/// @endcond

#include "oneapi/dnnl/dnnl_config.h"

#include "oneapi/dnnl/dnnl_common_types.h"

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_memory
/// @{

/// Memory format kind
typedef enum {
    /// Undefined memory format kind, used for empty memory descriptors.
    dnnl_format_kind_undef = 0,
    /// A special format kind that indicates that the actual format will be
    /// selected by a primitive automatically.
    dnnl_format_kind_any,
    /// A tensor in a generic format described by the stride and blocking
    /// values in each dimension.
    dnnl_blocked,
    /// A special format kind that indicates that tensor format is opaque.
    dnnl_format_kind_opaque,
#ifdef DNNL_EXPERIMENTAL_SPARSE
    /// Format kind for sparse tensors.
    dnnl_format_kind_sparse,
#endif
    /// Parameter to allow internal only format kinds without undefined
    /// behavior. This parameter is chosen to be valid for so long as
    /// sizeof(int) >= 2.
    dnnl_format_kind_max = 0x7fff,
} dnnl_format_kind_t;

#ifdef DNNL_EXPERIMENTAL_SPARSE
/// Sparse encodings.
typedef enum {
    /// Undefined sparse encoding kind, used for empty memory descriptors.
    dnnl_sparse_encoding_undef = 0,
    /// Compressed Sparse Row (CSR) encoding.
    dnnl_csr,
    /// An encoding that is used for an opaque storage schema for
    /// tensors with unstructured sparsity. A memory descriptor with the
    /// packed encoding cannot be used to create a memory object. It can
    /// only be used to create a primitive descriptor to query the
    /// actual memory descriptor (similar to the format tag `any`).
    dnnl_packed,
} dnnl_sparse_encoding_t;
#endif

#ifdef DNNL_EXPERIMENTAL_PROFILING
/// Profiling data kind.
typedef enum {
    /// Undefined profiling data kind.
    dnnl_profiling_data_kind_undef = 0,
    /// Data kind to query an execution time in nanoseconds.
    dnnl_profiling_data_kind_time,
} dnnl_profiling_data_kind_t;

#endif

/// Memory format tag specification.
///
/// oneDNN formats describe physical data layout. The physical layout
/// is described as a sequence of the dimensions as they are laid out in the
/// memory (from the outer-most to the inner-most). Note that this order
/// doesn't affect the logical order of the dimensions that is kept in the
/// `dims` field of the dnnl_memory_desc_t structure. The logical order of the
/// dimensions is specified by the primitive that uses the tensor.
///
/// For example, CNN 5D tensor always has its logical dimensions in the order
/// `(batch, channels, depth, height, width)`, while the physical layout might be
/// `NCDHW` (corresponds to #dnnl_ncdhw format tag) or
/// `NDHWC` (corresponds to #dnnl_ndhwc format tag).
///
/// ~~~cpp
/// int batch = 2, channels = 16, depth = 13, height = 13, width = 13;
///
/// int ndims = 5; // 5D tensor
/// dnnl_dims_t dims = {batch, channels, depth, height, width};
/// dnnl_memory_desc_t data_in_ncdhw;
/// dnnl_memory_desc_create_with_tag(
///      &data_in_ncdhw, 5, dims, dnnl_f32, dnnl_ncdhw);
///
/// // note that in both cases dims passed are the same
/// dnnl_memory_desc_t data_in_ndhwc;
/// dnnl_memory_desc_create_with_tag(
///      &data_in_ndhwc, 5, dims, dnnl_f32, dnnl_ndhwc);
///
/// dnnl_memory_desc_destroy(data_in_ncdhw);
/// dnnl_memory_desc_destroy(data_in_ndhwc);
/// ~~~
///
/// Memory format tags can be further divided into two categories:
///  - Domain-agnostic names, i.e. names the do not depend on the tensor usage
///    in the specific primitive. These names use letters from `a` to `l` to
///    denote logical dimension from 1 to 12, and form the order in which the
///    dimensions are laid in memory. For instance, #dnnl_ab is used to denote
///    2D tensor where the second logical dimension (aka `b`) is the innermost,
///    i.e. has stride = 1, and the first logical dimension (`a`) laid out in
///    memory with stride equal to the size of second dimension. On the other
///    hand, #dnnl_ba is just transposed version of the same tensor: the
///    first dimension (`a`) becomes the innermost one.
///  - Domain-specific names, i.e. names that make sense only in the context of
///    a certain domain, such as CNN. This names are just aliases to the
///    corresponding domain-agnostic tags and used mostly for the convenience.
///    For example, #dnnl_nc is used to denote 2D CNN activations tensor
///    memory format, where channels are the innermost dimension and batch is an
///    outermost one. Moreover, #dnnl_nc is just an alias to #dnnl_ab,
///    since for oneDNN CNN primitives the logical dimensions of
///    activations tensors come in order: batch, channels, spatial.
///    In other words, batch corresponds to the first logical dimension (`a`),
///    channels correspond to the second one (`b`).
///
/// The following domain-specific notation applies to memory format tags:
///  - @c 'n' denotes the mini-batch dimension
///  - @c 'c' denotes a channels dimension
///  - When there are multiple channel dimensions (for example, in convolution
///    weights tensor), @c 'i' and @c 'o' denote dimensions of input and output
///    channels
///  - @c 'd', @c 'h', and @c 'w' denote spatial depth, height, and width
///    respectively
///
/// Upper-case letters indicate that the data is laid out in blocks for a
/// particular dimension. In such cases, the format name contains both upper-
/// and lower-case letters for that dimension with a lower-case letter preceded
/// by the block size. For example: #dnnl_nChw8c describes a format where the
/// outermost dimension is mini-batch, followed by the channel block number,
/// followed by the spatial height and width, and finally followed by 8-element
/// channel blocks.
///
/// @sa @ref dev_guide_understanding_memory_formats
typedef enum {
    /// Undefined memory format tag
    dnnl_format_tag_undef = 0,
    /// Undefined memory format tag.
    /// The primitive selects a format automatically.
    dnnl_format_tag_any,

    // Semantic agnostic section
    // The physical order of dimensions is defined by the permutation of the
    // characters, assuming that ab..z defines the natural order.

    // Plain formats

    dnnl_a, ///< plain 1D tensor
    dnnl_ab, ///< plain 2D tensor
    dnnl_abc, ///< plain 3D tensor
    dnnl_abcd, ///< plain 4D tensor
    dnnl_abcde, ///< plain 5D tensor
    dnnl_abcdef, ///< plain 6D tensor
    dnnl_abcdefg, ///< plain 7D tensor
    dnnl_abcdefgh, ///< plain 8D tensor
    dnnl_abcdefghi, ///< plain 9D tensor
    dnnl_abcdefghij, ///< plain 10D tensor
    dnnl_abcdefghijk, ///< plain 11D tensor
    dnnl_abcdefghijkl, ///< plain 12D tensor

    // Permuted plain formats

    dnnl_ba, ///< permuted 2D tensor
    dnnl_acb, ///< permuted 3D tensor
    dnnl_bac, ///< permuted 3D tensor
    dnnl_bca, ///< permuted 3D tensor
    dnnl_cab, ///< permuted 3D tensor
    dnnl_cba, ///< permuted 3D tensor
    dnnl_abdc, ///< permuted 4D tensor
    dnnl_acbd, ///< permuted 4D tensor
    dnnl_acdb, ///< permuted 4D tensor
    dnnl_adbc, ///< permuted 4D tensor
    dnnl_adcb, ///< permuted 4D tensor
    dnnl_bacd, ///< permuted 4D tensor
    dnnl_bcda, ///< permuted 4D tensor
    dnnl_cdab, ///< permuted 4D tensor
    dnnl_cdba, ///< permuted 4D tensor
    dnnl_dcab, ///< permuted 4D tensor
    dnnl_abced, ///< permuted 5D tensor
    dnnl_abdec, ///< permuted 5D tensor
    dnnl_acbde, ///< permuted 5D tensor
    dnnl_acdeb, ///< permuted 5D tensor
    dnnl_adecb, ///< permuted 5D tensor
    dnnl_bacde, ///< permuted 5D tensor
    dnnl_bcdea, ///< permuted 5D tensor
    dnnl_cdeab, ///< permuted 5D tensor
    dnnl_cdeba, ///< permuted 5D tensor
    dnnl_decab, ///< permuted 5D tensor
    dnnl_abcdfe, ///< permuted 6D tensor
    dnnl_abdefc, ///< permuted 6D tensor
    dnnl_abdfce, ///< permuted 6D tensor
    dnnl_acbdef, ///< permuted 6D tensor
    dnnl_adefcb, ///< permuted 6D tensor
    dnnl_defcab, ///< permuted 6D tensor
    dnnl_abcdegf, ///< permuted 7D tensor
    dnnl_abcdefhg, ///< permuted 8D tensor
    dnnl_abcdefgih, ///< permuted 9D tensor
    dnnl_abcdefghji, ///< permuted 10D tensor
    dnnl_abcdefghikj, ///< permuted 11D tensor
    dnnl_abcdefghijlk, ///< permuted 12D tensor

    // Opaque blocked formats

    dnnl_Abc16a,
    dnnl_ABc16a16b,
    dnnl_ABc32a32b,
    dnnl_ABc4a4b,
    /// 3D tensor blocked by 2nd dimension with block size 16
    dnnl_aBc16b,
    dnnl_ABc16b16a,
    dnnl_Abc4a,
    /// 3D tensor blocked by 2nd dimension with block size 32
    dnnl_aBc32b,
    /// 3D tensor blocked by 2nd dimension with block size 4
    dnnl_aBc4b,
    dnnl_ABc4b16a4b,
    dnnl_ABc2b8a4b,
    dnnl_ABc16b16a4b,
    dnnl_ABc16b16a2b,
    dnnl_ABc4b4a,
    dnnl_ABc8a16b2a,
    dnnl_ABc8a8b,
    dnnl_ABc8a4b,
    /// 3D tensor blocked by 2nd dimension with block size 8
    dnnl_aBc8b,
    dnnl_ABc8b16a2b,
    dnnl_BAc8a16b2a,
    dnnl_ABc8b8a,
    dnnl_Abcd16a,
    dnnl_Abcd8a,
    dnnl_ABcd16a16b,
    dnnl_Abcd32a,
    dnnl_ABcd32a32b,
    /// 4D tensor blocked by 2nd dimension with block size 16
    dnnl_aBcd16b,
    dnnl_ABcd16b16a,
    dnnl_aBCd16b16c,
    dnnl_aBCd16c16b,
    dnnl_Abcd4a,
    /// 4D tensor blocked by 2nd dimension with block size 32
    dnnl_aBcd32b,
    /// 4D tensor blocked by 2nd dimension with block size 4
    dnnl_aBcd4b,
    dnnl_ABcd4b16a4b,
    dnnl_ABcd16b16a4b,
    dnnl_ABcd16b16a2b,
    dnnl_ABcd4b4a,
    dnnl_ABcd4a4b,
    dnnl_aBCd2c4b2c,
    dnnl_aBCd4b8c2b,
    dnnl_aBCd4c16b4c,
    dnnl_aBCd2c8b4c,
    dnnl_aBCd16c16b4c,
    dnnl_aBCd16c16b2c,
    dnnl_aBCd4c4b,
    dnnl_aBCd4b4c,
    dnnl_ABcd8a16b2a,
    dnnl_ABcd2b8a4b,
    dnnl_ABcd8a8b,
    dnnl_ABcd8a4b,
    /// 4D tensor blocked by 2nd dimension with block size 8
    dnnl_aBcd8b,
    dnnl_aBCd4c8b2c,
    dnnl_ABcd8b16a2b,
    dnnl_aBCd8b16c2b,
    dnnl_BAcd8a16b2a,
    /// 4D tensor blocked by 1st and 2nd dimension with block size 8
    dnnl_ABcd8b8a,
    dnnl_aBCd8b8c,
    dnnl_aBCd8b4c,
    dnnl_aBCd8c16b2c,
    dnnl_ABcde8a16b2a,
    dnnl_aCBd8b16c2b,
    dnnl_aBCd8c8b,
    dnnl_Abcde16a,
    dnnl_Abcde32a,
    dnnl_ABcde16a16b,
    dnnl_BAcde8a16b2a,
    /// 4D tensor blocked by 3rd dimension with block size 4
    dnnl_aBCd2b4c2b,
    /// 5D tensor blocked by 1st dimension with block size 16
    dnnl_ABcde4b16a4b,
    /// 5D tensor blocked by 1st dimension with block size 8
    dnnl_ABcde2b8a4b,
    /// 5D tensor blocked by 2nd dimension with block size 16
    dnnl_aBcde16b,
    dnnl_ABcde16b16a,
    dnnl_aBCde16b16c,
    dnnl_aBCde16c16b,
    dnnl_aBCde2c8b4c,
    dnnl_Abcde4a,
    /// 5D tensor blocked by 2nd dimension with block size 32
    dnnl_aBcde32b,
    /// 5D tensor blocked by 2nd dimension with block size 4
    dnnl_aBcde4b,
    dnnl_ABcde4b4a,
    dnnl_ABcde4a4b,
    dnnl_aBCde4b4c,
    dnnl_aBCde2c4b2c,
    dnnl_aBCde4b8c2b,
    dnnl_aBCde4c16b4c,
    dnnl_aBCde16c16b4c,
    dnnl_aBCde16c16b2c,
    dnnl_aBCde4c4b,
    dnnl_Abcde8a,
    dnnl_ABcde8a8b,
    dnnl_ABcde8a4b,
    dnnl_BAcde16b16a,
    /// 5D tensor blocked by 2nd dimension with block size 8
    dnnl_aBcde8b,
    dnnl_ABcde8b16a2b,
    dnnl_aBCde8b16c2b,
    dnnl_aBCde4c8b2c,
    dnnl_aCBde8b16c2b,
    dnnl_ABcde8b8a,
    dnnl_ABcde32a32b,
    dnnl_aBCde8b8c,
    dnnl_aBCde8b4c,
    dnnl_ABc4a8b8a4b,
    dnnl_ABcd4a8b8a4b,
    dnnl_ABcde4a8b8a4b,
    dnnl_BAc4b8a8b4a,
    dnnl_BAcd4b8a8b4a,
    dnnl_BAcde4b8a8b4a,
    dnnl_ABcd2a8b8a2b,
    dnnl_aBCd4b8c8b4c,
    dnnl_aBCde4b8c8b4c,
    dnnl_aBCde2b8c8b2c,
    dnnl_aBCde8c16b2c,
    dnnl_aBCde8c8b,
    /// 5D tensor blocked by 3rd dimension with block size 4
    dnnl_aBCde2b4c2b,
    /// 6D tensor blocked by 2nd dimension with block size 16
    dnnl_aBcdef16b,
    dnnl_aBCdef16b16c,
    dnnl_aBCdef16c16b,
    dnnl_aBCdef4c16b4c,
    /// 6D tensor blocked by 2nd dimension with block size 8
    dnnl_aBCdef2c8b4c,
    dnnl_aBCdef4c8b2c,
    /// 6D tensor blocked by 3rd dimension with block size 4
    dnnl_aBCdef2b4c2b,
    /// 6D tensor blocked by 2nd dimension with block size 4
    dnnl_aBcdef4b,
    dnnl_aBCdef4c4b,
    dnnl_aBCdef4b4c,
    dnnl_aBCdef2c4b2c,
    dnnl_aBCdef4b8c2b,
    dnnl_aBCdef8b8c,
    dnnl_aBCdef8b4c,
    dnnl_aBCdef8c16b2c,
    dnnl_aBCdef4b8c8b4c,
    dnnl_aBCdef8b16c2b,
    dnnl_aCBdef8b16c2b,
    dnnl_aBCdef8c8b,
    dnnl_aBdc16b,
    dnnl_aBdC16b2c,
    dnnl_aBdC16b4c,
    dnnl_aBdc4b,
    dnnl_aBdc8b,
    dnnl_aBdec16b,
    dnnl_aBdeC16b2c,
    dnnl_aBdeC16b4c,
    dnnl_aBdec32b,
    dnnl_aBdec4b,
    dnnl_aBdec8b,
    dnnl_aBdefc16b,
    dnnl_aBdefC16b2c,
    dnnl_aCBdef16c16b,
    dnnl_aBdefc4b,
    dnnl_aBdefc8b,
    dnnl_Abcdef16a,
    dnnl_Abcdef32a,
    dnnl_aBedc16b,
    dnnl_Acb16a,
    dnnl_AcB16a2b,
    dnnl_AcB16a4b,
    dnnl_Acb4a,
    dnnl_Acb8a,
    dnnl_aCBd16b16c,
    dnnl_aCBd16c16b,
    dnnl_aCBde16b16c,
    dnnl_aCBde16c16b,
    dnnl_Acdb16a,
    dnnl_AcdB16a2b,
    dnnl_AcdB16a4b,
    dnnl_Acdb32a,
    dnnl_Acdb4a,
    dnnl_Acdb8a,
    dnnl_Acdeb16a,
    dnnl_AcdeB16a2b,
    dnnl_Acdeb4a,
    dnnl_Acdeb8a,
    dnnl_Adcb16a,
    dnnl_BAc16a16b,
    dnnl_BAc16b16a,
    dnnl_BAcd16a16b,
    dnnl_BAcd16b16a,
    dnnl_aCBd4c8b8c4b,
    dnnl_aCBde4c8b8c4b,
    dnnl_aCBdef4c8b8c4b,
    dnnl_BAcde16a16b,
    dnnl_aCBdef16b16c,
    dnnl_ABc16b32a,
    dnnl_ABc16b64a,
    dnnl_ABc4b32a4b,
    dnnl_ABc4b64a4b,
    dnnl_ABc8b32a2b,
    dnnl_ABc8b64a2b,
    dnnl_AB16b16a,
    dnnl_AB16b32a,
    dnnl_AB16b64a,
    dnnl_AB8b16a2b,
    dnnl_AB8b32a2b,
    dnnl_AB8b64a2b,
    dnnl_AB4b16a4b,
    dnnl_AB4b32a4b,
    dnnl_AB4b64a4b,
    dnnl_AB16b16a4b,
    dnnl_ABcd16b32a,
    dnnl_ABcd16b64a,
    dnnl_ABcd4b32a4b,
    dnnl_ABcd4b64a4b,
    dnnl_ABcd8b32a2b,
    dnnl_ABcd8b64a2b,
    dnnl_ABcde4b32a4b,
    dnnl_ABcde4b64a4b,
    dnnl_ABcde16b16a4b,
    dnnl_ABcde16b16a2b,
    dnnl_ABcde16b32a,
    dnnl_ABcde16b64a,
    dnnl_ABcde8b32a2b,
    dnnl_ABcde8b64a2b,
    dnnl_aBCdef16c16b4c,
    dnnl_aBCdef16c16b2c,
    dnnl_AB32a32b8a4b,
    dnnl_AB8a4b,
    dnnl_AB32a32b8a2b,
    dnnl_AB8a2b,
    dnnl_abDc32d,
    dnnl_abDC32d4c,
    dnnl_abdEc32e,
    dnnl_abdEC32e2c,
    dnnl_abdEC32e4c,
    dnnl_aBdefC16b4c,
    dnnl_AcdeB16a4b,
    dnnl_ABcd16a16b2a,
    dnnl_ABc16a16b2a,
    dnnl_aBCd16b16c2b,
    dnnl_aBCde16b16c2b,
    dnnl_Acb32a,
    dnnl_AcB32a2b,
    dnnl_AcB32a4b,
    dnnl_Acb48a,
    dnnl_AcB48a2b,
    dnnl_AcB48a4b,
    dnnl_Acb64a,
    dnnl_AcB64a2b,
    dnnl_AcB64a4b,
    dnnl_cBa2b,
    dnnl_cBa4b,
    dnnl_aBdc32b,
    dnnl_aBdC32b2c,
    dnnl_aBdC32b4c,
    dnnl_aBdc48b,
    dnnl_aBdC48b2c,
    dnnl_aBdC48b4c,
    dnnl_aBdc64b,
    dnnl_aBdC64b2c,
    dnnl_aBdC64b4c,
    dnnl_adCb2c,
    dnnl_adCb4c,
    dnnl_AcdB32a2b,
    dnnl_AcdB32a4b,
    dnnl_Acdb48a,
    dnnl_AcdB48a2b,
    dnnl_AcdB48a4b,
    dnnl_Acdb64a,
    dnnl_AcdB64a2b,
    dnnl_AcdB64a4b,
    dnnl_cdBa2b,
    dnnl_cdBa4b,
    dnnl_aBdeC32b2c,
    dnnl_aBdeC32b4c,
    dnnl_aBdec48b,
    dnnl_aBdeC48b2c,
    dnnl_aBdeC48b4c,
    dnnl_aBdec64b,
    dnnl_aBdeC64b2c,
    dnnl_aBdeC64b4c,
    dnnl_adeCb2c,
    dnnl_adeCb4c,
    dnnl_Acdeb32a,
    dnnl_AcdeB32a2b,
    dnnl_AcdeB32a4b,
    dnnl_Acdeb48a,
    dnnl_AcdeB48a2b,
    dnnl_AcdeB48a4b,
    dnnl_Acdeb64a,
    dnnl_AcdeB64a2b,
    dnnl_AcdeB64a4b,
    dnnl_cdeBa2b,
    dnnl_cdeBa4b,
    dnnl_aBdefc32b,
    dnnl_aBdefC32b2c,
    dnnl_aBdefC32b4c,
    dnnl_aBdefc48b,
    dnnl_aBdefC48b2c,
    dnnl_aBdefC48b4c,
    dnnl_aBdefc64b,
    dnnl_aBdefC64b2c,
    dnnl_aBdefC64b4c,
    dnnl_adefCb2c,
    dnnl_adefCb4c,
    dnnl_AB16b32a4b,
    dnnl_AB16b48a4b,
    dnnl_AB16b64a4b,
    dnnl_AB16b16a2b,
    dnnl_AB16b32a2b,
    dnnl_AB16b48a2b,
    dnnl_AB16b64a2b,
    dnnl_ABc16b32a4b,
    dnnl_ABc16b48a4b,
    dnnl_ABc16b64a4b,
    dnnl_ABc16b32a2b,
    dnnl_ABc16b48a2b,
    dnnl_ABc16b64a2b,
    dnnl_ABcd16b32a4b,
    dnnl_ABcd16b48a4b,
    dnnl_ABcd16b64a4b,
    dnnl_ABcd16b32a2b,
    dnnl_ABcd16b48a2b,
    dnnl_ABcd16b64a2b,
    dnnl_ABcde16b32a4b,
    dnnl_ABcde16b48a4b,
    dnnl_ABcde16b64a4b,
    dnnl_ABcde16b32a2b,
    dnnl_ABcde16b48a2b,
    dnnl_ABcde16b64a2b,
    dnnl_ABc32a16b,
    dnnl_ABcd32a16b,
    dnnl_ABcde32a16b,
    dnnl_AB48a16b,
    dnnl_AB48a32b,
    dnnl_ABc40a16b,
    dnnl_ABc40a32b,
    dnnl_aBC48b16c,
    dnnl_aBC48b32c,
    dnnl_ABcd40a16b,
    dnnl_ABcd40a32b,
    dnnl_abCd32c,
    dnnl_abdCe32c,
    dnnl_abdCE32c2e,
    dnnl_BA16a16b2a,
    dnnl_BA16a32b2a,
    dnnl_BA16a48b2a,
    dnnl_BA16a64b2a,
    dnnl_BA16a16b4a,
    dnnl_BA16a32b4a,
    dnnl_BA16a48b4a,
    dnnl_BA16a64b4a,
    dnnl_ABcd8a2b,
    dnnl_aBdeC16c16b2c,
    dnnl_aBdeC16c16b4c,
    dnnl_aBdefC16c16b2c,
    dnnl_AcB16b16a2b,
    dnnl_AcB16b16a4b,
    dnnl_AcdB16b16a2b,
    dnnl_AcdB16b16a4b,
    dnnl_AcdeB16b16a2b,
    dnnl_aBdefC16c16b4c,
    dnnl_AcdeB16b16a4b,
    dnnl_AcB16b32a2b,
    dnnl_AcB16b32a4b,
    dnnl_AcB16b48a2b,
    dnnl_AcB16b48a4b,
    dnnl_AcB16b64a2b,
    dnnl_AcB16b64a4b,
    dnnl_aBdC16c16b2c,
    dnnl_aBdC16c16b4c,
    dnnl_aBdC16c32b2c,
    dnnl_aBdC16c32b4c,
    dnnl_aBdC16c48b2c,
    dnnl_aBdC16c48b4c,
    dnnl_aBdC16c64b2c,
    dnnl_aBdC16c64b4c,
    dnnl_AcdB16b32a2b,
    dnnl_AcdB16b32a4b,
    dnnl_AcdB16b48a2b,
    dnnl_AcdB16b48a4b,
    dnnl_AcdB16b64a2b,
    dnnl_AcdB16b64a4b,
    dnnl_aBdeC16c32b2c,
    dnnl_aBdeC16c32b4c,
    dnnl_aBdeC16c48b2c,
    dnnl_aBdeC16c48b4c,
    dnnl_aBdeC16c64b2c,
    dnnl_aBdeC16c64b4c,
    dnnl_AcdeB16b32a2b,
    dnnl_AcdeB16b32a4b,
    dnnl_AcdeB16b48a2b,
    dnnl_AcdeB16b48a4b,
    dnnl_AcdeB16b64a2b,
    dnnl_AcdeB16b64a4b,
    dnnl_aBdefC16c32b2c,
    dnnl_aBdefC16c32b4c,
    dnnl_aBdefC16c48b2c,
    dnnl_aBdefC16c48b4c,
    dnnl_aBdefC16c64b2c,
    dnnl_aBdefC16c64b4c,
    dnnl_decbA16a,
    dnnl_ABc4a2b,
    dnnl_ABc8a2b,
    dnnl_aBCd8b2c,
    dnnl_ABcde4a2b,
    dnnl_ABcde8a2b,
    dnnl_ABcde40a16b,
    dnnl_ABcde40a32b,
    dnnl_aBCde8b2c,
    dnnl_ABcde4a8b8a2b,
    dnnl_ABcd4a8b8a2b,
    dnnl_ABc4a8b8a2b,
    dnnl_aBCdef4b8c8b2c,
    dnnl_aBCde4b8c8b2c,
    dnnl_aBCd4b8c8b2c,
    dnnl_BAcde4b8a8b2a,
    dnnl_BAcd4b8a8b2a,
    dnnl_BAc4b8a8b2a,
    dnnl_aCBdef4c8b8c2b,
    dnnl_aCBde4c8b8c2b,
    dnnl_aCBd4c8b8c2b,
    dnnl_aBCdef8b2c,
    dnnl_AB32a16b,
    dnnl_AB32a32b,
    dnnl_BA4b8a8b2a,
    dnnl_BA4b8a8b4a,
    dnnl_aBC32b16c,
    dnnl_aBC32b32c,
    dnnl_aCB4c8b8c2b,
    dnnl_aCB4c8b8c4b,
    dnnl_ABcd4a2b,
    dnnl_ABc2b8a16b4a,
    dnnl_ABcd2b8a16b4a,
    dnnl_ABcde2b8a16b4a,
    dnnl_ABc2a8b16a4b,
    dnnl_ABc2a8b16a2b,
    dnnl_ABc2b32a8b,
    dnnl_ABcd2a8b16a4b,
    dnnl_ABcd2a8b16a2b,
    dnnl_aCBd2c8b16c2b,
    dnnl_ABcd2b32a8b,
    dnnl_aBCd2c8b16c2b,
    dnnl_ABcde2a8b16a4b,
    dnnl_ABcde2a8b16a2b,
    dnnl_aCBde2c8b16c2b,
    dnnl_ABcde2b32a8b,
    dnnl_aBC2b8c16b2c,
    dnnl_aBCd2b8c16b2c,
    dnnl_aBCde2b8c16b2c,
    dnnl_aBCdef2b8c16b2c,
    dnnl_BAcde2b8a16b4a,
    dnnl_BAcd2b8a16b4a,
    dnnl_BAc2b8a16b4a,
    dnnl_BAcde2b8a16b2a,
    dnnl_BAcd2b8a16b2a,
    dnnl_BAc2b8a16b2a,
    dnnl_aBCde2c8b16c2b,
    dnnl_aBCdef2c8b16c2b,
    dnnl_aCBdef2c8b16c2b,
    dnnl_aBCd2b8c16b4c,
    dnnl_aBCde2b8c16b4c,
    dnnl_BA4b8a16b2a,
    dnnl_BA4b8a16b4a,
    dnnl_aCB4c8b16c2b,
    dnnl_aCB4c8b16c4b,
    dnnl_BA16a16b,
    dnnl_BA16a32b,
    dnnl_BA16a48b,
    dnnl_BA16a64b,
    dnnl_aCB16c2b,
    dnnl_aCB16c4b,
    dnnl_BA16b2a,
    dnnl_BA16b4a,
    dnnl_aBC16b16c,
    dnnl_aBC16b32c,
    dnnl_AB16a16b,
    dnnl_AB16a32b,
    dnnl_ABcde16a16b2a,
    dnnl_aBCdef16b16c2b,
    dnnl_Acedb16a,
    dnnl_aBdfec16b,
    dnnl_abdEC64e2c,
    dnnl_abdEC64e4c,
    dnnl_aCB16b16c,
    dnnl_aCB16b32c,
    dnnl_aCB16b48c,
    dnnl_aCB16b64c,
    dnnl_aCB16b16c2b,
    dnnl_aCB16b32c2b,
    dnnl_aCB16b48c2b,
    dnnl_aCB16b64c2b,
    dnnl_aCB16b16c4b,
    dnnl_aCB16b32c4b,
    dnnl_aCB16b48c4b,
    dnnl_aCB16b64c4b,
    dnnl_abCd4c,
    dnnl_abCde4c,
    dnnl_abCdef4c,
    dnnl_abCde32c,
    dnnl_abCdef32c,
    dnnl_ABcd16a32b,
    dnnl_decbA8a,
    dnnl_aCdefB16b32c2b,
    dnnl_aCdefB16b32c4b,
    dnnl_aCdefB16b48c2b,
    dnnl_aCdefB16b48c4b,
    dnnl_aCdefB16b64c2b,
    dnnl_aCdefB16b64c4b,
    dnnl_BcdeA16a32b2a,
    dnnl_BcdeA16a32b4a,
    dnnl_BcdeA16a48b2a,
    dnnl_BcdeA16a48b4a,
    dnnl_BcdeA16a64b2a,
    dnnl_BcdeA16a64b4a,
    dnnl_aCdefb32c,
    dnnl_aCdefB32c2b,
    dnnl_aCdefB32c4b,
    dnnl_aCdefb48c,
    dnnl_aCdefB48c2b,
    dnnl_aCdefB48c4b,
    dnnl_aCdefb64c,
    dnnl_aCdefB64c2b,
    dnnl_aCdefB64c4b,
    dnnl_Bcdea32b,
    dnnl_BcdeA32b2a,
    dnnl_BcdeA32b4a,
    dnnl_Bcdea48b,
    dnnl_BcdeA48b2a,
    dnnl_BcdeA48b4a,
    dnnl_Bcdea64b,
    dnnl_BcdeA64b2a,
    dnnl_BcdeA64b4a,
    dnnl_Bca32b,
    dnnl_BcA32b2a,
    dnnl_BcA32b4a,
    dnnl_Bca48b,
    dnnl_BcA48b2a,
    dnnl_BcA48b4a,
    dnnl_Bca64b,
    dnnl_BcA64b2a,
    dnnl_BcA64b4a,
    dnnl_aCdb32c,
    dnnl_aCdB32c2b,
    dnnl_aCdB32c4b,
    dnnl_aCdb48c,
    dnnl_aCdB48c2b,
    dnnl_aCdB48c4b,
    dnnl_aCdb64c,
    dnnl_aCdB64c2b,
    dnnl_aCdB64c4b,
    dnnl_BcA16a16b2a,
    dnnl_BcA16a16b4a,
    dnnl_BcdA16a16b2a,
    dnnl_BcdA16a16b4a,
    dnnl_BcdeA16a16b2a,
    dnnl_BcdeA16a16b4a,
    dnnl_aCdB16b16c2b,
    dnnl_aCdB16b16c4b,
    dnnl_aCdeB16b16c2b,
    dnnl_aCdeB16b16c4b,
    dnnl_aCdefB16b16c2b,
    dnnl_aCdefB16b16c4b,
    dnnl_BcA16a32b2a,
    dnnl_BcA16a32b4a,
    dnnl_BcA16a48b2a,
    dnnl_BcA16a48b4a,
    dnnl_BcA16a64b2a,
    dnnl_BcA16a64b4a,
    dnnl_aCdB16b32c2b,
    dnnl_aCdB16b32c4b,
    dnnl_aCdB16b48c2b,
    dnnl_aCdB16b48c4b,
    dnnl_aCdB16b64c2b,
    dnnl_aCdB16b64c4b,
    dnnl_BcdA16a32b2a,
    dnnl_BcdA16a32b4a,
    dnnl_BcdA16a48b2a,
    dnnl_BcdA16a48b4a,
    dnnl_BcdA16a64b2a,
    dnnl_BcdA16a64b4a,
    dnnl_aCdeB16b32c2b,
    dnnl_aCdeB16b32c4b,
    dnnl_aCdeB16b48c2b,
    dnnl_aCdeB16b48c4b,
    dnnl_aCdeB16b64c2b,
    dnnl_aCdeB16b64c4b,
    dnnl_Bca16b,
    dnnl_BcA16b2a,
    dnnl_BcA16b4a,
    dnnl_Bcda16b,
    dnnl_BcdA16b2a,
    dnnl_BcdA16b4a,
    dnnl_Bcdea16b,
    dnnl_BcdeA16b2a,
    dnnl_BcdeA16b4a,
    dnnl_aCdb16c,
    dnnl_aCdB16c2b,
    dnnl_aCdB16c4b,
    dnnl_aCdeb16c,
    dnnl_aCdeB16c2b,
    dnnl_aCdeB16c4b,
    dnnl_aCdefb16c,
    dnnl_aCdefB16c2b,
    dnnl_aCdefB16c4b,
    dnnl_Bcda32b,
    dnnl_BcdA32b2a,
    dnnl_BcdA32b4a,
    dnnl_Bcda48b,
    dnnl_BcdA48b2a,
    dnnl_BcdA48b4a,
    dnnl_Bcda64b,
    dnnl_BcdA64b2a,
    dnnl_BcdA64b4a,
    dnnl_aCdeb32c,
    dnnl_aCdeB32c2b,
    dnnl_aCdeB32c4b,
    dnnl_aCdeb48c,
    dnnl_aCdeB48c2b,
    dnnl_aCdeB48c4b,
    dnnl_aCdeb64c,
    dnnl_aCdeB64c2b,
    dnnl_aCdeB64c4b,
    dnnl_Acb24a,
    dnnl_Acdb24a,
    dnnl_Acdeb24a,
    dnnl_aBdc24b,
    dnnl_aBdec24b,
    dnnl_aBdefc24b,
    dnnl_abDc16d,
    dnnl_abdEc16e,
    dnnl_abdCe16c,
    dnnl_AcB24a2b,
    dnnl_AcdB24a2b,
    dnnl_AcdeB24a2b,
    dnnl_aBdC24b2c,
    dnnl_aBdeC24b2c,
    dnnl_aBdefC24b2c,
    dnnl_AcB8a2b,
    dnnl_AcdB8a2b,
    dnnl_AcdeB8a2b,
    dnnl_aBdC8b2c,
    dnnl_aBdeC8b2c,
    dnnl_aBdefC8b2c,
    dnnl_AB8b32a,
    dnnl_ABc8b32a,
    dnnl_ABcd8b32a,
    dnnl_ABcde8b32a,
    dnnl_AB8b24a,
    dnnl_ABc8b24a,
    dnnl_ABcd8b24a,
    dnnl_ABcde8b24a,
    dnnl_AB8b16a,
    dnnl_ABc8b16a,
    dnnl_ABcd8b16a,
    dnnl_ABcde8b16a,
    dnnl_AB8b8a,
    dnnl_AB4b8a4b,
    dnnl_AB4b24a4b,
    dnnl_ABc4b8a4b,
    dnnl_ABc4b24a4b,
    dnnl_ABcd4b8a4b,
    dnnl_ABcd4b24a4b,
    dnnl_ABcde4b8a4b,
    dnnl_ABcde4b24a4b,
    dnnl_AB8b24a2b,
    dnnl_ABc8b24a2b,
    dnnl_ABcd8b24a2b,
    dnnl_ABcde8b24a2b,
    dnnl_AB8b8a2b,
    dnnl_ABc8b8a2b,
    dnnl_ABcd8b8a2b,
    dnnl_ABcde8b8a2b,
    dnnl_AcB24a4b,
    dnnl_AcdB24a4b,
    dnnl_AcdeB24a4b,
    dnnl_aBdC24b4c,
    dnnl_aBdeC24b4c,
    dnnl_aBdefC24b4c,
    dnnl_AcB8a4b,
    dnnl_AcdB8a4b,
    dnnl_AcdeB8a4b,
    dnnl_aBdC8b4c,
    dnnl_aBdeC8b4c,
    dnnl_aBdefC8b4c,
    dnnl_Bca8b,
    dnnl_BcA8b2a,
    dnnl_Bcda8b,
    dnnl_BcdA8b2a,
    dnnl_Bcdea8b,
    dnnl_BcdeA8b2a,
    dnnl_aCdb8c,
    dnnl_aCdB8c2b,
    dnnl_aCdeb8c,
    dnnl_aCdeB8c2b,
    dnnl_aCdefb8c,
    dnnl_aCdefB8c2b,
    dnnl_Bca24b,
    dnnl_BcA24b2a,
    dnnl_Bcda24b,
    dnnl_BcdA24b2a,
    dnnl_Bcdea24b,
    dnnl_BcdeA24b2a,
    dnnl_aCdb24c,
    dnnl_aCdB24c2b,
    dnnl_aCdeb24c,
    dnnl_aCdeB24c2b,
    dnnl_aCdefb24c,
    dnnl_aCdefB24c2b,
    dnnl_BcA8b4a,
    dnnl_BcdA8b4a,
    dnnl_BcdeA8b4a,
    dnnl_aCdB8c4b,
    dnnl_aCdeB8c4b,
    dnnl_aCdefB8c4b,
    dnnl_BcA24b4a,
    dnnl_BcdA24b4a,
    dnnl_BcdeA24b4a,
    dnnl_aCdB24c4b,
    dnnl_aCdeB24c4b,
    dnnl_aCdefB24c4b,
    dnnl_AB16b48a,
    dnnl_ABc16b48a,
    dnnl_ABcd16b48a,
    dnnl_ABcde16b48a,
    dnnl_ABc16a4b,
    dnnl_ABcd16a4b,
    dnnl_ABcde16a4b,
    dnnl_defcbA16a,
    dnnl_defcbA8a,
    dnnl_AcB16b64a,
    dnnl_AcdB16b64a,
    dnnl_AcdeB16b64a,
    dnnl_AcB16b48a,
    dnnl_AcdB16b48a,
    dnnl_AcdeB16b48a,
    dnnl_AcB16b32a,
    dnnl_AcdB16b32a,
    dnnl_AcdeB16b32a,
    dnnl_AcB16b16a,
    dnnl_AcdB16b16a,
    dnnl_AcdeB16b16a,
    dnnl_AcB8b32a,
    dnnl_AcdB8b32a,
    dnnl_AcdeB8b32a,
    dnnl_AcB8b24a,
    dnnl_AcdB8b24a,
    dnnl_AcdeB8b24a,
    dnnl_AcB8b16a,
    dnnl_AcdB8b16a,
    dnnl_AcdeB8b16a,
    dnnl_AcB8b8a,
    dnnl_AcdB8b8a,
    dnnl_AcdeB8b8a,
    dnnl_AcB8b64a2b,
    dnnl_AcdB8b64a2b,
    dnnl_AcdeB8b64a2b,
    dnnl_AcB8b32a2b,
    dnnl_AcdB8b32a2b,
    dnnl_AcdeB8b32a2b,
    dnnl_AcB8b24a2b,
    dnnl_AcdB8b24a2b,
    dnnl_AcdeB8b24a2b,
    dnnl_AcB8b16a2b,
    dnnl_AcdB8b16a2b,
    dnnl_AcdeB8b16a2b,
    dnnl_AcB8b8a2b,
    dnnl_AcdB8b8a2b,
    dnnl_AcdeB8b8a2b,
    dnnl_AcB4b64a4b,
    dnnl_AcdB4b64a4b,
    dnnl_AcdeB4b64a4b,
    dnnl_AcB4b32a4b,
    dnnl_AcdB4b32a4b,
    dnnl_AcdeB4b32a4b,
    dnnl_AcB4b24a4b,
    dnnl_AcdB4b24a4b,
    dnnl_AcdeB4b24a4b,
    dnnl_AcB4b16a4b,
    dnnl_AcdB4b16a4b,
    dnnl_AcdeB4b16a4b,
    dnnl_AcB4b8a4b,
    dnnl_AcdB4b8a4b,
    dnnl_AcdeB4b8a4b,
    dnnl_Ab4a,
    dnnl_Ab8a,
    dnnl_BA4b4a,
    dnnl_BA8b4a,
    dnnl_BA2a24b,
    dnnl_aCB2b24c,
    dnnl_BA2a8b,
    dnnl_aCB2b8c,
    dnnl_BA8a24b,
    dnnl_aCB8b24c,
    dnnl_BA8a16b,
    dnnl_aCB8b16c,
    dnnl_BA8a8b,
    dnnl_aCB8b8c,
    dnnl_bcad,
    dnnl_cabd,
    dnnl_dabc,

    /// Just a sentinel, not real memory format tag. Must be changed after new
    /// format tag is added.
    dnnl_format_tag_last,

    // Aliases

    /// 1D tensor, an alias to #dnnl_a
    dnnl_x = dnnl_a,
    /// 2D CNN activations tensor, an alias to #dnnl_ab
    dnnl_nc = dnnl_ab,
    /// 2D CNN activations tensor, an alias to #dnnl_ba
    dnnl_cn = dnnl_ba,
    /// 2D RNN statistics tensor, an alias to #dnnl_ab
    dnnl_tn = dnnl_ab,
    /// 2D RNN statistics tensor, an alias to #dnnl_ba
    dnnl_nt = dnnl_ba,
    /// 3D CNN activations tensor, an alias to #dnnl_abc
    dnnl_ncw = dnnl_abc,
    /// 3D CNN activations tensor, an alias to #dnnl_acb
    dnnl_nwc = dnnl_acb,
    /// 4D CNN activations tensor, an alias to #dnnl_abcd
    dnnl_nchw = dnnl_abcd,
    /// 4D CNN activations tensor, an alias to #dnnl_acdb
    dnnl_nhwc = dnnl_acdb,
    /// 4D CNN activations tensor, an alias to #dnnl_bcda
    dnnl_chwn = dnnl_bcda,
    /// 5D CNN activations tensor, an alias to #dnnl_abcde
    dnnl_ncdhw = dnnl_abcde,
    /// 5D CNN activations tensor, an alias to #dnnl_acdeb
    dnnl_ndhwc = dnnl_acdeb,

    /// 2D CNN weights tensor, an alias to #dnnl_ab
    dnnl_oi = dnnl_ab,
    /// 2D CNN weights tensor, an alias to #dnnl_ba
    dnnl_io = dnnl_ba,
    /// 3D CNN weights tensor, an alias to #dnnl_abc
    dnnl_oiw = dnnl_abc,
    /// 3D CNN weights tensor, an alias to #dnnl_acb
    dnnl_owi = dnnl_acb,
    /// 3D CNN weights tensor, an alias to #dnnl_cba
    dnnl_wio = dnnl_cba,
    /// 3D CNN weights tensor, an alias to #dnnl_cab
    dnnl_woi = dnnl_cab,
    /// 3D CNN weights tensor, an alias to #dnnl_bca
    dnnl_iwo = dnnl_bca,
    /// 4D CNN weights tensor, an alias to #dnnl_abcd
    dnnl_oihw = dnnl_abcd,
    /// 4D CNN weights tensor, an alias to #dnnl_cdba
    dnnl_hwio = dnnl_cdba,
    /// 4D CNN weights tensor, an alias to #dnnl_cdab
    dnnl_hwoi = dnnl_cdab,
    /// 4D CNN weights tensor, an alias to #dnnl_acdb
    dnnl_ohwi = dnnl_acdb,
    /// 4D CNN weights tensor, an alias to #dnnl_bcda
    dnnl_ihwo = dnnl_bcda,
    /// 4D CNN weights tensor, an alias to #dnnl_bacd
    dnnl_iohw = dnnl_bacd,
    /// 5D CNN weights tensor, an alias to #dnnl_abcde
    dnnl_oidhw = dnnl_abcde,
    /// 5D CNN weights tensor, an alias to #dnnl_bacde
    dnnl_iodhw = dnnl_bacde,
    /// 5D CNN weights tensor, an alias to #dnnl_cdeba
    dnnl_dhwio = dnnl_cdeba,
    /// 5D CNN weights tensor, an alias to #dnnl_cdeab
    dnnl_dhwoi = dnnl_cdeab,
    /// 5D CNN weights tensor, an alias to #dnnl_acdeb
    dnnl_odhwi = dnnl_acdeb,
    /// 5D CNN weights tensor, an alias to #dnnl_bcdea
    dnnl_idhwo = dnnl_bcdea,

    /// 4D CNN weights tensor (incl. groups), an alias to #dnnl_abcd
    dnnl_goiw = dnnl_abcd,
    /// 4D CNN weights tensor (incl. groups), an alias to #dnnl_abdc
    dnnl_gowi = dnnl_abdc,
    /// 4D CNN weights tensor (incl. groups), an alias to #dnnl_dcab
    dnnl_wigo = dnnl_dcab,
    /// 5D CNN weights tensor (incl. groups), an alias to #dnnl_abcde
    dnnl_goihw = dnnl_abcde,
    /// 5D CNN weights tensor (incl. groups), an alias to #dnnl_abdec
    dnnl_gohwi = dnnl_abdec,
    /// 5D CNN weights tensor (incl. groups), an alias to #dnnl_decab
    dnnl_hwigo = dnnl_decab,
    /// 5D CNN weights tensor (incl. groups), an alias to #dnnl_acbde
    dnnl_giohw = dnnl_acbde,
    /// 6D CNN weights tensor (incl. groups), an alias to #dnnl_abcdef
    dnnl_goidhw = dnnl_abcdef,
    /// 6D CNN weights tensor (incl. groups), an alias to #dnnl_abdefc
    dnnl_godhwi = dnnl_abdefc,
    /// 6D CNN weights tensor (incl. groups), an alias to #dnnl_acbdef
    dnnl_giodhw = dnnl_acbdef,
    /// 6D CNN weights tensor (incl. groups), an alias to #dnnl_defcab
    dnnl_dhwigo = dnnl_defcab,

    /// 3D RNN data tensor in the format (seq_length, batch, input channels),
    /// an alias to #dnnl_abc.
    dnnl_tnc = dnnl_abc,
    /// 3D RNN data tensor in the format (batch, seq_length, input channels),
    /// an alias to #dnnl_bac.
    dnnl_ntc = dnnl_bac,
    /// 4D RNN states tensor in the format (num_layers, num_directions,
    /// batch, state channels), an alias to #dnnl_abcd.
    dnnl_ldnc = dnnl_abcd,
    /// 5D RNN weights tensor in the format (num_layers, num_directions,
    /// input_channels, num_gates, output_channels), an alias to #dnnl_abcde.
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    dnnl_ldigo = dnnl_abcde,
    /// 5D RNN weights tensor in the format (num_layers, num_directions,
    /// num_gates, output_channels, input_channels), an alias to #dnnl_abdec.
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    dnnl_ldgoi = dnnl_abdec,
    /// 4D LSTM projection tensor in the format (num_layers, num_directions,
    /// num_channels_in_hidden_state, num_channels_in_recurrent_projection),
    /// an alias to #dnnl_abcd.
    dnnl_ldio = dnnl_abcd,
    /// 4D LSTM projection tensor in the format (num_layers, num_directions,
    /// num_channels_in_recurrent_projection, num_channels_in_hidden_state),
    /// an alias to #dnnl_abdc.
    dnnl_ldoi = dnnl_abdc,
    /// 4D RNN bias tensor in the format (num_layers, num_directions,
    /// num_gates, output_channels), an alias to #dnnl_abcd.
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    dnnl_ldgo = dnnl_abcd,
    /// 5D LSTM projection tensor
    dnnl_ldOi16o = dnnl_abDc16d,
    dnnl_ldOi32o = dnnl_abDc32d,
    dnnl_ldOI32o4i = dnnl_abDC32d4c,
    dnnl_ldIo32i = dnnl_abCd32c,
    /// 6D RNN weights tensor
    dnnl_ldgOi16o = dnnl_abdEc16e,
    dnnl_ldgOi32o = dnnl_abdEc32e,
    dnnl_ldgOI32o2i = dnnl_abdEC32e2c,
    dnnl_ldgOI32o4i = dnnl_abdEC32e4c,
    dnnl_ldgOI64o2i = dnnl_abdEC64e2c,
    dnnl_ldgOI64o4i = dnnl_abdEC64e4c,
    dnnl_ldgIo16i = dnnl_abdCe16c,
    dnnl_ldgIo32i = dnnl_abdCe32c,
    dnnl_ldgIO32i2o = dnnl_abdCE32c2e,

    // Opaque data types, are not to be used explicitly

    // data
    /// 5D CNN activations tensor blocked by channels with block size 32,
    /// an alias to #dnnl_aBcde32b
    dnnl_nCdhw32c = dnnl_aBcde32b,
    /// 5D CNN activations tensor blocked by channels with block size 16,
    /// an alias to #dnnl_aBcde16b
    dnnl_nCdhw16c = dnnl_aBcde16b,
    /// 5D CNN activations tensor blocked by channels with block size 4,
    /// an alias to #dnnl_aBcde4b
    dnnl_nCdhw4c = dnnl_aBcde4b,
    /// 5D CNN activations tensor blocked by channels with block size 8,
    /// an alias to #dnnl_aBcde8b
    dnnl_nCdhw8c = dnnl_aBcde8b,
    /// 4D CNN activations tensor blocked by channels with block size 32,
    /// an alias to #dnnl_aBcd32b
    dnnl_nChw32c = dnnl_aBcd32b,
    /// 4D CNN activations tensor blocked by channels with block size 16,
    /// an alias to #dnnl_aBcd16b
    dnnl_nChw16c = dnnl_aBcd16b,
    /// 4D CNN activations tensor blocked by channels with block size 4,
    /// an alias to #dnnl_aBcd4b
    dnnl_nChw4c = dnnl_aBcd4b,
    /// 4D CNN activations tensor blocked by channels with block size 8,
    /// an alias to #dnnl_aBcd8b
    dnnl_nChw8c = dnnl_aBcd8b,
    /// 3D CNN activations tensor blocked by channels with block size 32,
    /// an alias to #dnnl_aBc32b
    dnnl_nCw32c = dnnl_aBc32b,
    /// 3D CNN activations tensor blocked by channels with block size 16,
    /// an alias to #dnnl_aBc16b
    dnnl_nCw16c = dnnl_aBc16b,
    /// 3D CNN activations tensor blocked by channels with block size 4,
    /// an alias to #dnnl_aBc4b
    dnnl_nCw4c = dnnl_aBc4b,
    /// 3D CNN activations tensor blocked by channels with block size 8,
    /// an alias to #dnnl_aBc8b
    dnnl_nCw8c = dnnl_aBc8b,
    dnnl_NCw16n16c = dnnl_ABc16a16b,
    dnnl_NCdhw16n16c = dnnl_ABcde16a16b,
    dnnl_NChw16n16c = dnnl_ABcd16a16b,
    dnnl_NCw32n16c = dnnl_ABc32a16b,
    dnnl_NChw32n16c = dnnl_ABcd32a16b,
    dnnl_NChw16n32c = dnnl_ABcd16a32b,
    dnnl_NCdhw32n16c = dnnl_ABcde32a16b,
    dnnl_NCw32n32c = dnnl_ABc32a32b,
    dnnl_NChw32n32c = dnnl_ABcd32a32b,
    dnnl_NCdhw32n32c = dnnl_ABcde32a32b,

    // weights, 2D
    dnnl_OI16i16o = dnnl_AB16b16a,
    dnnl_OI16i32o = dnnl_AB16b32a,
    dnnl_OI16i48o = dnnl_AB16b48a,
    dnnl_OI16i64o = dnnl_AB16b64a,
    dnnl_OI8i8o2i = dnnl_AB8b8a2b,
    dnnl_OI8i16o2i = dnnl_AB8b16a2b,
    dnnl_OI8i24o2i = dnnl_AB8b24a2b,
    dnnl_OI8i32o2i = dnnl_AB8b32a2b,
    dnnl_OI8i64o2i = dnnl_AB8b64a2b,
    dnnl_OI4i8o4i = dnnl_AB4b8a4b,
    dnnl_OI4i16o4i = dnnl_AB4b16a4b,
    dnnl_OI4i24o4i = dnnl_AB4b24a4b,
    dnnl_OI4i32o4i = dnnl_AB4b32a4b,
    dnnl_OI4i64o4i = dnnl_AB4b64a4b,
    dnnl_OI16i16o4i = dnnl_AB16b16a4b,
    dnnl_OI8i32o = dnnl_AB8b32a,
    dnnl_OI8i24o = dnnl_AB8b24a,
    dnnl_OI8i16o = dnnl_AB8b16a,
    dnnl_OI8i8o = dnnl_AB8b8a,

    // weights, 3D
    dnnl_IOw16o16i = dnnl_BAc16a16b,
    dnnl_IOw16i16o = dnnl_BAc16b16a,
    dnnl_OIw16i16o = dnnl_ABc16b16a,
    dnnl_OwI16i16o = dnnl_AcB16b16a,
    dnnl_OIw16i32o = dnnl_ABc16b32a,
    dnnl_OwI16i32o = dnnl_AcB16b32a,
    dnnl_OIw16i48o = dnnl_ABc16b48a,
    dnnl_OwI16i48o = dnnl_AcB16b48a,
    dnnl_OIw16i64o = dnnl_ABc16b64a,
    dnnl_OwI16i64o = dnnl_AcB16b64a,
    dnnl_OIw16o16i = dnnl_ABc16a16b,
    dnnl_Oiw16o = dnnl_Abc16a,
    dnnl_OIw4i8o4i = dnnl_ABc4b8a4b,
    dnnl_OwI4i8o4i = dnnl_AcB4b8a4b,
    dnnl_OIw4i16o4i = dnnl_ABc4b16a4b,
    dnnl_OwI4i16o4i = dnnl_AcB4b16a4b,
    dnnl_OIw4i24o4i = dnnl_ABc4b24a4b,
    dnnl_OwI4i24o4i = dnnl_AcB4b24a4b,
    dnnl_OIw4i32o4i = dnnl_ABc4b32a4b,
    dnnl_OwI4i32o4i = dnnl_AcB4b32a4b,
    dnnl_OIw4i64o4i = dnnl_ABc4b64a4b,
    dnnl_OwI4i64o4i = dnnl_AcB4b64a4b,
    dnnl_OIw2i8o4i = dnnl_ABc2b8a4b,
    dnnl_OIw16i16o4i = dnnl_ABc16b16a4b,
    dnnl_OIw16i16o2i = dnnl_ABc16b16a2b,
    dnnl_OIw16o16i2o = dnnl_ABc16a16b2a,
    dnnl_OIw4i4o = dnnl_ABc4b4a,
    dnnl_OIw4o4i = dnnl_ABc4a4b,
    dnnl_Oiw4o = dnnl_Abc4a,
    dnnl_OIw8i8o2i = dnnl_ABc8b8a2b,
    dnnl_OwI8i8o2i = dnnl_AcB8b8a2b,
    dnnl_OIw8i16o2i = dnnl_ABc8b16a2b,
    dnnl_OwI8i16o2i = dnnl_AcB8b16a2b,
    dnnl_OIw8i24o2i = dnnl_ABc8b24a2b,
    dnnl_OwI8i24o2i = dnnl_AcB8b24a2b,
    dnnl_OIw8i32o2i = dnnl_ABc8b32a2b,
    dnnl_OwI8i32o2i = dnnl_AcB8b32a2b,
    dnnl_OIw8i64o2i = dnnl_ABc8b64a2b,
    dnnl_OwI8i64o2i = dnnl_AcB8b64a2b,
    dnnl_OIw8i8o = dnnl_ABc8b8a,
    dnnl_OwI8i8o = dnnl_AcB8b8a,
    dnnl_OIw8o16i2o = dnnl_ABc8a16b2a,
    dnnl_IOw8o16i2o = dnnl_BAc8a16b2a,
    dnnl_OIw8o8i = dnnl_ABc8a8b,
    dnnl_OIw8o4i = dnnl_ABc8a4b,
    dnnl_Owi16o = dnnl_Acb16a,
    dnnl_OwI16o2i = dnnl_AcB16a2b,
    dnnl_OwI16o4i = dnnl_AcB16a4b,
    dnnl_Iwo8i = dnnl_Bca8b,
    dnnl_IwO8i2o = dnnl_BcA8b2a,
    dnnl_IwO8i4o = dnnl_BcA8b4a,
    dnnl_Iwo16i = dnnl_Bca16b,
    dnnl_IwO16i2o = dnnl_BcA16b2a,
    dnnl_IwO16i4o = dnnl_BcA16b4a,
    dnnl_Iwo24i = dnnl_Bca24b,
    dnnl_IwO24i2o = dnnl_BcA24b2a,
    dnnl_IwO24i4o = dnnl_BcA24b4a,
    dnnl_Owi4o = dnnl_Acb4a,
    dnnl_Owi8o = dnnl_Acb8a,
    dnnl_OwI8o2i = dnnl_AcB8a2b,
    dnnl_OIw8i32o = dnnl_ABc8b32a,
    dnnl_OwI8i32o = dnnl_AcB8b32a,
    dnnl_OIw8i24o = dnnl_ABc8b24a,
    dnnl_OwI8i24o = dnnl_AcB8b24a,
    dnnl_OIw8i16o = dnnl_ABc8b16a,
    dnnl_OwI8i16o = dnnl_AcB8b16a,
    dnnl_OwI8o4i = dnnl_AcB8a4b,

    // weights, 4D
    dnnl_IOhw16i16o = dnnl_BAcd16b16a,
    dnnl_IOhw16o16i = dnnl_BAcd16a16b,
    dnnl_Ohwi16o = dnnl_Acdb16a,
    dnnl_OhwI16o2i = dnnl_AcdB16a2b,
    dnnl_OhwI16o4i = dnnl_AcdB16a4b,
    dnnl_Ihwo8i = dnnl_Bcda8b,
    dnnl_IhwO8i2o = dnnl_BcdA8b2a,
    dnnl_IhwO8i4o = dnnl_BcdA8b4a,
    dnnl_Ihwo16i = dnnl_Bcda16b,
    dnnl_IhwO16i2o = dnnl_BcdA16b2a,
    dnnl_IhwO16i4o = dnnl_BcdA16b4a,
    dnnl_Ihwo24i = dnnl_Bcda24b,
    dnnl_IhwO24i2o = dnnl_BcdA24b2a,
    dnnl_IhwO24i4o = dnnl_BcdA24b4a,
    dnnl_Ohwi24o = dnnl_Acdb24a,
    dnnl_Ohwi32o = dnnl_Acdb32a,
    dnnl_Ohwi4o = dnnl_Acdb4a,
    dnnl_Ohwi8o = dnnl_Acdb8a,
    dnnl_OhwI8o2i = dnnl_AcdB8a2b,
    dnnl_OhwI8o4i = dnnl_AcdB8a4b,
    dnnl_OIhw16i16o = dnnl_ABcd16b16a,
    dnnl_OhwI16i16o = dnnl_AcdB16b16a,
    dnnl_OIhw16i32o = dnnl_ABcd16b32a,
    dnnl_OhwI16i32o = dnnl_AcdB16b32a,
    dnnl_OIhw16i48o = dnnl_ABcd16b48a,
    dnnl_OhwI16i48o = dnnl_AcdB16b48a,
    dnnl_OIhw16i64o = dnnl_ABcd16b64a,
    dnnl_OhwI16i64o = dnnl_AcdB16b64a,
    dnnl_OIhw16o16i = dnnl_ABcd16a16b,
    dnnl_Oihw16o = dnnl_Abcd16a,
    dnnl_OIhw4i8o4i = dnnl_ABcd4b8a4b,
    dnnl_OhwI4i8o4i = dnnl_AcdB4b8a4b,
    dnnl_OIhw4i16o4i = dnnl_ABcd4b16a4b,
    dnnl_OhwI4i16o4i = dnnl_AcdB4b16a4b,
    dnnl_OIhw4i24o4i = dnnl_ABcd4b24a4b,
    dnnl_OhwI4i24o4i = dnnl_AcdB4b24a4b,
    dnnl_OIhw4i32o4i = dnnl_ABcd4b32a4b,
    dnnl_OhwI4i32o4i = dnnl_AcdB4b32a4b,
    dnnl_OIhw4i64o4i = dnnl_ABcd4b64a4b,
    dnnl_OhwI4i64o4i = dnnl_AcdB4b64a4b,
    dnnl_OIhw16i16o4i = dnnl_ABcd16b16a4b,
    dnnl_OIhw16i16o2i = dnnl_ABcd16b16a2b,
    dnnl_OIhw16o16i2o = dnnl_ABcd16a16b2a,
    dnnl_OIhw4i4o = dnnl_ABcd4b4a,
    dnnl_OIhw4o4i = dnnl_ABcd4a4b,
    dnnl_Oihw4o = dnnl_Abcd4a,
    dnnl_OIhw8i8o2i = dnnl_ABcd8b8a2b,
    dnnl_OhwI8i8o2i = dnnl_AcdB8b8a2b,
    dnnl_OIhw8i16o2i = dnnl_ABcd8b16a2b,
    dnnl_OhwI8i16o2i = dnnl_AcdB8b16a2b,
    dnnl_OIhw8i32o2i = dnnl_ABcd8b32a2b,
    dnnl_OhwI8i32o2i = dnnl_AcdB8b32a2b,
    dnnl_OIhw8i24o2i = dnnl_ABcd8b24a2b,
    dnnl_OhwI8i24o2i = dnnl_AcdB8b24a2b,
    dnnl_OIhw8i64o2i = dnnl_ABcd8b64a2b,
    dnnl_OhwI8i64o2i = dnnl_AcdB8b64a2b,
    dnnl_OIhw8i8o = dnnl_ABcd8b8a,
    dnnl_OhwI8i8o = dnnl_AcdB8b8a,
    dnnl_OIhw8o16i2o = dnnl_ABcd8a16b2a,
    dnnl_OIhw2i8o4i = dnnl_ABcd2b8a4b,
    dnnl_IOhw8o16i2o = dnnl_BAcd8a16b2a,
    dnnl_OIhw8o8i = dnnl_ABcd8a8b,
    dnnl_OIhw8o4i = dnnl_ABcd8a4b,
    dnnl_Owhi16o = dnnl_Adcb16a,
    dnnl_OIhw8i32o = dnnl_ABcd8b32a,
    dnnl_OhwI8i32o = dnnl_AcdB8b32a,
    dnnl_OIhw8i24o = dnnl_ABcd8b24a,
    dnnl_OhwI8i24o = dnnl_AcdB8b24a,
    dnnl_OIhw8i16o = dnnl_ABcd8b16a,
    dnnl_OhwI8i16o = dnnl_AcdB8b16a,

    // weights, 5D
    dnnl_Odhwi16o = dnnl_Acdeb16a,
    dnnl_OdhwI16o2i = dnnl_AcdeB16a2b,
    dnnl_OdhwI16o4i = dnnl_AcdeB16a4b,
    dnnl_Idhwo8i = dnnl_Bcdea8b,
    dnnl_IdhwO8i2o = dnnl_BcdeA8b2a,
    dnnl_IdhwO8i4o = dnnl_BcdeA8b4a,
    dnnl_Idhwo16i = dnnl_Bcdea16b,
    dnnl_IdhwO16i2o = dnnl_BcdeA16b2a,
    dnnl_IdhwO16i4o = dnnl_BcdeA16b4a,
    dnnl_Idhwo24i = dnnl_Bcdea24b,
    dnnl_IdhwO24i2o = dnnl_BcdeA24b2a,
    dnnl_IdhwO24i4o = dnnl_BcdeA24b4a,
    dnnl_Odhwi4o = dnnl_Acdeb4a,
    dnnl_Odhwi8o = dnnl_Acdeb8a,
    dnnl_OdhwI8o2i = dnnl_AcdeB8a2b,
    dnnl_OdhwI8o4i = dnnl_AcdeB8a4b,
    dnnl_Odwhi16o = dnnl_Acedb16a,
    dnnl_OIdhw16i16o = dnnl_ABcde16b16a,
    dnnl_OdhwI16i16o = dnnl_AcdeB16b16a,
    dnnl_OIdhw16i32o = dnnl_ABcde16b32a,
    dnnl_OdhwI16i32o = dnnl_AcdeB16b32a,
    dnnl_OIdhw16i48o = dnnl_ABcde16b48a,
    dnnl_OdhwI16i48o = dnnl_AcdeB16b48a,
    dnnl_OIdhw16i64o = dnnl_ABcde16b64a,
    dnnl_OdhwI16i64o = dnnl_AcdeB16b64a,
    dnnl_OIdhw16o16i = dnnl_ABcde16a16b,
    dnnl_Oidhw16o = dnnl_Abcde16a,
    dnnl_OIdhw4i4o = dnnl_ABcde4b4a,
    dnnl_OIdhw4o4i = dnnl_ABcde4a4b,
    dnnl_Oidhw4o = dnnl_Abcde4a,
    dnnl_OIdhw8i8o2i = dnnl_ABcde8b8a2b,
    dnnl_OdhwI8i8o2i = dnnl_AcdeB8b8a2b,
    dnnl_OIdhw8i16o2i = dnnl_ABcde8b16a2b,
    dnnl_OdhwI8i16o2i = dnnl_AcdeB8b16a2b,
    dnnl_OIdhw8i32o2i = dnnl_ABcde8b32a2b,
    dnnl_OdhwI8i32o2i = dnnl_AcdeB8b32a2b,
    dnnl_OIdhw8i24o2i = dnnl_ABcde8b24a2b,
    dnnl_OdhwI8i24o2i = dnnl_AcdeB8b24a2b,
    dnnl_OIdhw8i64o2i = dnnl_ABcde8b64a2b,
    dnnl_OdhwI8i64o2i = dnnl_AcdeB8b64a2b,
    dnnl_OIdhw8i8o = dnnl_ABcde8b8a,
    dnnl_OdhwI8i8o = dnnl_AcdeB8b8a,
    dnnl_OIdhw8o16i2o = dnnl_ABcde8a16b2a,
    dnnl_IOdhw8o16i2o = dnnl_BAcde8a16b2a,
    dnnl_OIdhw4i8o4i = dnnl_ABcde4b8a4b,
    dnnl_OdhwI4i8o4i = dnnl_AcdeB4b8a4b,
    dnnl_OIdhw4i16o4i = dnnl_ABcde4b16a4b,
    dnnl_OdhwI4i16o4i = dnnl_AcdeB4b16a4b,
    dnnl_OIdhw4i24o4i = dnnl_ABcde4b24a4b,
    dnnl_OdhwI4i24o4i = dnnl_AcdeB4b24a4b,
    dnnl_OIdhw4i32o4i = dnnl_ABcde4b32a4b,
    dnnl_OdhwI4i32o4i = dnnl_AcdeB4b32a4b,
    dnnl_OIdhw4i64o4i = dnnl_ABcde4b64a4b,
    dnnl_OdhwI4i64o4i = dnnl_AcdeB4b64a4b,
    dnnl_OIdhw16i16o4i = dnnl_ABcde16b16a4b,
    dnnl_OIdhw16i16o2i = dnnl_ABcde16b16a2b,
    dnnl_OIdhw2i8o4i = dnnl_ABcde2b8a4b,
    dnnl_OIdhw8o8i = dnnl_ABcde8a8b,
    dnnl_OIdhw8o4i = dnnl_ABcde8a4b,
    dnnl_IOdhw16i16o = dnnl_BAcde16b16a,
    dnnl_OIdhw4o8i8o4i = dnnl_ABcde4a8b8a4b,
    dnnl_IOdhw16o16i = dnnl_BAcde16a16b,
    dnnl_OIdhw16o16i2o = dnnl_ABcde16a16b2a,
    dnnl_OIdhw8i32o = dnnl_ABcde8b32a,
    dnnl_OdhwI8i32o = dnnl_AcdeB8b32a,
    dnnl_OIdhw8i24o = dnnl_ABcde8b24a,
    dnnl_OdhwI8i24o = dnnl_AcdeB8b24a,
    dnnl_OIdhw8i16o = dnnl_ABcde8b16a,
    dnnl_OdhwI8i16o = dnnl_AcdeB8b16a,

    // weights w/ groups, 3D
    dnnl_Goiw16g = dnnl_Abcd16a,
    dnnl_Goiw8g = dnnl_Abcd8a,
    dnnl_Goiw4g = dnnl_Abcd4a,
    dnnl_gIOw16o16i = dnnl_aCBd16b16c,
    dnnl_gIOw16i16o = dnnl_aCBd16c16b,
    dnnl_gOIw16i16o = dnnl_aBCd16c16b,
    dnnl_gOIw16o16i = dnnl_aBCd16b16c,
    dnnl_gOiw16o = dnnl_aBcd16b,
    dnnl_gOIw4i16o4i = dnnl_aBCd4c16b4c,
    dnnl_gOIw2i8o4i = dnnl_aBCd2c8b4c,
    dnnl_gOIw16i16o4i = dnnl_aBCd16c16b4c,
    dnnl_gOIw16i16o2i = dnnl_aBCd16c16b2c,
    dnnl_gOIw16o16i2o = dnnl_aBCd16b16c2b,
    dnnl_gOIw4i4o = dnnl_aBCd4c4b,
    dnnl_gOIw4o4i = dnnl_aBCd4b4c,
    dnnl_gOiw4o = dnnl_aBcd4b,
    dnnl_gOIw8i16o2i = dnnl_aBCd8c16b2c,
    dnnl_gOIw8i8o = dnnl_aBCd8c8b,
    dnnl_gOIw8o16i2o = dnnl_aBCd8b16c2b,
    dnnl_gIOw8o16i2o = dnnl_aCBd8b16c2b,
    dnnl_gOIw8o8i = dnnl_aBCd8b8c,
    dnnl_gOIw8o4i = dnnl_aBCd8b4c,
    dnnl_gOwi16o = dnnl_aBdc16b,
    dnnl_gOwI16o2i = dnnl_aBdC16b2c,
    dnnl_gOwI16o4i = dnnl_aBdC16b4c,
    dnnl_gIwo8i = dnnl_aCdb8c,
    dnnl_gIwO8i2o = dnnl_aCdB8c2b,
    dnnl_gIwO8i4o = dnnl_aCdB8c4b,
    dnnl_gIwo16i = dnnl_aCdb16c,
    dnnl_gIwO16i2o = dnnl_aCdB16c2b,
    dnnl_gIwO16i4o = dnnl_aCdB16c4b,
    dnnl_gIwo24i = dnnl_aCdb24c,
    dnnl_gIwO24i2o = dnnl_aCdB24c2b,
    dnnl_gIwO24i4o = dnnl_aCdB24c4b,
    dnnl_gOwi4o = dnnl_aBdc4b,
    dnnl_gOwi8o = dnnl_aBdc8b,
    dnnl_gOwI8o2i = dnnl_aBdC8b2c,
    dnnl_gOwI8o4i = dnnl_aBdC8b4c,
    dnnl_Goiw32g = dnnl_Abcd32a,
    dnnl_gOIw2i4o2i = dnnl_aBCd2c4b2c,
    dnnl_gOIw2o4i2o = dnnl_aBCd2b4c2b,
    dnnl_gOIw4i8o2i = dnnl_aBCd4c8b2c,
    dnnl_gOIw4o8i2o = dnnl_aBCd4b8c2b,
    dnnl_goIw4i = dnnl_abCd4c,
    dnnl_goIw32i = dnnl_abCd32c,

    // weights w/ groups, 4D
    dnnl_gIOhw16i16o = dnnl_aCBde16c16b,
    dnnl_gIOhw16o16i = dnnl_aCBde16b16c,
    dnnl_gOhwi16o = dnnl_aBdec16b,
    dnnl_gOhwI16o2i = dnnl_aBdeC16b2c,
    dnnl_gOhwI16o4i = dnnl_aBdeC16b4c,
    dnnl_gIhwo8i = dnnl_aCdeb8c,
    dnnl_gIhwO8i2o = dnnl_aCdeB8c2b,
    dnnl_gIhwO8i4o = dnnl_aCdeB8c4b,
    dnnl_gIhwo16i = dnnl_aCdeb16c,
    dnnl_gIhwO16i2o = dnnl_aCdeB16c2b,
    dnnl_gIhwO16i4o = dnnl_aCdeB16c4b,
    dnnl_gIhwo24i = dnnl_aCdeb24c,
    dnnl_gIhwO24i2o = dnnl_aCdeB24c2b,
    dnnl_gIhwO24i4o = dnnl_aCdeB24c4b,
    dnnl_gOhwi32o = dnnl_aBdec32b,
    dnnl_gOhwi24o = dnnl_aBdec24b,
    dnnl_gOhwI24o2i = dnnl_aBdeC24b2c,
    dnnl_gOhwI24o4i = dnnl_aBdeC24b4c,
    dnnl_gOhwi4o = dnnl_aBdec4b,
    dnnl_gOhwi8o = dnnl_aBdec8b,
    dnnl_gOhwI8o2i = dnnl_aBdeC8b2c,
    dnnl_gOhwI8o4i = dnnl_aBdeC8b4c,
    dnnl_Goihw16g = dnnl_Abcde16a,
    dnnl_gOIhw16i16o = dnnl_aBCde16c16b,
    dnnl_gOIhw16o16i = dnnl_aBCde16b16c,
    dnnl_gOihw16o = dnnl_aBcde16b,
    dnnl_gOIhw2i8o4i = dnnl_aBCde2c8b4c,
    dnnl_gOIhw4i16o4i = dnnl_aBCde4c16b4c,
    dnnl_gOIhw16i16o4i = dnnl_aBCde16c16b4c,
    dnnl_gOIhw16i16o2i = dnnl_aBCde16c16b2c,
    dnnl_gOIhw16o16i2o = dnnl_aBCde16b16c2b,
    dnnl_gOIhw4i4o = dnnl_aBCde4c4b,
    dnnl_gOIhw4o4i = dnnl_aBCde4b4c,
    dnnl_gOihw4o = dnnl_aBcde4b,
    dnnl_Goihw8g = dnnl_Abcde8a,
    dnnl_Goihw4g = dnnl_Abcde4a,
    dnnl_gOIhw8i16o2i = dnnl_aBCde8c16b2c,
    dnnl_gOIhw8i8o = dnnl_aBCde8c8b,
    dnnl_gOIhw8o16i2o = dnnl_aBCde8b16c2b,
    dnnl_gIOhw8o16i2o = dnnl_aCBde8b16c2b,
    dnnl_gOIhw8o8i = dnnl_aBCde8b8c,
    dnnl_gOIhw8o4i = dnnl_aBCde8b4c,
    dnnl_Goihw32g = dnnl_Abcde32a,
    dnnl_gOwhi16o = dnnl_aBedc16b,
    dnnl_goIhw4i = dnnl_abCde4c,
    dnnl_goIhw32i = dnnl_abCde32c,

    dnnl_OIw4o8i8o4i = dnnl_ABc4a8b8a4b,
    dnnl_OIhw4o8i8o4i = dnnl_ABcd4a8b8a4b,
    dnnl_IOw4i8o8i4o = dnnl_BAc4b8a8b4a,
    dnnl_IOhw4i8o8i4o = dnnl_BAcd4b8a8b4a,
    dnnl_IOdhw4i8o8i4o = dnnl_BAcde4b8a8b4a,

    dnnl_OIhw2o8i8o2i = dnnl_ABcd2a8b8a2b,
    dnnl_gOIw4o8i8o4i = dnnl_aBCd4b8c8b4c,
    dnnl_gOIhw4o8i8o4i = dnnl_aBCde4b8c8b4c,
    dnnl_gOIdhw4o8i8o4i = dnnl_aBCdef4b8c8b4c,
    dnnl_gIOw4i8o8i4o = dnnl_aCBd4c8b8c4b,
    dnnl_gIOhw4i8o8i4o = dnnl_aCBde4c8b8c4b,
    dnnl_gIOdhw4i8o8i4o = dnnl_aCBdef4c8b8c4b,
    dnnl_gOIhw2o8i8o2i = dnnl_aBCde2b8c8b2c,
    dnnl_gOIhw2i4o2i = dnnl_aBCde2c4b2c,
    dnnl_gOIhw2o4i2o = dnnl_aBCde2b4c2b,
    dnnl_gOIhw4i8o2i = dnnl_aBCde4c8b2c,
    dnnl_gOIhw4o8i2o = dnnl_aBCde4b8c2b,

    // weights w/ groups, 6D
    dnnl_gIOdhw16i16o = dnnl_aCBdef16c16b,
    dnnl_gIOdhw16o16i = dnnl_aCBdef16b16c,
    dnnl_gOdhwi16o = dnnl_aBdefc16b,
    dnnl_gOdhwI16o2i = dnnl_aBdefC16b2c,
    dnnl_gOdhwI16o4i = dnnl_aBdefC16b4c,
    dnnl_gIdhwo8i = dnnl_aCdefb8c,
    dnnl_gIdhwO8i2o = dnnl_aCdefB8c2b,
    dnnl_gIdhwO8i4o = dnnl_aCdefB8c4b,
    dnnl_gIdhwo16i = dnnl_aCdefb16c,
    dnnl_gIdhwO16i2o = dnnl_aCdefB16c2b,
    dnnl_gIdhwO16i4o = dnnl_aCdefB16c4b,
    dnnl_gIdhwo24i = dnnl_aCdefb24c,
    dnnl_gIdhwO24i2o = dnnl_aCdefB24c2b,
    dnnl_gIdhwO24i4o = dnnl_aCdefB24c4b,
    dnnl_gOdhwi4o = dnnl_aBdefc4b,
    dnnl_gOdhwi8o = dnnl_aBdefc8b,
    dnnl_gOdhwI8o2i = dnnl_aBdefC8b2c,
    dnnl_gOdhwI8o4i = dnnl_aBdefC8b4c,
    dnnl_gOdwhi16o = dnnl_aBdfec16b,
    dnnl_gOIdhw16i16o = dnnl_aBCdef16c16b,
    dnnl_gOIdhw4i16o4i = dnnl_aBCdef4c16b4c,
    dnnl_gOIdhw16i16o4i = dnnl_aBCdef16c16b4c,
    dnnl_gOIdhw2i8o4i = dnnl_aBCdef2c8b4c,
    dnnl_gOIdhw16i16o2i = dnnl_aBCdef16c16b2c,
    dnnl_gOIdhw16o16i = dnnl_aBCdef16b16c,
    dnnl_gOIdhw16o16i2o = dnnl_aBCdef16b16c2b,
    dnnl_gOidhw16o = dnnl_aBcdef16b,
    dnnl_gOIdhw4i4o = dnnl_aBCdef4c4b,
    dnnl_gOIdhw4o4i = dnnl_aBCdef4b4c,
    dnnl_gOidhw4o = dnnl_aBcdef4b,
    dnnl_gOIdhw8i16o2i = dnnl_aBCdef8c16b2c,
    dnnl_gOIdhw8i8o = dnnl_aBCdef8c8b,
    dnnl_gOIdhw8o16i2o = dnnl_aBCdef8b16c2b,
    dnnl_gIOdhw8o16i2o = dnnl_aCBdef8b16c2b,
    dnnl_gOIdhw8o8i = dnnl_aBCdef8b8c,
    dnnl_gOIdhw8o4i = dnnl_aBCdef8b4c,
    dnnl_Goidhw16g = dnnl_Abcdef16a,
    dnnl_Goidhw32g = dnnl_Abcdef32a,
    dnnl_gOIdhw2i4o2i = dnnl_aBCdef2c4b2c,
    dnnl_gOIdhw4i8o2i = dnnl_aBCdef4c8b2c,
    dnnl_gOIdhw2o4i2o = dnnl_aBCdef2b4c2b,
    dnnl_gOIdhw4o8i2o = dnnl_aBCdef4b8c2b,
    dnnl_goIdhw4i = dnnl_abCdef4c,
    dnnl_goIdhw32i = dnnl_abCdef32c,

    // weights, 3D
    dnnl_Owi24o = dnnl_Acb24a,
    dnnl_OwI24o2i = dnnl_AcB24a2b,
    dnnl_OwI24o4i = dnnl_AcB24a4b,
    dnnl_Owi32o = dnnl_Acb32a,
    dnnl_OwI32o2i = dnnl_AcB32a2b,
    dnnl_OwI32o4i = dnnl_AcB32a4b,
    dnnl_Owi48o = dnnl_Acb48a,
    dnnl_OwI48o2i = dnnl_AcB48a2b,
    dnnl_OwI48o4i = dnnl_AcB48a4b,
    dnnl_Owi64o = dnnl_Acb64a,
    dnnl_OwI64o2i = dnnl_AcB64a2b,
    dnnl_OwI64o4i = dnnl_AcB64a4b,
    dnnl_Iwo32i = dnnl_Bca32b,
    dnnl_IwO32i2o = dnnl_BcA32b2a,
    dnnl_IwO32i4o = dnnl_BcA32b4a,
    dnnl_Iwo48i = dnnl_Bca48b,
    dnnl_IwO48i2o = dnnl_BcA48b2a,
    dnnl_IwO48i4o = dnnl_BcA48b4a,
    dnnl_Iwo64i = dnnl_Bca64b,
    dnnl_IwO64i2o = dnnl_BcA64b2a,
    dnnl_IwO64i4o = dnnl_BcA64b4a,
    dnnl_wIo2i = dnnl_cBa2b,
    dnnl_wIo4i = dnnl_cBa4b,
    dnnl_gOwi24o = dnnl_aBdc24b,
    dnnl_gOwI24o2i = dnnl_aBdC24b2c,
    dnnl_gOwI24o4i = dnnl_aBdC24b4c,
    dnnl_gOwi32o = dnnl_aBdc32b,
    dnnl_gOwI32o2i = dnnl_aBdC32b2c,
    dnnl_gOwI32o4i = dnnl_aBdC32b4c,
    dnnl_gOwi48o = dnnl_aBdc48b,
    dnnl_gOwI48o2i = dnnl_aBdC48b2c,
    dnnl_gOwI48o4i = dnnl_aBdC48b4c,
    dnnl_gOwi64o = dnnl_aBdc64b,
    dnnl_gOwI64o2i = dnnl_aBdC64b2c,
    dnnl_gOwI64o4i = dnnl_aBdC64b4c,
    dnnl_gIwo32i = dnnl_aCdb32c,
    dnnl_gIwO32i2o = dnnl_aCdB32c2b,
    dnnl_gIwO32i4o = dnnl_aCdB32c4b,
    dnnl_gIwo48i = dnnl_aCdb48c,
    dnnl_gIwO48i2o = dnnl_aCdB48c2b,
    dnnl_gIwO48i4o = dnnl_aCdB48c4b,
    dnnl_gIwo64i = dnnl_aCdb64c,
    dnnl_gIwO64i2o = dnnl_aCdB64c2b,
    dnnl_gIwO64i4o = dnnl_aCdB64c4b,
    dnnl_gwio = dnnl_adcb,
    dnnl_gwIo2i = dnnl_adCb2c,
    dnnl_gwIo4i = dnnl_adCb4c,
    // weights, 4D
    dnnl_OhwI24o = dnnl_Acdb24a,
    dnnl_OhwI24o2i = dnnl_AcdB24a2b,
    dnnl_OhwI24o4i = dnnl_AcdB24a4b,
    dnnl_OhwI32o = dnnl_Acdb32a,
    dnnl_OhwI32o2i = dnnl_AcdB32a2b,
    dnnl_OhwI32o4i = dnnl_AcdB32a4b,
    dnnl_Ohwi48o = dnnl_Acdb48a,
    dnnl_OhwI48o2i = dnnl_AcdB48a2b,
    dnnl_OhwI48o4i = dnnl_AcdB48a4b,
    dnnl_Ohwi64o = dnnl_Acdb64a,
    dnnl_OhwI64o2i = dnnl_AcdB64a2b,
    dnnl_OhwI64o4i = dnnl_AcdB64a4b,
    dnnl_Ihwo32i = dnnl_Bcda32b,
    dnnl_IhwO32i2o = dnnl_BcdA32b2a,
    dnnl_IhwO32i4o = dnnl_BcdA32b4a,
    dnnl_Ihwo48i = dnnl_Bcda48b,
    dnnl_IhwO48i2o = dnnl_BcdA48b2a,
    dnnl_IhwO48i4o = dnnl_BcdA48b4a,
    dnnl_Ihwo64i = dnnl_Bcda64b,
    dnnl_IhwO64i2o = dnnl_BcdA64b2a,
    dnnl_IhwO64i4o = dnnl_BcdA64b4a,
    dnnl_hwIo2i = dnnl_cdBa2b,
    dnnl_hwIo4i = dnnl_cdBa4b,
    dnnl_gOhwI24o = dnnl_aBdec24b,
    dnnl_gOhwI32o = dnnl_aBdec32b,
    dnnl_gOhwI32o2i = dnnl_aBdeC32b2c,
    dnnl_gOhwI32o4i = dnnl_aBdeC32b4c,
    dnnl_gOhwi48o = dnnl_aBdec48b,
    dnnl_gOhwI48o2i = dnnl_aBdeC48b2c,
    dnnl_gOhwI48o4i = dnnl_aBdeC48b4c,
    dnnl_gOhwi64o = dnnl_aBdec64b,
    dnnl_gOhwI64o2i = dnnl_aBdeC64b2c,
    dnnl_gOhwI64o4i = dnnl_aBdeC64b4c,
    dnnl_gIhwo32i = dnnl_aCdeb32c,
    dnnl_gIhwO32i2o = dnnl_aCdeB32c2b,
    dnnl_gIhwO32i4o = dnnl_aCdeB32c4b,
    dnnl_gIhwo48i = dnnl_aCdeb48c,
    dnnl_gIhwO48i2o = dnnl_aCdeB48c2b,
    dnnl_gIhwO48i4o = dnnl_aCdeB48c4b,
    dnnl_gIhwo64i = dnnl_aCdeb64c,
    dnnl_gIhwO64i2o = dnnl_aCdeB64c2b,
    dnnl_gIhwO64i4o = dnnl_aCdeB64c4b,
    dnnl_ghwio = dnnl_adecb,
    dnnl_ghwIo2i = dnnl_adeCb2c,
    dnnl_ghwIo4i = dnnl_adeCb4c,
    // weights, 5D
    dnnl_Odhwi24o = dnnl_Acdeb24a,
    dnnl_OdhwI24o2i = dnnl_AcdeB24a2b,
    dnnl_OdhwI24o4i = dnnl_AcdeB24a4b,
    dnnl_Odhwi32o = dnnl_Acdeb32a,
    dnnl_OdhwI32o2i = dnnl_AcdeB32a2b,
    dnnl_OdhwI32o4i = dnnl_AcdeB32a4b,
    dnnl_Odhwi48o = dnnl_Acdeb48a,
    dnnl_OdhwI48o2i = dnnl_AcdeB48a2b,
    dnnl_OdhwI48o4i = dnnl_AcdeB48a4b,
    dnnl_Odhwi64o = dnnl_Acdeb64a,
    dnnl_OdhwI64o2i = dnnl_AcdeB64a2b,
    dnnl_OdhwI64o4i = dnnl_AcdeB64a4b,
    dnnl_Idhwo32i = dnnl_Bcdea32b,
    dnnl_IdhwO32i2o = dnnl_BcdeA32b2a,
    dnnl_IdhwO32i4o = dnnl_BcdeA32b4a,
    dnnl_Idhwo48i = dnnl_Bcdea48b,
    dnnl_IdhwO48i2o = dnnl_BcdeA48b2a,
    dnnl_IdhwO48i4o = dnnl_BcdeA48b4a,
    dnnl_Idhwo64i = dnnl_Bcdea64b,
    dnnl_IdhwO64i2o = dnnl_BcdeA64b2a,
    dnnl_IdhwO64i4o = dnnl_BcdeA64b4a,
    dnnl_dhwIo2i = dnnl_cdeBa2b,
    dnnl_dhwIo4i = dnnl_cdeBa4b,
    dnnl_gOdhwi24o = dnnl_aBdefc24b,
    dnnl_gOdhwI24o2i = dnnl_aBdefC24b2c,
    dnnl_gOdhwI24o4i = dnnl_aBdefC24b4c,
    dnnl_gOdhwi32o = dnnl_aBdefc32b,
    dnnl_gOdhwI32o2i = dnnl_aBdefC32b2c,
    dnnl_gOdhwI32o4i = dnnl_aBdefC32b4c,
    dnnl_gOdhwi48o = dnnl_aBdefc48b,
    dnnl_gOdhwI48o2i = dnnl_aBdefC48b2c,
    dnnl_gOdhwI48o4i = dnnl_aBdefC48b4c,
    dnnl_gOdhwi64o = dnnl_aBdefc64b,
    dnnl_gOdhwI64o2i = dnnl_aBdefC64b2c,
    dnnl_gOdhwI64o4i = dnnl_aBdefC64b4c,
    dnnl_gIdhwo32i = dnnl_aCdefb32c,
    dnnl_gIdhwO32i2o = dnnl_aCdefB32c2b,
    dnnl_gIdhwO32i4o = dnnl_aCdefB32c4b,
    dnnl_gIdhwo48i = dnnl_aCdefb48c,
    dnnl_gIdhwO48i2o = dnnl_aCdefB48c2b,
    dnnl_gIdhwO48i4o = dnnl_aCdefB48c4b,
    dnnl_gIdhwo64i = dnnl_aCdefb64c,
    dnnl_gIdhwO64i2o = dnnl_aCdefB64c2b,
    dnnl_gIdhwO64i4o = dnnl_aCdefB64c4b,
    dnnl_gdhwio = dnnl_adefcb,
    dnnl_gdhwIo2i = dnnl_adefCb2c,
    dnnl_gdhwIo4i = dnnl_adefCb4c,
    dnnl_OI16i32o4i = dnnl_AB16b32a4b,
    dnnl_OI16i48o4i = dnnl_AB16b48a4b,
    dnnl_OI16i64o4i = dnnl_AB16b64a4b,
    dnnl_OI16i16o2i = dnnl_AB16b16a2b,
    dnnl_OI16i32o2i = dnnl_AB16b32a2b,
    dnnl_OI16i48o2i = dnnl_AB16b48a2b,
    dnnl_OI16i64o2i = dnnl_AB16b64a2b,
    dnnl_OIw16i32o4i = dnnl_ABc16b32a4b,
    dnnl_OIw16i48o4i = dnnl_ABc16b48a4b,
    dnnl_OIw16i64o4i = dnnl_ABc16b64a4b,
    dnnl_OIw16i32o2i = dnnl_ABc16b32a2b,
    dnnl_OIw16i48o2i = dnnl_ABc16b48a2b,
    dnnl_OIw16i64o2i = dnnl_ABc16b64a2b,
    dnnl_OIhw16i32o4i = dnnl_ABcd16b32a4b,
    dnnl_OIhw16i48o4i = dnnl_ABcd16b48a4b,
    dnnl_OIhw16i64o4i = dnnl_ABcd16b64a4b,
    dnnl_OIhw16i32o2i = dnnl_ABcd16b32a2b,
    dnnl_OIhw16i48o2i = dnnl_ABcd16b48a2b,
    dnnl_OIhw16i64o2i = dnnl_ABcd16b64a2b,
    dnnl_OIdhw16i32o4i = dnnl_ABcde16b32a4b,
    dnnl_OIdhw16i48o4i = dnnl_ABcde16b48a4b,
    dnnl_OIdhw16i64o4i = dnnl_ABcde16b64a4b,
    dnnl_OIdhw16i32o2i = dnnl_ABcde16b32a2b,
    dnnl_OIdhw16i48o2i = dnnl_ABcde16b48a2b,
    dnnl_OIdhw16i64o2i = dnnl_ABcde16b64a2b,
    dnnl_OwI16i16o2i = dnnl_AcB16b16a2b,
    dnnl_OwI16i16o4i = dnnl_AcB16b16a4b,
    dnnl_OhwI16i16o2i = dnnl_AcdB16b16a2b,
    dnnl_OhwI16i16o4i = dnnl_AcdB16b16a4b,
    dnnl_OdhwI16i16o2i = dnnl_AcdeB16b16a2b,
    dnnl_OdhwI16i16o4i = dnnl_AcdeB16b16a4b,
    dnnl_IwO16o16i2o = dnnl_BcA16a16b2a,
    dnnl_IwO16o16i4o = dnnl_BcA16a16b4a,
    dnnl_IhwO16o16i2o = dnnl_BcdA16a16b2a,
    dnnl_IhwO16o16i4o = dnnl_BcdA16a16b4a,
    dnnl_IdhwO16o16i2o = dnnl_BcdeA16a16b2a,
    dnnl_IdhwO16o16i4o = dnnl_BcdeA16a16b4a,
    dnnl_gOwI16i16o2i = dnnl_aBdC16c16b2c,
    dnnl_gOwI16i16o4i = dnnl_aBdC16c16b4c,
    dnnl_gOhwI16i16o2i = dnnl_aBdeC16c16b2c,
    dnnl_gOhwI16i16o4i = dnnl_aBdeC16c16b4c,
    dnnl_gOdhwI16i16o2i = dnnl_aBdefC16c16b2c,
    dnnl_gOdhwI16i16o4i = dnnl_aBdefC16c16b4c,
    dnnl_gIwO16o16i2o = dnnl_aCdB16b16c2b,
    dnnl_gIwO16o16i4o = dnnl_aCdB16b16c4b,
    dnnl_gIhwO16o16i2o = dnnl_aCdeB16b16c2b,
    dnnl_gIhwO16o16i4o = dnnl_aCdeB16b16c4b,
    dnnl_gIdhwO16o16i2o = dnnl_aCdefB16b16c2b,
    dnnl_gIdhwO16o16i4o = dnnl_aCdefB16b16c4b,
    dnnl_OwI16i32o2i = dnnl_AcB16b32a2b,
    dnnl_OwI16i32o4i = dnnl_AcB16b32a4b,
    dnnl_OwI16i48o2i = dnnl_AcB16b48a2b,
    dnnl_OwI16i48o4i = dnnl_AcB16b48a4b,
    dnnl_OwI16i64o2i = dnnl_AcB16b64a2b,
    dnnl_OwI16i64o4i = dnnl_AcB16b64a4b,
    dnnl_IwO16o32i2o = dnnl_BcA16a32b2a,
    dnnl_IwO16o32i4o = dnnl_BcA16a32b4a,
    dnnl_IwO16o48i2o = dnnl_BcA16a48b2a,
    dnnl_IwO16o48i4o = dnnl_BcA16a48b4a,
    dnnl_IwO16o64i2o = dnnl_BcA16a64b2a,
    dnnl_IwO16o64i4o = dnnl_BcA16a64b4a,
    dnnl_gOwI16i32o2i = dnnl_aBdC16c32b2c,
    dnnl_gOwI16i32o4i = dnnl_aBdC16c32b4c,
    dnnl_gOwI16i48o2i = dnnl_aBdC16c48b2c,
    dnnl_gOwI16i48o4i = dnnl_aBdC16c48b4c,
    dnnl_gOwI16i64o2i = dnnl_aBdC16c64b2c,
    dnnl_gOwI16i64o4i = dnnl_aBdC16c64b4c,
    dnnl_gIwO16o32i2o = dnnl_aCdB16b32c2b,
    dnnl_gIwO16o32i4o = dnnl_aCdB16b32c4b,
    dnnl_gIwO16o48i2o = dnnl_aCdB16b48c2b,
    dnnl_gIwO16o48i4o = dnnl_aCdB16b48c4b,
    dnnl_gIwO16o64i2o = dnnl_aCdB16b64c2b,
    dnnl_gIwO16o64i4o = dnnl_aCdB16b64c4b,
    dnnl_OhwI16i32o2i = dnnl_AcdB16b32a2b,
    dnnl_OhwI16i32o4i = dnnl_AcdB16b32a4b,
    dnnl_OhwI16i48o2i = dnnl_AcdB16b48a2b,
    dnnl_OhwI16i48o4i = dnnl_AcdB16b48a4b,
    dnnl_OhwI16i64o2i = dnnl_AcdB16b64a2b,
    dnnl_OhwI16i64o4i = dnnl_AcdB16b64a4b,
    dnnl_IhwO16o32i2o = dnnl_BcdA16a32b2a,
    dnnl_IhwO16o32i4o = dnnl_BcdA16a32b4a,
    dnnl_IhwO16o48i2o = dnnl_BcdA16a48b2a,
    dnnl_IhwO16o48i4o = dnnl_BcdA16a48b4a,
    dnnl_IhwO16o64i2o = dnnl_BcdA16a64b2a,
    dnnl_IhwO16o64i4o = dnnl_BcdA16a64b4a,
    dnnl_gOhwI16i32o2i = dnnl_aBdeC16c32b2c,
    dnnl_gOhwI16i32o4i = dnnl_aBdeC16c32b4c,
    dnnl_gOhwI16i48o2i = dnnl_aBdeC16c48b2c,
    dnnl_gOhwI16i48o4i = dnnl_aBdeC16c48b4c,
    dnnl_gOhwI16i64o2i = dnnl_aBdeC16c64b2c,
    dnnl_gOhwI16i64o4i = dnnl_aBdeC16c64b4c,
    dnnl_gIhwO16o32i2o = dnnl_aCdeB16b32c2b,
    dnnl_gIhwO16o32i4o = dnnl_aCdeB16b32c4b,
    dnnl_gIhwO16o48i2o = dnnl_aCdeB16b48c2b,
    dnnl_gIhwO16o48i4o = dnnl_aCdeB16b48c4b,
    dnnl_gIhwO16o64i2o = dnnl_aCdeB16b64c2b,
    dnnl_gIhwO16o64i4o = dnnl_aCdeB16b64c4b,
    dnnl_OdhwI16i32o2i = dnnl_AcdeB16b32a2b,
    dnnl_OdhwI16i32o4i = dnnl_AcdeB16b32a4b,
    dnnl_OdhwI16i48o2i = dnnl_AcdeB16b48a2b,
    dnnl_OdhwI16i48o4i = dnnl_AcdeB16b48a4b,
    dnnl_OdhwI16i64o2i = dnnl_AcdeB16b64a2b,
    dnnl_OdhwI16i64o4i = dnnl_AcdeB16b64a4b,
    dnnl_IdhwO16o32i2o = dnnl_BcdeA16a32b2a,
    dnnl_IdhwO16o32i4o = dnnl_BcdeA16a32b4a,
    dnnl_IdhwO16o48i2o = dnnl_BcdeA16a48b2a,
    dnnl_IdhwO16o48i4o = dnnl_BcdeA16a48b4a,
    dnnl_IdhwO16o64i2o = dnnl_BcdeA16a64b2a,
    dnnl_IdhwO16o64i4o = dnnl_BcdeA16a64b4a,
    dnnl_gOdhwI16i32o2i = dnnl_aBdefC16c32b2c,
    dnnl_gOdhwI16i32o4i = dnnl_aBdefC16c32b4c,
    dnnl_gOdhwI16i48o2i = dnnl_aBdefC16c48b2c,
    dnnl_gOdhwI16i48o4i = dnnl_aBdefC16c48b4c,
    dnnl_gOdhwI16i64o2i = dnnl_aBdefC16c64b2c,
    dnnl_gOdhwI16i64o4i = dnnl_aBdefC16c64b4c,
    dnnl_gIdhwO16o32i2o = dnnl_aCdefB16b32c2b,
    dnnl_gIdhwO16o32i4o = dnnl_aCdefB16b32c4b,
    dnnl_gIdhwO16o48i2o = dnnl_aCdefB16b48c2b,
    dnnl_gIdhwO16o48i4o = dnnl_aCdefB16b48c4b,
    dnnl_gIdhwO16o64i2o = dnnl_aCdefB16b64c2b,
    dnnl_gIdhwO16o64i4o = dnnl_aCdefB16b64c4b,
    dnnl_hwioG16g = dnnl_decbA16a,
    dnnl_hwioG8g = dnnl_decbA8a,
    dnnl_dhwioG16g = dnnl_defcbA16a,
    dnnl_dhwioG8g = dnnl_defcbA8a,
    dnnl_NCdhw40n16c = dnnl_ABcde40a16b,
    dnnl_NCw40n16c = dnnl_ABc40a16b,
    dnnl_NChw40n16c = dnnl_ABcd40a16b,
    dnnl_NCw40n32c = dnnl_ABc40a32b,
    dnnl_NChw40n32c = dnnl_ABcd40a32b,
    dnnl_NCdhw40n32c = dnnl_ABcde40a32b,
    dnnl_OIdhw4o8i8o2i = dnnl_ABcde4a8b8a2b,
    dnnl_OIhw4o8i8o2i = dnnl_ABcd4a8b8a2b,
    dnnl_OIw4o8i8o2i = dnnl_ABc4a8b8a2b,
    dnnl_gOIdhw4o8i8o2i = dnnl_aBCdef4b8c8b2c,
    dnnl_gOIhw4o8i8o2i = dnnl_aBCde4b8c8b2c,
    dnnl_gOIw4o8i8o2i = dnnl_aBCd4b8c8b2c,
    dnnl_IOdhw4i8o8i2o = dnnl_BAcde4b8a8b2a,
    dnnl_IOhw4i8o8i2o = dnnl_BAcd4b8a8b2a,
    dnnl_IOw4i8o8i2o = dnnl_BAc4b8a8b2a,
    dnnl_gIOdhw4i8o8i2o = dnnl_aCBdef4c8b8c2b,
    dnnl_gIOhw4i8o8i2o = dnnl_aCBde4c8b8c2b,
    dnnl_gIOw4i8o8i2o = dnnl_aCBd4c8b8c2b,
    dnnl_NCw2c32n8c = dnnl_ABc2b32a8b,
    dnnl_NChw2c32n8c = dnnl_ABcd2b32a8b,
    dnnl_NCdhw2c32n8c = dnnl_ABcde2b32a8b,
    dnnl_OIw2i8o16i4o = dnnl_ABc2b8a16b4a,
    dnnl_OIhw2i8o16i4o = dnnl_ABcd2b8a16b4a,
    dnnl_OIdhw2i8o16i4o = dnnl_ABcde2b8a16b4a,
    dnnl_OIw2o8i16o4i = dnnl_ABc2a8b16a4b,
    dnnl_OIw2o8i16o2i = dnnl_ABc2a8b16a2b,
    dnnl_IOw2i8o16i4o = dnnl_BAc2b8a16b4a,
    dnnl_IOw2i8o16i2o = dnnl_BAc2b8a16b2a,
    dnnl_OIhw2o8i16o4i = dnnl_ABcd2a8b16a4b,
    dnnl_OIhw2o8i16o2i = dnnl_ABcd2a8b16a2b,
    dnnl_IOhw2i8o16i4o = dnnl_BAcd2b8a16b4a,
    dnnl_IOhw2i8o16i2o = dnnl_BAcd2b8a16b2a,
    dnnl_OIdhw2o8i16o4i = dnnl_ABcde2a8b16a4b,
    dnnl_OIdhw2o8i16o2i = dnnl_ABcde2a8b16a2b,
    dnnl_IOdhw2i8o16i4o = dnnl_BAcde2b8a16b4a,
    dnnl_IOdhw2i8o16i2o = dnnl_BAcde2b8a16b2a,
    dnnl_gOIw2o8i16o2i = dnnl_aBCd2b8c16b2c,
    dnnl_gIOw2i8o16i2o = dnnl_aCBd2c8b16c2b,
    dnnl_gIOhw2i8o16i2o = dnnl_aBCde2c8b16c2b,
    dnnl_gIOdhw2i8o16i2o = dnnl_aBCdef2c8b16c2b,
    dnnl_gOIhw2o8i16o2i = dnnl_aBCde2b8c16b2c,
    dnnl_gOIdhw2o8i16o2i = dnnl_aBCdef2b8c16b2c,
    dnnl_gOIw2o8i16o4i = dnnl_aBCd2b8c16b4c,
    dnnl_gOIhw2o8i16o4i = dnnl_aBCde2b8c16b4c,
} dnnl_format_tag_t;

/// @} dnnl_api_memory

/// @addtogroup dnnl_api_primitives
/// @{
/// @addtogroup dnnl_api_primitives_common
/// @{

/// Kinds of propagation.
typedef enum {
    // TODO: suggest renames
    /// Undefined propagation type.
    dnnl_prop_kind_undef = 0,
    /// Forward data propagation (training mode). In this mode primitives
    /// perform computations necessary for subsequent backward propagation.
    dnnl_forward_training = 64,
    /// Forward data propagation (inference mode). In this mode primitives
    /// perform only computations that are necessary for inference and omit
    /// computations that are necessary only for backward propagation.
    dnnl_forward_inference = 96,
    /// Forward data propagation (alias for @c dnnl_forward_training).
    dnnl_forward = dnnl_forward_training,
    /// Backward propagation (with respect to all parameters).
    dnnl_backward = 128,
    /// Backward data propagation.
    dnnl_backward_data = 160,
    /// Backward weights propagation.
    dnnl_backward_weights = 192,
    /// Backward bias propagation.
    dnnl_backward_bias = 193,
} dnnl_prop_kind_t;

/// Kinds of primitives. Used to implement a way to extend the library with new
/// primitives without changing the ABI.
typedef enum {
    /// Undefined primitive
    dnnl_undefined_primitive,
    /// A reorder primitive.
    dnnl_reorder,
    /// A shuffle primitive.
    dnnl_shuffle,
    /// A (out-of-place) concat primitive.
    dnnl_concat,
    /// A sum primitive.
    dnnl_sum,
    /// A convolution primitive.
    dnnl_convolution,
    /// A deconvolution primitive.
    dnnl_deconvolution,
    /// An element-wise primitive.
    dnnl_eltwise,
    /// An LRN primitive.
    dnnl_lrn,
    /// A batch normalization primitive.
    dnnl_batch_normalization,
    /// An inner product primitive.
    dnnl_inner_product,
    /// A rnn primitive.
    dnnl_rnn,
    /// A matrix multiplication primitive (internal).
    dnnl_gemm,
    /// A binary primitive.
    dnnl_binary,
    /// A matrix multiplication primitive.
    dnnl_matmul,
    /// A resampling primitive.
    dnnl_resampling,
    /// A pooling primitive.
    dnnl_pooling,
    /// A reduction primitive.
    dnnl_reduction,
    /// A PReLU primitive.
    dnnl_prelu,
    /// A softmax primitive.
    dnnl_softmax,
    /// A layer normalization primitive.
    dnnl_layer_normalization,
    /// A group normalization primitive.
    dnnl_group_normalization,

    /// Parameter to allow internal only primitives without undefined behavior.
    /// This parameter is chosen to be valid for so long as sizeof(int) >= 2.
    dnnl_primitive_kind_max = 0x7fff,
} dnnl_primitive_kind_t;

/// Kinds of algorithms.
typedef enum {
    dnnl_alg_kind_undef,
    /// Direct convolution
    dnnl_convolution_direct = 0x1,
    /// Winograd convolution
    dnnl_convolution_winograd = 0x2,
    /// Convolution algorithm(either direct or Winograd) is chosen just in time
    dnnl_convolution_auto = 0x3,
    /// Direct deconvolution
    dnnl_deconvolution_direct = 0xa,
    /// Winograd deconvolution
    dnnl_deconvolution_winograd = 0xb,
    /// Eltwise: ReLU
    dnnl_eltwise_relu = 0x20,
    /// Eltwise: hyperbolic tangent non-linearity (tanh)
    dnnl_eltwise_tanh,
    /// Eltwise: exponential linear unit (elu)
    dnnl_eltwise_elu,
    /// Eltwise: square
    dnnl_eltwise_square,
    /// Eltwise: abs
    dnnl_eltwise_abs,
    /// Eltwise: square root
    dnnl_eltwise_sqrt,
    /// Eltwise: linear
    dnnl_eltwise_linear,
    /// Eltwise: soft_relu
    dnnl_eltwise_soft_relu,
    /// Eltwise: hardsigmoid
    dnnl_eltwise_hardsigmoid,
    /// Eltwise: logistic
    dnnl_eltwise_logistic,
    /// Eltwise: exponent
    dnnl_eltwise_exp,
    /// Eltwise: gelu
    ///
    /// @note Tanh approximation formula is used to approximate
    /// the cumulative distribution function of a Gaussian here
    dnnl_eltwise_gelu_tanh,
    /// Eltwise: swish
    dnnl_eltwise_swish,
    /// Eltwise: natural logarithm
    dnnl_eltwise_log,
    /// Eltwise: clip
    dnnl_eltwise_clip,
    /// Eltwise: clip version 2
    dnnl_eltwise_clip_v2,
    /// Eltwise: pow
    dnnl_eltwise_pow,
    /// Eltwise: erf-based gelu
    dnnl_eltwise_gelu_erf,
    /// Eltwise: round
    dnnl_eltwise_round,
    /// Eltwise: mish
    dnnl_eltwise_mish,
    /// Eltwise: hardswish
    dnnl_eltwise_hardswish,
    /// Eltwise: ReLU (dst for backward)
    dnnl_eltwise_relu_use_dst_for_bwd = 0x100,
    /// Eltwise: hyperbolic tangent non-linearity (tanh) (dst for backward)
    dnnl_eltwise_tanh_use_dst_for_bwd,
    /// Eltwise: exponential linear unit (elu) (dst for backward)
    dnnl_eltwise_elu_use_dst_for_bwd,
    /// Eltwise: square root (dst for backward)
    dnnl_eltwise_sqrt_use_dst_for_bwd,
    /// Eltwise: logistic (dst for backward)
    dnnl_eltwise_logistic_use_dst_for_bwd,
    /// Eltwise: exp (dst for backward)
    dnnl_eltwise_exp_use_dst_for_bwd,
    /// Eltwise: clip version 2 (dst for backward)
    dnnl_eltwise_clip_v2_use_dst_for_bwd,
    /// Max pooling
    dnnl_pooling_max = 0x1ff,
    /// Average pooling include padding
    dnnl_pooling_avg_include_padding = 0x2ff,
    /// Average pooling exclude padding
    dnnl_pooling_avg_exclude_padding = 0x3ff,
    /// Local response normalization (LRN) across multiple channels
    dnnl_lrn_across_channels = 0xaff,
    /// LRN within a single channel
    dnnl_lrn_within_channel = 0xbff,
    /// RNN cell
    dnnl_vanilla_rnn = 0x1fff,
    /// LSTM cell
    dnnl_vanilla_lstm = 0x2fff,
    /// GRU cell
    dnnl_vanilla_gru = 0x3fff,
    /// GRU cell with linear before reset
    ///
    /// Modification of original GRU cell. Differs from #dnnl_vanilla_gru
    /// in how the new memory gate is calculated:
    /// \f[ c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h})) \f]
    /// Primitive expects 4 biases on input:
    /// \f$[b_{u}, b_{r}, b_{c_x}, b_{c_h}]\f$
    dnnl_lbr_gru = 0x4fff,
    /// AUGRU cell
    dnnl_vanilla_augru = 0x5fff,
    /// AUGRU cell with linear before reset
    dnnl_lbr_augru = 0x6fff,
    /// Binary add
    dnnl_binary_add = 0x1fff0,
    /// Binary mul
    dnnl_binary_mul = 0x1fff1,
    /// Binary max
    dnnl_binary_max = 0x1fff2,
    /// Binary min
    dnnl_binary_min = 0x1fff3,
    /// Binary div
    dnnl_binary_div = 0x1fff4,
    /// Binary sub
    dnnl_binary_sub = 0x1fff5,
    /// Binary greater or equal
    dnnl_binary_ge = 0x1fff6,
    /// Binary greater than
    dnnl_binary_gt = 0x1fff7,
    /// Binary less or equal
    dnnl_binary_le = 0x1fff8,
    /// Binary less than
    dnnl_binary_lt = 0x1fff9,
    /// Binary equal
    dnnl_binary_eq = 0x1fffa,
    /// Binary not equal
    dnnl_binary_ne = 0x1fffb,
    /// Nearest Neighbor Resampling Method
    dnnl_resampling_nearest = 0x2fff0,
    /// Linear Resampling Method
    dnnl_resampling_linear = 0x2fff1,
    /// Reduction using max
    dnnl_reduction_max,
    /// Reduction using min
    dnnl_reduction_min,
    /// Reduction using sum
    dnnl_reduction_sum,
    /// Reduction using mul
    dnnl_reduction_mul,
    /// Reduction using mean
    dnnl_reduction_mean,
    /// Reduction using lp norm
    dnnl_reduction_norm_lp_max,
    /// Reduction using lp norm
    dnnl_reduction_norm_lp_sum,
    /// Reduction using lp norm without final pth-root
    dnnl_reduction_norm_lp_power_p_max,
    /// Reduction using lp norm without final pth-root
    dnnl_reduction_norm_lp_power_p_sum,
    /// Softmax
    dnnl_softmax_accurate = 0x30000,
    /// Logsoftmax
    dnnl_softmax_log,
} dnnl_alg_kind_t;

/// Flags for normalization primitives.
typedef enum {
    /// Use no normalization flags
    ///
    /// If specified
    ///  - on forward training propagation mean and variance are computed and
    ///    stored as output
    ///  - on backward propagation compute full derivative wrt data
    ///  - on backward propagation prop_kind == #dnnl_backward_data has the same
    ///    behavior as prop_kind == #dnnl_backward
    dnnl_normalization_flags_none = 0x0U,

    /// Use global statistics
    ///
    /// If specified
    ///  - on forward propagation use mean and variance provided by user (input)
    ///  - on backward propagation reduces the amount of computations, since
    ///    mean and variance are considered as constants
    ///
    ///  If not specified:
    ///   - on forward propagation mean and variance are computed and stored as
    ///     output
    ///   - on backward propagation compute full derivative wrt data
    dnnl_use_global_stats = 0x1U,

    /// Use scale parameter
    ///
    /// If specified:
    ///  - on forward propagation use scale for the normalization results
    ///  - on backward propagation (for prop_kind == #dnnl_backward) compute
    ///    diff wrt scale (hence one extra output used)
    dnnl_use_scale = 0x2U,

    /// Use shift parameter
    ///
    /// If specified:
    ///  - on forward propagation use shift (aka bias) for the normalization
    ///    results
    ///  - on backward propagation (for prop_kind == #dnnl_backward) compute
    ///    diff wrt shift (hence one extra output used)
    dnnl_use_shift = 0x4U,

    /// Fuse with ReLU
    ///
    /// The flag implies negative slope being 0. On training this is the only
    /// configuration supported. For inference, to use non-zero negative slope
    /// consider using @ref dev_guide_attributes_post_ops.
    ///
    /// If specified:
    ///  - on inference this option behaves the same as if the primitive were
    ///    fused with ReLU using post ops API with zero negative slope.
    ///  - on training primitive requires workspace (required to be able to
    ///    perform backward pass)
    dnnl_fuse_norm_relu = 0x8U,

    /// Fuse with Add and then fuse with ReLU
    ///
    /// If specified:
    ///
    ///  - on forward propagation apply element-wise binary Add operation to
    ///    to the normalization results with an additional input tensor and then
    ///    apply ReLU with negative slope being 0.
    ///  - on training primitive requires workspace (required to be able to
    ///    perform backward pass).
    ///  - on backward propagation save the result of backward ReLU operation
    ///    with input tensor and workspace from forward pass to extra output
    ///    tensor and then perform backward normalization.
    dnnl_fuse_norm_add_relu = 0x10U,

} dnnl_normalization_flags_t;

/// @} dnnl_api_primitives_common
/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_memory
/// @{

/// A wildcard value for dimensions that are unknown at a primitive creation
/// time.
#define DNNL_RUNTIME_DIM_VAL INT64_MIN

/// A `size_t` counterpart of the DNNL_RUNTIME_DIM_VAL.
/// For instance, this value is returned by dnnl_memory_desc_get_size() if
/// either of the dimensions or strides equal to #DNNL_RUNTIME_DIM_VAL.
#define DNNL_RUNTIME_SIZE_VAL ((size_t)DNNL_RUNTIME_DIM_VAL)

/// @cond DO_NOT_DOCUMENT_THIS
/// Hex representation for a **special** quiet NAN (!= NAN from math.h)
static const union {
    unsigned u;
    float f;
} DNNL_RUNTIME_F32_VAL_REP = {0x7fc000d0};
/// @endcond

/// A wildcard value for floating point values that are unknown at a primitive
/// creation time.
#define DNNL_RUNTIME_F32_VAL (DNNL_RUNTIME_F32_VAL_REP.f)

/// @cond DO_NOT_DOCUMENT_THIS
static const int DNNL_RUNTIME_S32_VAL_REP = INT32_MIN;
/// @endcond

/// A wildcard value for int32_t values that are unknown at a primitive creation
/// time.
#define DNNL_RUNTIME_S32_VAL DNNL_RUNTIME_S32_VAL_REP

/// @struct dnnl_memory_desc
/// An opaque structure to describe a memory descriptor.
struct dnnl_memory_desc;

/// A memory descriptor handle.
typedef struct dnnl_memory_desc *dnnl_memory_desc_t;

/// A memory descriptor handle.
typedef const struct dnnl_memory_desc *const_dnnl_memory_desc_t;

/// @struct dnnl_memory
/// An opaque structure to describe a memory.
struct dnnl_memory;

/// A memory handle.
typedef struct dnnl_memory *dnnl_memory_t;

/// A constant memory handle.
typedef const struct dnnl_memory *const_dnnl_memory_t;

/// @} dnnl_api_memory

/// @addtogroup dnnl_api_primitives
/// @{

/// @addtogroup dnnl_api_rnn
/// @{

/// Flags for RNN cell.
typedef enum {
    /// Undefined RNN flags
    dnnl_rnn_flags_undef = 0x0,
    /// Do not add weights gradient to existing diff_weights memory
    dnnl_rnn_flags_diff_weights_overwrite = 0x1,
} dnnl_rnn_flags_t;

/// A direction of RNN primitive execution.
typedef enum {
    /// Undefined RNN direction.
    dnnl_rnn_direction_undef = 0,
    /// Unidirectional execution of RNN primitive from left to right.
    dnnl_unidirectional_left2right,
    /// Unidirectional execution of RNN primitive from right to left.
    dnnl_unidirectional_right2left,
    /// Bidirectional execution of RNN primitive with concatenation of the
    /// results.
    dnnl_bidirectional_concat,
    /// Bidirectional execution of RNN primitive with summation of the
    /// results.
    dnnl_bidirectional_sum,
} dnnl_rnn_direction_t;

/// @} dnnl_api_rnn

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_primitives
/// @{
/// @addtogroup dnnl_api_primitives_common
/// @{

/// @struct dnnl_primitive_desc
/// @brief An opaque structure to describe a primitive descriptor.
struct dnnl_primitive_desc;

/// @brief A primitive descriptor handle.
typedef struct dnnl_primitive_desc *dnnl_primitive_desc_t;

/// @brief A constant primitive descriptor handle.
typedef const struct dnnl_primitive_desc *const_dnnl_primitive_desc_t;

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_attributes
/// @{

/// Scratchpad mode
typedef enum {
    /// The library manages the scratchpad allocation according to the policy
    /// specified by the `DNNL_ENABLE_CONCURRENT_EXEC`
    /// [build option](@ref dev_guide_build_options) (default).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=OFF` (default), the library
    /// scratchpad is common to all primitives to reduce the memory footprint.
    /// This configuration comes with limited thread-safety properties, namely
    /// primitives can be created and executed in parallel but cannot migrate
    /// between threads (in other words, each primitive should be executed in
    /// the same thread it was created in).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=ON`, the library scratchpad is
    /// private to each primitive. The memory footprint is larger than when
    /// using `DNNL_ENABLE_CONCURRENT_EXEC=OFF` but different primitives can be
    /// created and run concurrently (the same primitive cannot be run
    /// concurrently from two different threads though).
    dnnl_scratchpad_mode_library,
    /// The user manages the scratchpad allocation by querying and providing
    /// the scratchpad memory to primitives. This mode is thread-safe as long
    /// as the scratchpad buffers are not used concurrently by two primitive
    /// executions.
    dnnl_scratchpad_mode_user,
} dnnl_scratchpad_mode_t;

/// @struct dnnl_primitive_attr
/// @brief An opaque structure for primitive descriptor attributes.
///
/// Attributes may contain:
///  - output scales (to scale the result prior to storing it to the memory)
struct dnnl_primitive_attr;

/// @brief A primitive descriptor attributes handle that controls primitive
/// behavior.
typedef struct dnnl_primitive_attr *dnnl_primitive_attr_t;

/// @brief A constant primitive descriptor attributes handle.
typedef const struct dnnl_primitive_attr *const_dnnl_primitive_attr_t;

/// @struct dnnl_post_ops
/// @brief An opaque structure for a chain of post operations.
///
/// dnnl_post_ops can be used to perform some (trivial) operations like
/// accumulation or eltwise after certain primitives like convolution.
///
/// Post operations might be combined together, making a chain of post
/// operations. For instance one can configure convolution followed by
/// accumulation followed by eltwise. This might be especially beneficial
/// for residual learning blocks.
///
/// @warning
///      Of course not all combinations are supported, so the user should handle
///      errors accordingly.
///
/// Supported post operations:
///  - accumulation (base primitive: convolution)
///  - eltwise (base primitive: convolution)
struct dnnl_post_ops;

/// @brief A post operation chain handle.
typedef struct dnnl_post_ops *dnnl_post_ops_t;

/// @brief A constant post operation chain handle.
typedef const struct dnnl_post_ops *const_dnnl_post_ops_t;

/// @} dnnl_api_attributes

/// @addtogroup dnnl_api_primitives_common
/// @{

/// @struct dnnl_primitive
/// An opaque structure to describe a primitive.
struct dnnl_primitive;
/// A primitive handle.
typedef struct dnnl_primitive *dnnl_primitive_t;
/// A constant primitive handle.
typedef const struct dnnl_primitive *const_dnnl_primitive_t;

/// Undefined argument.
#define DNNL_ARG_UNDEF 0
/// Source argument #0.
#define DNNL_ARG_SRC_0 1
/// A special mnemonic for source argument for primitives that have a
/// single source. An alias for #DNNL_ARG_SRC_0.
#define DNNL_ARG_SRC DNNL_ARG_SRC_0
/// A special mnemonic for RNN input vector. An alias for
/// #DNNL_ARG_SRC_0.
#define DNNL_ARG_SRC_LAYER DNNL_ARG_SRC_0
/// A special mnemonic for reorder source argument. An alias for
/// #DNNL_ARG_SRC_0.
#define DNNL_ARG_FROM DNNL_ARG_SRC_0

/// Source argument #1.
#define DNNL_ARG_SRC_1 2
/// A special mnemonic for RNN input recurrent hidden state vector. An alias
/// for #DNNL_ARG_SRC_1.
#define DNNL_ARG_SRC_ITER DNNL_ARG_SRC_1

/// Source argument #2.
#define DNNL_ARG_SRC_2 3
/// A special mnemonic for RNN input recurrent cell state vector. An alias for
/// #DNNL_ARG_SRC_2.
#define DNNL_ARG_SRC_ITER_C DNNL_ARG_SRC_2

/// Source argument #3.
#define DNNL_ARG_SRC_3 4
/// A special mnemonic for RNN input recurrent cell attention vector. An alias for
/// #DNNL_ARG_SRC_3.
#define DNNL_ARG_AUGRU_ATTENTION DNNL_ARG_SRC_3

/// Destination argument #0.
#define DNNL_ARG_DST_0 17
/// A special mnemonic for destination argument for primitives that have a
/// single destination. An alias for #DNNL_ARG_DST_0.
#define DNNL_ARG_DST DNNL_ARG_DST_0
/// A special mnemonic for reorder destination argument. An alias for
/// #DNNL_ARG_DST_0.
#define DNNL_ARG_TO DNNL_ARG_DST_0
/// A special mnemonic for RNN output vector. An alias for #DNNL_ARG_DST_0.
#define DNNL_ARG_DST_LAYER DNNL_ARG_DST_0

/// Destination argument #1.
#define DNNL_ARG_DST_1 18
/// A special mnemonic for RNN input recurrent hidden state vector. An
/// alias for #DNNL_ARG_DST_1.
#define DNNL_ARG_DST_ITER DNNL_ARG_DST_1

/// Destination argument #2.
#define DNNL_ARG_DST_2 19
/// A special mnemonic for LSTM output recurrent cell state vector. An
/// alias for #DNNL_ARG_DST_2.
#define DNNL_ARG_DST_ITER_C DNNL_ARG_DST_2

/// Weights argument #0.
#define DNNL_ARG_WEIGHTS_0 33
/// A special mnemonic for primitives that have a single weights
/// argument. Alias for #DNNL_ARG_WEIGHTS_0.
#define DNNL_ARG_WEIGHTS DNNL_ARG_WEIGHTS_0
/// A special mnemonic for RNN weights applied to the layer input. An
/// alias for #DNNL_ARG_WEIGHTS_0.
#define DNNL_ARG_WEIGHTS_LAYER DNNL_ARG_WEIGHTS_0

/// Weights argument #1.
#define DNNL_ARG_WEIGHTS_1 34
/// A special mnemonic for RNN weights applied to the recurrent input.
/// An alias for #DNNL_ARG_WEIGHTS_1.
#define DNNL_ARG_WEIGHTS_ITER DNNL_ARG_WEIGHTS_1

/// Weights argument #2.
#define DNNL_ARG_WEIGHTS_2 35
/// A special mnemonic for RNN weights applied to the peephole weights.
/// An alias for #DNNL_ARG_WEIGHTS_2.
#define DNNL_ARG_WEIGHTS_PEEPHOLE DNNL_ARG_WEIGHTS_2

/// Weights argument #3.
#define DNNL_ARG_WEIGHTS_3 36
/// A special mnemonic for RNN weights applied to the projection weights.
/// An alias for #DNNL_ARG_WEIGHTS_3.
#define DNNL_ARG_WEIGHTS_PROJECTION DNNL_ARG_WEIGHTS_3

/// Bias tensor argument.
#define DNNL_ARG_BIAS 41

/// Mean values tensor argument.
#define DNNL_ARG_MEAN 49
/// Variance values tensor argument.
#define DNNL_ARG_VARIANCE 50

/// A special mnemonic for scale argument of normalization primitives.
#define DNNL_ARG_SCALE 51
/// A special mnemonic for shift argument of normalization primitives.
#define DNNL_ARG_SHIFT 52

/// Workspace tensor argument. Workspace is used to pass information
/// from forward propagation to backward propagation computations.
#define DNNL_ARG_WORKSPACE 64
/// Scratchpad (temporary storage) tensor argument.
#define DNNL_ARG_SCRATCHPAD 80

/// Gradient (diff) of the source argument #0.
#define DNNL_ARG_DIFF_SRC_0 129
/// A special mnemonic for primitives that have a single diff source argument.
/// An alias for #DNNL_ARG_DIFF_SRC_0.
#define DNNL_ARG_DIFF_SRC DNNL_ARG_DIFF_SRC_0
/// A special mnemonic for gradient (diff) of RNN input vector. An alias for
/// #DNNL_ARG_DIFF_SRC_0.
#define DNNL_ARG_DIFF_SRC_LAYER DNNL_ARG_DIFF_SRC_0

/// Gradient (diff) of the source argument #1.
#define DNNL_ARG_DIFF_SRC_1 130
/// A special mnemonic for gradient (diff) of RNN input recurrent hidden state
/// vector. An alias for #DNNL_ARG_DIFF_SRC_1.
#define DNNL_ARG_DIFF_SRC_ITER DNNL_ARG_DIFF_SRC_1

/// Gradient (diff) of the source argument #2.
#define DNNL_ARG_DIFF_SRC_2 131
/// A special mnemonic for gradient (diff) of RNN input recurrent cell state
/// vector. An alias for #DNNL_ARG_DIFF_SRC_1.
#define DNNL_ARG_DIFF_SRC_ITER_C DNNL_ARG_DIFF_SRC_2

/// Gradient (diff) of the source argument #3.
#define DNNL_ARG_DIFF_SRC_3 132
/// A special mnemonic for gradient (diff) of RNN input recurrent cell attention
/// vector. An alias for #DNNL_ARG_DIFF_SRC_3.
#define DNNL_ARG_DIFF_AUGRU_ATTENTION DNNL_ARG_DIFF_SRC_3

/// Gradient (diff) of the destination argument #0.
#define DNNL_ARG_DIFF_DST_0 145
/// A special mnemonic for primitives that have a single diff destination
/// argument. An alias for #DNNL_ARG_DIFF_DST_0.
#define DNNL_ARG_DIFF_DST DNNL_ARG_DIFF_DST_0
/// A special mnemonic for gradient (diff) of RNN output vector. An alias for
/// #DNNL_ARG_DIFF_DST_0.
#define DNNL_ARG_DIFF_DST_LAYER DNNL_ARG_DIFF_DST_0

/// Gradient (diff) of the destination argument #1.
#define DNNL_ARG_DIFF_DST_1 146
/// A special mnemonic for gradient (diff) of RNN input recurrent hidden state
/// vector. An alias for #DNNL_ARG_DIFF_DST_1.
#define DNNL_ARG_DIFF_DST_ITER DNNL_ARG_DIFF_DST_1

/// Gradient (diff) of the destination argument #2.
#define DNNL_ARG_DIFF_DST_2 147
/// A special mnemonic for gradient (diff) of RNN input recurrent cell state
/// vector. An alias for #DNNL_ARG_DIFF_DST_2.
#define DNNL_ARG_DIFF_DST_ITER_C DNNL_ARG_DIFF_DST_2

/// Gradient (diff) of the weights argument #0.
#define DNNL_ARG_DIFF_WEIGHTS_0 161
/// A special mnemonic for primitives that have a single diff weights
/// argument. Alias for #DNNL_ARG_DIFF_WEIGHTS_0.
#define DNNL_ARG_DIFF_WEIGHTS DNNL_ARG_DIFF_WEIGHTS_0
/// A special mnemonic for diff of RNN weights applied to the layer input. An
/// alias for #DNNL_ARG_DIFF_WEIGHTS_0.
#define DNNL_ARG_DIFF_WEIGHTS_LAYER DNNL_ARG_DIFF_WEIGHTS_0

/// Gradient (diff) of the weights argument #1.
#define DNNL_ARG_DIFF_WEIGHTS_1 162
/// A special mnemonic for diff of RNN weights applied to the recurrent input.
/// An alias for #DNNL_ARG_DIFF_WEIGHTS_1.
#define DNNL_ARG_DIFF_WEIGHTS_ITER DNNL_ARG_DIFF_WEIGHTS_1

/// Gradient (diff) of the weights argument #2.
#define DNNL_ARG_DIFF_WEIGHTS_2 163
/// A special mnemonic for diff of RNN weights applied to the peephole weights.
/// An alias for #DNNL_ARG_DIFF_WEIGHTS_2.
#define DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE DNNL_ARG_DIFF_WEIGHTS_2

/// Gradient (diff) of the weights argument #3.
#define DNNL_ARG_DIFF_WEIGHTS_3 164
/// A special mnemonic for diff of RNN weights applied to the projection
/// weights. An alias for #DNNL_ARG_DIFF_WEIGHTS_3.
#define DNNL_ARG_DIFF_WEIGHTS_PROJECTION DNNL_ARG_DIFF_WEIGHTS_3

/// Gradient (diff) of the bias tensor argument.
#define DNNL_ARG_DIFF_BIAS 169

/// A special mnemonic for scale argument of normalization primitives.
#define DNNL_ARG_DIFF_SCALE 255
/// A special mnemonic for shift argument of normalization primitives.
#define DNNL_ARG_DIFF_SHIFT 256

/// Output scaling factors provided at execution time.
#define DNNL_ARG_ATTR_OUTPUT_SCALES 513

/// Starting index for source arguments for primitives that take a variable
/// number of source arguments.
#define DNNL_ARG_MULTIPLE_SRC 1024
/// Starting index for destination arguments for primitives that produce a
/// variable number of destination arguments.
#define DNNL_ARG_MULTIPLE_DST 2048

/// Scaling factors provided at execution time.
#define DNNL_ARG_ATTR_SCALES 4096

/// Zero points provided at execution time.
#define DNNL_ARG_ATTR_ZERO_POINTS 8192

/// Arguments for fused depthwise convolution.
/// See @ref dev_guide_attributes_post_ops_depthwise_fusion
#define DNNL_ARG_ATTR_POST_OP_DW 16384

/// Starting point for a binary post operation.
#define DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE 32768

/// Arguments for a binary post operation. Up to 32 arguments are supported.
/// See @ref dev_guide_attributes_post_ops_binary_fusion
#define DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) \
    (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * ((idx) + 1))

/// A structure that contains an index and a memory object, and is used to pass
/// arguments to dnnl_primitive_execute().
typedef struct {
    int arg; ///< An argument index, e.g. DNNL_ARG_SRC
    dnnl_memory_t memory; ///< Input/output memory
} dnnl_exec_arg_t;

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Primitive descriptor query specification
///
/// For generic function dnnl_primitive_desc_query(), the type of result must
/// agree with the queried argument. The correspondence table:
///
/// Query kind                      | Type of query result
/// --------------------------------|-----------------------------
/// dnnl_query_*_engine             | #dnnl_engine_t *
/// #dnnl_query_primitive_kind      | #dnnl_primitive_kind_t *
/// dnnl_query_*_s32                | int *
/// dnnl_query_*_s64                | #dnnl_dim_t * (same as int64_t *)
/// dnnl_query_*_f32                | float *
/// dnnl_query_*_f64                | double *
/// dnnl_query_*_str                | const char **
/// dnnl_query_*_md                 | #const_dnnl_memory_desc_t *
/// dnnl_query_*_pd                 | #const_dnnl_primitive_desc_t *
/// dnnl_query_cache_blob_id        | const uint8_t **
/// dnnl_query_strides              | const #dnnl_dims_t **
/// dnnl_query_dilations            | const #dnnl_dims_t **
/// dnnl_query_padding_l            | const #dnnl_dims_t **
/// dnnl_query_padding_r            | const #dnnl_dims_t **
/// dnnl_query_flags                | unsigned *
/// dnnl_query_alg_kind             | #dnnl_alg_kind_t *
/// dnnl_query_factors              | const float **
/// dnnl_query_cell_kind            | #dnnl_alg_kind_t *
/// dnnl_query_direction            | #dnnl_rnn_direction_t *
/// dnnl_query_activation_kind      | #dnnl_alg_kind_t *
/// dnnl_query_kernel               | const #dnnl_dims_t **
/// dnnl_query_dims                 | const #dnnl_dims_t **
/// dnnl_query_data_type            | #dnnl_data_type_t *
/// dnnl_query_padded_dims          | const #dnnl_dims_t **
/// dnnl_query_padded_offsets       | const #dnnl_dims_t **
/// dnnl_query_format_kind          | #dnnl_format_kind_t *
/// dnnl_query_inner_blks           | const #dnnl_dims_t **
/// dnnl_query_inner_idxs           | const #dnnl_dims_t **
/// dnnl_query_sparse_encoding      | #dnnl_sparse_encoding_t *
///
/// @note
///     Rule of thumb: all opaque types and structures are returned by
///     reference. All numbers are returned by value.
///
/// @warning
///     All returned references point to constant objects and are valid only
///     during the lifetime of the queried primitive descriptor. Returned objects
///     must not be destroyed by the user. If you need to keep the object longer
///     than the lifetime of the queried primitive descriptor, use
///     dnnl_primitive_desc_clone() to make a copy.
typedef enum {
    dnnl_query_undef = 0, ///< no query

    dnnl_query_engine, ///< execution engine
    dnnl_query_primitive_kind, ///< primitive kind

    dnnl_query_num_of_inputs_s32, ///< number of inputs expected
    dnnl_query_num_of_outputs_s32, ///< number of outputs expected

    dnnl_query_time_estimate_f64, ///< runtime estimation (seconds)
    dnnl_query_memory_consumption_s64, ///< memory consumption -- extra
    ///  (scratch) memory, additional to
    ///  all inputs and outputs memory
    ///  (bytes)

    dnnl_query_scratchpad_engine, ///< scratchpad engine -- engine to be used
    ///  for creating scratchpad memory

    dnnl_query_impl_info_str, ///< implementation name

    dnnl_query_reorder_src_engine, ///< source engine
    dnnl_query_reorder_dst_engine, ///< destination engine

    dnnl_query_prop_kind, ///< propagation kind

    dnnl_query_cache_blob_id_size_s64, ///< size of cache blob ID in bytes
    dnnl_query_cache_blob_id, ///< cache blob  ID (pointer to array)

    dnnl_query_strides, ///< strides
    dnnl_query_dilations, ///< dilations
    dnnl_query_padding_l, ///< left padding
    dnnl_query_padding_r, ///< right padding
    dnnl_query_epsilon_f32, ///< epsilon
    dnnl_query_flags, ///< flags
    dnnl_query_alg_kind, ///< algorithm kind
    dnnl_query_alpha_f32, ///< alpha
    dnnl_query_beta_f32, ///< beta
    dnnl_query_axis_s32, ///< axis
    dnnl_query_local_size_s64, ///< LRN parameter local size
    dnnl_query_k_f32, ///< LRN parameter K
    dnnl_query_p_f32, ///< Reduction parameter P
    dnnl_query_factors, ///< Resampling parameter factors
    dnnl_query_cell_kind, ///< RNN parameter cell kind
    dnnl_query_direction, ///< RNN parameter direction
    dnnl_query_activation_kind, ///< RNN parameter activation kind
    dnnl_query_kernel, ///< Pooling parameter kernel
    dnnl_query_group_size_s64, ///< Shuffle parameter group size

    // memory descriptor section
    dnnl_query_some_md = 128, ///< stub
    dnnl_query_src_md, ///< source memory desc
    dnnl_query_diff_src_md, ///< source gradient memory desc
    dnnl_query_weights_md, ///< weights memory descriptor desc
    dnnl_query_diff_weights_md, ///< weights grad. memory desc
    dnnl_query_dst_md, ///< destination memory desc
    dnnl_query_diff_dst_md, ///< destination grad. memory desc
    dnnl_query_workspace_md, ///< workspace memory desc
    dnnl_query_scratchpad_md, ///< scratchpad memory desc
    dnnl_query_exec_arg_md = 255, ///< memory desc of an execute argument

    dnnl_query_ndims_s32, ///< number of dimensions
    dnnl_query_dims, ///< vector of dimensions
    dnnl_query_data_type, ///< data type
    dnnl_query_submemory_offset_s64, ///< submemory offset
    dnnl_query_padded_dims, ///< vector of padded dimensions
    dnnl_query_padded_offsets, ///< vector of padded offsets
    dnnl_query_format_kind, ///< format kind
    dnnl_query_inner_nblks_s32, ///< number of innermost blocks
    dnnl_query_inner_blks, ///< vector of sizes of the innermost blocks
    dnnl_query_inner_idxs, ///< vector of logical indices of the blocks
#ifdef DNNL_EXPERIMENTAL_SPARSE
    dnnl_query_sparse_encoding, ///< Sparse encoding
    dnnl_query_nnz_s64, ///< Number of non-zero entries
    dnnl_query_num_handles_s32, ///< Number of buffers required for a memory
///  descriptor
#endif
    // Max value to prevent UB for internal use only dnnl_query_t
    dnnl_query_max = 0x7fff,
} dnnl_query_t;

/// @} dnnl_api_primitives_common

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_service
/// @{

/// Disable profiling completely
#define DNNL_JIT_PROFILE_NONE 0u

/// Enable VTune Profiler integration
#define DNNL_JIT_PROFILE_VTUNE 1u

/// Enable Linux perf integration via perfmap files
#define DNNL_JIT_PROFILE_LINUX_PERFMAP 2u

/// Enable Linux perf integration via jitdump files
#define DNNL_JIT_PROFILE_LINUX_JITDUMP 4u

/// Instruct Linux perf integration via jitdump files to use TSC. @ref
/// DNNL_JIT_PROFILE_LINUX_JITDUMP must be set too for this to take effect.
#define DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC 8u

/// Enable Linux perf integration (both jitdump and perfmap)
#define DNNL_JIT_PROFILE_LINUX_PERF \
    (DNNL_JIT_PROFILE_LINUX_JITDUMP | DNNL_JIT_PROFILE_LINUX_PERFMAP)

/// CPU instruction set flags
typedef enum {
    /// Library choice of ISA (excepting those listed as initial support)
    dnnl_cpu_isa_default = 0x0,

    /// Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)
    dnnl_cpu_isa_sse41 = 0x1,

    /// Intel Advanced Vector Extensions (Intel AVX)
    dnnl_cpu_isa_avx = 0x3,

    /// Intel Advanced Vector Extensions 2 (Intel AVX2)
    dnnl_cpu_isa_avx2 = 0x7,

    /// Intel AVX2 and Intel Deep Learning Boost (Intel DL Boost) support
    dnnl_cpu_isa_avx2_vnni = 0xf,

    /// Intel AVX2 and Intel Deep Learning Boost (Intel DL Boost)
    /// with 8-bit integer, float16 and bfloat16 support
    dnnl_cpu_isa_avx2_vnni_2 = 0x1f,

    /// Intel AVX-512 subset for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core = 0x27,

    /// Intel AVX-512 and Intel Deep Learning Boost (Intel DL Boost) support
    /// for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core_vnni = 0x67,

    /// Intel AVX-512, Intel DL Boost and bfloat16 support
    /// for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core_bf16 = 0xe7,

    /// Intel AVX-512 with float16, Intel DL Boost and bfloat16 support
    /// for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    // TODO: Align avx10_1 values to internal representation.
    dnnl_cpu_isa_avx10_1_512 = 0x1ef,
    /// @copydoc dnnl_cpu_isa_avx10_1_512
    dnnl_cpu_isa_avx512_core_fp16 = dnnl_cpu_isa_avx10_1_512,

    /// Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and
    /// Intel AMX with 8-bit integer and bfloat16 support
    // TODO: Align avx10_1 values to internal representation.
    dnnl_cpu_isa_avx10_1_512_amx = 0xfef,
    /// @copydoc dnnl_cpu_isa_avx10_1_512_amx
    dnnl_cpu_isa_avx512_core_amx = dnnl_cpu_isa_avx10_1_512_amx,

    /// Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and
    /// Intel AMX with 8-bit integer, bfloat16 and float16 support
    // TODO: Align avx10_1 values to internal representation.
    dnnl_cpu_isa_avx10_1_512_amx_fp16 = 0x1fef,
    /// @copydoc dnnl_cpu_isa_avx10_1_512_amx_fp16
    dnnl_cpu_isa_avx512_core_amx_fp16 = dnnl_cpu_isa_avx10_1_512_amx_fp16,
} dnnl_cpu_isa_t;

/// CPU ISA hints flags
typedef enum {
    /// No hints (use default features)
    dnnl_cpu_isa_no_hints = 0x0,

    /// Prefer to exclusively use Ymm registers for computations
    dnnl_cpu_isa_prefer_ymm = 0x1,
} dnnl_cpu_isa_hints_t;

/// @} dnnl_api_service

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif /* ONEAPI_DNNL_TYPES_H */
