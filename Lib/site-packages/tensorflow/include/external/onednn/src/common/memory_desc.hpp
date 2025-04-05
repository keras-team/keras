/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef COMMON_MEMORY_DESC_HPP
#define COMMON_MEMORY_DESC_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {

// Winograd-specific formats
enum class wino_memory_format_t {
    // Undefined memory format, used for empty memory descriptors.
    wino_undef,
    // Tensors of weights for 2x3 winograd convolutions.
    //
    // Internal weights format for 2x3 Winograd.
    wino_wei_aaOio,
    // Internal weights format for 2x3 Winograd.
    wino_wei_aaOBiOo,
    // Tensor of weights for 4x3 convolution.
    //
    // Internal weights format for 4x3 Winograd.
    wino_wei_OBaaIBOIio
};

enum class rnn_packed_memory_format_t { undef, ldigo_p, ldgoi_p, ldio_p };

// Create aliases for extra flags to preserve the old behavior.
// This should be removed and all places that are affected should use
// rnn_packed_memory_format_t::<flag name> syntax.
namespace rnn_packed_format {
const rnn_packed_memory_format_t undef = rnn_packed_memory_format_t::undef;
const rnn_packed_memory_format_t ldigo_p = rnn_packed_memory_format_t::ldigo_p;
const rnn_packed_memory_format_t ldgoi_p = rnn_packed_memory_format_t::ldgoi_p;
const rnn_packed_memory_format_t ldio_p = rnn_packed_memory_format_t::ldio_p;
} // namespace rnn_packed_format

// TODO: convert to 'enum class'.
// Flags for memory special features
enum memory_extra_flags_t {
    dnnl_memory_extra_flag_none = 0x0U,
    // Indicates the weights have an additional buffer, that depends on the
    // @p compensation_mask.
    //
    // For instance, in 4D case with the compensation mask equals (1 << 0)
    // the additional buffer would consist of OC values:
    // O[oc : 0,OC] =
    //  -128 * SUM(ic : 0,IC; kh : 0,KH; kw : 0,KW){ weights(oc, ic, kh, kw) }
    dnnl_memory_extra_flag_compensation_conv_s8s8 = 0x1U,
    dnnl_memory_extra_flag_scale_adjust = 0x2U,
    dnnl_memory_extra_flag_rnn_u8s8_compensation = 0x4U,
    dnnl_memory_extra_flag_gpu_rnn_u8s8_compensation
    = dnnl_memory_extra_flag_rnn_u8s8_compensation,
    dnnl_memory_extra_flag_compensation_conv_asymmetric_src = 0x8U,
    dnnl_memory_extra_flag_rnn_s8s8_compensation = 0x16U,
};

// Create aliases for extra flags to preserve the old behavior.
// This should be removed and all places that are affected should use
// memory_extra_flags_t::<flag name> syntax.
namespace memory_extra_flags {
const memory_extra_flags_t none = dnnl_memory_extra_flag_none;
const memory_extra_flags_t compensation_conv_s8s8
        = dnnl_memory_extra_flag_compensation_conv_s8s8;
const memory_extra_flags_t scale_adjust = dnnl_memory_extra_flag_scale_adjust;
const memory_extra_flags_t rnn_u8s8_compensation
        = dnnl_memory_extra_flag_rnn_u8s8_compensation;
const memory_extra_flags_t rnn_s8s8_compensation
        = dnnl_memory_extra_flag_rnn_s8s8_compensation;
const memory_extra_flags_t compensation_conv_asymmetric_src
        = dnnl_memory_extra_flag_compensation_conv_asymmetric_src;
} // namespace memory_extra_flags

// Generic description of blocked data layout for most memory formats.
struct blocking_desc_t {
    // The strides between the outermost blocks.
    // In case of plain (non-blocked) formats the strides between dimensions.
    dims_t strides;
    // Innermost section
    // ASSUMPTION: the innermost blocks are always dense
    // The number of innermost blocks, e.g. 3 in case of `OIhw_4i16o4i_`
    int inner_nblks;
    // The size of the blocks, e.g. `{4, 16, 4}` in case of `OIhw_4i16o4i`
    dims_t inner_blks;
    // The logical indices of the blocks, e.g. `{1, 0, 1}` in case of
    // `4i16o4i`, because `i` is the 1st dim and `o` is the 0st dim
    dims_t inner_idxs;
};

// Description of tensor of weights for winograd 2x3 convolution.
struct wino_desc_t {
    wino_memory_format_t wino_format;
    int r;
    int alpha;
    int ic;
    int oc;
    int ic_block;
    int oc_block;
    int ic2_block;
    int oc2_block;
    float adj_scale;
    size_t size;
};

#define DNNL_RNN_MAX_N_PARTS 4
// Description of tensor of packed weights for rnn.
struct rnn_packed_desc_t {
    // Maximum number of parts of RNN weights tensor that require separate
    // computation.
    const static int max_n_parts = 4;
    rnn_packed_memory_format_t format;
    int n_parts;
    int n;
    int ldb;
    int parts[max_n_parts];
    size_t part_pack_size[max_n_parts];
    unsigned pack_part[max_n_parts];
    size_t offset_compensation;
    size_t size;
};

struct sparse_desc_t {
    static constexpr int max_metadata_types = 2;
    // Each encoding defines the number of handles it requires and their
    // meaning.
    //
    // CSR: Number of handles is 3:
    //  - 0: values
    //  - 1: indices
    //  - 2: pointers
    //
    // packed: Number of handles is 3:
    //  - 0: values
    //  - 1: offsets
    //  - 2: bitmask
    sparse_encoding_t encoding;

    // Number of non-zero entries.
    dnnl_dim_t nnz;

    // Metadata types. Each encoding defines how to interpret these.
    // - CSR: 0th - index data type
    //        1st - pointer data type
    // - packed: N/A
    dnnl_data_type_t metadata_types[max_metadata_types];

    // The packed sparse encoding is described with `blocking_desc_t` and
    // can only be initialized by the implementation. The special encoding
    // `packed` will instruct the implementation to do that.
    // Storage schema description:
    //
    // The same tensor is described by `packed_desc` and `blocking` descriptors
    // in the absolutely the same way. However, the difference is how the tensor
    // is stored in the memory. When the tensor is described by `packed_desc`
    // its content is encoded meaning that there is metadata that is required to
    // decode the content.
    //
    // Encoding process:
    // - Reorder a dense tensor to the blocked format described by
    //  `packed_desc`
    // - Remove all zeroes from the reordered tensor
    // - Initialize metadata:
    //   * An array of offsets stores offsets for each block. The block is a
    //     product of all inner block, e.g. if the `packed_desc` describes a
    //     format tag BA16a64b4a then the size of the block is 4096 elements
    //     and the number of blocks is `padded_dims[0] * padded_dims[1] / 4096`.
    //     When the zeroes get removed we need to store the offset to the
    //     beginning of the block in the data without zeroes (packed data).
    //     For example, if the block size is 5 and there are two blocks:
    //     [01020][01203] then the array of offsets will have the following
    //     two values: [0,2] because the packed data is stored as [12][123].
    //     Tne offsets are stored as int64 values
    //   * A bitmask array stores a mask for all elements in the dense tensors
    //
    // Decoding process:
    // - Identify the block number that needs to be decoded (unpacked)
    // - Use the block number to find an offset in the packed data
    // - Use the bitmask to unpack the packed data
    blocking_desc_t packed_desc;
};

// Description of extra information stored in memory
struct memory_extra_desc_t {
    memory_extra_desc_t()
        : flags(0)
        , compensation_mask(0)
        , scale_adjust(0.0f)
        , asymm_compensation_mask(0) {}
    // The flags contain arbitrary extra information, such as compensation.
    // @sa dnnl_memory_extra_flags_t
    uint64_t flags;
    // Compensation mask
    int compensation_mask;
    // Scale applied to the data
    float scale_adjust;
    // Compensation mask for asymmetric quantization
    int asymm_compensation_mask;
};

status_t DNNL_API memory_desc_init_by_tag(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, format_tag_t tag);

status_t memory_desc_init_by_strides(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, const dims_t strides);

status_t memory_desc_init_submemory(memory_desc_t &memory_desc,
        const memory_desc_t &parent_memory_desc, const dims_t dims,
        const dims_t offsets);

status_t memory_desc_reshape(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, int ndims, const dims_t dims);

status_t memory_desc_permute_axes(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, const int *perm);

} // namespace impl
} // namespace dnnl

// Memory descriptor. The description is based on a number of dimensions,
// dimensions themselves, plus information about elements type and memory
// format. Additionally, contains format-specific descriptions of the data
// layout.
struct dnnl_memory_desc : public dnnl::impl::c_compatible {
    dnnl_memory_desc()
        : ndims(0)
        , dims {}
        , data_type(dnnl::impl::data_type::undef)
        , padded_dims {}
        , padded_offsets {}
        , offset0(0)
        , format_kind(dnnl::impl::format_kind::undef)
        , format_desc {}
        , extra {} {}
    // Number of dimensions
    int ndims;
    // Dimensions in the following order:
    // - CNN data tensors: mini-batch, channel, spatial
    //   (<code>{N, C, [[D,] H,] W}</code>)
    // - CNN weight tensors: group (optional), output channel, input channel,
    //   spatial (<code>{[G,] O, I, [[D,] H,] W}</code>)
    // - RNN data tensors: time, mini-batch, channels (<code>{T, N, C}</code>)
    //   or layers, directions, states, mini-batch, channels (<code>{L, D, S, N, C}</code>)
    // - RNN weight tensor: layers, directions, input channel, gates, output channels
    //   (<code>{L, D, I, G, O}</code>).
    //
    // @note
    //    The order of dimensions does not depend on the memory format, so
    //    whether the data is laid out in #dnnl_nchw or #dnnl_nhwc
    //    the dims for 4D CN data tensor would be <code>{N, C, H, W}</code>.
    dnnl::impl::dims_t dims;

    // Data type of the tensor elements.
    dnnl::impl::data_type_t data_type;

    // Size of the data including padding in each dimension.
    dnnl::impl::dims_t padded_dims;

    // Per-dimension offset from the padding to actual data, the top-level
    // tensor with offsets applied must lie within the padding area.
    dnnl::impl::dims_t padded_offsets;

    // Offset from memory origin to the current block, non-zero only in
    // a description of a memory sub-block.
    dnnl::impl::dim_t offset0;

    // Memory format kind.
    dnnl::impl::format_kind_t format_kind;
    union {
        // Description of the data layout for memory formats that use
        // blocking.
        dnnl::impl::blocking_desc_t blocking;
        // Tensor of weights for winograd convolution.
        dnnl::impl::wino_desc_t wino_desc;
        // Tensor of packed weights for RNN.
        dnnl::impl::rnn_packed_desc_t rnn_packed_desc;
        // Description of the sparse encodings.
        dnnl::impl::sparse_desc_t sparse_desc;
        // ... other descriptions possible
    } format_desc;

    dnnl::impl::memory_extra_desc_t extra;
};

#endif
