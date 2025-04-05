/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#ifndef COMMON_MEMORY_TRACKING_HPP
#define COMMON_MEMORY_TRACKING_HPP

#include <assert.h>
#include <unordered_map>

#include "memory_debug.hpp"
#include "memory_storage.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct exec_ctx_t;

namespace memory_tracking {

/* Memory tracking capabilities
 *
 * The main purpose of this header file is to provide uniform way to register
 * required memory for a scratchpad at a primitive descriptor creation time
 * and then easily access it having only the base address of the scratchpad.
 *
 * Primitives might contain multiple disjoint parts that require temporary
 * buffers (known as scratchpad) during their execution. A primitive descriptor
 * should summarize all the needs into one single number -- the buffer size
 * that would be requested from a user. At execution time, the corresponding
 * primitive will receive a base pointer to a scratchpad. It then needs to
 * provide each part of algorithm the corresponding piece of memory. Three main
 * challenges here are:
 * 1. Track correct offset (from the base scratchpad address) for each piece
 * 2. Algorithm might require that different memory pieces to be aligned, so
 *    the scratchpad size is no more just a sum of size of the corresponding
 *    subparts.
 * 3. While a primitive is responsible for its scratchpad, the implementation
 *    might use some other basic blocks (e.g. cpu_reducer) that also require
 *    scratchpad memory. So there should be a simple way of passing the
 *    information back and force between the main algorithm (a primitive) and
 *    auxiliary stuff that lives completely separately from it (e.g. reducer).
 *
 * To address these challenges this header file provides 3 structures:
 * 1. registry_t  -- the class the stores the information about requested
 *                   memory. The information includes required size and desired
 *                   alignment for each piece. This class is also responsible
 *                   for computing the right offset to a given piece using the
 *                   base pointer.
 *                   This class is basically a ledger with all entries.
 *                   Lives in primitive descriptors.
 *
 * 2. registrar_t -- the interface to a registry_t to book memory. Used at
 *                   primitive descriptor creation time only. Contains a
 *                   reference to the corresponding *mutable* registry.
 *                   Always modifiable.
 *                   Allows chaining (using prefixes).
 *
 * 3. grantor_t   -- the interface to a registry_t to access memory. Used at
 *                   primitive execution time only. Contains a reference to
 *                   the corresponding *constant* registry and base pointer.
 *                   Always constant.
 *                   Allows chaining (using prefixes).
 *
 * Both registrar_t and grantor_t allow chaining with extra prefix provided.
 * The feature is useful when a primitive offload a part of computations to
 * some other primitives which require their own scratchpad space
 * (e.g. reducer). Prefixes are used to avoid key collision in cases when
 * multiple sub-primitive (e.g. multiple reducers) are used.
 *
 * A short example below demonstrates how to use aforementioned classes. In it
 * the main primitive is convolution that uses scratchpad for keeping padded
 * bias. It also needs a reducer, that needs its own space as well.
 *
 *  ``` c++
 *  struct reducer_t {
 *      static void init(registrar_t &scratchpad) {
 *          // reserve space for 980*1024 floats (one page aligned)
 *          scratchpad.book<float>(key_space, 980 * 1024, 4096);
 *      }
 *
 *      void exec(const grantor_t &scratchpad) {
 *          // get the pointer to preserved space. scratchpad came from
 *          // upper primitive (convolution in this example)
 *          auto space = scratchpad.get<float>(key_reducer_space);
 *
 *          space[:] += ...;
 *      }
 *  };
 *
 *  struct conv_t {
 *      struct pd_t {
 *          void init() {
 *              registrar_t scratchpad(scratchpad_registry_);
 *
 *              // reserve space for 128 elements which are two bytes long that
 *              // require 4 byte alignment, but preferably have 64 byte
 *              // alignment for performance reasons
 *              // two alignment parameters are included for implementation
 *              // flexibility targeted at memory debugging purposes
 *              scratchpad.book(key_conv_padded_bias, 128, 2, 4, 64);
 *
 *              // create a proxy registrar for the reducer All entries made
 *              // by reducer would live in convolution's registry, but would
 *              // have their own `prefix`, so no interference with conv's
 *              // buffers.
 *              registrar_t reducer_scratchpad(scratchpad, prefix_reducer);
 *
 *              reducer_t::init(reducer_scratchpad);
 *          }
 *
 *          registry_t scratchpad_registry_;
 *      }
 *
 *      void exec() {
 *          // get the base pointer to a scratchpad memory from a user
 *          void *scratchpad_ptr = this->input(DNNL_MEM_SCRATCHPAD);
 *
 *          // create a grantor to the scratchpad (and provide the base
 *          // pointer).
 *          grantor_t scratchpad(pd()->scratchpad_registry_, scratchpad_ptr);
 *
 *          // access the padded_bias (need only key name and the grantor)
 *          auto padded_bias = scratchpad.get<float>(key_conv_padded_bias);
 *
 *          // to give the `right` grantor to reducer we need to add the
 *          // corresponding prefix, so that reducer would be able to access
 *          // its keys. The call is very similar to the one in pd_t::init
 *          // with only difference in types: grantor_t vs registrar_t.
 *          grantor_t reducer_scratchpad(scratchpad, prefix_reducer);
 *          reducer->exec(reducer_scratchpad);
 *      }
 *  };
 *  ```
 */

/* namespace with common keys and prefixes */
namespace names {
enum {
    key_none = 0,
    key_barrier,
    key_bnorm_cvt,
    key_bnorm_tmp_mean,
    key_bnorm_tmp_var,
    key_bnorm_tmp_diff_ss,
    key_bnorm_tmp_stats,
    key_bnorm_reduction,
    key_bnorm_reduction_shift,
    key_brgemm_primitive_batch,
    key_brgemm_primitive_buffer,
    key_brgemm_primitive_buffer_a,
    key_brgemm_primitive_buffer_b,
    key_brgemm_primitive_buffer_comp,
    key_brgemm_primitive_buffer_d,
    key_brgemm_primitive_zp_comp_a,
    key_brgemm_primitive_zp_comp_b,
    key_concat_iptrs,
    key_concat_istrides,
    key_concat_nelems,
    key_concat_optrs,
    key_concat_tent_dst,
    key_conv_adjusted_scales,
    key_conv_amx_inp_buffer,
    key_conv_amx_tilecfg,
    key_conv_amx_tile_buffer,
    key_conv_amx_wei_buffer,
    key_conv_amx_wsp_buffer,
    key_conv_bia_reduction,
    key_conv_bias_bf16_convert_wsp,
    key_conv_cudnn,
    key_conv_cudnn_algo,
    key_conv_cudnn_filter,
    key_conv_cudnn_temp,
    key_conv_dst_bf16_convert_wsp,
    key_conv_brgemm_addr_a,
    key_conv_brgemm_addr_b,
    key_conv_brgemm_batch,
    key_conv_brgemm_buffer,
    key_conv_brgemm_inp_buffer,
    key_conv_brgemm_inp_buffer_mask,
    key_conv_brgemm_out_buffer,
    key_conv_bwd_w_1st_bia_reorder,
    key_conv_bwd_w_1st_wei_reorder,
    key_conv_gemm_acc,
    key_conv_gemm_col,
    key_conv_gemm_imtr,
    key_conv_gemm_zp_src_comp,
    key_conv_int_dat_in_acc_dt,
    key_conv_padded_bias,
    key_conv_rtus_space,
    key_conv_store_wsp,
    key_conv_tails,
    key_conv_tr_diff_dst,
    key_conv_tr_diff_dst_bctx,
    key_conv_tr_src,
    key_conv_tr_src_bctx,
    key_conv_wei_reduction,
    key_conv_wei_reduction_bctx,
    key_conv_wei_bia_reduction,
    key_conv_wei_bia_reduction_bctx,
    key_conv_zero_point_flag,
    key_conv_zero_point_pad,
    key_conv_miopen_algo,
    key_conv_miopen_filter,
    key_deconv_bias,
    key_deconv_sum,
    key_deconv_zp,
    key_eltwise_diff_dst,
    key_eltwise_src,
    key_fusion_forward_scratchpad,
    key_fusion_inout_buffer,
    key_gemm_tmp_buffer,
    key_gemm_blocked_a,
    key_gemm_blocked_b,
    key_gemm_accumulator,
    key_gnorm_cvt,
    key_gnorm_reduction,
    key_gnorm_tmp_mean,
    key_gnorm_tmp_var,
    key_iprod_bias_bf16_convert_wsp,
    key_iprod_dst_bf16_convert_wsp,
    key_iprod_dst_reorder,
    key_iprod_int_dat_in_acc_dt,
    key_lnorm_inv_sqrtvar,
    key_lnorm_tmp_mean,
    key_lnorm_tmp_var,
    key_lnorm_tmp_diff_ss,
    key_lnorm_reduction,
    key_matmul_dst_in_acc_dt,
    key_pool_dst_bf16cvt,
    key_pool_dst_plain2blocked_cvt,
    key_pool_ind_plain2blocked_cvt,
    key_pool_src_bf16cvt,
    key_pool_src_plain2blocked_cvt,
    key_pool_reduction,
    key_precomputed_scales,
    key_prelu_reduction,
    key_reducer_space,
    key_reducer_space_bctx,
    key_reduction,
    key_reduction_1,
    key_reorder_cross_space,
    key_reorder_space,
    key_reorder_src_scales,
    key_reorder_dst_scales,
    key_reorder_wino_plain,
    key_reorder_wino_transform_space,
    key_reorder_precomputed_dst_scales,
    key_reorder_rnn_space,
    key_reorder_rnn_weights_quantization,
    key_reorder_rnn_weights_reduction,
    key_reorder_rnn_weights_transposition,
    key_reorder_rnn_weights_xf16_cvt,
    key_rnn_space,
    key_rnn_bf32_attention_trans,
    key_rnn_bf32_wei_layer_trans,
    key_rnn_bf32_wei_iter_trans,
    key_rnn_cell,
    key_rnn_diff_states,
    key_rnn_gates,
    key_rnn_gates_blocked,
    key_rnn_diff_gates,
    key_rnn_src_layer_trans,
    key_rnn_src_iter_trans,
    key_rnn_ht,
    key_rnn_diff_ht,
    key_rnn_ptrs_bia,
    key_rnn_ptrs_wei_layer,
    key_rnn_ptrs_wei_iter,
    key_rnn_ptrs_wei_projection,
    key_softmax_reduction,
    key_softmax_interim_store,
    key_sum_reduction,
    key_sum_srcs_cvt,
    key_wino_U,
    key_wino_V,
    key_wino_M,
    // These two keys should always be the last ones,
    // even though they are not in alphabetical order
    key_nested,
    key_nested_multiple,
};

enum {
    prefix_none = 0,
    prefix_fusion,
    prefix_reducer_bia,
    prefix_reducer_wei,
};
} // namespace names

// level 0: 00 00 00 xxx
// level 1: 00 00 aa xxx
// level 2: 00 aa bb xxx
// level 3: aa bb cc xxx
// max # of levels: 3 + 1 (base_level)
// here:
//      xxx        : [1 ..    MAX_KEY) : key
//      aa, bb, cc : [1 .. MAX_PREFIX) : prefixes for levels 1, 2, and 3

using key_t = uint32_t;
enum {
    MAX_KEY = (1u << 10),
    MAX_PREFIX = (1u << 7),
};

/// generates global key based on a prefix and a local key
inline key_t make_key(key_t prefix, key_t key) {
    return prefix + key;
}

/// generates global prefix based on the global parent and the local ones
inline key_t make_prefix(key_t parent_prefix, key_t prefix) {
    return MAX_PREFIX * parent_prefix + MAX_KEY * prefix;
}

struct registrar_t;
struct grantor_t;

enum { default_alignment = 128 };
inline size_t get_alignment(size_t alignment) {
    size_t minimal_alignment
            = memory_debug::is_mem_debug() ? getpagesize() : default_alignment;
    return nstl::max<size_t>(alignment, minimal_alignment);
}

inline size_t buffer_protect_size() {
    return memory_debug::is_mem_debug()
            ? memory_debug::protect_size() + getpagesize()
            : 0;
}

struct registry_t {
    struct entry_t {
        size_t offset, size, capacity, alignment;

        // apply offset and alignment + check memory_debug (host/cpu only)
        const void *compute_ptr(const void *base_ptr) const;
    };

    // perf_align is the desired alignment for performance.
    // data_align is the minimum data alignment required for functionality,
    //    this parameter is included for memory debugging purposes.
    void book(const key_t &key, size_t size, size_t data_align,
            size_t perf_align = default_alignment) {
        if (size == 0) return;
        assert(offset_map_.count(key) == 0);
        size_t alignment = memory_debug::is_mem_debug()
                ? data_align
                : nstl::max(data_align, perf_align);

        if (memory_debug::is_mem_debug() && size_ == 0)
            size_ += get_alignment(alignment) + buffer_protect_size();

        assert(alignment > 0 && (alignment & (alignment - 1)) == 0);
        size_t capacity
                = size + get_alignment(alignment) + buffer_protect_size();
        assert(capacity < (SIZE_MAX + INT_MIN));
        offset_map_[key] = entry_t {size_, size, capacity, alignment};

        size_ += capacity;
    }

    entry_t get(const key_t &key) const {
        if (size() == 0 || offset_map_.count(key) != 1)
            return entry_t {0, 0, 0, 0};
        return offset_map_.at(key);
    }

    size_t size() const { return size_; }

    registrar_t registrar();
    grantor_t grantor(const memory_storage_t *mem_storage,
            const exec_ctx_t &exec_ctx) const;

    template <typename return_type>
    class common_iterator_t {
    private:
        const void *base_ptr;
        std::unordered_map<key_t, entry_t>::const_iterator iter;

    public:
        common_iterator_t(const void *base_ptr_,
                const std::unordered_map<key_t, entry_t> &map,
                bool is_begin = true) {
            base_ptr = base_ptr_;
            if (is_begin) {
                iter = map.cbegin();
            } else {
                iter = map.cend();
            }
        }
        common_iterator_t &operator++(int) {
            iter++;
            return *this;
        }
        bool operator==(const common_iterator_t &rhs) const {
            return iter == rhs.iter;
        }
        bool operator!=(const common_iterator_t &rhs) const {
            return iter != rhs.iter;
        }
        std::pair<return_type, size_t> operator*() const {
            const entry_t &entry = iter->second;
            const void *ptr_start = entry.compute_ptr(base_ptr);
            return std::pair<return_type, size_t> {
                    (return_type)ptr_start, entry.size};
        }
    };
    typedef common_iterator_t<void *> iterator;
    typedef common_iterator_t<const void *> const_iterator;
    iterator begin(void *base_ptr_) const {
        return iterator(base_ptr_, offset_map_);
    }
    iterator end(void *base_ptr_) const {
        return iterator(base_ptr_, offset_map_, false);
    }
    const_iterator cbegin(const void *base_ptr_) const {
        return const_iterator(base_ptr_, offset_map_);
    }
    const_iterator cend(const void *base_ptr_) const {
        return const_iterator(base_ptr_, offset_map_, false);
    }

protected:
    std::unordered_map<key_t, entry_t> offset_map_;
    size_t size_ = 0;
};

struct registrar_t {
    registrar_t(registry_t &registry) : registry_(registry), prefix_(0) {}
    registrar_t(registrar_t &parent, const key_t &prefix)
        : registry_(parent.registry_)
        , prefix_(make_prefix(parent.prefix_, prefix)) {}

    void book(const key_t &key, size_t nelems, size_t data_size,
            size_t data_align = 0, size_t perf_align = default_alignment) {
        assert(nelems < (SIZE_MAX + INT_MIN));
        if (data_align == 0) data_align = data_size;
        registry_.book(make_key(prefix_, key), nelems * data_size, data_align,
                perf_align);
    }
    template <typename T>
    void book(const key_t &key, size_t nelems,
            size_t perf_align = default_alignment) {
        registry_.book(make_key(prefix_, key), nelems * sizeof(T), alignof(T),
                perf_align);
    }

    void book(const key_t &key, const registry_t &registry,
            size_t perf_align = default_alignment) {
        registry_.book(make_key(prefix_, key), registry.size(), 1, perf_align);
    }

    size_t size() const { return registry_.size(); }

protected:
    registry_t &registry_;
    const key_t prefix_;
};

struct grantor_t {
    grantor_t(const registry_t &registry,
            const memory_storage_t *base_mem_storage,
            const exec_ctx_t &exec_ctx)
        : registry_(registry)
        , prefix_(0)
        , base_mem_storage_(base_mem_storage)
        , exec_ctx_(&exec_ctx) {}
    grantor_t(const grantor_t &parent, const key_t &prefix)
        : registry_(parent.registry_)
        , prefix_(make_prefix(parent.prefix_, prefix))
        , base_mem_storage_(parent.base_mem_storage_)
        , exec_ctx_(parent.exec_ctx_) {}

    template <typename T = void>
    T *get(const key_t &key, size_t *size = nullptr) const {
        if (!base_mem_storage_) {
            assert(registry_.size() == 0);
            return nullptr;
        }
        auto e = registry_.get(make_key(prefix_, key));

        if (size) *size = e.size;
        if (e.size == 0) return nullptr;

        char *host_storage_ptr = get_host_storage_ptr(base_mem_storage_);
        char *base_ptr = host_storage_ptr + base_mem_storage_->base_offset();
        return (T *)e.compute_ptr(base_ptr);
    }

    std::unique_ptr<memory_storage_t> get_memory_storage(
            const key_t &key) const {
        if (!base_mem_storage_) {
            assert(registry_.size() == 0);
            return nullptr;
        }
        auto e = registry_.get(make_key(prefix_, key));
        if (e.size == 0) return nullptr;

        if (is_cpu_engine(base_mem_storage_)) {
            char *host_storage_ptr = get_host_storage_ptr(base_mem_storage_);
            char *base_ptr
                    = host_storage_ptr + base_mem_storage_->base_offset();
            char *aligned_ptr = (char *)e.compute_ptr(base_ptr);
            size_t aligned_offset = size_t(aligned_ptr - host_storage_ptr);
            return base_mem_storage_->get_sub_storage(aligned_offset, e.size);
        }

        const size_t aligned_offset
                = reinterpret_cast<size_t>(utils::align_ptr<char>(
                        reinterpret_cast<char *>(e.offset), e.alignment));
        assert(aligned_offset + e.size <= registry_.size());
        return base_mem_storage_->get_sub_storage(aligned_offset, e.size);
    }

    const memory_storage_t *get_base_storage() const {
        return base_mem_storage_;
    }
    const registry_t &get_registry() const { return registry_; }

protected:
    const registry_t &registry_;
    const key_t prefix_;
    const memory_storage_t *base_mem_storage_;
    const exec_ctx_t *exec_ctx_;

private:
    char *get_host_storage_ptr(const memory_storage_t *storage) const;
    bool is_cpu_engine(const memory_storage_t *mem_storage) const;
};

inline registrar_t registry_t::registrar() {
    return registrar_t(*this);
}
inline grantor_t registry_t::grantor(
        const memory_storage_t *mem_storage, const exec_ctx_t &exec_ctx) const {
    return grantor_t(*this, mem_storage, exec_ctx);
}

} // namespace memory_tracking
} // namespace impl
} // namespace dnnl

#endif
