/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_JIT_UNI_REORDER_HPP
#define CPU_AARCH64_JIT_UNI_REORDER_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace tr {

constexpr int max_ndims = DNNL_MAX_NDIMS;

struct node_t {
    static constexpr int64_t empty_field = -1;

    size_t n = 0;
    size_t tail_size = 0;
    int dim_id = empty_field;
    int parent_node_id = empty_field;
    bool is_zero_pad_needed = false;
    ptrdiff_t is = 0; // input stride
    ptrdiff_t os = 0; // output stride
    ptrdiff_t ss = 0; // scale stride
    ptrdiff_t cs = 0; // compensation stride

    bool is_dim_id_empty() const { return dim_id == empty_field; }
    bool is_parent_empty() const { return parent_node_id == empty_field; }
};

enum class scale_type_t { NONE, COMMON, MANY };

struct prb_t {
    /* The compensation mask value indicates how big an additional buffer should be.
     * Possible values for reorder:
     *     1) standard compensation = 1 = 0b01
     *     2) asymmetric compensation = 2 = 0b10
     *     3) compensation if tensor contains group = 3 = 0b11 */
    static constexpr int invalid_comp_mask = 0;
    static constexpr int standard_comp_mask = 0b1;
    static constexpr int asymmetric_comp_mask = 0b10;
    static constexpr int comp_mask_with_groups
            = standard_comp_mask + asymmetric_comp_mask;

    bool is_tail_in_one_of_child_nodes(int parent_node_id) const {
        for (int i = parent_node_id; i >= 0; i--) {
            if (nodes[i].parent_node_id == parent_node_id) {
                if (nodes[i].tail_size != 0)
                    return true;
                else
                    parent_node_id = i;
            }
        }

        return false;
    }

    int tail(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].tail_size);
    }

    int n(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].n);
    }
    int is(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].is);
    }
    int os(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].os);
    }
    int ss(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].ss);
    }

    int cs(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].cs);
    }

    data_type_t itype;
    data_type_t otype;
    int ndims;
    node_t nodes[max_ndims];
    ptrdiff_t ioff;
    ptrdiff_t ooff;
    scale_type_t src_scale_type;
    scale_type_t dst_scale_type;
    float beta;
    int full_ndims;
    bool is_tail_present = false;
    float scale_adjust = 1.f;
    int compensation_mask = invalid_comp_mask;
    bool req_s8s8_comp = false;
    bool req_asymmetric_comp = false;
    bool req_src_zp = false;
    bool req_dst_zp = false;
};

status_t prb_init(prb_t &prb, const memory_desc_t &imd,
        const memory_desc_t &omd, const primitive_attr_t *attr);

/** sorts the problem nodes so that output strides come in ascending order */
void prb_normalize(prb_t &p);

/** fill parent node info for blocked nodes */
void prb_node_dependency(prb_t &p);

/** folds nodes together if possible */
void prb_simplify(prb_t &p);

/** splits the node dim into two of sizes n1 and n / n1
 * @warning n must be multiple of n1 */
void prb_node_split(prb_t &p, int dim, size_t n1);

/** swaps d0 and d1 nodes */
void prb_node_swap(prb_t &p, int d0, int d1);

/** moves node d0 to the d1 position.
 * nodes (d0, d1] are shifted to the left if d0 < d1 or
 * to the right if d0 > d1 */
void prb_node_move(prb_t &p, int d0, int d1);

/** dumps the problem to stdout */
void prb_dump(const prb_t &p);

struct call_param_t {
    const void *in = nullptr;
    void *out = nullptr;
    const float *src_scales = nullptr;
    const float *dst_scales = nullptr;
    int32_t src_zp = 0;
    int32_t dst_zp = 0;
    int32_t *compensation_scratch = nullptr;
};

// The additional structure is needed because
// using a data structure with tail processing
// data for non-tail cases reduces kernel
// performance. This is because there is too
// much data that has to be transferred to the kernel.
struct tail_call_param_t {
    call_param_t base_params;
    int64_t curr_data_chunks[DNNL_MAX_NDIMS] = {-1};
    int64_t zeroing_data = static_cast<int64_t>(false);
    int64_t skip_kernel_execution = static_cast<int64_t>(false);
};

struct kernel_t {
    struct desc_t {
        int id;
        prb_t prb;
    };

    kernel_t(const desc_t &desc)
        : desc_(desc)
        , compensation_needed_(
                  desc.prb.req_s8s8_comp || desc.prb.req_asymmetric_comp) {}
    virtual void operator()(const call_param_t *c) const = 0;
    virtual void operator()(const tail_call_param_t *c) const = 0;
    virtual status_t create_kernel() = 0;
    virtual ~kernel_t() {}

    /** inits kernel descriptor:
     *      desc            -- kernel descriptor (output)
     *      prb             -- transposition problem (input)
     *      ndims_ker_max   -- limit the maximum number of dimensions kernel
     *                         will process (optional, 0 -- no limitation) */
    static status_t desc_init(
            desc_t &desc, const prb_t &prb, int ndims_ker_max = 0);

    /** creates kernel for the problem described in desc */
    static kernel_t *create(const desc_t &desc);

protected:
    const desc_t desc_;
    const prb_t &prb_ = desc_.prb;
    bool compensation_needed_ = false;
};

/* TODO: add trans_t class */

struct jit_single_blk_kernel_t;

} // namespace tr

struct jit_uni_reorder_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
        int nthr_;
        bool with_groups_ = false;
        dim_t D_mask_ = 0;

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

    private:
        status_t init_scratchpad();
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        friend dnnl::impl::impl_list_item_t;
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    enum { ndims_driver_max = 4 };

private:
    void omp_driver_0d(int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_1d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_2d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_3d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;
    void omp_driver_4d(int ithr, int nthr, int off, const char *in, char *out,
            const float *src_scales, const float *dst_scales, int src_zp,
            int dst_zp, int32_t *compensation_scratch) const;

    void omp_driver(const char *in, char *out, const float *src_scales,
            const float *dst_scales, int src_zp, int dst_zp,
            const memory_tracking::grantor_t &scratchpad) const;

    void fill_curr_data_chunks(const tr::prb_t &prb, const int off,
            const ptrdiff_t *omp_data_chunks, const int omp_ndims,
            tr::tail_call_param_t &c) const;

    void reduce_compensation(char *out,
            const int32_t *compensation_reduce_scratch, const int nthr,
            const dim_t wspace_per_thr_size) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::kernel_t> kernel_;
};

struct jit_blk_reorder_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;
        DECLARE_COMMON_PD_T("jit:blk", jit_blk_reorder_t);

        tr::prb_t prb_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        // Swap last two nodes, put block 4, 8, 16 nodes to first
        static void prb_tile_normalize(tr::prb_t &p);
        friend dnnl::impl::impl_list_item_t;
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    jit_blk_reorder_t(const pd_t *apd);
    ~jit_blk_reorder_t();

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::jit_single_blk_kernel_t> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
