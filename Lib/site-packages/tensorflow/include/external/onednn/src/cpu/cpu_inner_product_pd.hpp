/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef CPU_CPU_INNER_PRODUCT_PD_HPP
#define CPU_CPU_INNER_PRODUCT_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/inner_product_pd.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
inline bool dense_gemm_consitency_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace utils;

    auto strides_compatible = [&]() {
        bool ok = true;
        auto w_str = wei_d.blocking_desc().strides;
        auto d_str = src_d.blocking_desc().strides;
        for (int i = 1; i < src_d.ndims() - 1; i++) {
            ok = ok && w_str[i] / d_str[i] == w_str[i + 1] / d_str[i + 1];
        }
        return ok && one_of(w_str[1] / d_str[1], 1, wei_d.padded_dims()[0]);
    };

    auto inner_blk_compatible = [&]() {
        auto d_inner_blks = src_d.blocking_desc().inner_blks;
        auto w_inner_blks = wei_d.blocking_desc().inner_blks;
        auto d_inner_idxs = src_d.blocking_desc().inner_idxs;
        auto w_inner_idxs = wei_d.blocking_desc().inner_idxs;

        int d_inner_nblks = src_d.blocking_desc().inner_nblks;
        int w_inner_nblks = wei_d.blocking_desc().inner_nblks;

        bool ok = true;

        if ((wei_d.blocking_desc().strides[0] == 1) && (w_inner_nblks > 0)) {
            ok = ok && wei_d.dims()[0] / w_inner_blks[w_inner_nblks - 1] == 1
                    && w_inner_idxs[w_inner_nblks - 1] == 0;
            w_inner_nblks--;
        }
        ok = ok && d_inner_nblks == w_inner_nblks;

        for (int d = 0; d < w_inner_nblks; d++)
            ok = ok && (d_inner_blks[d] == w_inner_blks[d])
                    && (d_inner_idxs[d] == w_inner_idxs[d]);

        return ok;
    };

    return true && src_d.is_blocking_desc() && wei_d.is_blocking_desc()
            && src_d.ndims() == wei_d.ndims() && inner_blk_compatible()
            && strides_compatible() && dst_d.matches_tag(format_tag::nc)
            && src_d.only_padded_dim(1) && wei_d.only_padded_dim(1)
            && src_d.padded_dims()[1] == wei_d.padded_dims()[1]
            && src_d.is_dense(true) && dst_d.is_dense() && wei_d.is_dense(true);
}

void transpose_md(memory_desc_t &md) {
    // Note: we cannot directly use good leading dimension for a
    // in padded_dims.  This is because inner_blks does not
    // account for padding, and should divide the corresponding
    // padded_dim.
    auto put_a_last = [](memory_desc_t &md) {
        auto &md_blk = md.format_desc.blocking;
        md.padded_dims[0] = md.dims[0];
        md_blk.strides[0] = 1;
        for (int d = 1; d < md.ndims; d++)
            md_blk.strides[d] *= md.padded_dims[0];
        if (md_blk.inner_nblks > 0) {
            md_blk.inner_idxs[md_blk.inner_nblks] = 0;
            md_blk.inner_blks[md_blk.inner_nblks] = md.padded_dims[0];
            md_blk.inner_nblks++;
        }
    };

    auto put_a_first = [](memory_desc_t &md) {
        blocking_desc_t blk = md.format_desc.blocking;
        // make the stride for `a` bigger than any other stride and
        // use the fact that memory_desc_init_by_blocking_desc
        // preserves the strides order but actually changes them to
        // densify the descriptor
        blk.strides[0] = memory_desc_wrapper(md).size();
        memory_desc_init_by_blocking_desc(md, blk);
    };

    auto is_a_last = [](memory_desc_t &md) {
        auto &md_blk = md.format_desc.blocking;
        // The inner_blks condition makes sure that a is a non blocked dimension
        return (md_blk.strides[0] == 1) && (md_blk.inner_nblks == 0);
    };

    auto is_a_first = [&](memory_desc_t &md) {
        auto &md_blk = md.format_desc.blocking;
        for (int d = 1; d < md.ndims; d++)
            if (md_blk.strides[0] < md_blk.strides[d]) return false;
        return true;
    };

    if (is_a_last(md))
        put_a_first(md);
    else if (is_a_first(md))
        put_a_last(md);

    // here, by default we do not transpose md if it is not
}

format_tag_t get_tag(memory_desc_t &md) {
    using namespace format_tag;
    auto tag = memory_desc_matches_one_of_tag(md, ab, abc, abcd,
            abcde, // NCHW derivatives
            ba, bca, bcda, bcdea, cba, cdba,
            cdeba, // IO and spatial derivatives
            acb, acdb, acdeb, // NHWC derivatives
            aBcd16b, aBcde16b, aBcd8b, aBcde8b, aBcd4b,
            aBcde4b); // blocked layouts
    return tag;
}

inline bool is_ineff_lead_dim(const dim_t dim) {
    return dim % 1024 == 0; // check cache aliasing
}

/* Pick between M and K for the most efficient leading
 * dimension to compute GeMM. */
bool transpose_leading_dim(const dim_t M, const dim_t K) {
    return IMPLICATION(is_ineff_lead_dim(M), is_ineff_lead_dim(K) && M <= K);
}
} // namespace

#define INIT_MEM_BY_TAG(tag_init_f, md) \
    do { \
        auto tag = tag_init_f; \
        if (tag == format_tag::undef) return status::unimplemented; \
        CHECK(memory_desc_init_by_tag(md, tag)); \
    } while (0)

struct cpu_inner_product_fwd_pd_t : public inner_product_fwd_pd_t {
    using inner_product_fwd_pd_t::inner_product_fwd_pd_t;

protected:
    status_t set_default_params(bool allow_all_tags = false) {
        using namespace format_tag;

        auto set_default_src = [&]() {
            if (weights_md_.format_kind == format_kind::any) {
                INIT_MEM_BY_TAG(utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                        src_md_);
            } else {
                format_tag_t weights_tag = get_tag(weights_md_);
                if (allow_all_tags && weights_tag == undef) {
                    INIT_MEM_BY_TAG(
                            utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                            src_md_);
                } else {
                    INIT_MEM_BY_TAG(weights_tag, src_md_);
                }
                // transpose weights to improve efficiency of non-copy kernels
                if (src_md_.format_desc.blocking.strides[0] == 1)
                    transpose_md(src_md_);
            }
            return status::success;
        };

        auto set_default_weights = [&]() {
            format_tag_t src_tag = get_tag(src_md_);
            if (allow_all_tags && src_tag == undef) {
                INIT_MEM_BY_TAG(utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                        weights_md_);
            } else {
                INIT_MEM_BY_TAG(src_tag, weights_md_);
            }
            /* with batch = 1, no transpose to use the faster gemv kernels */
            /* otherwise, we transpose the weights to improve efficiency of
             * no-copy kernels */
            if (MB() > 1 && transpose_leading_dim(OC(), IC_total()))
                transpose_md(weights_md_);
            return status::success;
        };

        if (src_md_.format_kind == format_kind::any) CHECK(set_default_src());
        if (weights_md_.format_kind == format_kind::any)
            CHECK(set_default_weights());
        if (dst_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_md_, nc));
        if (bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, x));
        return status::success;
    }
};

struct cpu_inner_product_bwd_data_pd_t : public inner_product_bwd_data_pd_t {
    using inner_product_bwd_data_pd_t::inner_product_bwd_data_pd_t;

protected:
    status_t set_default_params(bool allow_all_tags = false) {
        using namespace format_tag;

        auto set_default_diff_src = [&]() {
            if (weights_md_.format_kind == format_kind::any) {
                INIT_MEM_BY_TAG(utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                        diff_src_md_);
            } else {
                format_tag_t weights_tag = get_tag(weights_md_);
                if (allow_all_tags && weights_tag == undef) {
                    INIT_MEM_BY_TAG(
                            utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                            diff_src_md_);
                } else {
                    INIT_MEM_BY_TAG(weights_tag, diff_src_md_);
                }
                if (diff_src_md_.format_desc.blocking.strides[0] == 1)
                    transpose_md(diff_src_md_);
            }
            return status::success;
        };

        auto set_default_weights = [&]() {
            format_tag_t diff_src_tag = get_tag(diff_src_md_);
            if (allow_all_tags && diff_src_tag == undef) {
                INIT_MEM_BY_TAG(utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                        weights_md_);
            } else {
                INIT_MEM_BY_TAG(diff_src_tag, weights_md_);
            }
            /* with batch = 1, no transpose to use the faster gemv kernels */
            /* otherwise, we transpose the weights to improve efficiency of
             * no-copy kernels */
            if (MB() == 1) transpose_md(weights_md_);

            return status::success;
        };

        if (diff_src_md_.format_kind == format_kind::any)
            CHECK(set_default_diff_src());
        if (weights_md_.format_kind == format_kind::any)
            CHECK(set_default_weights());
        if (diff_dst_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_md_, nc));
        return status::success;
    }
};

struct cpu_inner_product_bwd_weights_pd_t
    : public inner_product_bwd_weights_pd_t {
    using inner_product_bwd_weights_pd_t::inner_product_bwd_weights_pd_t;

protected:
    status_t set_default_params(bool allow_all_tags = false) {
        using namespace format_tag;

        auto set_default_src = [&]() {
            if (diff_weights_md_.format_kind == format_kind::any) {
                INIT_MEM_BY_TAG(utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                        src_md_);
            } else {
                format_tag_t diff_weights_tag = get_tag(diff_weights_md_);
                if (allow_all_tags && diff_weights_tag == undef) {
                    INIT_MEM_BY_TAG(
                            utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                            src_md_);
                } else {
                    INIT_MEM_BY_TAG(diff_weights_tag, src_md_);
                }
                if (src_md_.format_desc.blocking.strides[0] == 1)
                    transpose_md(src_md_);
            }
            return status::success;
        };

        auto set_default_diff_weights = [&]() {
            format_tag_t src_tag = get_tag(src_md_);
            if (allow_all_tags && src_tag == undef) {
                INIT_MEM_BY_TAG(utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                        diff_weights_md_);
            } else {
                INIT_MEM_BY_TAG(src_tag, diff_weights_md_);
            }
            // Here, we want diff_weights layout to match the fwd weights layout
            if (MB() > 1 && transpose_leading_dim(OC(), MB()))
                transpose_md(diff_weights_md_);
            return status::success;
        };

        if (src_md_.format_kind == format_kind::any) CHECK(set_default_src());
        if (diff_weights_md_.format_kind == format_kind::any)
            CHECK(set_default_diff_weights());
        if (diff_dst_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_md_, nc));
        if (diff_bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md_, x));
        return status::success;
    }
};
#undef INIT_MEM_BY_TAG

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
