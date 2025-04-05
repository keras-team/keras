/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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

#ifndef CPU_RNN_CPU_RNN_PD_HPP
#define CPU_RNN_CPU_RNN_PD_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/rnn_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_rnn_fwd_pd_t : public rnn_fwd_pd_t {
    using rnn_fwd_pd_t::rnn_fwd_pd_t;

protected:
    status_t set_default_params() {
        using namespace format_tag;
        if (src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
        if (dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

        if (is_augru()) {
            if (augru_attention_md().format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(augru_attention_md(), tnc));
        }

        // Optional parameters
        if (with_src_iter() && src_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
        if (with_src_iter_c() && src_iter_c_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
        if (is_lstm_peephole()
                && weights_peephole_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(weights_peephole_md_, ldgo));
        if (with_bias() && bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
        if (with_dst_iter() && dst_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
        if (with_dst_iter_c() && dst_iter_c_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

        return status::success;
    }

    status_t check_layout_consistency(bool is_brgemm) {
        using namespace format_tag;
        using namespace data_type;
        using namespace types;

        const auto is_blocked = [&](const memory_desc_t &md, int ndims,
                                        bool require_last_dim_contiguous) {
            return md.format_kind == format_kind::blocked && md.ndims == ndims
                    && IMPLICATION(require_last_dim_contiguous,
                            md.format_desc.blocking.strides[md.ndims - 1] == 1);
        };

        bool ok = true;
        ok = ok && is_blocked(src_layer_md_, 3, true)
                && is_blocked(dst_layer_md_, 3, true);
        ok = ok
                && IMPLICATION(!is_zero_md(&src_iter_md_),
                        is_blocked(src_iter_md_, 4, true))
                && IMPLICATION(!is_zero_md(&src_iter_c_md_),
                        is_blocked(src_iter_c_md_, 4, true))
                && IMPLICATION(!is_zero_md(&dst_iter_md_),
                        is_blocked(dst_iter_md_, 4, true))
                && IMPLICATION(!is_zero_md(&dst_iter_c_md_),
                        is_blocked(dst_iter_c_md_, 4, true));

        if (weights_layer_md_.format_kind == format_kind::rnn_packed)
            ok = ok
                    && (weights_layer_md_.format_desc.rnn_packed_desc.format
                            == rnn_packed_memory_format_t::ldigo_p);
        else
            ok = ok
                    && (rnn_utils::is_ldigo(&weights_layer_md_)
                            || rnn_utils::is_ldigo_blocked(&weights_layer_md_));

        if (weights_iter_md_.format_kind == format_kind::rnn_packed)
            ok = ok
                    && (weights_iter_md_.format_desc.rnn_packed_desc.format
                            == rnn_packed_memory_format_t::ldigo_p);
        else
            ok = ok
                    && (rnn_utils::is_ldigo(&weights_iter_md_)
                            || rnn_utils::is_ldigo_blocked(&weights_iter_md_));

        ok = ok
                && IMPLICATION(is_lstm_peephole(),
                        memory_desc_matches_tag(weights_peephole_md_, ldgo));

        if (is_lstm_projection()) {
            if (weights_projection_md_.format_kind == format_kind::rnn_packed)
                ok = ok
                        && (weights_projection_md_.format_desc.rnn_packed_desc
                                        .format
                                == rnn_packed_memory_format_t::ldio_p);
            else
                ok = ok
                        && (rnn_utils::is_ldio(&weights_projection_md_)
                                || rnn_utils::is_ldio_blocked(
                                        &weights_projection_md_));
        }

        ok = ok
                && IMPLICATION(
                        with_bias(), memory_desc_matches_tag(bias_md_, ldgo));

        /* Int8 is supported only for packed weights, if not BRGEMM version */
        const data_type_t weights_iter_dt = weights_iter_md_.data_type;
        const data_type_t weights_layer_dt = weights_layer_md_.data_type;
        if (!rnn_utils::is_ldigo_blocked(&weights_iter_md_))
            ok = ok
                    && IMPLICATION(weights_iter_dt == s8,
                            weights_iter_md_.format_kind
                                    == format_kind::rnn_packed);
        if (!rnn_utils::is_ldigo_blocked(&weights_layer_md_))
            ok = ok
                    && IMPLICATION(weights_layer_dt == s8,
                            weights_layer_md_.format_kind
                                    == format_kind::rnn_packed);
        return ok ? status::success : status::unimplemented;
    }
};

struct cpu_rnn_bwd_pd_t : public rnn_bwd_pd_t {
    using rnn_bwd_pd_t::rnn_bwd_pd_t;

protected:
    status_t set_default_params() {
        using namespace format_tag;
        if (src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
        if (dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

        if (is_augru()) {
            if (augru_attention_md().format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(augru_attention_md(), tnc));
            if (diff_augru_attention_md().format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(diff_augru_attention_md(), tnc));
        }

        if (diff_src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_src_layer_md_, tnc));
        if (diff_weights_layer_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(diff_weights_layer_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_layer_md_, ldigo));
        }
        if (diff_weights_iter_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(diff_weights_iter_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_iter_md_, ldigo));
        }
        if (diff_dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_layer_md_, tnc));

        // Optional parameters
        if (with_src_iter() && src_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
        if (with_src_iter_c() && src_iter_c_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
        if (is_lstm_peephole()
                && weights_peephole_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(weights_peephole_md_, ldgo));
        if (is_lstm_projection()
                && weights_projection_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(weights_projection_md_, ldoi));
        if (with_bias() && bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
        if (with_dst_iter() && dst_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
        if (with_dst_iter_c() && dst_iter_c_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

        if (with_src_iter()
                && diff_src_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_src_iter_md_, ldnc));
        if (with_src_iter_c()
                && diff_src_iter_c_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_src_iter_c_md_, ldnc));
        if (is_lstm_peephole()
                && diff_weights_peephole_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_weights_peephole_md_, ldgo));
        if (is_lstm_projection()
                && diff_weights_projection_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_weights_projection_md_, ldio));
        if (with_bias() && diff_bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md_, ldgo));
        if (with_dst_iter()
                && diff_dst_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_iter_md_, ldnc));
        if (with_dst_iter_c()
                && diff_dst_iter_c_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_iter_c_md_, ldnc));

        return status::success;
    }

    status_t check_layout_consistency(bool is_brgemm) {
        using namespace format_tag;
        using namespace types;

        const auto is_blocked = [&](const memory_desc_t &md, int ndims,
                                        bool require_last_dim_contiguous) {
            return md.format_kind == format_kind::blocked && md.ndims == ndims
                    && IMPLICATION(require_last_dim_contiguous,
                            md.format_desc.blocking.strides[md.ndims - 1] == 1);
        };

        bool ok = true;
        ok = ok && is_blocked(src_layer_md_, 3, true)
                && is_blocked(dst_layer_md_, 3, true);
        ok = ok
                && IMPLICATION(!is_zero_md(&src_iter_md_),
                        is_blocked(src_iter_md_, 4, true))
                && IMPLICATION(!is_zero_md(&src_iter_c_md_),
                        is_blocked(src_iter_c_md_, 4, true))
                && IMPLICATION(!is_zero_md(&dst_iter_md_),
                        is_blocked(dst_iter_md_, 4, true))
                && IMPLICATION(!is_zero_md(&dst_iter_c_md_),
                        is_blocked(dst_iter_c_md_, 4, true));

        const auto check_weights_consistency =
                [&](const memory_desc_t &weights_md) {
                    if (weights_md.format_kind == format_kind::rnn_packed)
                        return ok
                                && weights_md.format_desc.rnn_packed_desc.format
                                == rnn_packed_memory_format_t::ldgoi_p;
                    else if (is_brgemm)
                        return ok && rnn_utils::is_ldgoi_blocked(&weights_md);
                    else
                        return ok && rnn_utils::is_ldgoi(&weights_md);
                };

        ok = check_weights_consistency(weights_layer_md_);
        ok = check_weights_consistency(weights_iter_md_);

        ok = ok
                && IMPLICATION(is_augru(),
                        memory_desc_matches_tag(augru_attention_md(), tnc));
        ok = ok
                && IMPLICATION(is_lstm_peephole(),
                        memory_desc_matches_tag(weights_peephole_md_, ldgo));
        ok = ok
                && IMPLICATION(is_lstm_projection(),
                        memory_desc_matches_tag(weights_projection_md_, ldoi));
        ok = ok
                && IMPLICATION(
                        with_bias(), memory_desc_matches_tag(bias_md_, ldgo));

        ok = ok && is_blocked(diff_src_layer_md_, 3, true)
                && is_blocked(diff_dst_layer_md_, 3, true);
        ok = ok
                && IMPLICATION(!is_zero_md(&diff_src_iter_md_),
                        is_blocked(diff_src_iter_md_, 4, true))
                && IMPLICATION(!is_zero_md(&diff_src_iter_c_md_),
                        is_blocked(diff_src_iter_c_md_, 4, true))
                && IMPLICATION(!is_zero_md(&diff_dst_iter_md_),
                        is_blocked(diff_dst_iter_md_, 4, true))
                && IMPLICATION(!is_zero_md(&diff_dst_iter_c_md_),
                        is_blocked(diff_dst_iter_c_md_, 4, true));

        ok = ok
                && IMPLICATION(is_augru(),
                        memory_desc_matches_tag(
                                diff_augru_attention_md(), tnc));
        ok = ok && rnn_utils::is_ldigo(&diff_weights_layer_md_)
                && rnn_utils::is_ldigo(&diff_weights_iter_md_);
        ok = ok
                && IMPLICATION(is_lstm_peephole()
                                && !is_zero_md(&diff_weights_peephole_md_),
                        memory_desc_matches_tag(
                                diff_weights_peephole_md_, ldgo));
        ok = ok
                && IMPLICATION(!is_zero_md(&diff_weights_projection_md_),
                        memory_desc_matches_tag(
                                diff_weights_projection_md_, ldio));
        ok = ok
                && IMPLICATION(!is_zero_md(&diff_bias_md_),
                        memory_desc_matches_tag(diff_bias_md_, ldgo));

        return ok ? status::success : status::unimplemented;
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
