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

#ifndef COMMON_RNN_PD_HPP
#define COMMON_RNN_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "rnn.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define VDISPATCH_RNN(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, rnn, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_RNN_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, rnn, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct rnn_fwd_pd_t;

struct rnn_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::rnn;

    const rnn_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::cell_kind:
                *(alg_kind_t *)result = desc()->cell_kind;
                break;
            case query::activation_kind:
                *(alg_kind_t *)result = desc()->activation_kind;
                break;
            case query::direction:
                *(rnn_direction_t *)result = desc()->direction;
                break;
            case query::alpha_f32: *(float *)result = desc()->alpha; break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->src_layer_desc : &src_layer_md_;
        if (index == 1 && with_src_iter())
            return user_input ? &desc()->src_iter_desc : &src_iter_md_;
        if (index == 2 && with_src_iter_c())
            return user_input ? &desc()->src_iter_c_desc : &src_iter_c_md_;
        return &glob_zero_md;
    }

    memory_desc_t &augru_attention_md() {
        if (with_augru_attention()) return weights_peephole_md_;
        return glob_zero_md;
    }

    const memory_desc_t &const_augru_attention_md() const {
        if (with_augru_attention()) return weights_peephole_md_;
        return glob_zero_md;
    }

    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->weights_layer_desc
                              : &weights_layer_md_;
        if (index == 1)
            return user_input ? &desc()->weights_iter_desc : &weights_iter_md_;

        const int peephole_index = 2;
        if (is_lstm_peephole() && index == peephole_index)
            return user_input ? &desc()->weights_peephole_desc
                              : &weights_peephole_md_;

        const int projection_index = 2 + is_lstm_peephole();
        if (is_lstm_projection() && index == projection_index)
            return user_input ? &desc()->weights_projection_desc
                              : &weights_projection_md_;

        const int bias_index = 2 + is_lstm_peephole() + is_lstm_projection();
        if (with_bias() && index == bias_index)
            return user_input ? &desc()->bias_desc : &bias_md_;

        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->dst_layer_desc : &dst_layer_md_;
        if (index == 1 && with_dst_iter())
            return user_input ? &desc()->dst_iter_desc : &dst_iter_md_;
        if (index == 2 && with_dst_iter_c())
            return user_input ? &desc()->dst_iter_c_desc : &dst_iter_c_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *workspace_md(int index = 0) const override {
        return (index == 0) ? &ws_md_ : &glob_zero_md;
    }

    /* common aux functions */

    bool is_training() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::backward);
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    dim_t T() const { return desc_.src_layer_desc.dims[0]; }
    dim_t MB() const { return desc_.src_layer_desc.dims[1]; }

    dim_t L() const { return desc_.weights_layer_desc.dims[0]; }
    dim_t D() const { return desc_.weights_layer_desc.dims[1]; }

    dim_t SIC() const { return desc_.weights_iter_desc.dims[2]; }

    dim_t SLC() const { return desc_.weights_layer_desc.dims[2]; }
    dim_t G() const { return desc_.weights_layer_desc.dims[3]; }
    dim_t DHC() const { return desc_.weights_layer_desc.dims[4]; }

    // Returns the number of channels for the iter tensor.
    // Must be equal to the dst_iter.dims[3] if dst_iter is not zero.
    dim_t DIC() const {
        return is_lstm_projection() ? desc_.weights_projection_desc.dims[3]
                                    : DHC();
    }

    dim_t DLC() const { return desc_.dst_layer_desc.dims[2]; }

    bool with_bias() const {
        return !memory_desc_wrapper(desc_.bias_desc).is_zero();
    }

    bool with_augru_attention() const { return is_augru(); }

    bool with_src_iter() const {
        return !(memory_desc_wrapper(desc_.src_iter_desc).is_zero());
    }

    bool with_src_iter_c() const {
        return is_lstm()
                && !(memory_desc_wrapper(desc_.src_iter_desc).is_zero());
    }

    bool with_dst_iter() const {
        return !memory_desc_wrapper(desc_.dst_iter_desc).is_zero();
    }

    bool with_dst_iter_c() const {
        return is_lstm() && !memory_desc_wrapper(desc_.dst_iter_desc).is_zero();
    }

    dnnl::impl::alg_kind_t cell_kind() const { return desc_.cell_kind; }
    dnnl::impl::alg_kind_t activation_kind() const {
        return desc_.activation_kind;
    }

    bool is_lbr() const {
        return utils::one_of(cell_kind(), dnnl_lbr_gru, dnnl_lbr_augru);
    }

    bool is_augru() const {
        return utils::one_of(cell_kind(), dnnl_vanilla_augru, dnnl_lbr_augru);
    }

    bool is_lstm() const { return cell_kind() == dnnl_vanilla_lstm; }

    bool is_lstm_peephole() const {
        return is_lstm()
                && !memory_desc_wrapper(weights_peephole_md_).is_zero();
    }

    bool is_lstm_projection() const {
        return !memory_desc_wrapper(weights_projection_md_).is_zero();
    }

    bool diff_weights_overwrite() const {
        return desc_.flags & rnn_flags::diff_weights_overwrite;
    }

    dnnl_rnn_direction_t direction() const { return desc_.direction; }

protected:
    rnn_desc_t desc_;
    const rnn_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t src_layer_md_;
    memory_desc_t src_iter_md_;
    memory_desc_t src_iter_c_md_;
    memory_desc_t weights_layer_md_;
    memory_desc_t weights_iter_md_;
    memory_desc_t weights_peephole_md_;
    memory_desc_t weights_projection_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_layer_md_;
    memory_desc_t dst_iter_md_;
    memory_desc_t dst_iter_c_md_;

    memory_desc_t ws_md_;

    rnn_pd_t(const rnn_desc_t *adesc, const primitive_attr_t *attr,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , src_layer_md_(desc_.src_layer_desc)
        , src_iter_md_(desc_.src_iter_desc)
        , src_iter_c_md_(desc_.src_iter_c_desc)
        , weights_layer_md_(desc_.weights_layer_desc)
        , weights_iter_md_(desc_.weights_iter_desc)
        , weights_peephole_md_(desc_.weights_peephole_desc)
        , weights_projection_md_(desc_.weights_projection_desc)
        , bias_md_(desc_.bias_desc)
        , dst_layer_md_(desc_.dst_layer_desc)
        , dst_iter_md_(desc_.dst_iter_desc)
        , dst_iter_c_md_(desc_.dst_iter_c_desc)
        , ws_md_() {}
};

struct rnn_fwd_pd_t : public rnn_pd_t {
    typedef rnn_fwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC_LAYER) return arg_usage_t::input;

        if (arg == DNNL_ARG_AUGRU_ATTENTION && with_augru_attention())
            return arg_usage_t::input;

        if (arg == DNNL_ARG_SRC_ITER && with_src_iter())
            return arg_usage_t::input;

        if (arg == DNNL_ARG_SRC_ITER_C && with_src_iter_c())
            return arg_usage_t::input;

        if (utils::one_of(arg, DNNL_ARG_WEIGHTS_LAYER, DNNL_ARG_WEIGHTS_ITER))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_WEIGHTS_PEEPHOLE && is_lstm_peephole())
            return arg_usage_t::input;

        if (arg == DNNL_ARG_WEIGHTS_PROJECTION && is_lstm_projection())
            return arg_usage_t::input;

        if (arg == DNNL_ARG_BIAS && with_bias()) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST_LAYER) return arg_usage_t::output;

        if (arg == DNNL_ARG_DST_ITER && with_dst_iter())
            return arg_usage_t::output;

        if (arg == DNNL_ARG_DST_ITER_C && with_dst_iter() && is_lstm())
            return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && is_training())
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC_LAYER: return src_md(0);
            case DNNL_ARG_AUGRU_ATTENTION: return &const_augru_attention_md();
            case DNNL_ARG_SRC_ITER: return src_md(1);
            case DNNL_ARG_SRC_ITER_C: return src_md(2);
            case DNNL_ARG_WEIGHTS_LAYER: return weights_md(0);
            case DNNL_ARG_WEIGHTS_ITER: return weights_md(1);
            case DNNL_ARG_WEIGHTS_PEEPHOLE:
                return is_lstm_peephole() ? weights_md(2) : &glob_zero_md;
            case DNNL_ARG_WEIGHTS_PROJECTION:
                return is_lstm_projection() ? weights_md(2 + is_lstm_peephole())
                                            : &glob_zero_md;
            case DNNL_ARG_BIAS:
                return weights_md(
                        2 + is_lstm_peephole() + is_lstm_projection());
            case DNNL_ARG_DST_LAYER: return dst_md(0);
            case DNNL_ARG_DST_ITER: return dst_md(1);
            case DNNL_ARG_DST_ITER_C: return dst_md(2);
            default: return rnn_pd_t::arg_md(arg);
        }
    }

    int n_inputs() const override {
        return 3 + is_lstm_peephole() + is_lstm_projection() + with_bias()
                + with_src_iter() + with_src_iter_c() + is_augru();
    }
    int n_outputs() const override {
        return 1 + with_dst_iter() + with_dst_iter_c() + is_training();
    }

protected:
    rnn_fwd_pd_t(const rnn_desc_t *adesc, const primitive_attr_t *attr,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_pd_t(adesc, attr, hint_fwd_pd) {}
};

struct rnn_bwd_pd_t : public rnn_pd_t {
    typedef rnn_bwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC_LAYER, DNNL_ARG_DST_LAYER,
                    DNNL_ARG_DIFF_DST_LAYER, DNNL_ARG_WEIGHTS_LAYER,
                    DNNL_ARG_WEIGHTS_ITER))
            return arg_usage_t::input;

        if (utils::one_of(arg, DNNL_ARG_DIFF_SRC_LAYER,
                    DNNL_ARG_DIFF_WEIGHTS_LAYER, DNNL_ARG_DIFF_WEIGHTS_ITER))
            return arg_usage_t::output;

        if (with_augru_attention()) {
            if (arg == DNNL_ARG_AUGRU_ATTENTION) return arg_usage_t::input;
            if (arg == DNNL_ARG_DIFF_AUGRU_ATTENTION)
                return arg_usage_t::output;
        }

        if (is_lstm_peephole()) {
            if (arg == DNNL_ARG_WEIGHTS_PEEPHOLE) return arg_usage_t::input;

            if (arg == DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE)
                return arg_usage_t::output;
        }

        if (is_lstm_projection()) {
            if (arg == DNNL_ARG_WEIGHTS_PROJECTION) return arg_usage_t::input;

            if (arg == DNNL_ARG_DIFF_WEIGHTS_PROJECTION)
                return arg_usage_t::output;
        }

        if (with_bias()) {
            if (arg == DNNL_ARG_BIAS) return arg_usage_t::input;

            if (arg == DNNL_ARG_DIFF_BIAS) return arg_usage_t::output;
        }

        if (with_src_iter()) {
            if (arg == DNNL_ARG_SRC_ITER) return arg_usage_t::input;

            if (arg == DNNL_ARG_DIFF_SRC_ITER) return arg_usage_t::output;
        }

        if (with_src_iter_c()) {
            if (arg == DNNL_ARG_SRC_ITER_C) return arg_usage_t::input;

            if (arg == DNNL_ARG_DIFF_SRC_ITER_C) return arg_usage_t::output;
        }

        if (with_dst_iter()
                && utils::one_of(
                        arg, DNNL_ARG_DST_ITER, DNNL_ARG_DIFF_DST_ITER))
            return arg_usage_t::input;

        if (with_dst_iter_c()
                && utils::one_of(
                        arg, DNNL_ARG_DST_ITER_C, DNNL_ARG_DIFF_DST_ITER_C))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_WORKSPACE) return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC_LAYER: return src_md(0);
            case DNNL_ARG_AUGRU_ATTENTION: return &const_augru_attention_md();
            case DNNL_ARG_SRC_ITER: return src_md(1);
            case DNNL_ARG_SRC_ITER_C: return src_md(2);
            case DNNL_ARG_DIFF_SRC_LAYER: return diff_src_md(0);
            case DNNL_ARG_DIFF_AUGRU_ATTENTION:
                return &const_diff_augru_attention_md();
            case DNNL_ARG_DIFF_SRC_ITER: return diff_src_md(1);
            case DNNL_ARG_DIFF_SRC_ITER_C: return diff_src_md(2);
            case DNNL_ARG_WEIGHTS_LAYER: return weights_md(0);
            case DNNL_ARG_WEIGHTS_ITER: return weights_md(1);
            case DNNL_ARG_WEIGHTS_PEEPHOLE:
                return is_lstm_peephole() ? weights_md(2) : &glob_zero_md;
            case DNNL_ARG_WEIGHTS_PROJECTION:
                return is_lstm_projection() ? weights_md(2 + is_lstm_peephole())
                                            : &glob_zero_md;
            case DNNL_ARG_BIAS:
                return weights_md(
                        2 + is_lstm_peephole() + is_lstm_projection());
            case DNNL_ARG_DIFF_WEIGHTS_LAYER: return diff_weights_md(0);
            case DNNL_ARG_DIFF_WEIGHTS_ITER: return diff_weights_md(1);
            case DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE:
                return is_lstm_peephole() ? diff_weights_md(2) : &glob_zero_md;
            case DNNL_ARG_DIFF_WEIGHTS_PROJECTION:
                return is_lstm_projection()
                        ? diff_weights_md(2 + is_lstm_peephole())
                        : &glob_zero_md;
            case DNNL_ARG_DIFF_BIAS:
                return diff_weights_md(
                        2 + is_lstm_peephole() + is_lstm_projection());
            case DNNL_ARG_DST_LAYER: return dst_md(0);
            case DNNL_ARG_DST_ITER: return dst_md(1);
            case DNNL_ARG_DST_ITER_C: return dst_md(2);
            case DNNL_ARG_DIFF_DST_LAYER: return diff_dst_md(0);
            case DNNL_ARG_DIFF_DST_ITER: return diff_dst_md(1);
            case DNNL_ARG_DIFF_DST_ITER_C: return diff_dst_md(2);
            default: return rnn_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *diff_src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_src_layer_desc
                              : &diff_src_layer_md_;
        if (index == 1 && with_src_iter())
            return user_input ? &desc()->diff_src_iter_desc
                              : &diff_src_iter_md_;
        if (index == 2 && with_src_iter_c())
            return user_input ? &desc()->diff_src_iter_c_desc
                              : &diff_src_iter_c_md_;
        return &glob_zero_md;
    }
    memory_desc_t &diff_augru_attention_md() {
        if (with_augru_attention()) return diff_weights_peephole_md_;
        return glob_zero_md;
    }
    const memory_desc_t &const_diff_augru_attention_md() const {
        if (with_augru_attention()) return diff_weights_peephole_md_;
        return glob_zero_md;
    }
    const memory_desc_t *diff_weights_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_weights_layer_desc
                              : &diff_weights_layer_md_;
        if (index == 1)
            return user_input ? &desc()->diff_weights_iter_desc
                              : &diff_weights_iter_md_;

        const int peephole_index = 2;
        if (is_lstm_peephole() && index == peephole_index)
            return user_input ? &desc()->diff_weights_peephole_desc
                              : &diff_weights_peephole_md_;

        const int projection_index = 2 + is_lstm_peephole();
        if (is_lstm_projection() && index == projection_index)
            return user_input ? &desc()->diff_weights_projection_desc
                              : &diff_weights_projection_md_;

        const int bias_index = 2 + is_lstm_peephole() + is_lstm_projection();
        if (with_bias() && index == bias_index)
            return user_input ? &desc()->diff_bias_desc : &diff_bias_md_;

        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_dst_layer_desc
                              : &diff_dst_layer_md_;
        if (index == 1 && with_dst_iter())
            return user_input ? &desc()->diff_dst_iter_desc
                              : &diff_dst_iter_md_;
        if (index == 2 && with_dst_iter_c())
            return user_input ? &desc()->diff_dst_iter_c_desc
                              : &diff_dst_iter_c_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override {
        return 6 + with_src_iter() + with_src_iter_c()
                + 2 * (with_dst_iter() + with_dst_iter_c()) + is_lstm_peephole()
                + is_lstm_projection() + with_bias() + is_augru();
    }
    int n_outputs() const override {
        return 3 + with_src_iter() + with_src_iter_c() + is_lstm_peephole()
                + is_lstm_projection() + with_bias() + is_augru();
    }

protected:
    memory_desc_t diff_src_layer_md_;
    memory_desc_t diff_src_iter_md_;
    memory_desc_t diff_src_iter_c_md_;
    memory_desc_t diff_weights_layer_md_;
    memory_desc_t diff_weights_iter_md_;
    memory_desc_t diff_weights_peephole_md_;
    memory_desc_t diff_weights_projection_md_;
    memory_desc_t diff_bias_md_;
    memory_desc_t diff_dst_layer_md_;
    memory_desc_t diff_dst_iter_md_;
    memory_desc_t diff_dst_iter_c_md_;

    rnn_bwd_pd_t(const rnn_desc_t *adesc, const primitive_attr_t *attr,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_layer_md_(desc_.diff_src_layer_desc)
        , diff_src_iter_md_(desc_.diff_src_iter_desc)
        , diff_src_iter_c_md_(desc_.diff_src_iter_c_desc)
        , diff_weights_layer_md_(desc_.diff_weights_layer_desc)
        , diff_weights_iter_md_(desc_.diff_weights_iter_desc)
        , diff_weights_peephole_md_(desc_.diff_weights_peephole_desc)
        , diff_weights_projection_md_(desc_.diff_weights_projection_desc)
        , diff_bias_md_(desc_.diff_bias_desc)
        , diff_dst_layer_md_(desc_.diff_dst_layer_desc)
        , diff_dst_iter_md_(desc_.diff_dst_iter_desc)
        , diff_dst_iter_c_md_(desc_.diff_dst_iter_c_desc) {}
};

} // namespace impl
} // namespace dnnl

#endif
