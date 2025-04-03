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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_stream.hpp"
#include "oneapi/dnnl/dnnl_threadpool.h"

#include "graph/interface/backend.hpp"
#include "graph/interface/graph.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using ltw = logical_tensor_wrapper_t;
using op_ptr = std::shared_ptr<op_t>;
using registry_key = size_t;

class sdp_reorder {
public:
    status_t init(const dnnl::reorder::primitive_desc &pd) {
        auto src_desc = pd.src_desc();
        auto dst_desc = pd.dst_desc();
        if (src_desc == dst_desc) is_inplace_ = true;
        reorder_prim_ = reorder(pd);
        return status::success;
    }
    bool get_inplace() { return is_inplace_; }
    status_t execute(const stream &astream,
            const std::unordered_map<int, memory> &args) const {
        if (is_inplace_)
            args.at(DNNL_ARG_DST)
                    .set_data_handle(args.at(DNNL_ARG_SRC).get_data_handle());
        else
            reorder_prim_.execute(astream, args);
        return status::success;
    }

private:
    primitive reorder_prim_;
    bool is_inplace_ = false;
};
struct sdp_decomp_config_t {
public:
    sdp_decomp_config_t() = default;

    // SDP input dimension
    memory::dim batch_size, num_head, seq_len_q, size_per_head;

    // SDP input and output strides
    memory::dims src1_strides, wei1_strides, wei2_strides, dst_strides,
            post_add_strides;

    // Thread nums during the workflow
    int nthr;

    // Used to record the exact input offset in subgraph
    // [mm1_src,mm1_wei,mm1_scale,mm1_add,mm2_wei,select_condition,select_other_input]
    std::vector<int> graph_inport;

    // Primitives that actually perform calculations
    primitive sub_mm1_prim, sub_softmax_prim, sub_mm2_prim;
    sdp_reorder sub_reorder0, sub_reorder1, sub_reorder2, sub_reorder3;

    // Args used in the execution of primitives
    std::unordered_map<int, memory> sub_reorder0_args, sub_reorder1_args,
            sub_mm1_args, sub_softmax_args, sub_reorder2_args, sub_mm2_args,
            sub_reorder3_args;

    // A map from memory to registry key, used to record the internal memories
    // location inside of the whole buffer.
    std::unordered_map<dnnl_memory_t, registry_key> mem_key_map;

    // Internal memory objects for each primitive in each threads.
    // reorder0
    memory sub_src1;
    // reorder1
    memory sub_wei1_user, sub_wei1_zp;
    //mm1
    memory sub_mm1_src, sub_mm1_wei, sub_mm1_dst;
    // sub_mm1_post_mem contains [post_scale, attn_mask(optional), post_binary(from select)...]
    std::vector<memory> sub_mm1_post_mem;
    //softmax
    memory sub_softmax_dst;
    //reorder2
    memory sub_wei2_user, sub_wei2_zp;
    //mm2
    memory sub_mm2_wei, sub_mm2_dst;
    //reorder3
    memory sub_dst_user;
    //scratchped
    memory sub_scratchpad;
    // shared memory
    memory sub_max_src1_src2, sub_max_dst1_wei2;

    bool attention_mask = false, has_select = false;
    // Used to record the ops from select
    std::vector<op_ptr> select_op;
    std::vector<int> select_outop_index;

private:
    // Used to record the ops contained in SDP
    // sdp_op = [reorder1, mm1, softmax, reorder2, mm2]
    // reorder1 is using mm1 weight u8->s8
    // reorder2 is using mm2 weight u8->s8
    std::vector<op_ptr> sdp_op;

public:
    // The function is used to check if the configuration of SDP is supported by
    // current implementation of decomp kernel. Currently, this implementation
    // can handle 4-dims tensor and limits the numerical relationship between
    // batch_size, num_head and thread num.
    // If the check passes, initialize few members according to inputs
    // If no, return unimplemented status directly and fallback to large kernel
    bool initial_check(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs) {
        // The order of input logical tensors in inputs is not certain, we need
        // to record the input offset in a certain order of ops.
        auto op_status = record_input_offset(sg, inputs);
        if (op_status != status::success) return false;
        memory::dims src1_user_dims = ltw(inputs[graph_inport[0]]).vdims();
        if (src1_user_dims.size() != 4) return false;

        // Initialize SDP input dimension according to the src of mm1
        batch_size = src1_user_dims[0];
        num_head = src1_user_dims[1];
        seq_len_q = src1_user_dims[2];
        size_per_head = src1_user_dims[3];

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
// RATIO is an empirical value used to determine the numerical relationship
// between batch_size, num_head and thread number to determine whether to use
// decompose kernel. The key to the decompose kernel is that we do parallel in
// the batch_size and num_head dimensions. Therefore, if the batch_size or
// num_head is too small, it will cause many idle threads and affect efficiency
// which may even worse than the original sequential kernel. Here we set this
// ratio based on the experimental value to ensure that users do not have any
// regression when using the decompose kernel.
// TODO: Refine the inequation based on the relationship of cache size and sdp
// memory footprint requirements.
#define RATIO 2
        // Initialize nthr with current threads num
        nthr = dnnl_get_current_num_threads();
        return batch_size * num_head > RATIO * nthr;
#else
        return true;
#endif
    }

    // Used to construct all params that SDP need
    template <bool quantized = false,
            memory::data_type dt = memory::data_type::f32>
    impl::status_t construct_params(std::shared_ptr<subgraph_t> &sg,
            registry_t &sdp_registry, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs) {

        // Record the ops inside of SDP pattern for later usage
        record_sdp_ops(sg, quantized);

        // Update SDPA input params. Sequence length for query and key/value are
        // NOT always same.
        memory::dim seq_len_kv;
        const auto &lt_wei
                = sdp_op[1]->get_input_value(1)->get_logical_tensor();
        const ltw ltw_wei(lt_wei);
        seq_len_kv = ltw_wei.vdims()[3];

        // Acquire the data type from input param for later primitive creation.
        // The src and wei dt of both quantized sdp and float sdp are the same.
        memory::data_type dt_src_user = static_cast<memory::data_type>(
                ltw(inputs[graph_inport[0]]).data_type());
        memory::data_type dt_wei_user = static_cast<memory::data_type>(
                ltw(inputs[graph_inport[1]]).data_type());
        memory::data_type dt_wei
                = quantized ? memory::data_type::s8 : dt_src_user;
        memory::data_type dt_inter = quantized ? dt : dt_src_user;

        ////////////////////////////////////////////////////////////////////////
        ////////////// Start Creating primitives ///////////////////////////////
        ////////////////////////////////////////////////////////////////////////
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
        // TODO: Here we create primitive with single thread, no exact reason,
        // pending on primitive investigation and fix
        omp_set_num_threads(1);
#endif
        // intermediate md used to create primitives
        memory::desc sub_src1_md, sub_wei1_user_md, sub_wei1_md, sub_mm1_src_md,
                sub_mm1_wei_md, sub_mm1_dst_md, sub_softmax_dst_md,
                sub_wei2_user_md, sub_mm2_wei_md, sub_mm2_dst_md, sub_dst_md,
                sub_dst_user_md;
        std::vector<memory::desc> sub_mm1_post_md;

        // must use user mode to support concurrent execution
        primitive_attr sub_reorder0_attr;
        sub_reorder0_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        // per-head: reorder src1 to dense, for first matmul
        memory::dims sub_src1_dims = {1, 1, seq_len_q, size_per_head};
        src1_strides = ltw(inputs[graph_inport[0]]).vstrides();
        sub_src1_md = memory::desc(sub_src1_dims, dt_src_user,
                {1, 1, src1_strides[2], src1_strides[3]});
        auto sub_src1_d_md
                = memory::desc(sub_src1_dims, dt_src_user, tag::abcd);
        auto sub_reorder0_pd = reorder::primitive_desc(p_engine, sub_src1_md,
                p_engine, sub_src1_d_md, sub_reorder0_attr);
        sub_reorder0.init(sub_reorder0_pd);

        auto &mgr = sg->fusion_info_mgr_;

        // per-head: reorder u8->s8 wei for first matmul
        // create reorder1 primitive attr
        dnnl::primitive_attr sub_reorder1_attr
                = make_primitive_attr(sdp_op[0], mgr);
        memory::dims sub_wei1_dims = {1, 1, size_per_head, seq_len_kv};
        auto wei_md = make_dnnl_memory_desc(
                sdp_op[1]->get_input_value(1)->get_logical_tensor());
        wei1_strides = wei_md.get_strides();
        sub_wei1_user_md = memory::desc(sub_wei1_dims, dt_wei_user,
                {1, 1, wei1_strides[2], wei1_strides[3]});
        // Flip the format to have `ba` weights MBI item in per thread loop.
        sub_wei1_md = memory::desc(sub_wei1_dims, dt_wei, tag::abdc);
        auto sub_reorder1_pd = reorder::primitive_desc(p_engine,
                sub_wei1_user_md, p_engine, sub_wei1_md, sub_reorder1_attr);
        sub_reorder1.init(sub_reorder1_pd);

        // first matmul
        // create first matmul primitive attr
        dnnl::primitive_attr sub_matmul1_attr
                = make_primitive_attr(sdp_op[1], mgr);
        memory::dims sub_mm1_src_dims = {1, 1, seq_len_q, size_per_head};
        memory::dims sub_mm1_wei_dims = {1, 1, size_per_head, seq_len_kv};
        memory::dims sub_mm1_dst_dims = {1, 1, seq_len_q, seq_len_kv};

        sub_mm1_src_md = memory::desc(sub_mm1_src_dims, dt_src_user, tag::abcd);
        sub_mm1_wei_md = memory::desc(sub_mm1_wei_dims, dt_wei, tag::abdc);
        sub_mm1_dst_md = memory::desc(sub_mm1_dst_dims, dt_inter, tag::abcd);
        dnnl::post_ops dnnl_pops;
        auto ori_dnnl_pops = sub_matmul1_attr.get_post_ops();
        for (int i = 0; i < ori_dnnl_pops.get()->len(); i++) {
            auto alg = static_cast<algorithm>(
                    ori_dnnl_pops.get()->entry_[i].binary.alg);
            const dnnl::impl::memory_desc_t &ori_desc
                    = ori_dnnl_pops.get()->entry_[i].binary.user_src1_desc;
            auto post_shape = ori_desc.dims;
            auto post_stride = ori_desc.format_desc.blocking.strides;
            auto post_dt = static_cast<memory::data_type>(ori_desc.data_type);
            memory::dims post_stride_dims
                    = memory::dims(post_stride, post_stride + ori_desc.ndims);
            auto new_sub_md = memory::desc({1, 1, post_shape[2], post_shape[3]},
                    post_dt, post_stride_dims);
            sub_mm1_post_md.emplace_back(new_sub_md);
            dnnl_pops.append_binary(alg, new_sub_md);
        }
        sub_matmul1_attr.set_post_ops(std::move(dnnl_pops));
        auto sub_mm1_pd = matmul::primitive_desc(p_engine, sub_mm1_src_md,
                sub_mm1_wei_md, sub_mm1_dst_md, sub_matmul1_attr);
        sub_mm1_prim = matmul(sub_mm1_pd);

        // softmax
        // create softmax primitive attr
        dnnl::primitive_attr sub_softmax_attr
                = make_primitive_attr(sdp_op[2], mgr);
        sub_softmax_dst_md
                = memory::desc(sub_mm1_dst_dims, dt_src_user, tag::abcd);
        auto sub_softmax_pd = softmax_forward::primitive_desc(p_engine,
                prop_kind::forward_inference, algorithm::softmax_accurate,
                sub_mm1_dst_md, sub_softmax_dst_md,
                sub_mm1_dst_md.get_ndims() - 1, sub_softmax_attr);
        sub_softmax_prim = softmax_forward(sub_softmax_pd);

        // reorder u8->s8 wei for second matmul
        // create reorder2 primitive attr
        dnnl::primitive_attr sub_reorder2_attr
                = make_primitive_attr(sdp_op[3], mgr);
        memory::dims sub_wei2_dims = {1, 1, seq_len_kv, size_per_head};
        wei2_strides = ltw(inputs[graph_inport[4]]).vstrides();
        sub_wei2_user_md = memory::desc(sub_wei2_dims, dt_wei_user,
                {1, 1, wei2_strides[2], wei2_strides[3]});
        // The format is `abcd` due to performance of reorder to `abdc` is low.
        auto sub_wei2_md = memory::desc(sub_wei2_dims, dt_wei, tag::abcd);
        auto sub_reorder2_pd = reorder::primitive_desc(p_engine,
                sub_wei2_user_md, p_engine, sub_wei2_md, sub_reorder2_attr);
        sub_reorder2.init(sub_reorder2_pd);

        // second matmul
        // create second matmul primitive attr
        dnnl::primitive_attr sub_matmul2_attr
                = make_primitive_attr(sdp_op[4], mgr);
        memory::dims sub_mm2_src_dims = {1, 1, seq_len_q, seq_len_kv};
        memory::dims sub_mm2_wei_dims = {1, 1, seq_len_kv, size_per_head};
        memory::dims sub_mm2_dst_dims = {1, 1, seq_len_q, size_per_head};
        auto sub_mm2_src_md
                = memory::desc(sub_mm2_src_dims, dt_src_user, tag::abcd);
        sub_mm2_wei_md = memory::desc(sub_mm2_wei_dims, dt_wei, tag::abcd);
        sub_mm2_dst_md = memory::desc(sub_mm2_dst_dims, dt_src_user, tag::abcd);
        auto sub_mm2_pd = matmul::primitive_desc(p_engine, sub_mm2_src_md,
                sub_mm2_wei_md, sub_mm2_dst_md, sub_matmul2_attr);
        sub_mm2_prim = matmul(sub_mm2_pd);

        // per-head: reorder dst2 from dense to strided
        primitive_attr sub_reorder3_attr;
        sub_reorder3_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        memory::dims sub_dst_dims = {1, 1, seq_len_q, size_per_head};
        auto out_lt = sdp_op[4]->get_output_value(0)->get_logical_tensor();
        dst_strides = ltw(out_lt).vstrides();
        sub_dst_md = memory::desc(sub_dst_dims, dt_src_user, tag::abcd);
        sub_dst_user_md = memory::desc(sub_dst_dims, dt_src_user,
                {1, 1, dst_strides[2], dst_strides[3]});
        auto sub_reorder3_pd = reorder::primitive_desc(p_engine, sub_dst_md,
                p_engine, sub_dst_user_md, sub_reorder3_attr);
        sub_reorder3.init(sub_reorder3_pd);
        ////////////////////////////////////////////////////////////////////////
        /////////////// End Creating primitives ////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        /////////////// Start Constructing exec args ///////////////////////////
        ////////////////////////////////////////////////////////////////////////
        memory::desc max_scratchpad_md, sub_max_src1_src2_md,
                sub_max_dst1_wei2_md;
        size_t max_scratchpad_size = 0;
        // all the scratchpads required by the primitives.
        const std::vector<memory::desc> scratchpads {
                sub_reorder0_pd.scratchpad_desc(),
                sub_reorder1_pd.scratchpad_desc(), sub_mm1_pd.scratchpad_desc(),
                sub_softmax_pd.scratchpad_desc(),
                sub_reorder2_pd.scratchpad_desc(), sub_mm2_pd.scratchpad_desc(),
                sub_reorder3_pd.scratchpad_desc()};

        for (auto &sp : scratchpads) {
            const size_t size = sp.get_size();
            if (size > max_scratchpad_size) {
                max_scratchpad_size = size;
                max_scratchpad_md = sp;
            }
        }

        auto sub_src1_size = sub_src1_d_md.get_size();
        auto sub_src2_size = sub_softmax_dst_md.get_size();
        sub_max_src1_src2_md = sub_src1_size > sub_src2_size
                ? sub_src1_d_md
                : sub_softmax_dst_md;

        auto sub_dst1_size = sub_mm1_dst_md.get_size();
        auto sub_wei2_size = sub_mm2_wei_md.get_size();
        sub_max_dst1_wei2_md = sub_dst1_size > sub_wei2_size ? sub_mm1_dst_md
                                                             : sub_mm2_wei_md;

        // Initialize memory object with empty buffer
        sub_max_src1_src2 = memory(sub_max_src1_src2_md, p_engine, nullptr);
        sub_max_dst1_wei2 = memory(sub_max_dst1_wei2_md, p_engine, nullptr);
        // reorder0: 2d strided -> 2d ab
        sub_src1 = memory(sub_src1_md, p_engine, nullptr);
        // reorder1: 2d strided u8 -> 2d ba s8
        sub_wei1_user = memory(sub_wei1_user_md, p_engine, nullptr);
        // mm1
        sub_mm1_src = memory(sub_mm1_src_md, p_engine, nullptr);
        sub_mm1_wei = memory(sub_mm1_wei_md, p_engine, nullptr);
        sub_mm1_dst = memory(sub_mm1_dst_md, p_engine, nullptr);
        for (size_t i = 0; i < sub_mm1_post_md.size(); i++) {
            sub_mm1_post_mem.emplace_back(
                    memory(sub_mm1_post_md[i], p_engine, nullptr));
        }
        // softmax
        sub_softmax_dst = memory(sub_softmax_dst_md, p_engine, nullptr);
        // reorder2
        sub_wei2_user = memory(sub_wei2_user_md, p_engine, nullptr);
        // mm2
        sub_mm2_wei = memory(sub_mm2_wei_md, p_engine, nullptr);
        sub_mm2_dst = memory(sub_mm2_dst_md, p_engine, nullptr);
        //reorder3
        sub_dst_user = memory(sub_dst_user_md, p_engine, nullptr);

        // scratchpad, each thread will have a largest scratchpad.
        sub_scratchpad = memory(max_scratchpad_md, p_engine, nullptr);

        sub_reorder0_args
                = {{DNNL_ARG_SRC, sub_src1}, {DNNL_ARG_DST, sub_mm1_src},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder1_args
                = {{DNNL_ARG_SRC, sub_wei1_user}, {DNNL_ARG_DST, sub_mm1_wei},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_mm1_args = {{DNNL_ARG_SRC, sub_mm1_src},
                {DNNL_ARG_WEIGHTS, sub_mm1_wei}, {DNNL_ARG_DST, sub_mm1_dst},
                {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
        for (int i = 0; i < (int)sub_mm1_post_mem.size(); i++) {
            sub_mm1_args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                            sub_mm1_post_mem[i]});
        }

        sub_softmax_args
                = {{DNNL_ARG_SRC, sub_mm1_dst}, {DNNL_ARG_DST, sub_softmax_dst},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder2_args
                = {{DNNL_ARG_SRC, sub_wei2_user}, {DNNL_ARG_DST, sub_mm2_wei},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_mm2_args = {{DNNL_ARG_SRC, sub_softmax_dst},
                {DNNL_ARG_WEIGHTS, sub_mm2_wei}, {DNNL_ARG_DST, sub_mm2_dst},
                {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder3_args
                = {{DNNL_ARG_SRC, sub_mm2_dst}, {DNNL_ARG_DST, sub_dst_user},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        // add scales and zps for mm1, softmax, mm2
        prepare_sdp_scales_zps(mgr, sdp_op[0], 1, sub_reorder1_args, p_engine);
        prepare_sdp_scales_zps(mgr, sdp_op[1], 2, sub_mm1_args, p_engine);
        prepare_sdp_scales_zps(mgr, sdp_op[2], 1, sub_softmax_args, p_engine);
        prepare_sdp_scales_zps(mgr, sdp_op[3], 1, sub_reorder2_args, p_engine);
        prepare_sdp_scales_zps(mgr, sdp_op[4], 2, sub_mm2_args, p_engine);
        ////////////////////////////////////////////////////////////////////////
        /////////////// End Constructing exec args /////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        // memory planing for buffer sharing
        memory_planning(sdp_registry, p_engine);
        // TODO: remove this when primitive new API ready
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
        omp_set_num_threads(nthr);
#endif
        return status::success;
    }

    impl::status_t record_select_ops(std::shared_ptr<subgraph_t> &sg,
            std::vector<op_ptr> &select_out_ops) {

        //post scale isn't from select.
        //so the post binary number from select is post_op's size - 1
        const auto select_out_ops_size = sub_mm1_post_mem.size() - 1;
        select_out_ops.resize(select_out_ops_size);
        //sdp_op[1] is mm1.
        size_t input_size = sdp_op[1]->num_inputs();
        /*
            src wei   post_scale attn_mask* post_binary...(from select)
              \   \       /       /         /
               \   \     /     /         /
                 \  \   /   /        /
                   \ \ / /       /
                     mm1
        */
        // input_size - select_out_ops_size is the starting index of post ops
        // from select.
        for (size_t i = 0; i < select_out_ops_size; i++) {
            select_out_ops[i] = sdp_op[1]
                                        ->get_input_value(input_size
                                                - select_out_ops_size + i)
                                        ->get_producer()
                                        .shared_from_this();
        }

        const std::unordered_set<op_kind_t> select_kind
                = {op_kind::dnnl_eltwise, op_kind::dnnl_binary};
        return topo_order_visit(
                sg->get_output_ops(), [&select_kind, this](op_t *op) {
                    bool is_select = false;
                    if (select_kind.count(op->get_kind())) is_select = true;
                    if (op->get_kind() == op_kind::dnnl_reorder
                            || op->get_kind() == op_kind::dnnl_unsqueeze) {
                        auto post_op = get_post_op(op->shared_from_this());
                        if (post_op != nullptr
                                && select_kind.count(post_op->get_kind()))
                            is_select = true;
                    }
                    if (is_select)
                        this->select_op.emplace_back(op->shared_from_this());
                    return status::success;
                });
    }
    impl::status_t record_select_out_index(
            const std::shared_ptr<subgraph_t> &sg,
            const std::vector<op_ptr> &select_out_ops) {
        // select_outop_index is used to record the topo order index of output
        // ops from the new select subgraph. -1 means this array isn't
        // initialized.
        select_outop_index.resize(select_out_ops.size(), -1);
        int temp = 0;
        return topo_order_visit(
                sg->get_output_ops(), [&temp, this, &select_out_ops](op_t *op) {
                    for (size_t i = 0; i < select_out_ops.size(); i++) {
                        if (select_out_ops[i].get() == op) {
                            select_outop_index[i] = temp;
                            break;
                        }
                    }
                    temp++;
                    return status::success;
                });
    }

private:
    op_ptr get_post_op(const op_ptr &op) const {
        const auto out_val = op->get_output_value(0);
        const auto &consumers = out_val->get_consumers();
        if (consumers.size() != 1) return nullptr;
        return consumers[0].get_op().shared_from_this();
    }

    impl::status_t record_input_offset(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs) {
        auto find_graph_inport = [&](std::shared_ptr<value_t> val) {
            // for quantized matmul, it has producer such as add_zp,sub_zp,mul_scale.
            if (val->get_consumers()[0].get_op().get_kind()
                    == graph::op_kind::MatMul) {
                while (val->has_producer()) {
                    val = val->get_producer().get_input_value(0);
                }
            }
            for (int i = 0; i < (int)inputs.size(); i++) {
                if (val->get_logical_tensor().id == inputs[i].id) { return i; }
            }
            // If the corresponding input is not found, return an invalid value
            return -1;
        };
        op_ptr mm1, mm2, scale, add, select;
        for (const auto &cur_op : sg->get_ops()) {
            if (mm1 != nullptr && mm2 != nullptr) break;
            if (cur_op->get_kind() != graph::op_kind::MatMul) continue;
            auto post_op = get_post_op(cur_op);
            if (post_op
                    && (post_op->get_kind() == graph::op_kind::Divide
                            || post_op->get_kind()
                                    == graph::op_kind::Multiply)) {
                mm1 = cur_op;
                scale = post_op;
                const auto pop = get_post_op(post_op);
                if (pop->get_kind() == graph::op_kind::Add) {
                    add = pop;
                    attention_mask = true;
                } else if (pop->get_kind() == graph::op_kind::Select) {
                    select = pop;
                    has_select = true;
                } else {
                    add = nullptr;
                    select = nullptr;
                }
            } else if (post_op
                    && (post_op->get_kind() == graph::op_kind::Select)) {
                return status::unimplemented;
            } else
                mm2 = cur_op;
        }
        if (impl::utils::one_of(nullptr, mm1, mm2))
            return status::invalid_graph;

        int src1_id = find_graph_inport(mm1->get_input_value(0));
        graph_inport.emplace_back(src1_id);
        int wei1_id = find_graph_inport(mm1->get_input_value(1));
        graph_inport.emplace_back(wei1_id);
        // for scale and add op. The input order is uncertain.
        int scale_id = find_graph_inport(scale->get_input_value(1));
        if (scale_id == -1)
            scale_id = find_graph_inport(scale->get_input_value(0));
        graph_inport.emplace_back(scale_id);
        if (add) {
            int add_id = find_graph_inport(add->get_input_value(1));
            if (add_id == -1)
                add_id = find_graph_inport(add->get_input_value(0));
            graph_inport.emplace_back(add_id);
        } else {
            //placeholder
            graph_inport.emplace_back(-1);
        }
        int wei2_id = find_graph_inport(mm2->get_input_value(1));
        graph_inport.emplace_back(wei2_id);
        if (select) {
            int cond_id = find_graph_inport(select->get_input_value(0));
            int src0_id = find_graph_inport(select->get_input_value(1));
            graph_inport.emplace_back(cond_id);
            graph_inport.emplace_back(src0_id);
        } else {
            //placeholder
            graph_inport.emplace_back(-1);
            graph_inport.emplace_back(-1);
        }
        return status::success;
    }

    impl::status_t record_sdp_ops(
            std::shared_ptr<subgraph_t> &sg, bool is_quantize) {
        const auto get_wei_pre_op = [](const op_ptr &op) -> op_ptr {
            const auto out_val = op->get_input_value(1);
            if (out_val->has_producer()) {
                auto &producer = out_val->get_producer();
                if (producer.get_kind() != op_kind::dnnl_reorder)
                    return nullptr;
                return producer.shared_from_this();
            } else
                return nullptr;
        };

        subgraph_rewriter_t rewriter(sg);

        for (const auto &cur_op : sg->get_ops()) {
            if (!cur_op || cur_op->get_kind() != op_kind::dnnl_matmul) continue;
            auto post_op = get_post_op(cur_op);
            if (!post_op || post_op->get_kind() != op_kind::dnnl_softmax)
                continue;
            auto ppost_op = get_post_op(post_op);
            if (!ppost_op) return status::invalid_graph;

            op_ptr reorder1;
            op_ptr reorder2;
            if (is_quantize) {
                reorder1 = get_wei_pre_op(cur_op);
                reorder2 = get_wei_pre_op(ppost_op);
            }

            this->sdp_op = {reorder1, cur_op, post_op, reorder2, ppost_op};
            break;
        }
        return status::success;
    }

    void memory_planning(registry_t &sdp_registry, dnnl::engine p_engine) {
        // Registry is used to do the memory planning for sdp decompostion
        // algorithm. We reused some internal memory to reduce the memory
        // footprint for better cache hit. And here the key in registar of each
        // memory is planned in a specific order.
        registrar_t temporary_registrar = sdp_registry.registrar();

        // Here we initialize the map based on certain memory reuse logic. Those
        // memories(mds) who share the same buffer have the same registar key in
        // this map. So if we want to change the memory reuse logic, we need to
        // change the value of map here.
        mem_key_map = {{sub_max_src1_src2.get(), 0}, {sub_mm1_wei.get(), 1},
                {sub_max_dst1_wei2.get(), 2}, {sub_softmax_dst.get(), 0},
                {sub_mm2_dst.get(), 3}, {sub_scratchpad.get(), 4}};

        temporary_registrar.book(mem_key_map[sub_max_src1_src2.get()],
                sub_max_src1_src2.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_mm1_wei.get()],
                sub_mm1_wei.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_max_dst1_wei2.get()],
                sub_max_dst1_wei2.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_mm2_dst.get()],
                sub_mm2_dst.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_scratchpad.get()],
                sub_scratchpad.get_desc().get_size());
    }

    impl::status_t prepare_sdp_scales_zps(const fusion_info_mgr_t &mgr,
            std::shared_ptr<op_t> &op, int index,
            std::unordered_map<int, memory> &args,
            const dnnl::engine &p_engine) {
        const auto dt_scale = memory::data_type::f32,
                   dt_zp = memory::data_type::s32;
        // scale zp order:
        // 1. src scale, wei scale
        // 2. src zp, wei zp
        // 3. dst scale, dst zp
        if (op && op->has_attr(op_attr::fusion_info_key)
                && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info_t fusion_info = mgr.get_info(key);
            if (fusion_info.with_runtime_scales(true, 0)) {
                memory::desc sub_src_scale_md
                        = memory::desc({1}, dt_scale, tag::x);
                memory sub_src_scale = memory(sub_src_scale_md, p_engine);
                float *src_scale_val_ptr = reinterpret_cast<float *>(
                        sub_src_scale.get_data_handle());
                src_scale_val_ptr[0] = get_attr_value<float, float>(
                        op, index++, op_attr::scales);

                args.insert(
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sub_src_scale});
            }
            if (fusion_info.with_runtime_scales(true, 1)) {
                memory::desc sub_wei_scale_md
                        = memory::desc({1}, dt_scale, tag::x);
                memory sub_wei_scale = memory(sub_wei_scale_md, p_engine);
                float *wei_scale_val_ptr = reinterpret_cast<float *>(
                        sub_wei_scale.get_data_handle());
                wei_scale_val_ptr[0] = get_attr_value<float, float>(
                        op, index++, op_attr::scales);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                        sub_wei_scale});
            }

            // src_zp and wei_zp
            if (fusion_info.with_runtime_zero_points(true, 0)) {
                memory::desc sub_src_zp_md = memory::desc({1}, dt_zp, tag::x);
                memory sub_src_zp = memory(sub_src_zp_md, p_engine);
                int *src_zp_val_ptr
                        = reinterpret_cast<int *>(sub_src_zp.get_data_handle());
                src_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                        op, index++, op_attr::zps);
                args.insert(
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, sub_src_zp});
            }
            if (fusion_info.with_runtime_zero_points(true, 1)) {
                memory::desc sub_wei_zp_md = memory::desc({1}, dt_zp, tag::x);
                memory sub_wei_zp = memory(sub_wei_zp_md, p_engine);
                int *wei_zp_val_ptr
                        = reinterpret_cast<int *>(sub_wei_zp.get_data_handle());
                wei_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                        op, index++, op_attr::zps);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                        sub_wei_zp});
            }

            // dst scale, dst zp
            if (fusion_info.with_runtime_scales(false, 0)) {
                memory::desc sub_dst_scale_md
                        = memory::desc({1}, dt_scale, tag::x);
                memory sub_dst_scale = memory(sub_dst_scale_md, p_engine);
                float *dst_scale_val_ptr = reinterpret_cast<float *>(
                        sub_dst_scale.get_data_handle());
                dst_scale_val_ptr[0] = get_attr_value<float, float>(
                        op, index++, op_attr::scales);
                args.insert(
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, sub_dst_scale});
            }
            if (fusion_info.with_runtime_zero_points(false, 0)) {
                memory::desc sub_dst_zp_md = memory::desc({1}, dt_zp, tag::x);
                memory sub_dst_zp = memory(sub_dst_zp_md, p_engine);
                int *dst_zp_val_ptr
                        = reinterpret_cast<int *>(sub_dst_zp.get_data_handle());
                dst_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                        op, index++, op_attr::zps);
                args.insert(
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_dst_zp});
            }
        }
        if (op && op->get_kind() == op_kind::dnnl_reorder) {
            if (op->has_attr(op_attr::with_runtime_dst_zps)
                    && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
                memory::desc sub_dst_zp_md = memory::desc({1}, dt_zp, tag::x);
                memory sub_dst_zp = memory(sub_dst_zp_md, p_engine);
                int *dst_zp_val_ptr
                        = reinterpret_cast<int *>(sub_dst_zp.get_data_handle());
                dst_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                        op, index++, op_attr::zps);
                args.insert(
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_dst_zp});
            }
        }
        return status::success;
    }

    template <typename attr_dt, typename target_dt>
    target_dt get_attr_value(
            std::shared_ptr<op_t> &op, int i, op_attr_t attr_name) {
        const auto in_val = op->get_input_value(i);
        auto &producer = in_val->get_producer();
        return static_cast<target_dt>(
                producer.get_attr<std::vector<attr_dt>>(attr_name)[0]);
    }

    dnnl::primitive_attr make_primitive_attr(
            std::shared_ptr<op_t> &op, fusion_info_mgr_t &mgr) {
        fusion_info_t fusion_info;
        dnnl::primitive_attr attr;
        if (op && op->has_attr(op_attr::fusion_info_key)
                && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info = mgr.get_info(key);
            attr = make_dnnl_primitive_attr(op, fusion_info);
        }
        if (op && op->get_kind() == op_kind::dnnl_reorder) {
            // generate mask
            int mask = 0;
            if (op->has_attr(op_attr::axis) && op->has_attr(op_attr::qtype)) {
                int64_t axis = op->get_attr<int64_t>(op_attr::axis);
                std::string qtype = op->get_attr<std::string>(op_attr::qtype);
                mask = qtype == "per_tensor" ? 0 : 1 << axis;
            }

            if (op->has_attr(op_attr::with_runtime_dst_zps)
                    && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
                // runtime dst zps
                attr.set_zero_points_mask(DNNL_ARG_TO, mask);
            } else if (op->has_attr(op_attr::dst_zps)) {
                assertm(false, "only support runtime dst zero points.\n");
            }
        }
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        return attr;
    }
};

// The second template param dt is used to indicate the internal data type of
// int8 sdp pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
class sdp_decomp_kernel_t : public kernel_base_t {
private:
    allocator_t *g_alloc_ = nullptr;
    // used for sdp internal memory planning
    registry_t sdp_registry_;

    // we create 2 subgraph_ for graph which include select op The sdp part is
    // the first subgraph_ and the select part which didn't fused by sdp is the
    // second select_subgraph_. The sdp subgraph_ uses decompostion algorithm.
    // the select_subgraph_ uses sequential algorithm
    std::shared_ptr<subgraph_t> subgraph_;
    std::shared_ptr<subgraph_t> select_subgraph_;
    std::function<std::shared_ptr<execution_args_set_t>()>
            select_resource_ctor_;
    memory_planner_t memory_planner_;
    subgraph_visualizer_t vis_;

    // SDP-related params
    sdp_decomp_config_t sdp_cfg_;

public:
    sdp_decomp_kernel_t() {
        thread_local_cache_t<sdp_args_set_t> res_cache;
        res_cache.retain();

        thread_local_cache_t<execution_args_set_t> select_res_cache;
        select_res_cache.retain();
    }

    ~sdp_decomp_kernel_t() override {
        thread_local_cache_t<sdp_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();

        thread_local_cache_t<execution_args_set_t> select_res_cache;
        select_res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        select_res_cache.release();
    }

    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = reinterpret_cast<graph::allocator_t *>(
                g_engine->get_allocator());

        // get subgraph from the deep copied partition
        subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
                part->get_fpmath_mode(), part->get_use_blocked_layout(), true);
        BACKEND_DNNL_CHECK(
                set_given_inputs_outputs(subgraph_, inputs, outputs));

        // Check if it's supported by decompostion kernel
        if (!sdp_cfg_.initial_check(subgraph_, inputs))
            return status::unimplemented;

        subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pass_pipeline_t pipeline = pass_pipeline_t(vis);
        pass_pipeline_t select_pipeline = pass_pipeline_t(vis);
        BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
        // Fusion and canonicalization passes begin
        if (quantized) {
            BACKEND_DNNL_ADD_PASS(pipeline, lift_up_typecast);
            BACKEND_DNNL_ADD_PASS(pipeline, lift_up_quantize);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_typecast_to_matmul_or_conv);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_typecast_to_predecessor);
            BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_scales);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_scales);
            BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_zero_points);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_zero_points);
            BACKEND_DNNL_ADD_PASS(pipeline, insert_runtime_u8_to_s8_for_matmul);
        }
        BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
        BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);
        if (quantized) {
            BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_scales);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_scales);
            BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_zero_points);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_zero_points);
            BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
        }
        pipeline.reset_visualize_arg(true, false);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_transpose_to_matmul);
        BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

        // Run the added passes
        BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

        // fill information for inputs logical tensors
        for (size_t i = 0; i < inputs.size(); i++) {
            auto &in = const_cast<logical_tensor_t &>(inputs[i]);
            in = subgraph_->ins_[i];
        }

        // fill information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            auto &out = const_cast<logical_tensor_t &>(outputs[i]);
            out = subgraph_->outs_[i];
        }

        resource_ctor_
                = [this]() { return std::make_shared<sdp_args_set_t>(this); };

        // Initialize and construct kernel params
        sdp_cfg_.construct_params<quantized, dt>(
                subgraph_, sdp_registry_, p_engine_, inputs);

        // Create a new subgraph for select. the select_out_ops is the new
        // subgraph's output ops. The out values of these out_ops are the
        // connection between two graphs
        std::vector<op_ptr> select_out_ops;
        if (sdp_cfg_.has_select) {
            sdp_cfg_.record_select_ops(subgraph_, select_out_ops);
            select_subgraph_ = std::make_shared<subgraph_t>(sdp_cfg_.select_op,
                    p_engine_, part->get_fpmath_mode(),
                    part->get_use_blocked_layout(), false);

            const std::vector<logical_tensor_t> select_inputs
                    = {inputs[sdp_cfg_.graph_inport[5]],
                            inputs[sdp_cfg_.graph_inport[6]]};

            select_subgraph_->ins_ = select_inputs;
            BACKEND_DNNL_ADD_PASS(select_pipeline, replace_select_values);

            // do constant propagation again since layout propagation may
            // insert/delete operators
            if (enabled_constant_cache()) {
                BACKEND_DNNL_ADD_PASS(select_pipeline, constant_propagation);
            }

            // bind the memory for each op
            auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
                return memory_planner_.run(sg);
            };
            select_pipeline.reset_visualize_arg(true, true);
            BACKEND_DNNL_ADD_PASS(select_pipeline, memory_plan);
            BACKEND_DNNL_ADD_PASS(select_pipeline, compile_ops);

            BACKEND_DNNL_CHECK(select_pipeline.run(select_subgraph_));

            sdp_cfg_.record_select_out_index(select_subgraph_, select_out_ops);
            select_resource_ctor_ = [this]() {
                return this->memory_planner_.get_exec_args_set().clone();
            };
        }

        return status::success;
    }

    void prepare_sub_args(const grantor_t &var_grantor, const int id,
            const size_t block_size,
            std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map) {
        auto size_offset = id * block_size;
        mem_map[sdp_cfg_.sub_mm1_wei.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm1_wei.get()])
                + size_offset);
        // mm1
        mem_map[sdp_cfg_.sub_mm1_src.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
                + size_offset);
        mem_map[sdp_cfg_.sub_mm1_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
                + size_offset);
        // softmax
        mem_map[sdp_cfg_.sub_softmax_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
                + size_offset);
        // mm2
        mem_map[sdp_cfg_.sub_mm2_wei.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
                + size_offset);
        mem_map[sdp_cfg_.sub_mm2_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm2_dst.get()])
                + size_offset);
        // scratchpad, each thread will have a largest scratchpad.
        mem_map[sdp_cfg_.sub_scratchpad.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_scratchpad.get()])
                + size_offset);
    }

    void prepare_args_set(const execution_args_set_t *res,
            const std::vector<tensor_t> &inputs,
            const scratchpad_t &scratchpad) {
        // update the data of partition in/outputs args
        for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
            mem_idx.first.set_data_handle(
                    inputs[mem_idx.second].get_data_handle());
        }

        grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
                scratchpad.get_buffer());

        for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
            mem_offkey.first.set_data_handle(
                    var_grantor.get(mem_offkey.second));
        }
    }

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        dnnl::stream strm = make_dnnl_stream(p_engine_, *g_stream);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        auto *tp_stream
                = dnnl::impl::utils::downcast<dnnl::impl::cpu::cpu_stream_t *>(
                        const_cast<stream_t *>(g_stream));
        tp_stream->before_exec_hook();
        int thread_num = 1;
        dnnl_threadpool_interop_get_max_concurrency(&thread_num);
        sdp_cfg_.nthr = thread_num;
#endif

        // each thread's own local resource
        thread_local_cache_t<execution_args_set_t> select_res_cache;
        execution_args_set_t *select_res = nullptr;
        if (sdp_cfg_.has_select)
            select_res = select_res_cache.get_or_add(
                    reinterpret_cast<size_t>(this), select_resource_ctor_);
        thread_local_cache_t<sdp_args_set_t> res_cache;
        sdp_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        int MBO = sdp_cfg_.batch_size, MBI = sdp_cfg_.num_head;

        char *src1_user_pointer = static_cast<char *>(
                inputs[sdp_cfg_.graph_inport[0]].get_data_handle());
        char *wei1_user_pointer = static_cast<char *>(
                inputs[sdp_cfg_.graph_inport[1]].get_data_handle());
        char *wei2_user_pointer = static_cast<char *>(
                inputs[sdp_cfg_.graph_inport[4]].get_data_handle());
        char *dst2_user_pointer
                = static_cast<char *>(outputs[0].get_data_handle());

        // allocate the select internal memory
        temporary_scratchpad_t select_scratchpad(
                memory_planner_.total_internal_temporary_size(), p_engine_,
                *g_alloc_);
        assertm(select_scratchpad.size()
                        >= memory_planner_.total_internal_temporary_size(),
                "no enough scratchpad memory");
        if (sdp_cfg_.has_select) {
            const std::vector<tensor_t> select_inputs
                    = {inputs[sdp_cfg_.graph_inport[5]],
                            inputs[sdp_cfg_.graph_inport[6]]};
            prepare_args_set(select_res, select_inputs, select_scratchpad);
        }
        size_t block_size = sdp_registry_.size();
        temporary_scratchpad_t scratchpad(
                block_size * sdp_cfg_.nthr, p_engine_, *g_alloc_);
        assertm(scratchpad.size() >= sdp_registry_.size(),
                "no enough scratchpad memory");
        grantor_t var_grantor = sdp_registry_.grantor(scratchpad.get_buffer());

        const auto get_mem_dt_size = [](const memory &m) -> size_t {
            return memory::data_type_size(m.get_desc().get_data_type());
        };

        const auto loop = [&](int tid, int nthr, dim_t bo, dim_t bi) {
            // prepare execution args and allocate real memory
            prepare_sub_args(var_grantor, tid, block_size, res->mem_map);

            // reorder0
            auto &sub_src1_tid = res->mem_map[sdp_cfg_.sub_src1.get()][tid];
            // reorder1:
            auto &sub_wei1_user_tid
                    = res->mem_map[sdp_cfg_.sub_wei1_user.get()][tid];

            // matmul1
            auto &sub_mm1_post_scale_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_mem[0].get()][tid];
            sub_mm1_post_scale_tid.set_data_handle(
                    inputs[sdp_cfg_.graph_inport[2]].get_data_handle());

            //The first post_op is post_scale, so it starts from 1.
            size_t start_index = 1;
            if (sdp_cfg_.attention_mask) {
                auto &sub_mm1_post_add_tid
                        = res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                               .get()][tid];
                auto mask_input = inputs[sdp_cfg_.graph_inport[3]];
                auto mask_strides
                        = ltw(mask_input.get_logical_tensor()).vstrides();
                sub_mm1_post_add_tid.set_data_handle(
                        static_cast<char *>(mask_input.get_data_handle())
                        + bo * mask_strides[1]
                                * get_mem_dt_size(sub_mm1_post_add_tid));
            }
            if (sdp_cfg_.has_select) {
                //connect select_graph and sdp_graph
                for (size_t i = start_index;
                        i < sdp_cfg_.sub_mm1_post_mem.size(); i++) {
                    auto &sub_mm1_post_tid
                            = res->mem_map[sdp_cfg_.sub_mm1_post_mem[i].get()]
                                          [tid];
                    const auto &select_res_args = select_res->get_exec_args();
                    auto out_mem = select_res_args[sdp_cfg_.select_outop_index[i
                                                           - 1]]
                                           .at(DNNL_ARG_DST);
                    auto out_strides = out_mem.get_desc().get_strides();
                    sub_mm1_post_tid.set_data_handle(
                            static_cast<char *>(out_mem.get_data_handle())
                            + bo * out_strides[0]
                                    * get_mem_dt_size(sub_mm1_post_tid));
                }
            }
            // reorder2:
            auto &sub_wei2_user_tid
                    = res->mem_map[sdp_cfg_.sub_wei2_user.get()][tid];

            //reorder3
            auto &sub_dst_user_tid
                    = res->mem_map[sdp_cfg_.sub_dst_user.get()][tid];

            // matmul2
            auto &sub_mm2_dst_tid
                    = res->mem_map[sdp_cfg_.sub_mm2_dst.get()][tid];

            const size_t sub_src1_offset
                    = (bo * sdp_cfg_.src1_strides[0]
                              + bi * sdp_cfg_.src1_strides[1])
                    * get_mem_dt_size(sub_src1_tid);
            const size_t sub_wei1_offset
                    = (bo * sdp_cfg_.wei1_strides[0]
                              + bi * sdp_cfg_.wei1_strides[1])
                    * get_mem_dt_size(sub_wei1_user_tid);
            const size_t sub_wei2_offset
                    = (bo * sdp_cfg_.wei2_strides[0]
                              + bi * sdp_cfg_.wei2_strides[1])
                    * get_mem_dt_size(sub_wei2_user_tid);
            const size_t sub_dst_user_offset
                    = (bo * sdp_cfg_.dst_strides[0]
                              + bi * sdp_cfg_.dst_strides[1])
                    * get_mem_dt_size(sub_dst_user_tid);

            sub_wei1_user_tid.set_data_handle(
                    wei1_user_pointer + sub_wei1_offset);
            sub_src1_tid.set_data_handle(src1_user_pointer + sub_src1_offset);
            sub_wei2_user_tid.set_data_handle(
                    wei2_user_pointer + sub_wei2_offset);
            sub_dst_user_tid.set_data_handle(
                    dst2_user_pointer + sub_dst_user_offset);

            // If the last reorder is inplace, it means we don't have to do
            // extra reorder, thus we should set matmul's output to the user's
            // output directly.
            if (sdp_cfg_.sub_reorder3.get_inplace()) {
                sub_mm2_dst_tid.set_data_handle(
                        dst2_user_pointer + sub_dst_user_offset);
            }

            // in parallel region - these primitives should use single thread.
            sdp_cfg_.sub_reorder0.execute(strm, res->sub_reorder0_args[tid]);
            sdp_cfg_.sub_reorder1.execute(strm, res->sub_reorder1_args[tid]);
            sdp_cfg_.sub_mm1_prim.execute(strm, res->sub_mm1_args[tid]);

            sdp_cfg_.sub_softmax_prim.execute(strm, res->sub_softmax_args[tid]);

            sdp_cfg_.sub_reorder2.execute(strm, res->sub_reorder2_args[tid]);

            sdp_cfg_.sub_mm2_prim.execute(strm, res->sub_mm2_args[tid]);
            sdp_cfg_.sub_reorder3.execute(strm, res->sub_reorder3_args[tid]);
        };
        if (sdp_cfg_.has_select) {
            for (size_t i = 0; i < select_subgraph_->execs_.size(); i++) {
                select_subgraph_->execs_[i]->execute(
                        strm, select_res->get_exec_args()[i]);
            }
        }
        parallel_nd_ext(sdp_cfg_.nthr, MBO, MBI, loop);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        tp_stream->after_exec_hook();
#endif
        return status::success;
    }

    class sdp_args_set_t {
    public:
        sdp_args_set_t(sdp_decomp_kernel_t<quantized, dt> *sdp_kernel) {
            int nthr = sdp_kernel->sdp_cfg_.nthr;
            //consrtuct new args
            auto args_ctor = [this, nthr](const std::unordered_map<int, memory>
                                                  &ori_args,
                                     std::vector<std::unordered_map<int,
                                             memory>> &args) {
                args.resize(nthr);
                for (auto iter : ori_args) {
                    memory ori_mem = iter.second;
                    if (mem_map.count(ori_mem.get()) == 0) {
                        //consrtuct new memorys
                        mem_map[ori_mem.get()] = std::vector<memory>(nthr);
                        for (int tid = 0; tid < nthr; tid++) {
                            mem_map[ori_mem.get()][tid]
                                    = memory(ori_mem.get_desc(),
                                            ori_mem.get_engine(), nullptr);
                            if (iter.first >= DNNL_ARG_ATTR_SCALES
                                    && iter.first <= DNNL_ARG_ATTR_POST_OP_DW) {
                                mem_map[ori_mem.get()][tid].set_data_handle(
                                        ori_mem.get_data_handle());
                            }
                        }
                    }
                    for (int tid = 0; tid < nthr; tid++) {
                        args[tid].insert(
                                {iter.first, mem_map[ori_mem.get()][tid]});
                    }
                }
            };
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder0_args, sub_reorder0_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder1_args, sub_reorder1_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_mm1_args, sub_mm1_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_softmax_args, sub_softmax_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder2_args, sub_reorder2_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_mm2_args, sub_mm2_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder3_args, sub_reorder3_args);
        }
        std::unordered_map<dnnl_memory_t, std::vector<memory>> mem_map;
        // execution args for each op in the subgraph
        std::vector<std::unordered_map<int, memory>> sub_reorder0_args,
                sub_reorder1_args, sub_mm1_args, sub_softmax_args,
                sub_reorder2_args, sub_mm2_args, sub_reorder3_args;
    };

    std::function<std::shared_ptr<sdp_args_set_t>()> resource_ctor_;

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        UNUSED(g_stream);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(sycl_deps);
        UNUSED(sycl_event);
        return status::unimplemented;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &cl_deps,
            cl_event *ret_event) override {
        UNUSED(g_stream);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(cl_deps);
        UNUSED(ret_event);
        return status::unimplemented;
    }
#endif
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
