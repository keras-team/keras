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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_MQA_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_MQA_HPP

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

class mqa_reorder {
public:
    status_t init(const dnnl::reorder::primitive_desc &pd) {
        is_inplace_ = pd.src_desc() == pd.dst_desc();
        reorder_prim_ = reorder(pd);
        return status::success;
    }
    status_t execute(const stream &astream,
            const std::unordered_map<int, memory> &args) const {
        // If the src and dst are the same, we just set the src arg to dst
        // directly instead of the real exection.
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
struct mqa_decomp_config_t {
public:
    mqa_decomp_config_t() = default;

    // MQA input dimension
    memory::dim batch_size, num_head, seq_len, size_per_head;

    // Thread nums during the workflow
    int nthr;

    // Used to record the exact input offset of the MQA subgraph
    // [mm1_src, mm1_wei, mm1_add, mm2_src]
    std::vector<int> graph_inport;

    // Primitives that actually perform calculations
    primitive sub_mm1_prim, sub_softmax_prim, sub_mm2_prim;
    mqa_reorder sub_reorder0, sub_reorder1, sub_reorder2, sub_reorder3;

    // Args used in the execution of primitives
    std::unordered_map<int, memory> sub_reorder0_args, sub_reorder1_args,
            sub_mm1_args, sub_softmax_args, sub_reorder2_args, sub_mm2_args,
            sub_reorder3_args;

    // A map from memory to registry key, used to record the internal memories
    // location inside of the whole buffer.
    std::unordered_map<dnnl_memory_t, registry_key> mem_key_map;

    /// Internal memory objects for each primitive in each threads.
    // reorder0
    memory sub_src1;
    // reorder1
    memory sub_wei1_user;
    //mm1
    memory sub_mm1_src, sub_mm1_wei, sub_mm1_dst, sub_mm1_post_add;
    //softmax
    memory sub_softmax_dst;
    //reorder2
    memory sub_src2_user;
    //mm2
    memory sub_mm2_src, sub_mm2_dst;
    //reorder3
    memory sub_dst_user;
    //scratchped
    memory sub_scratchpad;
    // shared memory
    memory sub_max_src1_src2, sub_max_dst1_dst2;

private:
    // Used to record the ops contained in MQA
    std::vector<std::shared_ptr<op_t>> mqa_op;

public:
    // The function is used to check if the configuration of MQA is supported by
    // current implementation of decomp kernel. Currently, this implementation
    // can handle 3-dims tensor and limits the numerical relationship between
    // batch_size, num_head and thread num.
    // If the check passes, initialize few members according to inputs
    // If no, return unimplemented status directly and fallback to large kernel
    bool initial_check(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs) {
        // The order of input logical tensors in inputs is not certain, we need
        // to record the input offset in a certain order of ops.
        record_input_offset(sg, inputs);

        // Key(3-dims): batch_size * seq_len * size_per_head
        memory::dims src1_user_dims = ltw(inputs[graph_inport[0]]).vdims();
        // Query(3-dims): batch_size * size_per_head * (num_head * seq_len)
        memory::dims wei1_user_dims = ltw(inputs[graph_inport[1]]).vdims();
        if (src1_user_dims.size() != 3 || wei1_user_dims.size() != 3)
            return false;

        // Initialize MQA input dimension according to the src of mm1
        batch_size = src1_user_dims[0];
        seq_len = src1_user_dims[1];
        size_per_head = src1_user_dims[2];
        num_head = wei1_user_dims[2] / seq_len;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
// RATIO is an empirical value used to determine the numerical relationship
// between batch_size, num_head and thread number to determine whether to use
// decompose kernel. The key to the decompose kernel is that we do parallel in
// the batch_size and num_head dimensions. Therefore, if the batch_size or
// num_head is too small, it will cause many idle threads and affect efficiency
// which may even worse than the original sequential kernel. Here we set this
// ratio based on the experimental value to ensure that users do not have any
// regression when using the decompose kernel.
// TODO: Refine the inequation based on the relationship of cache size and mqa
// memory footprint requirements.
#define RATIO 2
        // Initialize nthr with current threads num
        nthr = dnnl_get_current_num_threads();
        return batch_size * num_head > RATIO * nthr;
#else
        return true;
#endif
    }

    // Used to construct all params that MQA need
    template <bool quantized = false,
            memory::data_type dt = memory::data_type::f32>
    impl::status_t construct_params(std::shared_ptr<subgraph_t> &sg,
            registry_t &mqa_registry, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs) {

        // Record the ops inside of MQA pattern in a specific order.
        record_mqa_ops(sg);

        // Acquire the data type from input param for later primitive creation.
        // The src and wei dt of both quantized mqa and float mqa are the same.
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
                sub_mm1_wei_md, sub_mm1_dst_md, sub_mm1_post_add_md,
                sub_softmax_dst_md, sub_src2_user_md, sub_mm2_src_md,
                sub_mm2_dst_md, sub_dst_md, sub_dst_user_md;

        // must use user mode to support concurrent execution
        primitive_attr sub_reorder0_attr;
        sub_reorder0_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        // per-head: reorder src1 to dense, for first matmul
        memory::dims sub_src1_dims = {1, seq_len, size_per_head};
        sub_src1_md = memory::desc(
                sub_src1_dims, dt_src_user, {1, size_per_head, 1});
        auto sub_src1_d_md = memory::desc(sub_src1_dims, dt_src_user, tag::abc);
        auto sub_reorder0_pd = reorder::primitive_desc(p_engine, sub_src1_md,
                p_engine, sub_src1_d_md, sub_reorder0_attr);
        sub_reorder0.init(sub_reorder0_pd);

        auto &mgr = sg->fusion_info_mgr_;

        // per-head: reorder wei1 to dense, first matmul
        // create reorder1 primitive attr
        auto original_reorder1 = mqa_op[0];
        dnnl::primitive_attr sub_reorder1_attr
                = make_primitive_attr(original_reorder1, mgr);
        memory::dims sub_wei1_dims = {1, size_per_head, seq_len};

        auto original_matmul1 = mqa_op[1];
        auto wei_md = make_dnnl_memory_desc(
                original_matmul1->get_input_value(1)->get_logical_tensor());
        sub_wei1_user_md = memory::desc(
                sub_wei1_dims, dt_wei_user, {1, seq_len * num_head, 1});
        // Flip the format to have `ba` weights MBI item in per thread loop.
        sub_wei1_md = memory::desc(sub_wei1_dims, dt_wei, tag::abc);
        auto sub_reorder1_pd = reorder::primitive_desc(p_engine,
                sub_wei1_user_md, p_engine, sub_wei1_md, sub_reorder1_attr);
        sub_reorder1.init(sub_reorder1_pd);

        // first matmul
        // create first matmul primitive attr
        dnnl::primitive_attr sub_matmul1_attr
                = make_primitive_attr(original_matmul1, mgr);
        memory::dims sub_mm1_src_dims = {1, seq_len, size_per_head};
        memory::dims sub_mm1_wei_dims = {1, size_per_head, seq_len};
        memory::dims sub_mm1_dst_dims = {1, seq_len, seq_len};

        sub_mm1_src_md = memory::desc(sub_mm1_src_dims, dt_src_user, tag::abc);
        sub_mm1_wei_md = memory::desc(sub_mm1_wei_dims, dt_wei, tag::abc);
        sub_mm1_dst_md = memory::desc(sub_mm1_dst_dims, dt_inter, tag::abc);
        dnnl::post_ops dnnl_pops;
        auto mask_dt = static_cast<dnnl::memory::data_type>(
                ltw(inputs[graph_inport[2]]).data_type());
        sub_mm1_post_add_md
                = memory::desc({1, seq_len, seq_len}, mask_dt, tag::abc);
        auto ori_dnnl_pops = sub_matmul1_attr.get_post_ops();
        auto alg = static_cast<algorithm>(
                ori_dnnl_pops.get()->entry_[0].binary.alg);
        dnnl_pops.append_binary(alg, sub_mm1_post_add_md);
        sub_matmul1_attr.set_post_ops(std::move(dnnl_pops));
        auto sub_mm1_pd = matmul::primitive_desc(p_engine, sub_mm1_src_md,
                sub_mm1_wei_md, sub_mm1_dst_md, sub_matmul1_attr);
        sub_mm1_prim = matmul(sub_mm1_pd);

        // Here in the original graph, we have reshape and transpose op to
        // change the dimesion and layout of matmul's output. But with the
        // decompose kernel, no need to reshape or transpose the internal buffer.

        // softmax
        // create softmax primitive attr
        auto original_softmax = mqa_op[2];
        dnnl::primitive_attr sub_softmax_attr
                = make_primitive_attr(original_softmax, mgr);
        sub_softmax_dst_md
                = memory::desc(sub_mm1_dst_dims, dt_src_user, tag::abc);
        auto sub_softmax_pd = softmax_forward::primitive_desc(p_engine,
                prop_kind::forward_inference, algorithm::softmax_accurate,
                sub_mm1_dst_md, sub_softmax_dst_md,
                sub_mm1_dst_md.get_ndims() - 1, sub_softmax_attr);
        sub_softmax_prim = softmax_forward(sub_softmax_pd);

        // reorder src of second matmul (Value)
        // create reorder2 primitive attr
        auto original_reorder2 = mqa_op[3];
        dnnl::primitive_attr sub_reorder2_attr
                = make_primitive_attr(original_reorder2, mgr);
        memory::dims sub_src2_dims = {1, size_per_head, seq_len};
        sub_src2_user_md
                = memory::desc(sub_src2_dims, dt_src_user, {1, seq_len, 1});
        // The format is `abc` due to performance of reorder to `acb` is low.
        auto sub_src2_md = memory::desc(sub_src2_dims, dt_src_user, tag::abc);
        auto sub_reorder2_pd = reorder::primitive_desc(p_engine,
                sub_src2_user_md, p_engine, sub_src2_md, sub_reorder2_attr);
        sub_reorder2.init(sub_reorder2_pd);

        // second matmul
        // create second matmul primitive attr
        auto original_matmul2 = mqa_op[4];
        dnnl::primitive_attr sub_matmul2_attr
                = make_primitive_attr(original_matmul2, mgr);
        memory::dims sub_mm2_src_dims = {1, size_per_head, seq_len};
        memory::dims sub_mm2_wei_dims = {1, seq_len, seq_len};
        memory::dims sub_mm2_dst_dims = {1, size_per_head, seq_len};
        sub_mm2_src_md = memory::desc(sub_mm2_src_dims, dt_src_user, tag::abc);
        auto sub_mm2_wei_md
                = memory::desc(sub_mm2_wei_dims, dt_src_user, tag::abc);
        sub_mm2_dst_md = memory::desc(sub_mm2_dst_dims, dt_src_user, tag::abc);
        auto sub_mm2_pd = matmul::primitive_desc(p_engine, sub_mm2_src_md,
                sub_mm2_wei_md, sub_mm2_dst_md, sub_matmul2_attr);
        sub_mm2_prim = matmul(sub_mm2_pd);

        // per-head: reorder dst2 from dense to strided
        primitive_attr sub_reorder3_attr;
        sub_reorder3_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        memory::dims sub_dst_dims = {1, size_per_head, seq_len};
        sub_dst_md = memory::desc(sub_dst_dims, dt_src_user, tag::abc);
        sub_dst_user_md = memory::desc(
                sub_dst_dims, dt_src_user, {1, seq_len * num_head, 1});
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
                sub_max_dst1_dst2_md;
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
        auto sub_src2_size = sub_mm2_src_md.get_size();
        sub_max_src1_src2_md = sub_src1_size > sub_src2_size ? sub_src1_d_md
                                                             : sub_mm2_src_md;

        auto sub_dst1_size = sub_mm1_dst_md.get_size();
        auto sub_dst2_size = sub_mm2_dst_md.get_size();
        sub_max_dst1_dst2_md = sub_dst1_size > sub_dst2_size ? sub_mm1_dst_md
                                                             : sub_mm2_dst_md;

        // Initialize memory object with empty buffer
        sub_max_src1_src2 = memory(sub_max_src1_src2_md, p_engine, nullptr);
        sub_max_dst1_dst2 = memory(sub_max_dst1_dst2_md, p_engine, nullptr);
        // reorder0: 2d strided -> 2d ab
        sub_src1 = memory(sub_src1_md, p_engine, nullptr);
        // reorder1: 2d strided u8 -> 2d ba s8
        sub_wei1_user = memory(sub_wei1_user_md, p_engine, nullptr);
        // mm1
        sub_mm1_src = memory(sub_mm1_src_md, p_engine, nullptr);
        sub_mm1_wei = memory(sub_mm1_wei_md, p_engine, nullptr);
        sub_mm1_dst = memory(sub_mm1_dst_md, p_engine, nullptr);
        // sub_mm1_post_scale = memory(sub_mm1_post_scale_md, p_engine, nullptr);
        sub_mm1_post_add = memory(sub_mm1_post_add_md, p_engine, nullptr);
        // softmax
        sub_softmax_dst = memory(sub_softmax_dst_md, p_engine, nullptr);
        // reorder2
        sub_src2_user = memory(sub_src2_user_md, p_engine, nullptr);
        // mm2
        sub_mm2_src = memory(sub_mm2_src_md, p_engine, nullptr);
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
                {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                        sub_mm1_post_add},
                {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_softmax_args
                = {{DNNL_ARG_SRC, sub_mm1_dst}, {DNNL_ARG_DST, sub_softmax_dst},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder2_args
                = {{DNNL_ARG_SRC, sub_src2_user}, {DNNL_ARG_DST, sub_mm2_src},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_mm2_args = {{DNNL_ARG_SRC, sub_mm2_src},
                {DNNL_ARG_WEIGHTS, sub_softmax_dst},
                {DNNL_ARG_DST, sub_mm2_dst},
                {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder3_args
                = {{DNNL_ARG_SRC, sub_mm2_dst}, {DNNL_ARG_DST, sub_dst_user},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        ////////////////////////////////////////////////////////////////////////
        /////////////// End Constructing exec args /////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        // memory planing for buffer sharing
        memory_planning(mqa_registry, p_engine);
        return status::success;
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
        op_ptr mm1, mm2, add;
        for (const auto &cur_op : sg->get_ops()) {
            if (mm1 != nullptr && mm2 != nullptr) break;
            if (cur_op->get_kind() != graph::op_kind::MatMul) continue;
            auto post_op = get_post_op(cur_op);
            if (post_op != nullptr
                    && post_op->get_kind() == graph::op_kind::StaticReshape) {
                mm1 = cur_op;
                auto transpose = get_post_op(post_op);
                if (transpose != nullptr
                        && transpose->get_kind()
                                == graph::op_kind::StaticTranspose) {
                    add = get_post_op(transpose);
                }
            } else
                mm2 = cur_op;
        }
        if (impl::utils::one_of(nullptr, mm1, mm2, add))
            return status::invalid_graph;

        int src1_id = find_graph_inport(mm1->get_input_value(0));
        graph_inport.emplace_back(src1_id);
        int wei1_id = find_graph_inport(mm1->get_input_value(1));
        graph_inport.emplace_back(wei1_id);
        // for scale and add op. The input order is uncertain.
        int add_id = find_graph_inport(add->get_input_value(0));
        if (add_id == -1) add_id = find_graph_inport(add->get_input_value(1));
        graph_inport.emplace_back(add_id);

        int src2_id = find_graph_inport(mm2->get_input_value(0));
        graph_inport.emplace_back(src2_id);
        return status::success;
    }

    impl::status_t record_mqa_ops(std::shared_ptr<subgraph_t> &sg) {
        subgraph_rewriter_t rewriter(sg);
        op_ptr reorder1, reorder2, matmul1, softmax, matmul2;
        for (const auto &cur_op : sg->get_ops()) {
            if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
            if (get_post_op(cur_op) != nullptr) {
                matmul1 = cur_op;
                auto reshape = get_post_op(cur_op);
                auto transpose = get_post_op(reshape);
                softmax = get_post_op(transpose);
            } else {
                matmul2 = cur_op;
            }
        }
        this->mqa_op = {reorder1, matmul1, softmax, reorder2, matmul2};
        return status::success;
    }

    void memory_planning(registry_t &mqa_registry, dnnl::engine p_engine) {
        // Registry is used to do the memory planning for mqa decompostion
        // algorithm. We reused some internal memory to reduce the memory
        // footprint for better cache hit. And here the key in registar of each
        // memory is planned in a specific order.
        registrar_t temporary_registrar = mqa_registry.registrar();

        // Here we initialize the map based on certain memory reuse logic. Those
        // memories(mds) who share the same buffer have the same registar key in
        // this map. So if we want to change the memory reuse logic, we need to
        // change the value of map here.
        mem_key_map = {{sub_max_src1_src2.get(), 0}, {sub_mm1_wei.get(), 1},
                {sub_max_dst1_dst2.get(), 2}, {sub_softmax_dst.get(), 3},
                {sub_scratchpad.get(), 4}};

        temporary_registrar.book(mem_key_map[sub_max_src1_src2.get()],
                sub_max_src1_src2.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_mm1_wei.get()],
                sub_mm1_wei.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_max_dst1_dst2.get()],
                sub_max_dst1_dst2.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_softmax_dst.get()],
                sub_softmax_dst.get_desc().get_size());
        temporary_registrar.book(mem_key_map[sub_scratchpad.get()],
                sub_scratchpad.get_desc().get_size());
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
// int8 mqa pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
class mqa_decomp_kernel_t : public kernel_base_t {
private:
    allocator_t *g_alloc_ = nullptr;
    // used for mqa internal memory planning
    registry_t mqa_registry_;
    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;
    subgraph_visualizer_t vis_;

    // MQA-related params
    mqa_decomp_config_t mqa_cfg_;

public:
    mqa_decomp_kernel_t() {
        thread_local_cache_t<mqa_args_set_t> res_cache;
        res_cache.retain();
    }

    ~mqa_decomp_kernel_t() override {
        thread_local_cache_t<mqa_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();
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
        if (!mqa_cfg_.initial_check(subgraph_, inputs))
            return status::unimplemented;

        subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pass_pipeline_t pipeline = pass_pipeline_t(vis);

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
        // MQA pattern fusion
        BACKEND_DNNL_ADD_PASS(pipeline, lift_up_post_add_for_matmul);

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
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_transpose_to_matmul);
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
                = [this]() { return std::make_shared<mqa_args_set_t>(this); };

        // Initialize and construct kernel params
        mqa_cfg_.construct_params<quantized, dt>(
                subgraph_, mqa_registry_, p_engine_, inputs);

        return status::success;
    }

    void prepare_sub_args(const grantor_t &var_grantor, const int id,
            const size_t block_size,
            std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map) {
        auto size_offset = id * block_size;
        mem_map[mqa_cfg_.sub_mm1_wei.get()][id].set_data_handle(
                var_grantor.get(
                        mqa_cfg_.mem_key_map[mqa_cfg_.sub_mm1_wei.get()])
                + size_offset);
        // mm1
        mem_map[mqa_cfg_.sub_mm1_src.get()][id].set_data_handle(
                var_grantor.get(
                        mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_src1_src2.get()])
                + size_offset);
        mem_map[mqa_cfg_.sub_mm1_dst.get()][id].set_data_handle(
                var_grantor.get(
                        mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_dst1_dst2.get()])
                + size_offset);
        // softmax
        mem_map[mqa_cfg_.sub_softmax_dst.get()][id].set_data_handle(
                var_grantor.get(
                        mqa_cfg_.mem_key_map[mqa_cfg_.sub_softmax_dst.get()])
                + size_offset);
        // mm2
        mem_map[mqa_cfg_.sub_mm2_src.get()][id].set_data_handle(
                var_grantor.get(
                        mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_src1_src2.get()])
                + size_offset);
        mem_map[mqa_cfg_.sub_mm2_dst.get()][id].set_data_handle(
                var_grantor.get(
                        mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_dst1_dst2.get()])
                + size_offset);
        // scratchpad, each thread will have a largest scratchpad.
        mem_map[mqa_cfg_.sub_scratchpad.get()][id].set_data_handle(
                var_grantor.get(
                        mqa_cfg_.mem_key_map[mqa_cfg_.sub_scratchpad.get()])
                + size_offset);
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
        mqa_cfg_.nthr = thread_num;
#endif

        // each thread's own local resource
        thread_local_cache_t<mqa_args_set_t> res_cache;
        mqa_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        int MBO = mqa_cfg_.batch_size, MBI = mqa_cfg_.num_head,
            M1 = mqa_cfg_.seq_len, K1 = mqa_cfg_.size_per_head,
            N1 = mqa_cfg_.seq_len, M2 = mqa_cfg_.size_per_head,
            K2 = mqa_cfg_.seq_len, N2 = mqa_cfg_.seq_len;

        char *src1_user_pointer = static_cast<char *>(
                inputs[mqa_cfg_.graph_inport[0]].get_data_handle());
        char *wei1_user_pointer = static_cast<char *>(
                inputs[mqa_cfg_.graph_inport[1]].get_data_handle());
        char *post_add_user_pointer = static_cast<char *>(
                inputs[mqa_cfg_.graph_inport[2]].get_data_handle());
        char *src2_user_pointer = static_cast<char *>(
                inputs[mqa_cfg_.graph_inport[3]].get_data_handle());
        char *dst2_user_pointer
                = static_cast<char *>(outputs[0].get_data_handle());

        // allocate the internal memory
        size_t block_size = mqa_registry_.size();
        temporary_scratchpad_t scratchpad(
                block_size * mqa_cfg_.nthr, p_engine_, *g_alloc_);
        assertm(scratchpad.size() >= mqa_registry_.size(),
                "no enough scratchpad memory");
        grantor_t var_grantor = mqa_registry_.grantor(scratchpad.get_buffer());

        const auto get_mem_dt_size = [](const memory &m) -> size_t {
            return memory::data_type_size(m.get_desc().get_data_type());
        };

        const auto loop = [&](int tid, int nthr, dim_t bo, dim_t bi) {
            // prepare execution args and allocate real memory
            prepare_sub_args(var_grantor, tid, block_size, res->mem_map);

            // reorder0
            auto &sub_src1_tid = res->mem_map[mqa_cfg_.sub_src1.get()][tid];
            // reorder1:
            auto &sub_wei1_user_tid
                    = res->mem_map[mqa_cfg_.sub_wei1_user.get()][tid];

            auto &sub_mm1_post_add_tid
                    = res->mem_map[mqa_cfg_.sub_mm1_post_add.get()][tid];

            // reorder2:
            auto &sub_src2_user_tid
                    = res->mem_map[mqa_cfg_.sub_src2_user.get()][tid];

            //reorder3
            auto &sub_dst_user_tid
                    = res->mem_map[mqa_cfg_.sub_dst_user.get()][tid];

            const size_t sub_src1_offset
                    = bo * M1 * K1 * get_mem_dt_size(sub_src1_tid);
            const size_t sub_wei1_offset = (bo * MBI * K1 * N1 + bi * N1)
                    * get_mem_dt_size(sub_wei1_user_tid);
            const size_t sub_src2_offset
                    = bo * M2 * K2 * get_mem_dt_size(sub_src2_user_tid);
            const size_t sub_post_add_offset
                    = (bo * MBI * M1 * N1 + bi * M1 * N1)
                    * get_mem_dt_size(sub_mm1_post_add_tid);
            const size_t sub_dst_user_offset = (bo * MBI * M2 * N2 + bi * N2)
                    * get_mem_dt_size(sub_dst_user_tid);

            sub_wei1_user_tid.set_data_handle(
                    wei1_user_pointer + sub_wei1_offset);
            sub_src1_tid.set_data_handle(src1_user_pointer + sub_src1_offset);
            sub_src2_user_tid.set_data_handle(
                    src2_user_pointer + sub_src2_offset);
            sub_mm1_post_add_tid.set_data_handle(
                    post_add_user_pointer + sub_post_add_offset);
            sub_dst_user_tid.set_data_handle(
                    dst2_user_pointer + sub_dst_user_offset);

            // in parallel region - these primitives should use single thread.
            mqa_cfg_.sub_reorder0.execute(strm, res->sub_reorder0_args[tid]);
            mqa_cfg_.sub_reorder1.execute(strm, res->sub_reorder1_args[tid]);
            mqa_cfg_.sub_mm1_prim.execute(strm, res->sub_mm1_args[tid]);

            mqa_cfg_.sub_softmax_prim.execute(strm, res->sub_softmax_args[tid]);

            mqa_cfg_.sub_reorder2.execute(strm, res->sub_reorder2_args[tid]);

            mqa_cfg_.sub_mm2_prim.execute(strm, res->sub_mm2_args[tid]);
            mqa_cfg_.sub_reorder3.execute(strm, res->sub_reorder3_args[tid]);
        };
        // TODO: remove this when primitive new API ready
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
        omp_set_num_threads(mqa_cfg_.nthr);
#endif

        parallel_nd_ext(mqa_cfg_.nthr, MBO, MBI, loop);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        tp_stream->after_exec_hook();
#endif
        return status::success;
    }

    class mqa_args_set_t {
    public:
        mqa_args_set_t(mqa_decomp_kernel_t<quantized, dt> *mqa_kernel) {
            int nthr = mqa_kernel->mqa_cfg_.nthr;
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
                    mqa_kernel->mqa_cfg_.sub_reorder0_args, sub_reorder0_args);
            args_ctor(
                    mqa_kernel->mqa_cfg_.sub_reorder1_args, sub_reorder1_args);
            args_ctor(mqa_kernel->mqa_cfg_.sub_mm1_args, sub_mm1_args);
            args_ctor(mqa_kernel->mqa_cfg_.sub_softmax_args, sub_softmax_args);
            args_ctor(
                    mqa_kernel->mqa_cfg_.sub_reorder2_args, sub_reorder2_args);
            args_ctor(mqa_kernel->mqa_cfg_.sub_mm2_args, sub_mm2_args);
            args_ctor(
                    mqa_kernel->mqa_cfg_.sub_reorder3_args, sub_reorder3_args);
        }
        std::unordered_map<dnnl_memory_t, std::vector<memory>> mem_map;
        // execution args for each op in the subgraph
        std::vector<std::unordered_map<int, memory>> sub_reorder0_args,
                sub_reorder1_args, sub_mm1_args, sub_softmax_args,
                sub_reorder2_args, sub_mm2_args, sub_reorder3_args;
    };

    std::function<std::shared_ptr<mqa_args_set_t>()> resource_ctor_;

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
